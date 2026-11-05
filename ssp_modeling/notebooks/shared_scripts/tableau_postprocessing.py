"""
tableau_postprocessing.py
-------------------------
Python port of the R post-processing pipeline that produces Tableau-ready
CSVs from a SISEPUEDE run output, adapted for Morocco. Replaces:

  - ssp_modeling/output_postprocessing/scr/data_prep_new_mapping.r
  - ssp_modeling/output_postprocessing/scr/data_prep_drivers.r
  - ssp_modeling/output_postprocessing/levers_and_jobs_table/* (levers + jobs tables)

Usage from a manager notebook:

    from shared_scripts.tableau_postprocessing import run_tableau_postprocessing

    run_tableau_postprocessing(
        run_dir         = RUN_ID_OUTPUT_DIR_PATH,
        project_dir     = PROJECT_DIR,
        region          = "morocco",
        iso_code3       = "MAR",
        year_ref        = 2018,
    )

Outputs land in `<project_dir>/ssp_modeling/tableau/data/`:
  - decomposed_emissions_<region>_<year_ref>.csv   (HP-smoothed, historical + SISEPUEDE)
  - drivers_<region>.csv                            (driver variables + GDP history)
  - tableau_levers_table_complete.csv               (levers + stakeholder merge)
  - jobs_demand_<region>.csv                        (employment subset)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


# ---------------------------------------------------------------------------
# HP filter (port of R hp_filter_subsec)
# ---------------------------------------------------------------------------

def _hp_smooth_series(values: np.ndarray, lambda_hp: float) -> np.ndarray:
    """Anchored, non-negative HP trend matching the R hp_filter_subsec output."""
    if len(values) < 2:
        return values.copy()
    _, trend = hpfilter(values.astype(float), lamb=lambda_hp)
    sm = np.maximum(trend, 0.0)
    sm = np.maximum(sm + (values[0] - sm[0]), 0.0)
    sm[0] = values[0]
    return sm


def _apply_hp_to_subsec_gas(
    df: pd.DataFrame,
    subsec: str,
    gases,
    lambda_hp: float,
    by_cols=("primary_id", "strategy_id", "design_id", "future_id", "Code"),
    time_col: str = "Year",
    value_col: str = "value",
    cutoff_year: int | None = 2025,
) -> pd.DataFrame:
    """Apply HP smoothing per group for matching (subsector, gas) rows.

    If `cutoff_year` is set, the HP filter is applied only to years
    >= cutoff_year, anchored at cutoff_year so the smoothed series starts
    exactly at the original value of that year (preserving continuity).
    """
    if isinstance(gases, str):
        gases = [gases]
    df["value_original"] = df["value"].astype(float)
    mask = (
        (df["subsector"] == subsec)
        & (df["Gas"].isin(gases))
        & df["strategy_id"].notna()
    )
    if not mask.any():
        return df

    sub = df.loc[mask]
    smoothed = pd.Series(np.nan, index=sub.index, dtype=float, name="value_hp")
    for _, idx in sub.groupby(list(by_cols), dropna=False).indices.items():
        ordered = sub.iloc[idx].sort_values(time_col)
        if cutoff_year is not None:
            window = ordered.loc[ordered[time_col] >= cutoff_year]
            if len(window) < 2:
                continue
            v = window[value_col].astype(float).to_numpy()
            sm = _hp_smooth_series(v, lambda_hp)
            smoothed.loc[window.index] = sm
        else:
            v = ordered[value_col].astype(float).to_numpy()
            sm = _hp_smooth_series(v, lambda_hp)
            smoothed.loc[ordered.index] = sm

    df.loc[smoothed.index, "value_hp"] = smoothed.values
    valid = smoothed.notna()
    if valid.any():
        df.loc[smoothed.index[valid], value_col] = smoothed[valid].values
    not_smoothed = smoothed.isna()
    if not_smoothed.any():
        df.loc[smoothed.index[not_smoothed], "value_hp"] = (
            df.loc[smoothed.index[not_smoothed], value_col].astype(float).values
        )
    return df


# HP smoothing schedule tuned for Morocco's IPCC subsector naming.
# Format: (subsector_label, [gases], lambda_hp)
HP_SCHEDULE = [

    ("3.A.1 - Enteric Fermentation",                       ["CH4"], 200),
    ("3.A.2 - Manure Management",                          ["CH4"], 200),
    ("3.C.4 - Direct N2O Emissions from managed soils",    ["N2O"], 200),
    
    ("4.A.1 - Managed Waste Disposal Sites",               ["CH4"], 400),
    ("4.A.2 - Unmanaged Waste Disposal Sites",             ["CH4"], 400),
    ("4.D.1 - Domestic Wastewater",                        ["CH4"], 400),
]


# ---------------------------------------------------------------------------
# Emissions table (data_prep_new_mapping.r)
# ---------------------------------------------------------------------------

def build_emissions_table(
    run_dir: Path,
    targets_path: Path,
    edgar_path: Path,
    iso_code3: str,
    region: str,
    year_ref: int,
    out_path: Path,
    hp_schedule: Iterable[tuple] = HP_SCHEDULE,
    hp_cutoff_year: int | None = 2025,
) -> pd.DataFrame:
    """Port of data_prep_new_mapping.r."""

    # 1 — load mapping
    mapping = pd.read_csv(targets_path)
    mapping = mapping.drop(columns=[c for c in ("id", "ids", iso_code3) if c in mapping.columns])
    mapping = mapping.reset_index().rename(columns={"index": "_row"})
    mapping["ids"] = (
        mapping["_row"].astype(str)
        + "_" + mapping["subsector_ssp"].astype(str)
        + "_" + mapping["gas"].astype(str)
    )

    # 2 — load historical inventory, filter to country
    edgar = pd.read_csv(edgar_path)
    edgar = edgar[edgar["Code"] == iso_code3].copy()
    edgar["ID"] = edgar["subsector"].astype(str) + ":" + edgar["Gas"].astype(str)

    # 3 — load decomposed_ssp_output, filter to region, sort
    data = pd.read_csv(run_dir / "decomposed_ssp_output.csv")
    data = data[data["region"] == region].copy()
    data = data.sort_values(["primary_id", "time_period", "region"]).reset_index(drop=True)

    # 4 — build mapped (rowSum) columns according to mapping$vars (colon-separated)
    id_vars = ["region", "time_period", "primary_id"]
    data_cols = set(data.columns)

    for _, row in mapping.iterrows():
        vars_list = [v.strip() for v in str(row["vars"]).split(":") if v.strip()]
        present = [v for v in vars_list if v in data_cols]
        if len(present) > 1:
            data[row["ids"]] = data[present].sum(axis=1)
        elif len(present) == 1:
            data[row["ids"]] = data[present[0]]
        else:
            data[row["ids"]] = 0.0

    data_new = data[id_vars + mapping["ids"].tolist()].copy()

    # 5 — wide → long
    data_new = data_new.melt(id_vars=id_vars, var_name="ids", value_name="value")

    # 6 — merge with mapping; rename CSC sector/subsector
    map_keep = mapping.drop(columns=["vars"]).rename(
        columns={"sector": "CSC.Sector", "subsector": "CSC.Subsector"}
    )
    data_new = data_new.merge(map_keep, on="ids", how="left")

    # 7 — aggregate to inventory level
    data_new = (
        data_new.groupby(
            ["primary_id", "time_period", "ID", "CSC.Sector", "CSC.Subsector"],
            dropna=False,
            as_index=False,
        )["value"]
        .sum()
        .rename(columns={"CSC.Sector": "sector", "CSC.Subsector": "subsector"})
    )

    data_new["Year"] = data_new["time_period"] + 2015
    data_new["Gas"] = data_new["ID"].astype(str).str.split(":").str[1]

    # 8 — merge with primary + strategy attributes
    att_primary = pd.read_csv(run_dir / "ATTRIBUTE_PRIMARY.csv")
    data_new = data_new.merge(att_primary, on="primary_id", how="left")

    att_strategy = pd.read_csv(run_dir / "ATTRIBUTE_STRATEGY.csv")[["strategy_id", "strategy"]]
    data_new = data_new.merge(att_strategy, on="strategy_id", how="left")

    # 9 — melt historic inventory wide to long
    id_vars_edgar = ["Code", "sector", "subsector", "Gas", "ID"]
    measure_cols = [c for c in edgar.columns if re.fullmatch(r"X?\d{4}", str(c))]
    edgar_long = edgar.melt(
        id_vars=id_vars_edgar,
        value_vars=measure_cols,
        var_name="variable",
        value_name="value",
    )
    edgar_long["Year"] = edgar_long["variable"].astype(str).str.lstrip("X").astype(int)
    edgar_long = edgar_long.drop(columns=["variable"])
    edgar_long["strategy_id"] = np.nan
    edgar_long["primary_id"]  = np.nan
    edgar_long["design_id"]   = np.nan
    edgar_long["future_id"]   = np.nan
    edgar_long["Contry"]      = region
    edgar_long["strategy"]    = "Historical"
    edgar_long["source"]      = "NIR"

    # 10 — prepare data_new for rbind
    data_new = data_new.drop(columns=["time_period"])
    data_new["Code"]   = iso_code3
    data_new["Contry"] = region
    data_new["source"] = "SISEPUEDE"
    edgar_max_year = int(edgar_long["Year"].max())
    data_new = data_new[data_new["Year"] >= edgar_max_year].copy()

    combined = pd.concat([data_new, edgar_long], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["strategy_id", "sector", "subsector", "Gas", "Year"], na_position="last"
    ).reset_index(drop=True)

    # 11 — HP filter
    combined["value_hp"] = np.nan
    for subsec, gases, lam in hp_schedule:
        combined = _apply_hp_to_subsec_gas(
            combined, subsec, gases, lam, cutoff_year=hp_cutoff_year,
        )

    # 12 — write CSV with the column order expected by Tableau
    final_cols = [
        "strategy_id", "primary_id", "ID", "sector", "subsector", "value",
        "Year", "Gas", "design_id", "future_id", "strategy", "Code",
        "Contry", "source", "value_original", "value_hp",
    ]
    final_cols = [c for c in final_cols if c in combined.columns]
    out = combined[final_cols].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[emissions] {out_path}  ({len(out):,} rows)")
    return out


# ---------------------------------------------------------------------------
# Drivers table (data_prep_drivers.r)
# ---------------------------------------------------------------------------

def build_drivers_table(
    run_dir: Path,
    drivers_taxonomy_path: Path,
    wide_inputs_outputs_path: Path,
    iso_code3: str,
    region: str,
    year_ref: int,
    out_path: Path,
) -> pd.DataFrame:
    """Port of data_prep_drivers.r."""

    data = pd.read_csv(run_dir / "decomposed_ssp_output.csv")
    id_vars = ["region", "time_period", "primary_id"]
    measure_cols = [c for c in data.columns if c not in id_vars]
    long_df = data.melt(id_vars=id_vars, value_vars=measure_cols,
                        var_name="variable", value_name="value")

    drivers = pd.read_csv(drivers_taxonomy_path).rename(columns={"field": "variable"})

    long_df = long_df[long_df["variable"].isin(drivers["variable"].unique())].copy()
    long_df = long_df.merge(drivers, on="variable", how="left")

    long_df["Year"] = long_df["time_period"] + 2015
    long_df = long_df.drop(columns=["time_period"])
    long_df = long_df[long_df["Year"] >= year_ref].copy()

    att_primary = pd.read_csv(run_dir / "ATTRIBUTE_PRIMARY.csv")
    long_df = long_df.merge(att_primary, on="primary_id", how="left")

    att_strategy = pd.read_csv(run_dir / "ATTRIBUTE_STRATEGY.csv")[["strategy_id", "strategy"]]
    long_df = long_df.merge(att_strategy, on="strategy_id", how="left")

    long_df["Units"]           = "NA"
    long_df["Data_Type"]       = "sisepuede simulation"
    long_df["iso_code3"]       = iso_code3
    long_df["Country"]         = region
    long_df["output_type"]     = "drivers"
    long_df["gas"]             = np.nan
    long_df = long_df.drop(columns=[c for c in ("region", "subsector_total_field",
                                                "model_variable_information") if c in long_df.columns])

    # energy_subsector classification
    energy_keywords = {
        "ccsq": "Carbon Capture and Sequestration",
        "inen": "Industrial Energy",
        "entc": "Power(electricity/heat)",
        "trns": "Transportation",
        "scoe": "Buildings",
    }
    long_df["energy_subsector"] = pd.Series(pd.NA, index=long_df.index, dtype=object)
    energy_mask = long_df["variable"].str.contains("energy", regex=False, na=False)
    es = pd.Series("TBD", index=long_df.index, dtype=object)
    for kw, label in energy_keywords.items():
        es = es.where(~long_df["variable"].str.contains(kw, regex=False, na=False), label)
    long_df.loc[energy_mask, "energy_subsector"] = es[energy_mask]

    # GDP history
    gdp = pd.read_csv(wide_inputs_outputs_path, usecols=["primary_id", "time_period", "gdp_mmm_usd"])
    gdp = gdp[gdp["primary_id"] == 0].copy()
    gdp["year"] = gdp["time_period"] + 2015
    gdp = gdp[gdp["year"] <= year_ref][["year", "gdp_mmm_usd"]]

    strategies_df = (
        long_df[["strategy_id", "design_id", "future_id", "strategy"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    gdp = gdp.assign(_k=1).rename(columns={"year": "Year", "gdp_mmm_usd": "value"})
    strategies_df = strategies_df.assign(_k=1)
    drivers_hist = strategies_df.merge(gdp, on="_k").drop(columns="_k")
    drivers_hist["variable"]         = "gdp_mmm_usd"
    drivers_hist["primary_id"]       = 0
    drivers_hist["sector"]           = "Socioeconomic"
    drivers_hist["subsector"]        = "Economy"
    drivers_hist["model_variable"]   = "GDP"
    drivers_hist["category_value"]   = "('', '')"
    drivers_hist["category_name"]    = "cat_economy"
    drivers_hist["gas"]              = np.nan
    drivers_hist["gas_name"]         = ""
    drivers_hist["Units"]            = "NA"
    drivers_hist["Data_Type"]        = "historical"
    drivers_hist["iso_code3"]        = iso_code3
    drivers_hist["Country"]          = region
    drivers_hist["output_type"]      = "drivers"
    drivers_hist["energy_subsector"] = np.nan

    years_in_sim = long_df.loc[long_df["variable"] == "gdp_mmm_usd", "Year"].dropna().unique()
    if len(years_in_sim):
        last_year_in_sim = int(min(years_in_sim))
        drivers_hist = drivers_hist[drivers_hist["Year"] < last_year_in_sim].copy()

    out_cols = [
        "variable", "strategy_id", "primary_id", "value", "sector", "subsector",
        "model_variable", "category_value", "category_name", "gas", "gas_name",
        "Year", "design_id", "future_id", "strategy", "Units", "Data_Type",
        "iso_code3", "Country", "output_type", "energy_subsector",
    ]
    for c in out_cols:
        if c not in long_df.columns:
            long_df[c] = np.nan
        if c not in drivers_hist.columns:
            drivers_hist[c] = np.nan

    final = pd.concat([long_df[out_cols], drivers_hist[out_cols]], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)
    print(f"[drivers]   {out_path}  ({len(final):,} rows)")
    return final


# ---------------------------------------------------------------------------
# Levers table
# ---------------------------------------------------------------------------

# R make.names() converts spaces and parentheses to dots.
_R_MAKE_NAMES_RENAME = {
    "Sector (output)":            "Sector..output.",
    "Subsector (output)":         "Subsector..output.",
    "Example government policies": "Example.government.policies",
}


def build_levers_table(
    run_dir: Path,
    descriptions_path: Path,
    stakeholder_codes_path: Path,
    out_path: Path,
    levers_filename: str = "levers_implementation_morocco.csv",
) -> pd.DataFrame:
    """Merges levers_implementation_<region>.csv with descriptions + stakeholder codes."""

    ssp_table = pd.read_csv(run_dir / levers_filename)
    ssp_table["transformation_code"] = ssp_table["transformer_code"].str.replace("TFR:", "", regex=False)

    desp = pd.read_csv(descriptions_path)
    scodes = pd.read_csv(stakeholder_codes_path).rename(columns=_R_MAKE_NAMES_RENAME)
    scodes["transformation_code"] = scodes["transformation_code"].str.replace("TX:", "", regex=False)

    merged = ssp_table.merge(desp, on="transformation_code", how="inner")
    merged = merged.merge(
        scodes[["transformation_code", "transformation_name_stakeholder",
                "Sector..output.", "Subsector..output.", "Example.government.policies"]],
        on="transformation_code",
        how="inner",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[levers]    {out_path}  ({len(merged):,} rows)")
    return merged


# ---------------------------------------------------------------------------
# Jobs table
# ---------------------------------------------------------------------------

def build_jobs_table(
    employment_path: Path,
    iso_code3: str,
    out_path: Path,
) -> pd.DataFrame:
    """Splits the ":"-encoded Strategy column and filters to country."""

    jobs = pd.read_csv(employment_path)
    parts = jobs["Strategy"].astype(str).str.split(":", n=1, expand=True)
    jobs["ssp_sector"]                 = parts[0]
    jobs["ssp_transformation_name"]    = parts[1] if parts.shape[1] > 1 else ""
    jobs = jobs[jobs["Country"] == iso_code3].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(out_path, index=False)
    print(f"[jobs]      {out_path}  ({len(jobs):,} rows)")
    return jobs


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_tableau_postprocessing(
    run_dir: str | Path,
    project_dir: str | Path,
    region: str,
    iso_code3: str,
    year_ref: int,
    wide_inputs_outputs_filename: str | None = None,
    out_dir: str | Path | None = None,
    hp_cutoff_year: int | None = 2025,
    drivers_taxonomy_filename: str = "driver_variables_taxonomy_20251013.csv",
    levers_filename: str | None = None,
) -> dict:
    """Generate every Tableau-ready CSV in one call (Morocco-adapted).

    Parameters
    ----------
    run_dir : run output directory (must contain decomposed_ssp_output.csv,
        ATTRIBUTE_PRIMARY.csv, ATTRIBUTE_STRATEGY.csv,
        levers_implementation_<region>.csv, and the WIDE_INPUTS_OUTPUTS.csv).
    project_dir : repo root.
    region : region label inside the model output, e.g. "morocco".
    iso_code3 : ISO 3-letter country code, e.g. "MAR".
    year_ref : reference / calibration year (e.g. 2018).
    wide_inputs_outputs_filename : explicit filename of the WIDE_INPUTS_OUTPUTS
        file inside run_dir; if None, auto-detected.
    out_dir : where to write the Tableau CSVs (defaults to
        <project_dir>/ssp_modeling/tableau/data).
    hp_cutoff_year : first year where HP smoothing kicks in (default 2025).
    drivers_taxonomy_filename : filename of the drivers taxonomy file under
        output_postprocessing/data/ (Morocco currently has
        driver_variables_taxonomy_20240117.csv).
    levers_filename : explicit name for the levers implementation CSV
        (default: levers_implementation_<region>.csv).
    """
    run_dir     = Path(run_dir)
    project_dir = Path(project_dir)
    out_dir     = Path(out_dir) if out_dir else project_dir / "ssp_modeling" / "tableau" / "data"

    # Auto-detect WIDE_INPUTS_OUTPUTS file
    if wide_inputs_outputs_filename is None:
        candidates = sorted(run_dir.glob("*WIDE_INPUTS_OUTPUTS.csv"))
        if not candidates:
            raise FileNotFoundError(f"No *WIDE_INPUTS_OUTPUTS.csv found in {run_dir}")
        wide_path = candidates[-1]
    else:
        wide_path = run_dir / wide_inputs_outputs_filename

    pp_root = project_dir / "ssp_modeling" / "output_postprocessing"
    pp_data = pp_root / "data"
    levers_dir = pp_root / "levers_and_jobs_table"

    targets_path           = pp_data / "invent" / f"emission_targets_{iso_code3.lower()}_{year_ref}.csv"
    edgar_path             = pp_data / "invent" / f"invent_historic_{iso_code3.lower()}.csv"
    drivers_taxonomy_path  = pp_data / drivers_taxonomy_filename
    descriptions_path      = levers_dir / "ssp_descriptions.csv"
    stakeholder_codes_path = levers_dir / "stakeholder_codes.csv"
    employment_path        = levers_dir / "Sisepuede - Employment Results - WB (SECTOR).csv"

    levers_filename = levers_filename or f"levers_implementation_{region}.csv"

    emissions = build_emissions_table(
        run_dir        = run_dir,
        targets_path   = targets_path,
        edgar_path     = edgar_path,
        iso_code3      = iso_code3,
        region         = region,
        year_ref       = year_ref,
        out_path       = out_dir / f"decomposed_emissions_{region}_{year_ref}.csv",
        hp_cutoff_year = hp_cutoff_year,
    )
    drivers = build_drivers_table(
        run_dir                  = run_dir,
        drivers_taxonomy_path    = drivers_taxonomy_path,
        wide_inputs_outputs_path = wide_path,
        iso_code3                = iso_code3,
        region                   = region,
        year_ref                 = year_ref,
        out_path                 = out_dir / f"drivers_{region}.csv",
    )
    levers = build_levers_table(
        run_dir                = run_dir,
        descriptions_path      = descriptions_path,
        stakeholder_codes_path = stakeholder_codes_path,
        out_path               = out_dir / "tableau_levers_table_complete.csv",
        levers_filename        = levers_filename,
    )
    jobs = build_jobs_table(
        employment_path = employment_path,
        iso_code3       = iso_code3,
        out_path        = out_dir / f"jobs_demand_{region}.csv",
    )

    return {
        "emissions": emissions,
        "drivers":   drivers,
        "levers":    levers,
        "jobs":      jobs,
    }
