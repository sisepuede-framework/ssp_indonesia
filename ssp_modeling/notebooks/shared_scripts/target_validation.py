"""
target_validation.py
--------------------
Validates SISEPUEDE Morocco model outputs against documented climate-pathway targets
(LT-LEDS / SNBC, NDC 3.0, Methane Roadmap, Kigali Amendment).

Loads target definitions from morocco_ndc_targets.yaml, computes corresponding
indicators from the SSP output dataframes, and produces a status report.

Status legend:
- PASS:                 |pct_diff| <= tolerance_pct (default 5%)
- WARN:                 5% < |pct_diff| <= 15%
- FAIL:                 |pct_diff| > 15%
- WAVE_A_KNOWN_ISSUE:   pre-flight model bug flagged in targets YAML
- STRUCTURAL_GAP:       no SISEPUEDE lever can represent this target
- INSUFFICIENT_DATA:    indicator cannot be computed from provided dataframes
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml
import numpy as np
import pandas as pd

# Year base for time_period mapping
T0_YEAR = 2015

# Map document sector names to SSP subsector codes
SECTOR_TO_SUBSECTORS = {
    "Energy": ["entc", "inen", "scoe", "fgtv"],
    "Transport": ["trns"],
    "Industry": ["ippu"],
    "AFOLU": ["lvst", "lsmm", "agrc", "soil", "frst", "lndu"],
    "Waste": ["waso", "trww"],
    "Economy": ["entc", "inen", "trns", "scoe", "fgtv", "ippu", "lvst", "lsmm", "agrc", "soil", "frst", "lndu", "waso", "trww", "ccsq"],
}

# Map subsector_ssp codes (as in target file) to friendly labels in tableau output
SUBSECTOR_TO_TABLEAU_SUBSECTOR_HINT = {
    "entc": "1.A.1 - Energy Industries",
    "inen": "1.A.2 - Manufacturing Industries and Construction",
    "trns": "1.A.3 - Transport",
    "scoe": "1.A.4 - Other Sectors",
    "fgtv": "1.B",
    "ippu": "2",
    "lvst": "3.A.1 - Enteric Fermentation",
    "waso": "4.A",
    "trww": "4.D",
}


def _classify(pct_diff: float, tolerance_pct: float) -> str:
    if np.isnan(pct_diff):
        return "INSUFFICIENT_DATA"
    a = abs(pct_diff)
    if a <= tolerance_pct:
        return "PASS"
    if a <= 3 * tolerance_pct:
        return "WARN"
    return "FAIL"


def _compute_indicator(
    target: dict,
    df_decomposed: pd.DataFrame,
    df_export: pd.DataFrame,
    strategy_id: int,
    region: str,
) -> Optional[float]:
    """
    Compute the modeled value of an indicator. Returns None if the indicator
    cannot be computed.

    df_decomposed: from Tableau decomposed_emissions_<region>_<year>.csv (long format)
                   columns: strategy_id, primary_id, ID, sector, subsector, value, Year, Gas, ...
    df_export:     wide WIDE_INPUTS_OUTPUTS dataframe with primary_id, time_period, region columns + emission/driver columns
    """
    indicator = target["indicator"]
    year = target["year"]
    sector = target.get("sector")
    subsector = target.get("subsector")

    # 1) subsector_emissions_mt (most common, direct lookup in df_decomposed)
    if indicator == "subsector_emissions_mt":
        subsector_filter = subsector
        if subsector_filter:
            hint = SUBSECTOR_TO_TABLEAU_SUBSECTOR_HINT.get(subsector_filter, subsector_filter)
            mask = (
                (df_decomposed["strategy_id"] == strategy_id)
                & (df_decomposed["Year"] == year)
                & (df_decomposed["source"] == "SISEPUEDE")
                & (df_decomposed["subsector"].astype(str).str.startswith(hint.split(" -")[0]))
            )
            return float(df_decomposed.loc[mask, "value"].sum()) if mask.any() else np.nan
        return np.nan

    # 2) total_emissions_mt
    if indicator == "total_emissions_mt":
        mask = (
            (df_decomposed["strategy_id"] == strategy_id)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["source"] == "SISEPUEDE")
        )
        return float(df_decomposed.loc[mask, "value"].sum()) if mask.any() else np.nan

    # 3) total_co2e_reduction_pct_vs_ref (uses baseline scenario primary_id=0 as reference)
    if indicator == "total_co2e_reduction_pct_vs_ref":
        baseline = df_decomposed[
            (df_decomposed["strategy_id"] == 0)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["source"] == "SISEPUEDE")
        ]["value"].sum()
        strategy = df_decomposed[
            (df_decomposed["strategy_id"] == strategy_id)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["source"] == "SISEPUEDE")
        ]["value"].sum()
        if baseline == 0 or np.isnan(baseline):
            return np.nan
        return (strategy - baseline) / baseline

    # 4) net_emissions_mt_co2e (= total in long file; same as #2)
    if indicator == "net_emissions_mt_co2e":
        return _compute_indicator({**target, "indicator": "total_emissions_mt"}, df_decomposed, df_export, strategy_id, region)

    # 5) subsector_emissions_reduction_pct_vs_baseline
    if indicator == "subsector_emissions_reduction_pct_vs_baseline":
        # default subsector = "trns" for transport
        sub_filter = subsector or "trns"
        hint = SUBSECTOR_TO_TABLEAU_SUBSECTOR_HINT.get(sub_filter, sub_filter)
        baseline = df_decomposed[
            (df_decomposed["strategy_id"] == 0)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["source"] == "SISEPUEDE")
            & (df_decomposed["subsector"].astype(str).str.startswith(hint.split(" -")[0]))
        ]["value"].sum()
        strategy = df_decomposed[
            (df_decomposed["strategy_id"] == strategy_id)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["source"] == "SISEPUEDE")
            & (df_decomposed["subsector"].astype(str).str.startswith(hint.split(" -")[0]))
        ]["value"].sum()
        if baseline == 0 or np.isnan(baseline):
            return np.nan
        return (strategy - baseline) / baseline

    # 6) ag_emissions_mt
    if indicator == "ag_emissions_mt":
        agrc_subsectors = SECTOR_TO_SUBSECTORS["AFOLU"]
        ssp_hints = {SUBSECTOR_TO_TABLEAU_SUBSECTOR_HINT.get(s, s).split(" -")[0] for s in agrc_subsectors}
        mask = (
            (df_decomposed["strategy_id"] == strategy_id)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["source"] == "SISEPUEDE")
        )
        # Approximate: sum all AFOLU rows
        afolu_codes = ("3.A", "3.B", "3.C", "3.D")
        sub_mask = df_decomposed["subsector"].astype(str).str.startswith(afolu_codes)
        return float(df_decomposed.loc[mask & sub_mask, "value"].sum()) if (mask & sub_mask).any() else np.nan

    # 7) hfc_emissions_reduction_pct
    if indicator == "hfc_emissions_reduction_pct":
        baseline = df_decomposed[
            (df_decomposed["strategy_id"] == 0)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["Gas"] == "HFCS")
            & (df_decomposed["source"] == "SISEPUEDE")
        ]["value"].sum()
        strategy = df_decomposed[
            (df_decomposed["strategy_id"] == strategy_id)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["Gas"] == "HFCS")
            & (df_decomposed["source"] == "SISEPUEDE")
        ]["value"].sum()
        if baseline == 0 or np.isnan(baseline):
            return np.nan
        return (strategy - baseline) / baseline

    # 8) methane_reduction_pct_vs_baseline (across all CH4 sources except fgtv)
    if indicator == "methane_reduction_pct_vs_baseline":
        ch4_mask_base = (
            (df_decomposed["strategy_id"] == 0)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["Gas"] == "CH4")
            & (df_decomposed["source"] == "SISEPUEDE")
        )
        ch4_mask_str = (
            (df_decomposed["strategy_id"] == strategy_id)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["Gas"] == "CH4")
            & (df_decomposed["source"] == "SISEPUEDE")
        )
        baseline = df_decomposed.loc[ch4_mask_base, "value"].sum()
        strategy = df_decomposed.loc[ch4_mask_str, "value"].sum()
        if baseline == 0 or np.isnan(baseline):
            return np.nan
        return (strategy - baseline) / baseline

    # 9) forest_sink_mt_co2_per_year (negative value = sink)
    if indicator == "forest_sink_mt_co2_per_year":
        mask = (
            (df_decomposed["strategy_id"] == strategy_id)
            & (df_decomposed["Year"] == year)
            & (df_decomposed["source"] == "SISEPUEDE")
            & (df_decomposed["subsector"].astype(str).str.startswith("3.B.1"))
        )
        return float(df_decomposed.loc[mask, "value"].sum()) if mask.any() else np.nan

    # Indicators requiring activity drivers from df_export (renewable share, EV stock, areas, etc.)
    # These are stubs; full implementation requires wide-format column mapping.
    structural_indicators = {
        "renewable_electricity_share",
        "coal_generation_share",
        "rooftop_pv_capacity_gw",
        "industrial_energy_intensity_reduction_pct",
        "land_transport_demand_reduction_pct",
        "ev_stock_millions",
        "passenger_car_electric_share",
        "h2_truck_stock",
        "rail_green_energy_share",
        "phosphate_co2_capture_rate",
        "microalgas_cdr_mt",
        "reforestation_area_mha",
        "no_till_area_mha",
        "organic_farming_area_kha",
        "arboriculture_area_mha",
        "soil_n_reduction_pct",
        "recycling_rate",
        "recycling_rate_principal_sectors",
        "energy_related_emissions_reduction_pct",
    }
    if indicator in structural_indicators:
        return np.nan  # Will be classified as INSUFFICIENT_DATA or STRUCTURAL_GAP

    return None


def validate_targets(
    df_decomposed: pd.DataFrame,
    df_export: pd.DataFrame,
    targets_yaml_path: Path,
    strategy_id: int,
    region: str = "morocco",
    default_tolerance_pct: float = 0.05,
) -> pd.DataFrame:
    """Validate SSP-Morocco model outputs against documented climate pathway targets.

    Parameters
    ----------
    df_decomposed : long-format Tableau emissions dataframe
    df_export :    wide WIDE_INPUTS_OUTPUTS dataframe
    targets_yaml_path : path to morocco_ndc_targets.yaml
    strategy_id : SISEPUEDE strategy_id to validate (e.g., 6005)
    region : region label (default "morocco")
    default_tolerance_pct : default tolerance band for PASS (default 5%)

    Returns
    -------
    DataFrame with one row per target, status per validation logic.
    """
    with open(targets_yaml_path) as f:
        cfg = yaml.safe_load(f)
    targets = cfg.get("targets", [])

    rows = []
    for t in targets:
        modeled = _compute_indicator(t, df_decomposed, df_export, strategy_id, region)
        target_value = t["target_value"]
        tolerance = t.get("tolerance_pct", default_tolerance_pct)

        if t.get("irreducible_gap", False):
            status = "STRUCTURAL_GAP"
            abs_diff = np.nan
            pct_diff = np.nan
        elif t.get("wave_a_known_issue", False):
            status = "WAVE_A_KNOWN_ISSUE"
            abs_diff = (modeled - target_value) if modeled is not None and not np.isnan(modeled) else np.nan
            pct_diff = (abs_diff / abs(target_value)) if (target_value != 0 and not np.isnan(abs_diff)) else np.nan
        elif modeled is None or np.isnan(modeled):
            status = "INSUFFICIENT_DATA"
            abs_diff = np.nan
            pct_diff = np.nan
        else:
            abs_diff = modeled - target_value
            pct_diff = abs_diff / abs(target_value) if target_value != 0 else (np.inf if abs_diff != 0 else 0.0)
            status = _classify(pct_diff, tolerance)

        rows.append({
            "target_id": t["id"],
            "sector": t.get("sector"),
            "subsector": t.get("subsector", ""),
            "indicator": t["indicator"],
            "year": t["year"],
            "unit": t.get("unit", ""),
            "target_value": target_value,
            "modeled_value": modeled if modeled is not None else np.nan,
            "abs_diff": abs_diff,
            "pct_diff": pct_diff,
            "status": status,
            "tolerance_pct": tolerance,
            "source_citation": t.get("source_citation", ""),
            "notes": t.get("notes", ""),
            "strategy_id": strategy_id,
        })

    return pd.DataFrame(rows)


def print_validation_report(df_validation: pd.DataFrame) -> None:
    """Print a sector-grouped summary of validation results."""
    if df_validation.empty:
        print("No targets validated.")
        return

    strategy_id = df_validation["strategy_id"].iloc[0]
    total = len(df_validation)
    counts = df_validation["status"].value_counts()

    print(f"\n{'=' * 70}")
    print(f"  TARGET VALIDATION REPORT - Strategy {strategy_id}")
    print(f"{'=' * 70}")
    print(f"  Total targets: {total}")
    for status in ["PASS", "WARN", "FAIL", "WAVE_A_KNOWN_ISSUE", "STRUCTURAL_GAP", "INSUFFICIENT_DATA"]:
        n = counts.get(status, 0)
        if n:
            print(f"  {status:<22s}: {n}")

    print(f"\n  By sector:")
    for sector, group in df_validation.groupby("sector"):
        print(f"  -- {sector} ({len(group)} targets)")
        for _, r in group.iterrows():
            icon = {"PASS": "[OK]", "WARN": "[!]", "FAIL": "[X]", "WAVE_A_KNOWN_ISSUE": "[W]", "STRUCTURAL_GAP": "[G]", "INSUFFICIENT_DATA": "[?]"}.get(r["status"], "[.]")
            tv = r["target_value"]
            mv = r["modeled_value"]
            mv_str = f"{mv:.4g}" if (mv is not None and not pd.isna(mv)) else "--"
            print(f"     {icon} [{r['status']:<22s}] {r['target_id']:<35s} target={tv} modeled={mv_str}")
    print(f"{'=' * 70}\n")
