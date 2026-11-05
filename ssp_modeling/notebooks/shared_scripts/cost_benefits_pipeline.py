"""
cost_benefits_pipeline.py
--------------------------
Full cost-benefit pipeline for Morocco SSP runs.

Wraps costs_benefits_ssp.cb_calculate.CostBenefits and the Tableau-reshaping
logic that lives in ssp_modeling/cost-benefits/cb.ipynb, so the manager
notebook can call a single function instead of repeating the steps inline.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from costs_benefits_ssp.cb_calculate import CostBenefits


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_base_code(strategy_code_base: str, att_strategy: pd.DataFrame) -> str:
    """Return *strategy_code_base* if it exists in att_strategy, else 'BASE'."""
    if strategy_code_base in att_strategy["strategy_code"].values:
        return strategy_code_base
    fallback = "BASE"
    print(
        f"[cb_pipeline] '{strategy_code_base}' not found in ATTRIBUTE_STRATEGY — "
        f"falling back to '{fallback}'."
    )
    return fallback


def _build_strategy_maps(
    att_primary: pd.DataFrame,
    att_strategy: pd.DataFrame,
) -> tuple:
    """
    Build (strategy_id_map, primary_id_map) dicts keyed by strategy_code.

    strategy_id_map  : strategy_code -> strategy_id  (from att_strategy)
    primary_id_map   : strategy_code -> primary_id   (from att_primary join)
    """
    sid_map = dict(zip(att_strategy["strategy_code"], att_strategy["strategy_id"]))

    # Invert: strategy_id -> primary_id
    sid_to_pid = dict(zip(att_primary["strategy_id"], att_primary["primary_id"]))

    pid_map = {code: sid_to_pid.get(sid) for code, sid in sid_map.items()}
    return sid_map, pid_map


def _reshape_for_tableau(
    results_shifted: pd.DataFrame,
    ssp_data: pd.DataFrame,
    att_strategy: pd.DataFrame,
    strategy_id_map: dict,
    primary_id_map: dict,
) -> pd.DataFrame:
    """
    Reshape cost-benefit results to Tableau-ready format.

    Steps
    -----
    1.  Split variable string into (name, sector, cb_type, item_1, item_2).
    2.  Scale USD → billions USD.
    3.  Drop pre-2025 shifted rows.
    4.  Add Year, strategy label, strategy_id, primary_id, ids, gdp_mmm_usd.
    """
    cb = results_shifted.copy()

    # 1. Decompose variable
    parts = cb["variable"].astype(str).str.split(":", n=4, expand=True)
    parts.columns = ["name", "sector", "cb_type", "item_1", "item_2"]
    cb = pd.concat([cb, parts], axis=1)

    # 2. USD → billions
    cb["value"] = cb["value"] / 1e9

    # 3. Remove shifted entries (costs that were redistrubuted pre-2025)
    cb = cb[~cb["item_2"].astype(str).str.contains("shifted", na=False)]
    cb = cb[~cb["variable"].astype(str).str.contains("shifted2", na=False)]

    # 4. Year
    cb["Year"] = cb["time_period"] + 2015

    # 5. Strategy metadata from the maps (dynamic — no hardcoding)
    cb["strategy_id"] = cb["strategy_code"].map(strategy_id_map)
    cb["primary_id"]  = cb["strategy_code"].map(primary_id_map)

    # Human-readable label: use strategy_code as-is (can be overridden later in notebook)
    cb["strategy"] = cb["strategy_code"]

    # 6. Unique identifier
    cb["ids"] = cb["variable"].astype(str) + ":" + cb["strategy_id"].astype(str)

    # 7. Merge GDP
    gdp = ssp_data[["primary_id", "time_period", "gdp_mmm_usd"]].drop_duplicates()
    cb = cb.merge(gdp, on=["primary_id", "time_period"], how="left")

    # 8. TEMP PATCH: drop ENTC technical-cost rows with the wrong sign
    #    (capex, transmission, ...). Costs are stored negative (Tableau plots
    #    SUM(value)*-1), so value > 0 here renders as a *negative* cost in the
    #    dashboard, which is not meaningful. Remove until the upstream ENTC
    #    technical-cost calculation is fixed.
    bad_sign = (
        (cb["sector"] == "entc")
        & (cb["cb_type"] == "technical_cost")
        & (cb["value"] > 0)
    )
    if bad_sign.any():
        dropped = cb.loc[bad_sign, ["variable", "strategy_code", "Year"]]
        print(
            f"[cb_pipeline] TEMP PATCH: dropping {bad_sign.sum()} wrong-sign "
            "ENTC technical-cost rows:"
        )
        for var, grp in dropped.groupby("variable"):
            years = ", ".join(
                f"{s} {y}"
                for s, y in grp[["strategy_code", "Year"]].itertuples(index=False, name=None)
            )
            print(f"    {var}: {years}")
        cb = cb[~bad_sign].copy()

    return cb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cost_benefits(
    df_decomposed: pd.DataFrame,
    att_primary: pd.DataFrame,
    att_strategy: pd.DataFrame,
    cb_config_path: Path,
    run_output_dir: Path,
    project_dir: Path,
    strategy_code_base: str = "BASE",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the full cost-benefit pipeline and return a Tableau-ready DataFrame.

    Parameters
    ----------
    df_decomposed      : wide-format decomposed SSP output (from run_decomposition)
    att_primary        : ATTRIBUTE_PRIMARY table as DataFrame
    att_strategy       : ATTRIBUTE_STRATEGY table as DataFrame
    cb_config_path     : path to cb_config_params.xlsx
    run_output_dir     : directory of the current SSP run (for context; not written to here)
    project_dir        : repo root (unused currently; kept for API symmetry)
    strategy_code_base : strategy_code that acts as the BAU/BASE reference
    output_path        : if given, writes the result CSV to this path

    Returns
    -------
    pd.DataFrame  Tableau-ready cost-benefit results
    """
    cb_config_path = Path(cb_config_path)

    # Resolve base strategy (fall back to BASE if specified code is absent)
    base_code = _resolve_base_code(strategy_code_base, att_strategy)

    # Build dynamic strategy → id maps
    strategy_id_map, primary_id_map = _build_strategy_maps(att_primary, att_strategy)

    # Instantiate and configure
    cb_obj = CostBenefits(df_decomposed, att_primary, att_strategy, base_code)
    cb_obj.load_cb_parameters(str(cb_config_path))

    # Compute
    results_system = cb_obj.compute_system_cost_for_all_strategies()
    results_tx     = cb_obj.compute_technical_cost_for_all_strategies()
    results_all    = pd.concat([results_system, results_tx], ignore_index=True)

    # Post-process interactions and shift pre-2025 costs
    results_pp      = cb_obj.cb_process_interactions(results_all)
    results_shifted = cb_obj.cb_shift_costs(results_pp)

    # Reshape for Tableau
    cb_data = _reshape_for_tableau(
        results_shifted,
        df_decomposed,
        att_strategy,
        strategy_id_map,
        primary_id_map,
    )

    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cb_data.to_csv(output_path, index=False)
        print(f"[cb_pipeline] Saved {len(cb_data):,} rows → {output_path}")

    return cb_data
