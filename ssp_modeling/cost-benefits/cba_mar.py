# Load packages
from costs_benefits_ssp.cb_calculate import CostBenefits
import pandas as pd
import os
import pathlib

# ---------------------------------------------------------------------------
# Paths
# Run this script from the project root:
#   python ssp_modeling/cost-benefits/cba_mar.py
# ---------------------------------------------------------------------------
SSP_PATH = pathlib.Path(os.getcwd())

RUN_ID = "2026-05-15T20;31;35.553357"
SSP_RUN = SSP_PATH / "ssp_modeling" / "ssp_run_output" / f"sisepuede_results_sisepuede_run_{RUN_ID}"

CB_CONFIG_PATH = SSP_PATH / "ssp_modeling" / "cost-benefits" / "cb_config_files" / "cb_config_params.xlsx"
CB_OUTPUT_PATH = SSP_PATH / "ssp_modeling" / "cost-benefits" / "output"
CB_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load SSP run data
# ---------------------------------------------------------------------------
ssp_data    = pd.read_csv(SSP_RUN / "decomposed_ssp_output.csv")
att_primary  = pd.read_csv(SSP_RUN / "ATTRIBUTE_PRIMARY.csv")
att_strategy = pd.read_csv(SSP_RUN / "ATTRIBUTE_STRATEGY.csv")

strategy_code_base = "BASE"

# ---------------------------------------------------------------------------
# Instantiate and configure CostBenefits
# ---------------------------------------------------------------------------
cb = CostBenefits(ssp_data, att_primary, att_strategy, strategy_code_base)

# Uncomment to export an initial config template:
# cb.export_db_to_excel(str(CB_CONFIG_PATH))

cb.load_cb_parameters(str(CB_CONFIG_PATH))

# ---------------------------------------------------------------------------
# Compute costs
# ---------------------------------------------------------------------------
results_system = cb.compute_system_cost_for_all_strategies()
results_tx     = cb.compute_technical_cost_for_all_strategies()

results_all = pd.concat([results_system, results_tx], ignore_index=True)

# Post-process interactions and shift pre-2025 costs forward
results_all_pp        = cb.cb_process_interactions(results_all)
results_all_pp_shifted = cb.cb_shift_costs(results_all_pp)

# ---------------------------------------------------------------------------
# Reshape for Tableau
# ---------------------------------------------------------------------------
cb_data = results_all_pp_shifted.copy()

# Split variable into components (cb : sector : cb_type : item_1 : item_2)
cb_chars = cb_data["variable"].astype(str).str.split(":", n=4, expand=True)
cb_chars.columns = ["name", "sector", "cb_type", "item_1", "item_2"]
cb_data = pd.concat([cb_data, cb_chars], axis=1)

# Scale USD → billions USD
cb_data["value"] = cb_data["value"] / 1e9

# Remove pre-2025 shifted entries (these are already captured in the 2025/2035 rows)
cb_data = cb_data[~cb_data["item_2"].astype(str).str.contains("shifted", na=False)]
cb_data = cb_data[~cb_data["variable"].astype(str).str.contains("shifted2", na=False)]

# Year column
cb_data["Year"] = cb_data["time_period"] + 2015

# ---------------------------------------------------------------------------
# Strategy labels  –  update when new strategies are added to the run
# ---------------------------------------------------------------------------
STRATEGY_LABELS = {
    "PFLO:NDC":  "NDC",
}

STRATEGY_IDS = {
    "BASE":     0,
    "PFLO:NDC": 6003,
}

PRIMARY_IDS = {
    "BASE":     0,
    "PFLO:NDC": 72072,
}

cb_data["strategy"]    = cb_data["strategy_code"].map(STRATEGY_LABELS)
cb_data["strategy_id"] = cb_data["strategy_code"].map(STRATEGY_IDS)
cb_data["primary_id"]  = cb_data["strategy_code"].map(PRIMARY_IDS)

# Identifier column
cb_data["ids"] = cb_data["variable"].astype(str) + ":" + cb_data["strategy_id"].astype(str)

# Merge GDP
gdp = ssp_data[["primary_id", "time_period", "gdp_mmm_usd"]]
cb_data = cb_data.merge(gdp, on=["primary_id", "time_period"], how="left")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_file = CB_OUTPUT_PATH / f"cb_{RUN_ID}.csv"
cb_data.to_csv(out_file, index=False)
print(f"Saved {len(cb_data):,} rows → {out_file}")
