"""
bur3_calibration.py
-------------------
Sector-by-sector calibration of the SISEPUEDE Indonesia baseline (primary_id 0)
against the BUR3 national GHG inventory.

Mechanism
=========
Derive a multiplicative factor per sub-sector at the anchor year (2019) and apply
it to that sub-sector's INPUT fields, then export a calibrated base input. Because
most emission factors enter the model linearly, scaling a sub-sector's EF (or energy
demand) by f scales its emissions by ~f -- so the calibrated result can be predicted
analytically (fast) and, optionally, confirmed with one full re-run.

Lever per sub-sector
--------------------
  'ef'      -> scale ef_<sub>_* fields         (linear; exact for EF-driven sub-sectors)
  'demand'  -> scale scalar_<sub>_*energy_demand* fields (combustion via energy model)
  'manual'  -> factor reported but NOT auto-applied (entc: needs generation-mix change,
               not a blind EF scale; ccsq ~ 0)

FOLU / peat
-----------
Indonesia's FOLU is peat-dominated (~854 Mt in 2019) and the model has no explicit
peat variable. The forest sink (frst) is kept as modelled and the land-use source
(lndu) is scaled so that frst + lndu*f hits the FOLU target. The peat gap is thus
folded into lndu -> flagged as a STRUCTURAL_GAP, not a physical calibration.

Run `python -I bur3_calibration.py` for the self-check.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

T0_YEAR = 2015
EMISSION_PREFIX = "emission_co2e_subsector_total_"

# IPCC sector -> SISEPUEDE sub-sector codes (matches target_validation.SECTOR_TO_SUBSECTORS)
IPCC_GROUPS = {
    "Energy": ["ccsq", "entc", "fgtv", "inen", "scoe", "trns"],
    "IPPU": ["ippu"],
    "Agriculture": ["agrc", "lsmm", "lvst", "soil"],
    "FOLU": ["frst", "lndu"],
    "Waste": ["trww", "waso"],
}
SUB_TO_SECTOR = {s: g for g, subs in IPCC_GROUPS.items() for s in subs}

# Lever type per sub-sector
LEVERS = {
    "fgtv": "ef", "ippu": "ef", "soil": "ef", "waso": "ef", "lsmm": "ef",
    "trww": "ef", "lvst": "ef", "agrc": "ef", "trns": "ef",
    "frst": "ef", "lndu": "ef",
    "inen": "demand", "scoe": "demand",
    "entc": "manual", "ccsq": "manual",
}


# --------------------------------------------------------------------------- #
# Reference + baseline loading
# --------------------------------------------------------------------------- #
def load_reference(yaml_path) -> dict:
    with open(yaml_path) as fh:
        return yaml.safe_load(fh)


def find_latest_output(dir_out) -> Path:
    """Newest MODEL_OUTPUT.csv under a SISEPUEDE output directory."""
    pat = os.path.join(str(dir_out), "sisepuede_run_*", "*_output_database", "MODEL_OUTPUT.csv")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(f"No MODEL_OUTPUT.csv under {dir_out} (pattern: {pat})")
    return Path(max(hits, key=os.path.getmtime))


def load_baseline(model_output_csv, primary_id: int = 0, t0_year: int = T0_YEAR) -> pd.DataFrame:
    """Baseline sub-sector emissions by year: columns = ['year', <subsector codes...>]."""
    df = pd.read_csv(model_output_csv)
    df = df[df["primary_id"] == primary_id].sort_values("time_period").copy()
    fields = [c for c in df.columns if c.startswith(EMISSION_PREFIX)]
    out = pd.DataFrame({"year": t0_year + df["time_period"].astype(int).values})
    for f in fields:
        out[f.replace(EMISSION_PREFIX, "")] = df[f].values
    return out.reset_index(drop=True)


def _row(df_base: pd.DataFrame, year: int) -> pd.Series:
    hit = df_base[df_base["year"] == year]
    if hit.empty:
        raise ValueError(f"year {year} not in baseline (have {df_base['year'].min()}-{df_base['year'].max()})")
    return hit.iloc[0]


def aggregate_ipcc(df_base: pd.DataFrame) -> pd.DataFrame:
    """IPCC-sector totals per year (Energy/IPPU/Agriculture/FOLU/Waste + totals)."""
    out = pd.DataFrame({"year": df_base["year"].values})
    for g, subs in IPCC_GROUPS.items():
        present = [s for s in subs if s in df_base.columns]
        out[g] = df_base[present].sum(axis=1).values
    out["Total_incl_FOLU"] = out[list(IPCC_GROUPS)].sum(axis=1)
    out["Total_excl_FOLU"] = out["Total_incl_FOLU"] - out["FOLU"]
    out["AFOLU"] = out["Agriculture"] + out["FOLU"]
    return out


# --------------------------------------------------------------------------- #
# Factors
# --------------------------------------------------------------------------- #
def build_comparison(df_base: pd.DataFrame, ref: dict, year: int = 2019) -> pd.DataFrame:
    """Per sub-sector: model vs BUR3 target at `year`, the implied factor, and lever.

    Target resolution:
      - explicit value in subsector_targets_<year> -> use it (Energy sub-sectors, IPPU)
      - Agriculture/Waste member -> sector factor (scale all members equally to sector total)
      - frst -> keep as modelled (sink); lndu -> absorbs FOLU gap incl. peat
    """
    row = _row(df_base, year)
    sub_tgt = ref.get(f"subsector_targets_{year}", {})
    sec_tgt = ref.get(f"sector_targets_{year}", {})
    folu = ref.get(f"folu_{year}", {})
    agg = aggregate_ipcc(df_base)
    sec_model = _row(agg, year)

    recs = []
    for sub, lever in LEVERS.items():
        if sub not in df_base.columns:
            continue
        model = float(row[sub])
        sector = SUB_TO_SECTOR[sub]

        if sub == "frst":
            target, factor, note = model, 1.0, "forest sink kept as modelled"
        elif sub == "lndu":
            target_folu = float(folu.get("net_total", sec_tgt.get("FOLU", np.nan)))
            frst_model = float(row.get("frst", 0.0))
            target = target_folu - frst_model           # lndu must supply the rest
            factor = target / model if model else 1.0
            note = "STRUCTURAL_GAP: peat folded into lndu (no explicit peat variable)"
        elif sub_tgt.get(sub) is not None:
            target = float(sub_tgt[sub])
            factor = target / model if model else 1.0
            note = ""
        elif sector in ("Agriculture", "Waste") and sec_tgt.get(sector) is not None:
            factor = float(sec_tgt[sector]) / float(sec_model[sector]) if sec_model[sector] else 1.0
            target = model * factor
            note = f"{sector} sector factor"
        else:
            target, factor, note = model, 1.0, "no target"

        recs.append({
            "subsector": sub, "ipcc_sector": sector, "lever": lever,
            f"model_{year}": round(model, 1), f"bur3_{year}": round(target, 1),
            "factor": factor, "applied": lever in ("ef", "demand"),  # full precision
            "note": note,
        })

    df = pd.DataFrame(recs)
    order = {"Energy": 0, "IPPU": 1, "Agriculture": 2, "FOLU": 3, "Waste": 4}
    return df.sort_values(["ipcc_sector", "subsector"], key=lambda s: s.map(order).fillna(9) if s.name == "ipcc_sector" else s).reset_index(drop=True)


def factors_dict(comparison: pd.DataFrame) -> dict:
    return dict(zip(comparison["subsector"], comparison["factor"]))


# --------------------------------------------------------------------------- #
# Apply to inputs + predict
# --------------------------------------------------------------------------- #
def ef_fields(cols, sub):
    return [c for c in cols if c.startswith(f"ef_{sub}_")]


def demand_fields(cols, sub):
    return [c for c in cols if re.search(rf"scalar_{sub}.*energy_demand", c)]


def fields_for(cols, sub: str, lever: str):
    if lever == "ef":
        return ef_fields(cols, sub)
    if lever == "demand":
        return demand_fields(cols, sub)
    return []


def apply_factors(df_input: pd.DataFrame, comparison: pd.DataFrame):
    """Return (calibrated_input, change_log). Only 'ef'/'demand' levers are applied."""
    df_new = df_input.copy()
    cols = list(df_input.columns)
    log = []
    for r in comparison.itertuples(index=False):
        f = float(r.factor)
        flds = fields_for(cols, r.subsector, r.lever) if r.applied else []
        if r.applied and flds and not np.isclose(f, 1.0):
            df_new[flds] = df_new[flds] * f
        log.append({
            "subsector": r.subsector, "lever": r.lever, "factor": round(f, 4),
            "n_fields": len(flds), "applied": bool(r.applied and flds and not np.isclose(f, 1.0)),
            "example_field": flds[0] if flds else "",
        })
    return df_new, pd.DataFrame(log)


def predict(df_base: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """Analytic calibrated sub-sector emissions = model * applied factor (linear).

    'manual' sub-sectors are left unchanged (factor reported but not applied).
    """
    out = df_base.copy()
    for r in comparison.itertuples(index=False):
        if r.subsector in out.columns and r.applied:
            out[r.subsector] = out[r.subsector] * float(r.factor)
    return out


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def _selfcheck():
    # synthetic baseline: one year, three sub-sectors
    df_base = pd.DataFrame({"year": [2019], "fgtv": [300.0], "inen": [250.0], "frst": [-100.0], "lndu": [500.0]})
    ref = {
        "subsector_targets_2019": {"fgtv": 20.0, "inen": 125.0},
        "sector_targets_2019": {"Energy": 999, "FOLU": 900},
        "folu_2019": {"net_total": 900.0},
    }
    comp = build_comparison(df_base, ref, 2019)
    fac = factors_dict(comp)
    assert np.isclose(fac["fgtv"], 20.0 / 300.0), fac["fgtv"]        # ef factor
    assert np.isclose(fac["inen"], 125.0 / 250.0), fac["inen"]        # demand factor
    assert np.isclose(fac["frst"], 1.0)                              # sink kept
    assert np.isclose(fac["lndu"], (900.0 - (-100.0)) / 500.0), fac["lndu"]  # absorbs FOLU gap

    # apply to a synthetic wide input
    df_in = pd.DataFrame({
        "ef_fgtv_x": [1.0, 1.0], "ef_fgtv_y": [2.0, 2.0],
        "scalar_inen_energy_demand_z": [1.0, 1.0],
        "ef_frst_a": [5.0, 5.0], "ef_lndu_b": [3.0, 3.0],
        "unrelated": [7.0, 7.0],
    })
    df_cal, log = apply_factors(df_in, comp)
    assert np.allclose(df_cal["ef_fgtv_x"], 1.0 * fac["fgtv"])        # fgtv EF scaled
    assert np.allclose(df_cal["scalar_inen_energy_demand_z"], 1.0 * fac["inen"])  # demand scaled
    assert np.allclose(df_cal["ef_frst_a"], 5.0)                     # frst factor 1 -> untouched
    assert np.allclose(df_cal["ef_lndu_b"], 3.0 * fac["lndu"])       # lndu scaled
    assert np.allclose(df_cal["unrelated"], 7.0)                     # other cols untouched

    # predicted calibrated emissions hit targets
    pred = predict(df_base, comp)
    assert np.isclose(pred["fgtv"].iloc[0], 20.0, atol=0.1)
    assert np.isclose(pred["inen"].iloc[0], 125.0, atol=0.1)
    assert np.isclose(pred["frst"].iloc[0] + pred["lndu"].iloc[0], 900.0, atol=0.1)  # FOLU net
    print("bur3_calibration self-check: OK")


if __name__ == "__main__":
    _selfcheck()
