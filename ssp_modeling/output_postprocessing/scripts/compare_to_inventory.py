#!/usr/bin/env python3
"""
Compare SISEPUEDE model output to national GHG inventory.

Country-agnostic diagnostic tool. Works with any emission_targets_{country}_{year}.csv.

Usage (CLI):
  python compare_to_inventory.py --targets targets.csv --output WIDE_INPUTS_OUTPUTS.csv --tp 7
  python compare_to_inventory.py ... --top 5 --explain     Show top 5 with annotations

Usage (Python API):
  from compare_to_inventory import compare, DiagnosticConfig, DAG_AFFECTS
  config = DiagnosticConfig(tp=7, threshold=0.15)
  diff, flagged, diag = compare("targets.csv", "WIDE.csv", config, verbose=False)

Outputs (saved to {output_dir}/diagnostics/):
  diff_report.csv      Full comparison with impact rank, components, trajectory
  flagged.csv          Component breakdown for flagged rows
  diagnostics.csv      Structural warnings (zero outputs, sign mismatches, etc.)
"""

import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

MAGNITUDE_RATIO = 10.0      # Model/inventory ratio triggering MAGNITUDE_10X
DOMINANCE_FRAC = 0.95       # Single component share triggering SINGLE_DOMINANCE
TRAJECTORY_GROW = 1.0       # >100% growth triggers TRAJECTORY warning
TRAJECTORY_SHRINK = -0.5    # >50% decline triggers TRAJECTORY warning
GDP_MIN_GROWTH = 0.02       # Minimum GDP growth to run elasticity checks
DECLINE_THRESH = -0.05      # Sector decline triggering DECLINING_WITH_GDP_GROWTH
ELASTICITY_LAG = 0.30       # Implicit elasticity below this triggers GROWTH_LAG
ELASTICITY_HIGH = 2.50      # Implicit elasticity above this triggers GROWTH_EXCEEDS_GDP
WASTE_POP_RATIO = 3.0       # Waste/population growth ratio triggering POPULATION_MISMATCH
GAS_RATIO_TOL = 0.20        # CO2 accuracy tolerance for GAS_RATIO check
GAS_RATIO_DEVIATE = 0.50    # CH4/N2O deviation triggering GAS_RATIO
BASE_YEAR = 2015

DAG_ORDER = {
    'lvst': 1, 'lsmm': 1, 'agrc': 1, 'soil': 1, 'lndu': 1, 'frst': 1,
    'waso': 2, 'trww': 2, 'wali': 2, 'ippu': 3,
    'inen': 4, 'scoe': 4, 'trns': 4, 'entc': 5, 'enfu': 5, 'fgtv': 6, 'ccsq': 7,
}

DAG_AFFECTS: Dict[str, List[str]] = {
    # AFOLU (Step 1) — source: sisepuede/models/circular_economy.py:248, energy_consumption.py:299
    'lvst': ['lsmm', 'soil', 'lndu', 'agrc', 'waso', 'inen'],  # waso: animal mass/demand; inen: ag energy
    'lsmm': ['soil', 'entc'],               # entc: recovered manure biogas (energy_production.py:358)
    'agrc': ['soil', 'inen', 'entc', 'waso'],  # waso: food loss to MSW (circular_economy.py:249)
    'soil': [], 'lndu': ['frst', 'soil', 'agrc'], 'frst': [],
    # Circular Economy (Step 2) — source: ippu.py:234, energy_production.py:360-367
    'waso': ['entc', 'ippu'],               # ippu: recycled waste reduces production; entc: biogas+incineration
    'trww': ['entc'],                        # entc: WW recovered biogas (energy_production.py:360)
    'wali': [],
    # IPPU (Step 3)
    'ippu': ['inen', 'entc'],
    # Energy (Steps 4-6)
    'inen': ['entc', 'fgtv'], 'scoe': ['entc', 'fgtv'], 'trns': ['entc', 'fgtv'],
    'entc': ['fgtv'], 'fgtv': [], 'ccsq': [],
}

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    'Electricity and Heat Generation': 'CO2/CH4/N2O from burning coal, gas, oil for electricity',
    'Fuel Production': 'Emissions from mining, refining, and processing fossil fuels',
    'Industrial Combustion': 'Fuel burning in factories (cement kilns, steel furnaces)',
    'Transportation': 'Vehicle exhaust from road, rail, aviation, and maritime',
    'Other Combustion': 'Fuel burning in homes, offices, and farms',
    'Fugitive Emissions': 'Gas leaks from pipelines, mines, and fuel storage',
    'IPPU': 'Chemical reactions in industry (calcination, smelting, refrigerant leaks)',
    'Livestock': 'Methane from enteric fermentation and manure decomposition',
    'Agriculture and Managed Soil': 'N2O from fertilizer, manure, crop residues, urea',
    'Solid Waste': 'Methane from organic waste in landfills and open dumps',
    'Wastewater Treatment': 'CH4 and N2O from sewage treatment and discharge',
    'CCSQ': 'Carbon captured and sequestered (negative emissions)',
}


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic comparison."""
    tp: int = 7
    strategy: int = 0
    threshold: float = 0.15
    min_magnitude: float = 0.01
    show_top: int = 10
    explain: bool = False


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_targets(path: Path) -> pd.DataFrame:
    """Load targets CSV, auto-detect country value column, rename to 'inventory'."""
    df = pd.read_csv(path)
    standard = {'subsector_ssp', 'sector', 'subsector', 'category',
                'aggregation_category', 'gas', 'ID', 'vars', 'ids',
                'target_source', 'description', 'fixability', 'notes'}
    val_cols = [c for c in df.columns if c not in standard]
    if not val_cols:
        raise ValueError(f"No country value column found in {path}")
    return df.rename(columns={val_cols[-1]: 'inventory'})


def load_model_row(path: Path, tp: int, strategy: int = 0) -> pd.Series:
    """Load a single time-period row from WIDE_INPUTS_OUTPUTS.csv."""
    df = pd.read_csv(path)
    if 'time_period' not in df.columns:
        raise FileNotFoundError(
            f"'{Path(path).name}' has no 'time_period' column. "
            "--output must point to WIDE_INPUTS_OUTPUTS.csv"
        )
    mask = df['time_period'] == tp
    if 'primary_id' in df.columns:
        mask &= df['primary_id'] == strategy
    if mask.sum() == 0:
        raise ValueError(f"No data for tp={tp}, strategy={strategy}")
    return df.loc[mask].iloc[0]


def load_full_model(path: Path, strategy: int = 0) -> Optional[pd.DataFrame]:
    """Load full model output for trajectory analysis. Returns None on error."""
    try:
        df = pd.read_csv(path)
        return df[df['primary_id'] == strategy] if 'primary_id' in df.columns else df
    except Exception:
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_vars(row) -> List[str]:
    """Parse colon-separated variable list from a targets row."""
    v = row.get('vars', '')
    return [x.strip() for x in str(v).split(':') if x.strip()] if not pd.isna(v) and v else []


def sum_vars(var_list: List[str], row: pd.Series) -> float:
    """Sum model values for variables present in row."""
    return sum(row.get(v, 0.0) for v in var_list if v in row.index)


def short_name(v: str) -> str:
    """Remove common prefixes for readable output."""
    return v.replace('emission_co2e_', '').replace('_nbmass', '')


def _category_label(row: pd.Series) -> str:
    """Format 'ID (Description)' from a diff row."""
    cat = row.get('category', row['ID'])
    desc = str(cat).split(' - ', 1)[1] if ' - ' in str(cat) else ''
    return f"{row['ID']} ({desc})" if desc else str(row['ID'])


# ── Core Comparison ───────────────────────────────────────────────────────────

def build_diff(targets: pd.DataFrame, model_row: pd.Series,
               full_model: Optional[pd.DataFrame], tp: int,
               min_mag: float) -> pd.DataFrame:
    """Build comparison DataFrame: targets vs model at given time period."""
    tp0_row = None
    if full_model is not None:
        tp0 = full_model[full_model['time_period'] == 0]
        if len(tp0) > 0:
            tp0_row = tp0.iloc[0]

    rows = []
    for _, t in targets.iterrows():
        vl = parse_vars(t)
        found = [v for v in vl if v in model_row.index]
        total = sum(model_row[v] for v in found)
        top_var, top_val, top_share = '', 0.0, 0.0
        if found and abs(total) > 1e-8:
            vals = {v: model_row[v] for v in found}
            top_var = max(vals, key=lambda k: abs(vals[k]))
            top_val = vals[top_var]
            top_share = abs(top_val) / abs(total) * 100
        rows.append({
            'model': total,
            'model_tp0': sum_vars(vl, tp0_row) if tp0_row is not None else np.nan,
            'n_vars': len(found), 'n_missing': len(vl) - len(found),
            'top_component': short_name(top_var) if top_var else '',
            'top_component_value': top_val, 'top_component_share': round(top_share, 1),
        })

    diff = targets.copy()
    for col, vals in pd.DataFrame(rows).items():
        diff[col] = vals.values

    diff['diff'] = diff['model'] - diff['inventory']
    mask = diff['inventory'].abs() > min_mag
    diff['error_pct'] = np.where(mask, diff['diff'].abs() / diff['inventory'].abs() * 100, np.nan)
    tp0_mask = diff['model_tp0'].abs() > min_mag
    diff['model_growth_pct'] = np.where(
        tp0_mask, (diff['model'] - diff['model_tp0']) / diff['model_tp0'].abs() * 100, np.nan
    )
    total_inv = diff['inventory'].abs().sum()
    diff['inventory_share'] = diff['inventory'].abs() / total_inv * 100 if total_inv > 0 else 0
    diff['direction'] = np.where(diff['diff'] > min_mag * 0.1, 'over',
                         np.where(diff['diff'] < -min_mag * 0.1, 'under', 'match'))
    diff['abs_impact_rank'] = diff['diff'].abs().rank(ascending=False, method='min').astype(int)
    for col in ('target_source', 'fixability'):
        diff[col] = targets[col].values if col in targets.columns else ''

    col_order = ['abs_impact_rank', 'ID', 'subsector_ssp', 'sector', 'category',
                 'aggregation_category', 'gas', 'inventory', 'model', 'model_tp0',
                 'diff', 'error_pct', 'direction', 'model_growth_pct', 'inventory_share',
                 'top_component', 'top_component_value', 'top_component_share',
                 'fixability', 'n_vars', 'n_missing', 'target_source', 'vars']
    existing = [c for c in col_order if c in diff.columns]
    return diff[existing + [c for c in diff.columns if c not in col_order]]


def get_components(vars_str: str, model_row: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Extract top N component variables by absolute value."""
    vl = parse_vars({'vars': vars_str})
    c = [{'var': v, 'val': model_row.get(v, 0.0)} for v in vl if v in model_row.index]
    df = pd.DataFrame(c)
    return df.reindex(df['val'].abs().sort_values(ascending=False).index).head(top_n) if len(df) else df


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _check_row_diagnostics(row, targets_row, model_row, full_model, tp, min_mag):
    """Per-row checks: ZERO_OUTPUT, SIGN_MISMATCH, MAGNITUDE_10X, SINGLE_DOMINANCE, MISSING_VARS, TRAJECTORY."""
    warnings = []
    inv, mod, rid = row['inventory'], row['model'], row['ID']
    cat, gas, sec = row.get('category', ''), row.get('gas', ''), row.get('sector', '')
    vl = parse_vars(targets_row)

    if abs(inv) < min_mag and abs(mod) < min_mag:
        return warnings

    def w(issue, sev, detail):
        warnings.append(dict(ID=rid, category=cat, gas=gas, sector=sec,
                             inventory=inv, model=mod, issue=issue, severity=sev, detail=detail))

    if abs(mod) < 1e-6 and abs(inv) > min_mag:
        zv = sum(1 for v in vl if v in model_row.index and abs(model_row[v]) < 1e-8)
        w('ZERO_OUTPUT', 'HIGH', f'Target={inv:.4f} but model=0. {zv}/{len(vl)} vars zero. Needs input values, not rescaling.')
        return warnings

    if inv * mod < 0 and abs(inv) > min_mag and abs(mod) > min_mag:
        w('SIGN_MISMATCH', 'HIGH', f'Inventory={inv:+.3f}, model={mod:+.3f}. Disagree on source vs sink.')

    if abs(inv) > min_mag and abs(mod) > min_mag:
        ratio = abs(mod / inv)
        if ratio > MAGNITUDE_RATIO or ratio < 1 / MAGNITUDE_RATIO:
            w('MAGNITUDE_10X', 'HIGH', f'Ratio={ratio:.1f}x. Possible unit error or missing parameter.')

    if len(vl) > 1 and abs(mod) > min_mag:
        vals = {v: model_row.get(v, 0.0) for v in vl if v in model_row.index}
        if vals:
            mx = max(abs(x) for x in vals.values())
            if mx / abs(mod) > DOMINANCE_FRAC:
                dom = max(vals, key=lambda k: abs(vals[k]))
                zeros = [v for v, x in vals.items() if abs(x) < 1e-8 and v != dom]
                if zeros:
                    w('SINGLE_DOMINANCE', 'MEDIUM',
                      f'{short_name(dom)[:40]} is {mx/abs(mod)*100:.0f}% of total. {len(zeros)} other vars are zero.')

    missing = [v for v in vl if v not in model_row.index]
    if missing:
        w('MISSING_VARS', 'LOW', f'{len(missing)} expected vars not in model output.')

    if full_model is not None and tp > 0 and abs(inv) > min_mag:
        tp0 = full_model[full_model['time_period'] == 0]
        if len(tp0) > 0:
            v0 = sum_vars(vl, tp0.iloc[0])
            if abs(v0) > min_mag:
                g = (mod - v0) / abs(v0)
                if abs(g) > TRAJECTORY_GROW or g < TRAJECTORY_SHRINK:
                    w('TRAJECTORY', 'MEDIUM', f'Changed {g*100:+.0f}% from tp=0 ({v0:.3f}) to tp={tp} ({mod:.3f}).')

    return warnings


def _check_growth_diagnostics(targets, diff, full_model, tp, min_mag):
    """Sector-level growth checks: DECLINING_WITH_GDP, GROWTH_LAG, GROWTH_EXCEEDS_GDP, POPULATION_MISMATCH."""
    warnings = []
    if full_model is None or tp == 0:
        return warnings

    tp0_df = full_model[full_model['time_period'] == 0]
    tp_df = full_model[full_model['time_period'] == tp]
    if len(tp0_df) == 0 or len(tp_df) == 0:
        return warnings
    tp0_r, tp_r = tp0_df.iloc[0], tp_df.iloc[0]

    gdp0, gdp_t = tp0_r.get('gdp_mmm_usd', 0), tp_r.get('gdp_mmm_usd', 0)
    gdp_growth = (gdp_t / gdp0 - 1) if gdp0 > 0 else 0
    pop0 = tp0_r.get('population_gnrl_rural', 0) + tp0_r.get('population_gnrl_urban', 0)
    pop_t = tp_r.get('population_gnrl_rural', 0) + tp_r.get('population_gnrl_urban', 0)
    pop_growth = (pop_t / pop0 - 1) if pop0 > 0 else 0

    def w(sec, issue, detail):
        warnings.append(dict(ID=sec, category=sec, gas='ALL', sector=sec,
                             inventory=0, model=0, issue=issue, severity='MEDIUM', detail=detail))

    if gdp_growth > GDP_MIN_GROWTH:
        for sec_name, sec_grp in diff.groupby('sector'):
            vl_all = list(set(v for _, r in sec_grp.iterrows()
                              for t in [targets[targets['ID'] == r['ID']]]
                              if len(t) > 0 for v in parse_vars(t.iloc[0])))
            if not vl_all:
                continue
            s0 = sum(tp0_r.get(v, 0) for v in vl_all if v in tp0_r.index)
            st = sum(tp_r.get(v, 0) for v in vl_all if v in tp_r.index)
            if abs(s0) < min_mag:
                continue
            sec_g = (st / s0 - 1)
            elast = sec_g / gdp_growth
            ann_s = ((st / s0) ** (1 / tp) - 1) * 100 if s0 > 0 else 0
            ann_g = ((gdp_t / gdp0) ** (1 / tp) - 1) * 100

            if sec_g < DECLINE_THRESH and gdp_growth > 0.05:
                w(sec_name, 'DECLINING_WITH_GDP_GROWTH',
                  f'{sec_name} emissions {ann_s:+.1f}%/yr while GDP grows {ann_g:+.1f}%/yr (elasticity={elast:+.2f}).')
            elif 0 < elast < ELASTICITY_LAG and sec_g > 0.01:
                w(sec_name, 'GROWTH_LAG',
                  f'{sec_name} grows {ann_s:+.1f}%/yr vs GDP {ann_g:+.1f}%/yr (elasticity={elast:+.2f}).')
            elif elast > ELASTICITY_HIGH and sec_g > 0.10:
                w(sec_name, 'GROWTH_EXCEEDS_GDP',
                  f'{sec_name} grows {ann_s:+.1f}%/yr vs GDP {ann_g:+.1f}%/yr (elasticity={elast:+.2f}).')

    if pop_growth > 0.01:
        for sec_name in ['4 - Waste']:
            sec_rows = diff[diff['sector'].str.startswith(sec_name[:3])]
            vl_pop = list(set(v for _, r in sec_rows.iterrows()
                              for t in [targets[targets['ID'] == r['ID']]]
                              if len(t) > 0 for v in parse_vars(t.iloc[0])))
            if not vl_pop:
                continue
            w0 = sum(tp0_r.get(v, 0) for v in vl_pop if v in tp0_r.index)
            wt = sum(tp_r.get(v, 0) for v in vl_pop if v in tp_r.index)
            if abs(w0) < min_mag:
                continue
            ratio = (wt / w0 - 1) / pop_growth if pop_growth > 0.01 else 0
            if ratio > WASTE_POP_RATIO:
                w(sec_name, 'POPULATION_MISMATCH',
                  f'{sec_name} grows {(wt/w0-1)*100:+.0f}% vs population {pop_growth*100:+.0f}% ({ratio:.1f}x faster).')

    return warnings


def _check_gas_ratios(diff, min_mag):
    """Cross-gas check: if CO2 matches but CH4/N2O don't."""
    warnings = []
    for (_, sec), grp in diff.groupby(['subsector', 'sector']):
        co2 = grp[grp['gas'] == 'CO2']
        if len(co2) == 0 or co2.iloc[0]['inventory'] < min_mag:
            continue
        r_co2 = co2.iloc[0]['model'] / co2.iloc[0]['inventory']
        if abs(r_co2) < 0.01:
            continue
        for gn, gr in [('CH4', grp[grp['gas'] == 'CH4']), ('N2O', grp[grp['gas'] == 'N2O'])]:
            if len(gr) == 0 or gr.iloc[0]['inventory'] < min_mag * 0.1:
                continue
            if abs(gr.iloc[0]['inventory']) < 1e-6:
                if abs(gr.iloc[0]['model']) > min_mag:
                    warnings.append(dict(
                        ID=gr.iloc[0]['ID'], category=gr.iloc[0].get('category', ''),
                        gas=gn, sector=sec, inventory=gr.iloc[0]['inventory'],
                        model=gr.iloc[0]['model'], issue='GAS_RATIO', severity='MEDIUM',
                        detail=f'Inventory {gn}=0 but model={gr.iloc[0]["model"]:.4f}. Check if inventory excludes {gn}.'))
                continue
            r_g = gr.iloc[0]['model'] / gr.iloc[0]['inventory']
            if abs(r_co2 - 1) < GAS_RATIO_TOL and abs(r_g - 1) > GAS_RATIO_DEVIATE:
                warnings.append(dict(
                    ID=gr.iloc[0]['ID'], category=gr.iloc[0].get('category', ''),
                    gas=gn, sector=sec, inventory=gr.iloc[0]['inventory'],
                    model=gr.iloc[0]['model'], issue='GAS_RATIO', severity='MEDIUM',
                    detail=f'CO2 ratio={r_co2:.0%} but {gn} ratio={r_g:.0%}. Check {gn} EFs.'))
    return warnings


def run_diagnostics(targets: pd.DataFrame, diff: pd.DataFrame, model_row: pd.Series,
                    full_model: Optional[pd.DataFrame], tp: int, min_mag: float) -> pd.DataFrame:
    """Run all diagnostic checks. Returns DataFrame of warnings."""
    w = []
    for idx, row in diff.iterrows():
        w.extend(_check_row_diagnostics(row, targets.iloc[idx], model_row, full_model, tp, min_mag))
    w.extend(_check_growth_diagnostics(targets, diff, full_model, tp, min_mag))
    w.extend(_check_gas_ratios(diff, min_mag))
    return pd.DataFrame(w) if w else pd.DataFrame()


# ── Public API ────────────────────────────────────────────────────────────────

def compare(targets_path, output_path, config: Optional[DiagnosticConfig] = None,
            save_dir: Optional[str] = None, verbose: bool = True
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare SISEPUEDE output to inventory. Returns (diff, flagged, diagnostics).

    Parameters
    ----------
    targets_path : path to emission_targets_{country}_{year}.csv
    output_path : path to WIDE_INPUTS_OUTPUTS.csv
    config : DiagnosticConfig (defaults to DiagnosticConfig())
    save_dir : directory for CSV output (None = don't save)
    verbose : print console report
    """
    cfg = config or DiagnosticConfig()
    targets = load_targets(Path(targets_path))
    model_row = load_model_row(Path(output_path), cfg.tp, cfg.strategy)
    full_model = load_full_model(Path(output_path), cfg.strategy)

    diff = build_diff(targets, model_row, full_model, cfg.tp, cfg.min_magnitude)
    sig = diff[diff['inventory'].abs() > cfg.min_magnitude].copy()
    threshold_pct = cfg.threshold * 100
    flagged = sig[sig['error_pct'] > threshold_pct].sort_values('diff', key=abs, ascending=False)
    diag = run_diagnostics(targets, diff, model_row, full_model, cfg.tp, cfg.min_magnitude)

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        diff.to_csv(out / 'diff_report.csv', index=False)
        if len(diag) > 0:
            diag.to_csv(out / 'diagnostics.csv', index=False)

    if verbose:
        _print_report(Path(targets_path), Path(output_path), cfg, diff, sig,
                      flagged, diag, model_row)

    return diff, flagged, diag


# ── Console Output ────────────────────────────────────────────────────────────

def _hdr(title: str, w: int = 80):
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def _sec(title: str):
    print(f"\n  {title}\n  {'-' * 76}")


def _print_report(targets_path, output_path, cfg, diff, sig, flagged, diag, model_row):
    """Print the full console diagnostic report."""
    total_err = sig['diff'].abs().sum()
    n_15 = (sig['error_pct'] <= 15).sum()
    n_25 = (sig['error_pct'] <= 25).sum()
    n_flag = (sig['error_pct'] > cfg.threshold * 100).sum()
    status = "GOOD" if total_err < 10 and n_15 > len(sig) * 0.5 else "NEEDS WORK"
    show_top = cfg.show_top if cfg.show_top > 0 else 9999

    print(f"\n  CALIBRATION: {total_err:.2f} MtCO2e | {n_15}/{len(sig)} within 15% | {status}")
    _hdr("SISEPUEDE CALIBRATION DIAGNOSTIC REPORT")
    print(f"  Targets:    {targets_path.name}\n  Model run:  {output_path.parent.name}")
    print(f"  Period:     tp={cfg.tp} (year {BASE_YEAR + cfg.tp})\n  Threshold:  {cfg.threshold*100:.0f}%")

    _hdr("1. SUMMARY")
    for label, val in [("Total absolute error", f"{total_err:8.2f} MtCO2e"),
                       ("Categories evaluated", f"{len(sig):8d}"),
                       ("Within 15%", f"{n_15:8d}  ({n_15/len(sig)*100:.0f}%)"),
                       ("Within 25%", f"{n_25:8d}  ({n_25/len(sig)*100:.0f}%)"),
                       ("Exceeding threshold", f"{n_flag:8d}")]:
        print(f"  {label + ':':25s}{val}")

    _hdr("2. SECTOR TOTALS")
    print(f"  {'Sector':<35s} {'Inventory':>10s} {'Model':>10s} {'Diff':>10s} {'Error':>8s}")
    print("  " + "-" * 75)
    for sec in sorted(diff['sector'].unique()):
        sub = diff[diff['sector'] == sec]
        inv_t, mod_t = sub['inventory'].sum(), sub['model'].sum()
        if abs(inv_t) > 0.01:
            print(f"  {sec:<35s} {inv_t:10.3f} {mod_t:10.3f} {mod_t-inv_t:+10.3f} {(mod_t-inv_t)/abs(inv_t)*100:+7.1f}%")

    flagged_rows = []
    if len(flagged) > 0:
        shown = min(show_top, len(flagged))
        _hdr(f"3. FLAGGED CATEGORIES (showing {shown} of {len(flagged)})")
        print(f"  {'Category':<50s} {'Inventory':>10s} {'Model':>10s} {'Diff':>10s} {'Error':>8s}")
        print("  " + "-" * 90)
        for i, (_, row) in enumerate(flagged.iterrows()):
            if i >= show_top:
                break
            inv, mod, d, e = row['inventory'], row['model'], row['diff'], row['error_pct']
            print(f"  {_category_label(row):<50s} {inv:10.3f} {mod:10.3f} {d:+10.3f} {e:7.1f}%")
            comps = get_components(row.get('vars', ''), model_row, top_n=5)
            for _, c in comps.iterrows():
                pct = c['val'] / mod * 100 if abs(mod) > 1e-6 else 0
                print(f"    {short_name(c['var']):<55s} {c['val']:8.4f}  ({pct:5.1f}%)")
                flagged_rows.append({'parent_ID': row['ID'], 'inventory': inv, 'model': mod,
                                     'diff': d, 'error_pct': e, 'component': c['var'],
                                     'value': c['val'], 'share_pct': pct})
            if cfg.explain:
                _print_explain(row)
            else:
                print()
        if shown < len(flagged):
            print(f"  ... {len(flagged) - shown} more flagged categories. Use --top 0 to see all.")

    if len(diag) > 0:
        for sev, label in [('HIGH', '4. HIGH-PRIORITY DIAGNOSTICS'),
                           ('MEDIUM', '5. MEDIUM-PRIORITY DIAGNOSTICS'),
                           ('LOW', '6. LOW-PRIORITY DIAGNOSTICS')]:
            sev_df = diag[diag['severity'] == sev]
            if len(sev_df) > 0:
                _hdr(f"{label} ({len(sev_df)})")
                for issue, grp in sev_df.groupby('issue'):
                    _sec(f"{issue} ({len(grp)})")
                    for _, w in grp.iterrows():
                        print(f"    {w['ID']:<30s}  {str(w['detail'])[:65]}")

    if len(flagged) > 0:
        _hdr("7. SUGGESTED NEXT STEPS")
        for i, (_, row) in enumerate(flagged.head(3).iterrows()):
            ssp = row.get('subsector_ssp', '')
            action = "increase" if row['diff'] < 0 else "decrease"
            tc = row.get('top_component', '')
            cascade = DAG_AFFECTS.get(ssp, [])
            note = f" (also affects: {', '.join(cascade)})" if cascade else ""
            print(f"  {i+1}. {_category_label(row)}: {action} {tc or ssp} by ~{abs(row['error_pct']):.0f}%{note}")
        print()

    _hdr("8. OUTPUT FILES")
    print(f"  (Use save_dir parameter or --out-dir flag to save CSV outputs)")
    print()

    return flagged_rows


def _print_explain(row: pd.Series):
    """Print learning annotations (WHAT/TRAJECTORY/DAG/FIX) for a flagged row."""
    agg = row.get('aggregation_category', '')
    desc = CATEGORY_DESCRIPTIONS.get(agg, '')
    if desc:
        print(f"      WHAT:  {desc}")
    tp0, growth = row.get('model_tp0', np.nan), row.get('model_growth_pct', np.nan)
    if not np.isnan(tp0) and not np.isnan(growth):
        print(f"      TRAJECTORY: tp=0 model={tp0:.3f}, model={row['model']:.3f} ({growth:+.0f}% growth)")
    affected = DAG_AFFECTS.get(row.get('subsector_ssp', ''), [])
    if affected:
        print(f"      DAG:   Also affects: {', '.join(affected)}")
    fix = row.get('fixability', '')
    if fix:
        print(f"      FIX:   {fix}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Compare SISEPUEDE output to GHG inventory')
    p.add_argument('--targets', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tp', type=int, default=7)
    p.add_argument('--strategy', type=int, default=0)
    p.add_argument('--threshold', type=float, default=0.15)
    p.add_argument('--min-magnitude', type=float, default=0.01)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--top', type=int, default=10, help='Top N flagged (0=all)')
    p.add_argument('--explain', action='store_true', help='Add learning annotations')
    args = p.parse_args()

    out_dir = args.out_dir or str(Path(args.output).parent / 'diagnostics')
    cfg = DiagnosticConfig(tp=args.tp, strategy=args.strategy, threshold=args.threshold,
                           min_magnitude=args.min_magnitude, show_top=args.top, explain=args.explain)
    compare(args.targets, args.output, cfg, save_dir=out_dir, verbose=True)


if __name__ == '__main__':
    main()
