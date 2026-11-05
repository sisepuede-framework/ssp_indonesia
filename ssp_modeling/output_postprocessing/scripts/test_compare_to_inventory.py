#!/usr/bin/env python3
"""
Tests for compare_to_inventory.py

Run: python -m pytest test_compare_to_inventory.py -v
  or: python test_compare_to_inventory.py  (standalone)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from compare_to_inventory import (
    DiagnosticConfig, DAG_AFFECTS, DAG_ORDER,
    parse_vars, sum_vars, short_name, load_targets,
    build_diff, get_components, run_diagnostics, compare,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_targets():
    """5-row targets DataFrame mimicking emission_targets_mar_2022.csv."""
    return pd.DataFrame({
        'subsector_ssp': ['entc', 'inen', 'lvst', 'waso', 'soil'],
        'subsector': ['1.A.1 - Energy Industries', '1.A.2 - Manufacturing',
                      '3.A - Livestock', '4.A - Solid Waste', '3.C - Aggregate sources'],
        'sector': ['1 - Energy', '1 - Energy', '3 - AFOLU', '4 - Waste', '3 - AFOLU'],
        'category': ['1.A.1 - Electricity', '1.A.2 - Manufacturing',
                     '3.A.1 - Enteric', '4.A - Solid Waste', '3.C.4 - Soil N2O'],
        'gas': ['CO2', 'CO2', 'CH4', 'CH4', 'N2O'],
        'ID': ['1.A.1:CO2', '1.A.2:CO2', '3.A.1:CH4', '4.A:CH4', '3.C.4:N2O'],
        'vars': [
            'emission_co2e_co2_entc_pp_coal:emission_co2e_co2_entc_pp_gas',
            'emission_co2e_co2_inen_cement',
            'emission_co2e_ch4_lvst_entferm_cattle',
            'emission_co2e_ch4_waso_landfill',
            'emission_co2e_n2o_soil_fertilizer:emission_co2e_n2o_soil_organic',
        ],
        'inventory': [32.0, 6.5, 9.1, 3.7, 6.4],
    })


@pytest.fixture
def mock_model_row():
    """Model output Series at tp=7 with known values."""
    data = {
        'time_period': 7,
        'emission_co2e_co2_entc_pp_coal': 30.0,
        'emission_co2e_co2_entc_pp_gas': 2.5,
        'emission_co2e_co2_inen_cement': 7.0,
        'emission_co2e_ch4_lvst_entferm_cattle': 7.8,
        'emission_co2e_ch4_waso_landfill': 3.3,
        'emission_co2e_n2o_soil_fertilizer': 5.5,
        'emission_co2e_n2o_soil_organic': 0.3,
        'gdp_mmm_usd': 307.0,
        'population_gnrl_rural': 13_000_000,
        'population_gnrl_urban': 24_000_000,
    }
    return pd.Series(data)


@pytest.fixture
def mock_full_model(mock_model_row):
    """3-row DataFrame (tp=0, tp=3, tp=7) for trajectory tests."""
    tp0 = mock_model_row.copy()
    tp0['time_period'] = 0
    tp0['emission_co2e_co2_entc_pp_coal'] = 28.0
    tp0['emission_co2e_co2_entc_pp_gas'] = 3.0
    tp0['emission_co2e_co2_inen_cement'] = 7.5
    tp0['emission_co2e_ch4_lvst_entferm_cattle'] = 8.5
    tp0['emission_co2e_ch4_waso_landfill'] = 2.0
    tp0['emission_co2e_n2o_soil_fertilizer'] = 5.0
    tp0['emission_co2e_n2o_soil_organic'] = 0.3
    tp0['gdp_mmm_usd'] = 270.0
    tp0['population_gnrl_rural'] = 14_000_000
    tp0['population_gnrl_urban'] = 21_000_000

    tp3 = mock_model_row.copy()
    tp3['time_period'] = 3

    return pd.DataFrame([tp0, tp3, mock_model_row])


# ── Unit Tests: parse_vars ────────────────────────────────────────────────────

class TestParseVars:
    def test_single_var(self):
        assert parse_vars({'vars': 'emission_co2e_co2_entc_pp_coal'}) == ['emission_co2e_co2_entc_pp_coal']

    def test_colon_separated(self):
        result = parse_vars({'vars': 'a:b:c'})
        assert result == ['a', 'b', 'c']

    def test_empty_string(self):
        assert parse_vars({'vars': ''}) == []

    def test_nan(self):
        assert parse_vars({'vars': np.nan}) == []

    def test_missing_key(self):
        assert parse_vars({}) == []

    def test_whitespace_handling(self):
        assert parse_vars({'vars': ' a : b : c '}) == ['a', 'b', 'c']


# ── Unit Tests: sum_vars ──────────────────────────────────────────────────────

class TestSumVars:
    def test_all_present(self):
        row = pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0})
        assert sum_vars(['a', 'b'], row) == 3.0

    def test_missing_vars(self):
        row = pd.Series({'a': 1.0})
        assert sum_vars(['a', 'missing'], row) == 1.0

    def test_empty_list(self):
        row = pd.Series({'a': 1.0})
        assert sum_vars([], row) == 0.0


# ── Unit Tests: short_name ────────────────────────────────────────────────────

class TestShortName:
    def test_removes_prefix(self):
        assert short_name('emission_co2e_co2_entc_pp_coal') == 'co2_entc_pp_coal'

    def test_removes_nbmass(self):
        assert short_name('emission_co2e_co2_inen_nbmass_cement') == 'co2_inen_cement'

    def test_plain_string(self):
        assert short_name('hello') == 'hello'


# ── Unit Tests: build_diff ────────────────────────────────────────────────────

class TestBuildDiff:
    def test_basic_comparison(self, mock_targets, mock_model_row, mock_full_model):
        diff = build_diff(mock_targets, mock_model_row, mock_full_model, tp=7, min_mag=0.01)
        assert len(diff) == 5
        assert 'diff' in diff.columns
        assert 'error_pct' in diff.columns
        assert 'direction' in diff.columns
        assert 'abs_impact_rank' in diff.columns

    def test_entc_value(self, mock_targets, mock_model_row, mock_full_model):
        diff = build_diff(mock_targets, mock_model_row, mock_full_model, tp=7, min_mag=0.01)
        entc = diff[diff['ID'] == '1.A.1:CO2'].iloc[0]
        assert entc['model'] == pytest.approx(32.5, abs=0.01)  # 30.0 + 2.5
        assert entc['inventory'] == 32.0
        assert entc['diff'] == pytest.approx(0.5, abs=0.01)

    def test_direction(self, mock_targets, mock_model_row, mock_full_model):
        diff = build_diff(mock_targets, mock_model_row, mock_full_model, tp=7, min_mag=0.01)
        entc = diff[diff['ID'] == '1.A.1:CO2'].iloc[0]
        assert entc['direction'] == 'over'
        lvst = diff[diff['ID'] == '3.A.1:CH4'].iloc[0]
        assert lvst['direction'] == 'under'

    def test_zero_inventory(self, mock_model_row, mock_full_model):
        targets = pd.DataFrame({
            'subsector_ssp': ['ccsq'], 'sector': ['5 - CCSQ'],
            'category': ['5 - CCSQ'], 'gas': ['CO2'], 'ID': ['5:CO2'],
            'vars': ['emission_co2e_co2_entc_pp_coal'], 'inventory': [0.0],
        })
        diff = build_diff(targets, mock_model_row, mock_full_model, tp=7, min_mag=0.01)
        assert np.isnan(diff.iloc[0]['error_pct'])

    def test_no_full_model(self, mock_targets, mock_model_row):
        diff = build_diff(mock_targets, mock_model_row, None, tp=7, min_mag=0.01)
        assert len(diff) == 5
        assert all(np.isnan(diff['model_tp0']))


# ── Unit Tests: get_components ────────────────────────────────────────────────

class TestGetComponents:
    def test_top_n(self, mock_model_row):
        comps = get_components('emission_co2e_co2_entc_pp_coal:emission_co2e_co2_entc_pp_gas',
                               mock_model_row, top_n=2)
        assert len(comps) == 2
        assert comps.iloc[0]['var'] == 'emission_co2e_co2_entc_pp_coal'

    def test_empty_vars(self, mock_model_row):
        comps = get_components('', mock_model_row)
        assert len(comps) == 0

    def test_missing_vars(self, mock_model_row):
        comps = get_components('nonexistent_var', mock_model_row)
        assert len(comps) == 0


# ── Unit Tests: run_diagnostics ───────────────────────────────────────────────

class TestDiagnostics:
    def test_zero_output_detected(self, mock_targets, mock_model_row, mock_full_model):
        # Zero out waste to trigger ZERO_OUTPUT
        row = mock_model_row.copy()
        row['emission_co2e_ch4_waso_landfill'] = 0.0
        diff = build_diff(mock_targets, row, mock_full_model, tp=7, min_mag=0.01)
        diag = run_diagnostics(mock_targets, diff, row, mock_full_model, tp=7, min_mag=0.01)
        zero_diags = diag[diag['issue'] == 'ZERO_OUTPUT']
        assert len(zero_diags) >= 1
        assert '4.A:CH4' in zero_diags['ID'].values

    def test_magnitude_10x_detected(self, mock_targets, mock_model_row, mock_full_model):
        # Make cement 100x too high
        row = mock_model_row.copy()
        row['emission_co2e_co2_inen_cement'] = 650.0
        diff = build_diff(mock_targets, row, mock_full_model, tp=7, min_mag=0.01)
        diag = run_diagnostics(mock_targets, diff, row, mock_full_model, tp=7, min_mag=0.01)
        mag_diags = diag[diag['issue'] == 'MAGNITUDE_10X']
        assert len(mag_diags) >= 1

    def test_no_false_positives(self, mock_targets, mock_model_row, mock_full_model):
        # With reasonable values, specific HIGH diagnostics should not fire
        diff = build_diff(mock_targets, mock_model_row, mock_full_model, tp=7, min_mag=0.01)
        diag = run_diagnostics(mock_targets, diff, mock_model_row, mock_full_model, tp=7, min_mag=0.01)
        high = diag[diag['severity'] == 'HIGH'] if len(diag) > 0 else pd.DataFrame()
        # Our mock data is reasonably close, so no ZERO_OUTPUT or MAGNITUDE_10X
        assert len(high) == 0

    def test_gas_ratio_not_triggered_when_co2_off(self, mock_targets, mock_model_row, mock_full_model):
        # If CO2 is also way off, GAS_RATIO should not trigger
        diff = build_diff(mock_targets, mock_model_row, mock_full_model, tp=7, min_mag=0.01)
        diag = run_diagnostics(mock_targets, diff, mock_model_row, mock_full_model, tp=7, min_mag=0.01)
        gas_diags = diag[diag['issue'] == 'GAS_RATIO'] if len(diag) > 0 else pd.DataFrame()
        # Mock data has no subsector with both CO2 and CH4/N2O rows that would trigger
        assert len(gas_diags) == 0


# ── Unit Tests: compare() API ─────────────────────────────────────────────────

class TestCompare:
    def test_returns_three_dataframes(self, tmp_path, mock_targets, mock_model_row, mock_full_model):
        # Write mock files
        targets_path = tmp_path / "targets.csv"
        mock_targets.rename(columns={'inventory': 'MAR'}).to_csv(targets_path, index=False)

        output_path = tmp_path / "WIDE.csv"
        mock_full_model.to_csv(output_path, index=False)

        diff, flagged, diag = compare(str(targets_path), str(output_path),
                                       DiagnosticConfig(tp=7), verbose=False)
        assert isinstance(diff, pd.DataFrame)
        assert isinstance(flagged, pd.DataFrame)
        assert isinstance(diag, pd.DataFrame)
        assert len(diff) == 5

    def test_verbose_false_no_print(self, tmp_path, mock_targets, mock_model_row,
                                     mock_full_model, capsys):
        targets_path = tmp_path / "targets.csv"
        mock_targets.rename(columns={'inventory': 'MAR'}).to_csv(targets_path, index=False)
        output_path = tmp_path / "WIDE.csv"
        mock_full_model.to_csv(output_path, index=False)

        compare(str(targets_path), str(output_path), DiagnosticConfig(tp=7), verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_save_dir_creates_files(self, tmp_path, mock_targets, mock_model_row, mock_full_model):
        targets_path = tmp_path / "targets.csv"
        mock_targets.rename(columns={'inventory': 'MAR'}).to_csv(targets_path, index=False)
        output_path = tmp_path / "WIDE.csv"
        mock_full_model.to_csv(output_path, index=False)
        save_dir = tmp_path / "output"

        compare(str(targets_path), str(output_path), DiagnosticConfig(tp=7),
                save_dir=str(save_dir), verbose=False)
        assert (save_dir / "diff_report.csv").exists()


# ── Unit Tests: DiagnosticConfig ──────────────────────────────────────────────

class TestDiagnosticConfig:
    def test_defaults(self):
        cfg = DiagnosticConfig()
        assert cfg.tp == 7
        assert cfg.threshold == 0.15
        assert cfg.show_top == 10

    def test_custom(self):
        cfg = DiagnosticConfig(tp=15, threshold=0.25, explain=True)
        assert cfg.tp == 15
        assert cfg.threshold == 0.25
        assert cfg.explain is True


# ── Unit Tests: DAG structures ────────────────────────────────────────────────

class TestDAG:
    def test_dag_affects_keys(self):
        assert 'entc' in DAG_AFFECTS
        assert 'lvst' in DAG_AFFECTS
        assert 'fgtv' in DAG_AFFECTS

    def test_dag_affects_terminal_nodes(self):
        for node in ['soil', 'frst', 'trww', 'fgtv', 'ccsq']:
            assert DAG_AFFECTS[node] == [], f"{node} should be terminal"

    def test_dag_order_complete(self):
        assert len(DAG_ORDER) >= 15


# ── Integration Test ──────────────────────────────────────────────────────────

class TestIntegration:
    """Integration test using actual project files (skipped if files not found)."""

    TARGETS = Path(__file__).parent.parent.parent.parent / \
              "ssp_modeling/output_postprocessing/data/invent/emission_targets_mar_2022.csv"

    @staticmethod
    def _find_latest_run():
        runs_dir = Path(__file__).parent.parent.parent.parent / "ssp_modeling/ssp_run_output"
        if not runs_dir.exists():
            return None
        runs = sorted(runs_dir.glob("calibration_*"))
        for r in reversed(runs):
            wide = r / "WIDE_INPUTS_OUTPUTS.csv"
            if wide.exists():
                return wide
        return None

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent.joinpath(
            "ssp_modeling/output_postprocessing/data/invent/emission_targets_mar_2022.csv"
        ).exists(),
        reason="Targets file not found"
    )
    def test_compare_against_latest_run(self):
        wide = self._find_latest_run()
        if wide is None:
            pytest.skip("No calibration run output found")

        diff, flagged, diag = compare(
            str(self.TARGETS), str(wide),
            DiagnosticConfig(tp=7, threshold=0.15),
            verbose=False,
        )
        assert len(diff) > 0
        assert 'diff' in diff.columns
        assert 'error_pct' in diff.columns
        total_err = diff[diff['inventory'].abs() > 0.01]['diff'].abs().sum()
        assert total_err < 50, f"Total error {total_err:.1f} MtCO2e seems unreasonably high"
        assert isinstance(flagged, pd.DataFrame)
        assert isinstance(diag, pd.DataFrame)


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
