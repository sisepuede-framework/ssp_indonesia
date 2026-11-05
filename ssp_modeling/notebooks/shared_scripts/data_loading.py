"""
data_loading.py
---------------
Thin utilities for loading SSP run artefacts from a run output directory.
"""

import pandas as pd
from pathlib import Path
from typing import List


def load_attribute_tables(run_dir: Path) -> tuple:
    """Return (att_primary, att_strategy) DataFrames from *run_dir*."""
    run_dir = Path(run_dir)
    att_primary  = pd.read_csv(run_dir / "ATTRIBUTE_PRIMARY.csv")
    att_strategy = pd.read_csv(run_dir / "ATTRIBUTE_STRATEGY.csv")
    return att_primary, att_strategy


# ---------------------------------------------------------------------------
# Strategy metadata + wide export (used by the tornado / whirlpool runners)
# ---------------------------------------------------------------------------

def parse_strategy_metadata(att_strategy: pd.DataFrame) -> pd.DataFrame:
    """
    Derive sector, transformation_code, transformation_name, and
    transformation_name_sector columns from strategy_code / strategy text.

    Uses ``transformation_specification`` (when present) as the primary source
    because it already contains the canonical TX:SECTOR:NAME path, regardless of
    whatever prefix the strategy_code carries (TORNADO_BASE:, WHIRLPOOL_..., etc.).

    Only single-transformation rows (no ``|`` separator) get sector /
    transformation_code; composite strategies remain NaN.

    transformation_specification examples:
      Single   : "TX:AGRC:DEC_LOSSES_SUPPLY_CHAIN_STRATEGY_CONDITIONAL"
                 → transformation_code = "TX:AGRC:DEC_LOSSES_SUPPLY_CHAIN"
                 → sector              = "AGRC"
      Singleton: "TX:AGRC:DEC_CH4_RICE"  (no suffix to strip)
      Composite: "TX:A|TX:B|..."         → NaN (multiple transformations)
      BASE     : "TX:BASE"               → NaN (no sector segment)
    """
    df = att_strategy.copy()

    has_spec = "transformation_specification" in df.columns

    if has_spec:
        # Only use single-transformation entries (no pipe separator)
        is_single = ~df["transformation_specification"].str.contains(r"\|", na=True)
        spec = df["transformation_specification"].where(is_single)

        df["sector"] = spec.str.extract(r"^TX:([A-Z]+):", expand=False)
        df["transformation_code"] = spec.str.replace(
            r"_STRATEGY_[A-Z0-9]+$", "", regex=True
        )

        # Composite whirlpool strategies (e.g. WHIRLPOOL_...:TX:SECTOR:CODE_STRATEGY_...)
        # represent "all-but-one" portfolios; derive the removed transformation from
        # strategy_code so downstream merges on (transformation_code, sector) work.
        whirlpool_tc = (
            df["strategy_code"]
            .str.extract(r"[^:]+:(TX:[A-Z]+:[A-Z0-9_]+)", expand=False)
            .str.replace(r"_STRATEGY_[A-Z0-9]+$", "", regex=True)
        )
        df["transformation_code"] = df["transformation_code"].fillna(whirlpool_tc)
        df["sector"] = df["sector"].fillna(
            df["transformation_code"].str.extract(r"^TX:([A-Z]+):", expand=False)
        )
    else:
        # Fallback: parse strategy_code directly
        df["sector"] = (
            df["strategy_code"]
            .str.extract(r"(?:[A-Z0-9_]+:TX:)?([A-Z]+):[A-Z0-9_]+$", expand=False)
        )
        tx_code = (
            df["strategy_code"]
            .str.extract(r"(TX:[A-Z0-9:_]+)", expand=False)
            .str.replace(r"_STRATEGY_[A-Z0-9]+$", "", regex=True)
        )
        fallback = df["strategy_code"].str.extract(r":([A-Z0-9_]+)$", expand=False)
        df["transformation_code"] = tx_code.where(tx_code.notna(), fallback)

    # Extract human-readable name from singleton rows (pattern: "... - SECTOR: Name")
    df['transformation_name'] = (
        df['strategy']
        .str.extract(r'-\s*[A-Z]+:\s*(.+)$', expand=False)
    )

    # Fill in transformation_name for tornado/whirlpool rows by looking up the
    # matching singleton (same transformation_code, name already resolved above)
    name_lookup = (
        df.dropna(subset=['transformation_code', 'transformation_name'])
        .drop_duplicates('transformation_code')
        .set_index('transformation_code')['transformation_name']
    )
    missing = df['transformation_name'].isna() & df['transformation_code'].notna()
    df.loc[missing, 'transformation_name'] = (
        df.loc[missing, 'transformation_code'].map(name_lookup)
    )

    # Last-resort fallback: some transformations used by the tornado / whirlpool
    # sets have no singleton row in the Morocco strategy list (e.g.
    # TX:FRST:INCREASE_SEQUESTRATION, TX:SCOE:INC_EFFICIENCY_HEAT,
    # TX:TRNS:SHIFT_FUEL_PUBLIC).  Build a readable label from the code itself so
    # they are not blank in the Tableau chart.
    still_missing = df['transformation_name'].isna() & df['transformation_code'].notna()
    df.loc[still_missing, 'transformation_name'] = (
        df.loc[still_missing, 'transformation_code']
        .str.split(':').str[-1]
        .str.replace('_', ' ', regex=False)
        .str.capitalize()
    )

    df['transformation_name_sector'] = (
        df['sector'].fillna('') + ': ' + df['transformation_name'].fillna('')
    ).where(df['transformation_name'].notna())

    return df


def load_wide_export(
    run_dir: Path,
    primary_ids_filter: List[int],
) -> pd.DataFrame:
    """
    Load the wide-format CSV from *run_dir* and filter to the requested
    primary_ids.

    Naming conventions supported (in order of preference):
      - ``*_WIDE_INPUTS_OUTPUTS.csv``  (standard sisepuede output)
      - ``WIDE_INPUTS_OUTPUTS.csv``    (legacy / manual runs)
    """
    matches = list(run_dir.glob("*_WIDE_INPUTS_OUTPUTS.csv"))

    if not matches:
        fallback = run_dir / "WIDE_INPUTS_OUTPUTS.csv"
        assert fallback.exists(), (
            f"No wide-format CSV found in {run_dir}. "
            "Expected '*_WIDE_INPUTS_OUTPUTS.csv' or 'WIDE_INPUTS_OUTPUTS.csv'."
        )
        matches = [fallback]

    df = pd.read_csv(matches[0])
    df = df[df["primary_id"].isin(primary_ids_filter)].reset_index(drop=True)
    return df
