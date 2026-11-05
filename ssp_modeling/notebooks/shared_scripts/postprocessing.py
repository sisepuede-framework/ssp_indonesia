"""
postprocessing.py
-----------------
Intertemporal decomposition / rescaling of SSP simulation output for Morocco.

Wraps ssp_modeling.output_postprocessing.intertemporal_decomposition,
which is the Python translation of:
  - ssp_modeling/output_postprocessing/scr/run_script_baseline_run_new.r
  - ssp_modeling/output_postprocessing/scr/intertemporal_decomposition.r
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Optional


def run_decomposition(
    df_export: pd.DataFrame,
    project_dir: Path,
    targets_path: Path,
    iso_code3: str,
    year_ref: int,
    region: str,
    output_path: Optional[Path] = None,
    initial_conditions_id: str = "_0",
) -> pd.DataFrame:
    """
    Rescale *df_export* against Morocco inventory targets and return the
    decomposed DataFrame.  Also writes the result to *output_path* if given.

    Parameters
    ----------
    df_export             : wide-format SSP simulation output (filtered primary_ids)
    project_dir           : repo root — added to sys.path so the module resolves
    targets_path          : path to emission_targets_mar_<year>.csv
    iso_code3             : 3-letter ISO country code (e.g. "MAR")
    year_ref              : reference / calibration year (e.g. 2022)
    region                : region label used inside the model (e.g. "morocco")
    output_path           : where to write the decomposed CSV (None = skip)
    initial_conditions_id : primary_id suffix of the baseline scenario (default "_0")
    """
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    from ssp_modeling.output_postprocessing.intertemporal_decomposition import (
        run_postprocessing,
    )

    df_decomposed = run_postprocessing(
        df_ssp_output         = df_export,
        targets_path          = targets_path,
        iso_code3             = iso_code3,
        year_ref              = year_ref,
        region                = region,
        initial_conditions_id = initial_conditions_id,
        output_path           = output_path,
    )

    return df_decomposed
