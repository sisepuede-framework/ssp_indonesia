# SSP_Indonesia

This repository contains notebooks and supporting files used to run the
**SISEPUEDE** model on Indonesia's mitigation scenarios. All modeling resources
reside in the `ssp_modeling` folder described below.


## Instructions: Setting Up the SISEPUEDE Environment

### 1. **Go to the `environment.yml` file**

Obtain the provided `environment.yml` file for SISEPUEDE.

### 2. **Set a Custom Environment Name**

Open the `environment.yml` file in your preferred text editor (such as VS Code, Atom, nano, or even Notepad).
At the very top, you'll see a line like:

```yaml
name: sisepuede
```

**Change `sisepuede` to your preferred environment name, usually related to the region you are working with** (e.g., `ssp-indonesia`, `ssp-usa`, or whatever you'd like).

For example:

```yaml
name: ssp-indonesia
```

### 3. **Create the Environment from the `.yml` File**

In your terminal, navigate to the directory containing your `environment.yml` file, then run:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment with the name you set in the file.

### 4. **Activate the Environment**

After installation, activate your new environment with:

```bash
conda activate <your_env_name>
```

*(Replace `<your_env_name>` with the name you specified in the `.yml` file, e.g., `ssp-indonesia`)*


### 5. **Done!**

Your environment is now ready to use, with all dependencies (including those installed via pip) preconfigured.


#### **Tips:**

* If you update the `environment.yml` file later, you can update your environment with:

  ```bash
  conda env update -f environment.yml --prune
  ```
* You can list all your environments with:

  ```bash
  conda env list
  ```
## Project Structure

The most relevant files are inside the `ssp_modeling` directory:
  - `notebooks/` – Jupyter notebooks that manage the modeling runs, plus `config_files/` (YAML configs) and `shared_scripts/` (the Python pipeline modules: calibration, post-processing, cost-benefits).
  - `input_data/` – Raw data for the model.
  - `ssp_run_output/` – Output folders created after executing a scenario which store the simulation output (gitignored).
  - `scenario_mapping/` – Spreadsheets with the mapping between SSP transformations and region-specific measures. This is where custom scenarios and transformation intensities are defined.
  - `transformations/` – CSVs and YAML files describing the transformations and strategies applied by the model.
  - `output_postprocessing/` – Python post-processing modules (e.g. `intertemporal_decomposition.py`) and the `levers_and_jobs_table/` input files used to build the Tableau tables.
  - `tableau/` – Stores the Tableau workbook (`.twbx`) and, under `data/`, the post-processed CSVs loaded into the dashboards.
  - `cost-benefits/` – Scripts and config (`cb_config_files/`) to run the cost-benefit analysis on a specific SISEPUEDE run.

## Steps to run the model and load data to Tableau for analysis

The whole pipeline is driven from the manager notebook
`ssp_modeling/notebooks/indonesia_manager_wb.ipynb`, which calls the Python
modules in `ssp_modeling/notebooks/shared_scripts/`. The old R post-processing
scripts have been replaced by these modules. All paths below are inside the
`ssp_modeling` directory unless noted.

1. **Configure the run**
    * Edit `notebooks/config_files/config.yaml` — it sets `country_name`,
      `country_code` (`IDN`), the calibrated SSP input file (`ssp_input_file_name`),
      the transformation crosswalk (`ssp_transformation_cw`), and the model flags.
    * Set up / adjust the transformation crosswalk spreadsheet in `scenario_mapping/`
      and the strategy definitions in `transformations/`.

2. **(Optional) Calibrate the baseline against the BUR3 inventory**
    * Run `notebooks/indonesia_calibration_wb.ipynb`. It diagnoses the baseline
      against Indonesia's BUR3 inventory, applies per-subsector calibration knobs,
      and exports the calibrated input CSV that `config.yaml`'s `ssp_input_file_name`
      points to. This replaces the previous `ssp_emission_targets` step.

3. **Run the model**
    * Open `notebooks/indonesia_manager_wb.ipynb` and run the cells: it builds the
      transformations and strategies, instantiates SISEPUEDE, projects the scenarios,
      and exports the wide results into a run folder under `ssp_run_output/`.

4. **Generate the Tableau tables (post-processing)**
    * In the same notebook, the "Post-processing pipeline — Tableau tables" section
      calls `run_tableau_postprocessing(...)` from
      `notebooks/shared_scripts/tableau_postprocessing.py`.
    * It reads the run output directly (the `*WIDE_INPUTS_OUTPUTS.csv`,
      `ATTRIBUTE_PRIMARY.csv`, `ATTRIBUTE_STRATEGY.csv`, and
      `levers_implementation_<region>.csv` files) — no R script and no separate
      levers/jobs step — and writes four CSVs to `tableau/data/`:
        * `emissions_<region>.csv`
        * `drivers_<region>.csv`
        * `tableau_levers_table_complete.csv`
        * `jobs_demand_<region>.csv`

5. **Run the cost-benefit analysis**
    * In the manager notebook, the "Cost-benefit analysis" section calls
      `run_cost_benefits(...)` from `notebooks/shared_scripts/cost_benefits_pipeline.py`,
      using `cost-benefits/cb_config_files/cb_config_params_indonesia.xlsx` for the
      cost factors and parameters. (The standalone `cost-benefits/cb.ipynb` is still
      available for running it outside the manager notebook.)

6. **Load the data into Tableau**
    * All post-processed CSVs land in `tableau/data/`.
    * Open the Tableau workbook in `tableau/` and point its data sources at those CSVs.



