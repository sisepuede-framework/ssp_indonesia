# SSP_Morroco

This repository contains notebooks and supporting files used to run the
**SISEPUEDE** model on Morroco's mitigation scenarios. All modeling resources
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

**Change `sisepuede` to your preferred environment name, usually related to the region you are working with** (e.g., `ssp-morroco`, `ssp-usa`, or whatever you'd like).

For example:

```yaml
name: ssp-morroco
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

*(Replace `<your_env_name>` with the name you specified in the `.yml` file, e.g., `ssp-morroco`)*


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
  - `config_files/` – YAML configuration files used by the notebooks.
  - `input_data/` – Raw data for the model.
  - `notebooks/` – Jupyter notebooks that manage the modeling runs.
  - `ssp_run_output/` – Output folders created after executing a scenario which store the simulation output.
  - `scenario_mapping/` – Spreadsheets with the mapping between SSP transformations and region-specific measures. This is where custom scenarios and transformation intensities are defined.
  - `transformations/` – CSVs and YAML files describing the transformations and strategies applied by the model.
  - `output_postprocessing/` – R scripts used to calibrate the model results and generate post-processed outputs. It also generates the levers and jobs tables.
  - `tableau/` – Stores tableau files and also the post-processed data that is loaded into the dashboards.
  - `cost-benefits/` – Contains scripts to run the cost-benefits analysis in specific SISEPUEDE simulations.

## Steps to run the model and load data to Tableau for analysis

All files and folders referenced here are inside the `ssp_modeling` directory.

1. **Run the model in the notebook**

   * Edit or create a configuration file in `config_files/`.
   * Set up the transformations spreadsheet for custom Strategies and save it in `scenario_mapping/`.
   * Open the manager notebook inside `notebooks/` and run the cells to execute the model.

2. **Post-process results**
    * If you require emission targets, generate the targets file by following the instructions in the [ssp_emission_targets repository](https://github.com/sisepuede-framework/ssp_emission_targets).
    * Run the `postprocessing_250820.r` script, editing it to point to the correct data.
    * The script will generate three files; two of them are used in Tableau.

3. **Load emissions data into Tableau**

   * In the `tableau` directory, locate the Tableau dashboard file and the `data` folder.
   * Copy the postprocessing output files into the `data` folder.
   * Load the files beginning with `decomposed_emissions_` and `drivers_` into Tableau.

4. **Create Levers table and Jobs table**
    * In the `output_postprocessing/`directory you can find the `levers_and_jobs_table/` folder which contains scripts and additional files to create the levers table and jobs table that need to be also loaded to Tableau.
    * The levers table uses a csv file that is created in the region manager notebook (where the simulation was executed), this csv file is stored in the specific run folder in the `ssp_run_output/` directory.
    * The jobs table uses a csv that is already in the `levers_and_jobs_table/` directory.

5. **Run cost and benefits analysis**
    * In the `cost-benefits/`directory you can fine a notebook called `cb.ipynb` which has the code to run the cost-benefit analysis in a specific SISEPUEDE run.
    * The code uses a configuration excel that can be found in `cost-benefits/cb_config_file/cb_config_params.xlsx`. Here you can configure cost factors and additional parameters for the cost-benefit analysis.



