#################################################
# Post processing process
#################################################

# load packages
library(data.table)
library(reshape2)
library(mFilter)
library(ggplot2)

rm(list=ls())

#ouputfile
dir.output  <- "ssp_modeling/ssp_run_output/sisepuede_results_sisepuede_run_2025-09-30T16;13;13.113881/"
output.file <- "sisepuede_results_sisepuede_run_2025-09-30T16;13;13.113881_WIDE_INPUTS_OUTPUTS.csv"

region <- "morocco" 
iso_code3 <- "MAR"

year_ref <- 2022

source('ssp_modeling/output_postprocessing/scr/run_script_baseline_run_new.r')

source('ssp_modeling/output_postprocessing/scr/data_prep_new_mapping.r')

source('ssp_modeling/output_postprocessing/scr/data_prep_drivers.r')

