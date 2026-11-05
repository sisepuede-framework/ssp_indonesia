# This script prepares the data for emissions mapping in Uganda
file.name <- paste0("decomposed_ssp_output.csv")
iso_code3 <- iso_code3
Country <- region

mapping <- read.csv(paste0("ssp_modeling/output_postprocessing/data/emission_targets_",region,"_",year_ref,"_LULUCF_update.csv"))

# drop "id" directly
mapping$id <- NULL

# drop the column whose name is stored in iso_code3
mapping[[iso_code3]] <- NULL

# names(mapping)

# modification of AG - Livestock:N2O subsector matching
# mapping$Vars[3] <- "emission_co2e_n2o_lsmm_direct_anaerobic_digester:emission_co2e_n2o_lsmm_direct_anaerobic_lagoon:emission_co2e_n2o_lsmm_direct_composting:emission_co2e_n2o_lsmm_direct_daily_spread:emission_co2e_n2o_lsmm_direct_deep_bedding:emission_co2e_n2o_lsmm_direct_dry_lot:emission_co2e_n2o_lsmm_direct_incineration:emission_co2e_n2o_lsmm_direct_liquid_slurry:emission_co2e_n2o_lsmm_direct_paddock_pasture_range:emission_co2e_n2o_lsmm_direct_poultry_manure:emission_co2e_n2o_lsmm_direct_storage_solid:emission_co2e_n2o_lsmm_indirect_anaerobic_digester:emission_co2e_n2o_lsmm_indirect_anaerobic_lagoon:emission_co2e_n2o_lsmm_indirect_composting:emission_co2e_n2o_lsmm_indirect_daily_spread:emission_co2e_n2o_lsmm_indirect_deep_bedding:emission_co2e_n2o_lsmm_indirect_dry_lot:emission_co2e_n2o_lsmm_indirect_incineration:emission_co2e_n2o_lsmm_indirect_liquid_slurry:emission_co2e_n2o_lsmm_indirect_paddock_pasture_range:emission_co2e_n2o_lsmm_indirect_poultry_manure:emission_co2e_n2o_lsmm_indirect_storage_solid"

# add edgar
edgar <- read.csv(paste0("ssp_modeling/output_postprocessing/data/CSC-GHG_emissions-April2024_to_calibrate_LULUCF_update.csv"))
dim(edgar)
edgar <- subset(edgar,Code==iso_code3)
dim(edgar)
edgar$Edgar_Class<- paste(edgar$CSC.Subsector,edgar$Gas,sep=":")

#load data  
data <- read.csv(paste0(dir.output,file.name)) 
data <- subset(data,region==Country)
dim(data)
#order data
setorder(data, primary_id, time_period, region)

#emission vars only 
id_vars <-c('region','time_period',"primary_id")
vars <- subset(colnames(data),!(colnames(data)%in%id_vars))
target_vars <- subset(vars,grepl("co2e_",vars)==TRUE)
total_vars <- subset(target_vars,grepl("emission_co2e_subsector_total",target_vars)==TRUE)
target_vars <- subset(target_vars,!(target_vars%in%total_vars))

#load inventory mapping table 
mapping$ids <- paste(row.names(mapping),mapping$Subsector,mapping$Gas,sep="_")
#now create those new columns in the simulation data set 
for  (i in 1:nrow(mapping))
{
#i<- 63
tvars <- mapping$Vars[i]
tvars <- unlist(strsplit(tvars,":"))
tvars <- subset(tvars,tvars%in%colnames(data))
if (length(tvars)>1) {
 data [,mapping$ids[i]] <- rowSums(data[,tvars])
} else if (length(tvars)==1 ) 
{ 
  data [,mapping$ids[i]] <- data[,tvars]
} else {
  data [,mapping$ids[i]] <- 0
} 
}
#now we just keep the new variables and the time period which we will reduce to above 2022
data_new <- data [,c(id_vars,mapping$ids)]
dim(data_new)

#convert from wide to long 
data_new <- data.table(data_new)
data_new <- reshape2::melt(data_new, id.vars = id_vars,
                   measure.vars = mapping$ids,
             )
data_new <- data.frame(data_new)
data_new$ids <- as.character(data_new$variable)

#merge with mapping 
 mapping$Vars <- NULL
 colnames(mapping) <- gsub("Edgar_Sector","CSC.Sector",colnames(mapping))
 colnames(mapping) <- gsub("Edgar_Subsector","CSC.Subsector",colnames(mapping)) 
 data_new <- merge(data_new,mapping,by="ids")

#now aggregare at inventory level 
data_new <- aggregate(list(value=data_new$value),by=list(primary_id=data_new$primary_id,
                                             time_period=data_new$time_period,
                                             Edgar_Class=data_new$Edgar_Class,
                                             CSC.Sector=data_new$CSC.Sector,
                                             CSC.Subsector=data_new$CSC.Subsector),sum)


data_new$Year <- data_new$time_period + 2015
data_new$Gas <- do.call("rbind",strsplit(data_new$Edgar_Class,":"))[,2]


#merge additional files  
att <- read.csv(paste0(dir.output,"ATTRIBUTE_PRIMARY.csv"))
head(att)

dim(data_new)

data_new <- merge(data_new,att,by="primary_id")
dim(data_new)

atts <- read.csv(paste0(dir.output,"ATTRIBUTE_STRATEGY.csv"))

#merge 
dim(data_new)
data_new <- merge(data_new,atts[c("strategy_id","strategy")],by="strategy_id")
dim(data_new)

#melt edgar data 
id_varsEd <- c("Code","CSC.Sector","CSC.Subsector","Gas","Edgar_Class")
measure.vars_Ed <- subset(colnames(edgar),grepl("X",colnames(edgar))==TRUE)
edgar <- data.table(edgar)
edgar <- melt(edgar, id.vars = id_varsEd, measure.vars =measure.vars_Ed)
edgar <- data.frame(edgar)
edgar$Year <- as.numeric(gsub("X","",edgar$variable))

#make sure both data frames have the same columns 
#edgar 
edgar$variable <- NULL
edgar$strategy_id <- NA
edgar$primary_id <- NA 
edgar$design_id <- NA 
edgar$future_id <- NA 
edgar$Contry <- Country
edgar$strategy <- "Historical" 
edgar$source <- "EDGAR"
edgar <- subset(edgar,Year<=max(edgar$Year))

#data_new 
data_new$time_period <- NULL 
data_new$Code <- iso_code3 
data_new$Contry <- Country
data_new$source <- "SISEPUEDE"
data_new <- subset(data_new,Year>=max(edgar$Year))


#rbind both 
data_new <- rbind(data_new,edgar)
data_new <- data_new[order(data_new$strategy_id,data_new$CSC.Subsector,data_new$Gas,data_new$Year),]




# HP filter

library(data.table)
library(mFilter)
library(ggplot2)

hp_filter_subsec <- function(data,
                             subsec_target,
                             gas_target,
                             lambda_hp = 100,
                             time_col = "Year",
                             value_col = "value",
                             by_cols = c("primary_id", "strategy_id", "design_id", "future_id", "Code"),
                             replace_original = TRUE,
                             facet_scale = "free_y") {
  library(data.table)
  library(ggplot2)
  library(mFilter)
  
  # Ensure data.table
  dt <- copy(data)
  setDT(dt)
  
  # Keep original for plotting
  dt[, value_original := get(value_col)]
  
  # Apply HP filter only if strategy_id is not NA, anchored to first year
  dt[`CSC.Subsector` == subsec_target & Gas %in% gas_target & !is.na(strategy_id),
     value_hp := {
       # order by time within group
       o  <- order(get(time_col))
       v  <- as.numeric(get(value_col))[o]
       if (length(v) < 2L) {
         # Not enough points to smooth; just return original in place
         out <- rep(NA_real_, .N)
         out[o][1] <- v[1]  # anchor first value
         out
       } else {
         hp <- mFilter::hpfilter(v, freq = lambda_hp, type = "lambda")
         sm <- pmax(hp$trend, 0)  # base smooth, non-negative
         
         # Anchor: shift entire smooth so it passes through the first observed value
         shift   <- v[1] - sm[1]
         sm_adj  <- sm + shift
         
         # (optional) keep non-negativity after shift, then enforce anchor again
         sm_adj  <- pmax(sm_adj, 0)
         sm_adj[1] <- v[1]  # ensure exact match at first point
         
         # put back in original row order
         out <- rep(NA_real_, .N)
         out[o] <- sm_adj
         out
       }
     },
     by = by_cols
  ]
  
  # Plot data (incluye históricos con strategy_id NA para ver original completo)
  plot_dt <- dt[`CSC.Subsector` == subsec_target & Gas %in% gas_target]
  
  p <- ggplot(plot_dt, aes(x = .data[[time_col]])) +
    geom_line(aes(y = value_original, colour = "Original"), size = 1) +
    geom_line(aes(y = value_hp,      colour = "HP (anchored)"), size = 1, na.rm = TRUE) +
    scale_colour_manual(values = c("Original" = "steelblue", "HP (anchored)" = "red")) +
    labs(
      x = time_col,
      y = value_col,
      title = paste("HP Filter anchored at first year (λ =", lambda_hp, ") -",
                    subsec_target, "-", paste(gas_target, collapse = ", ")),
      colour = "Series"
    ) +
    facet_wrap(~strategy_id, scales = facet_scale) +
    theme_minimal()
  
  # Replace original values with anchored smooth ONLY where it exists
  if (replace_original) {
    dt[`CSC.Subsector` == subsec_target & Gas %in% gas_target &
         !is.na(strategy_id) & !is.na(value_hp),
       (value_col) := value_hp]
  }
  
  list(data = dt, plot = p)
}


# AG - Crops

res <- hp_filter_subsec(
  data = data_new,
  subsec_target = "AG - Crops",
  gas_target = "N2O",
  lambda_hp = 1600
)

# plot
print(res$plot)



# AG - Livestock

res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "AG - Livestock",
  gas_target = "CH4",
  lambda_hp = 1600
)

res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "AG - Livestock",
  gas_target = c('HFC','N2O','PFC','SF6'),
  lambda_hp = 1600
)


# plot
print(res$plot)


#  EN - Building 


res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "EN - Building",
  gas_target = "CH4",
  lambda_hp = 1600
)

# plot
print(res$plot)

#  EN - Electricity/Heat


res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "EN - Electricity/Heat",
  gas_target = "CH4",
  lambda_hp = 1600
)

# plot
print(res$plot)


res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "EN - Electricity/Heat",
  gas_target = "CO2",
  lambda_hp = 1600
)

# plot
print(res$plot)


res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "EN - Electricity/Heat",
  gas_target = c('HFC','N2O','PFC','SF6'),
  lambda_hp = 1600
)

# plot
print(res$plot)



#   EN - Manufacturing/Construction


res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "EN - Manufacturing/Construction",
  gas_target = "CH4",
  lambda_hp = 1600
)

# plot
print(res$plot)

res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "EN - Manufacturing/Construction",
  gas_target = "CO2",
  lambda_hp = 1600
)

# plot
print(res$plot)


#   LULUCF - Deforestation


res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "LULUCF - Deforestation",
  gas_target = "CO2",
  lambda_hp = 1600
)

# plot
print(res$plot)

# Waste - Solid Waste 

res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "Waste - Solid Waste",
  gas_target = c('HFC','N2O','PFC','SF6'),
  lambda_hp = 1600
)

# plot
print(res$plot)


# LULUCF - Other Lan

res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "LULUCF - Other Land",
  gas_target = "CO2",
  lambda_hp = 1600
)

# plot
print(res$plot)



# Waste - Wastewater Treatment

res <- hp_filter_subsec(
  data = res$data,
  subsec_target = "Waste - Wastewater Treatment",
  gas_target = "CH4",
  lambda_hp = 1600
)

# plot
print(res$plot)






table(data_new$CSC.Subsector)
table(data_new$Gas)


#write file
dir.tableau <- paste0("ssp_modeling/tableau/data/")
file.name <- paste0("decomposed_emissions_", region, "_", year_ref, "_", output.file)

write.csv(res$data,paste0(dir.tableau,file.name),row.names=FALSE)

print('Finish:data_prep_new_mapping process')



