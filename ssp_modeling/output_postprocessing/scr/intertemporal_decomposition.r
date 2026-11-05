rescale <-function(z,rall,data_all,te_all,initial_conditions_id,dir.output,time_period_ref)
{
#z<-1
tregion<-rall[z]

#subset data and targets 
data <- subset(data_all, region==tregion)

#emissions total
tv1_all <- subset(colnames(data),grepl("co2e_",colnames(data))==TRUE)
#remove subsector totals 
tv1_all <- subset(tv1_all,grepl("emission_co2e_subsector_total_",tv1_all)==FALSE)


#first determine percent change across time  
data$Index <- paste0(data$region,"_",data$primary_id)
inds<-unique(data$Index)
ref_inds <- paste0(tregion,initial_conditions_id)
#this has to be done for every single  var 
pct_diffs<- list()
for (i in 1:length(inds))
{
#i<-3
step1<-list()
for (j in 1:length(tv1_all))
{ 
#j<-268
pivot <- data[data$Index==inds[i],c("Index","time_period",tv1_all[j])]
#remove single cell NAs
pivot[,tv1_all[j]] <- ifelse(is.na(pivot[,tv1_all[j]])==TRUE,mean(pivot[,tv1_all[j]],na.rm=TRUE),pivot[,tv1_all[j]])
#remove full variables NAs
if (is.na(mean(pivot[,tv1_all[j]]))==TRUE) {
 pivot[,tv1_all[j]] <- 0} else {pivot[,tv1_all[j]] <- pivot[,tv1_all[j]]  }

if (mean(unique(pivot[,tv1_all[j]]))==0) {
pivot[,paste0("pct_diff_",tv1_all[j])] <- 0
} else {
pivot$diff <- c(diff(pivot[,tv1_all[j]]),0)
pivot[,paste0("pct_diff_",tv1_all[j])] <- c(0,pivot$diff[1:(nrow(pivot)-1)]/pivot[,tv1_all[j]][1:(nrow(pivot)-1)])
pivot[,paste0("pct_diff_",tv1_all[j])] <- ifelse(is.na(pivot[,paste0("pct_diff_",tv1_all[j])])==TRUE,0,pivot[,paste0("pct_diff_",tv1_all[j])])
pivot[,paste0("pct_diff_",tv1_all[j])] <- ifelse(pivot[,paste0("pct_diff_",tv1_all[j])]==Inf,0,pivot[,paste0("pct_diff_",tv1_all[j])])
}
pivot[,paste0("diff_",tv1_all[j])] <-c(0,diff(pivot[,tv1_all[j]]))
pivot <- pivot[,c("Index","time_period",paste0("pct_diff_",tv1_all[j]),paste0("diff_",tv1_all[j]))]
step1<-append(step1,list(pivot))
}
step1 <- Reduce(function(...) merge(...,), step1) 
pct_diffs<- append(pct_diffs,list(step1))
}
pct_diffs <- do.call("rbind",pct_diffs)
pct_diffs <- pct_diffs[order(pct_diffs$Index,pct_diffs$time_period),]
dim(pct_diffs)
dim(data)
#check percent differences 

#apply this differences to every combination of gas and sector
#te_all$sector_gas <- paste(te_all$Subsector,te_all$Gas,sep="-")
te_all$sector_gas <- paste(row.names(te_all),te_all$Subsector,te_all$Gas,sep="-")
sector_gas_all <- unique(te_all$sector_gas)

for (w in 1:length(sector_gas_all))
{
#w<-64
sector_gas_i <- sector_gas_all[w] #set target sector-gas
tv1 <- unlist(strsplit(subset(te_all,sector_gas==sector_gas_i)$Vars,":")) #identify variables associayed with sector_gas
target_total <- subset(te_all,sector_gas==sector_gas_i)[,"tvalue"] #estimate target total
uncalibrated_total <- sum(data [data$time_period==time_period_ref & data$Index==ref_inds,tv1] , na.rm=TRUE) #estimate uncalibrated total
deviation_factor <- ifelse(uncalibrated_total==0,1.0,(target_total/uncalibrated_total)) #estimate deviation factor
data [data$time_period==time_period_ref,tv1] <- data [data$time_period==time_period_ref,tv1]* deviation_factor #apply deviation factor
round(sum(data [data$time_period==time_period_ref & data$Index==ref_inds,tv1] ),4) == round(target_total,4) #check sums are equal 

#if initial value is different than zero use corrected percent differences, otherwise, use percent differences 
#finally for every var and every index and every time period  
  for (i in 1:length(inds))
  {
  #i<-3
   for (j in 1:length(tv1))
   { 
   #j<-1
   # init_value <-data[data$Index==inds[i] & data$time_period==time_period_ref, tv1[j]]
   init_value <-data[data$Index==paste0(rall[z],initial_conditions_id) & data$time_period==time_period_ref, tv1[j]]
    if (init_value==0) {
       data[data$Index==inds[i],tv1[j]]<-init_value+cumsum(pct_diffs[pct_diffs$Index==inds[i], paste0("diff_",tv1[j])])* deviation_factor
    } else {
       time_change <- cumprod((1+pct_diffs[pct_diffs$Index==inds[i], paste0("pct_diff_",tv1[j])]))
       data[data$Index==inds[i],tv1[j]] <- init_value*time_change
    }
   }
  }
}

#estimate sector totals 
subsectors <- unique(te_all$Subsector)

for (a in 1:length(subsectors))
{
subsector_vars <- unlist(lapply(subset(te_all,Subsector==subsectors[a])$Vars,function(x){strsplit(x,":")}))
data[,paste0("emission_co2e_subsector_total_",subsectors[a])] <- rowSums(data[,subsector_vars])
}
#print file  
data$Index <- NULL 
dim(data)
write.csv(data,paste0(dir.output,"decomposed_ssp_output.csv"),row.names=FALSE)

rm(data)
print(rall[z])
}
