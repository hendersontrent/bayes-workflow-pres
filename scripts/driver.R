#-------------------------------------
# This script sets out to produce all
# analysis for the presentation in
# a format that is easily testable
# during development of the talk
#-------------------------------------

#-------------------------------------
# Author: Trent Henderson, 1 July 2021
#-------------------------------------

library(data.table)
library(dplyr)
library(magrittr)
library(tidyr)
library(ggplot2)
library(scales)
library(tibble)
library(janitor)
library(foreign)
library(Rcatch22)
library(theft)

#-------------------------------------------------------------------------------
#--------------------------INTERACTIVE DIST EXAMPLES----------------------------
#-------------------------------------------------------------------------------

# Named palette of colours to ensure they get consistent colours in all plots

palette <- c("#E494D3", "#87DCC0", "#88BBE4", "#998AD3", "#D4BBDD")

names(palette) <- c("Prior", 
                    "Likelihood of seeing 5/10 students in Go8",
                    "Unstandardised Posterior",
                    "Standardised Posterior",
                    "Random Sample of Students")

x <- seq(0, 1, length.out = 11)
the_xlab <- "Possible values of actual proportion"

pr <- data.frame(x = x,
                 y = dbeta(x, shape1 = 3.5, shape2 = 6.5),
                 category = "Prior") %>%
  mutate(y = y / sum(y))

pr %>%
  ggplot(aes(x = x, y = y, colour = category)) +
  geom_line(size = 1.25) +
  labs(title = "Our prior",
       x = the_xlab,
       y = "Probability Density") +
  theme(legend.position = "none")

# Sample of data

lh <- data.frame(x = 0:10) %>%
  mutate(y = dbinom(x = x, prob = 0.5, size = 10),
         category = names(palette)[2]) %>%
  mutate(x = x / max(x))

pr %>%
  ggplot(aes(x = x, y = y)) +
  geom_line(aes(colour = category), size = 1.25) +
  geom_line(data = lh, size = 1.25, aes(colour = category)) +
  labs(title = "Our prior and a random sample of students",
       x = the_xlab,
       y = "Probability Density",
       colour = NULL) +
  scale_colour_manual(values = palette) +
  theme(legend.position = "bottom",
        legend.key = element_blank())

# Posterior

posterior <- data.frame(x = x,
                        y = pr$y*lh$y,
                        category = "Unstandardised Posterior")

pr %>%
  ggplot(aes(x = x, y = y)) +
  geom_line(aes(colour = category), size = 1.25) +
  geom_line(data = lh, size = 1.25, aes(colour = category)) +
  geom_line(data = posterior, size = 1.25, aes(colour = category)) +
  labs(title = "Our prior, random sample, and posterior update",
       x = the_xlab,
       y = "Probability Density",
       colour = NULL) +
  scale_colour_manual(values = palette) +
  theme(legend.position = "bottom",
        legend.key = element_blank())

# Standardised posterior

st_post <- posterior %>%
  mutate(y = y / sum(y),
         category = "Standardised Posterior")

pr %>%
  ggplot(aes(x = x, y = y)) +
  geom_line(aes(colour = category), size = 1.25) +
  geom_line(data = lh, size = 1.25, aes(colour = category)) +
  geom_line(data = posterior, size = 1.25, aes(colour = category)) +
  geom_line(data = st_post, size = 1.25, aes(colour = category)) +
  labs(title = "Our prior, random sample, and standardised posterior update",
       x = the_xlab,
       y = "Probability Density",
       colour = NULL) +
  scale_colour_manual(values = palette) +
  theme(legend.position = "bottom",
        legend.key = element_blank())

# Impact of sample size

do_bayes <- function(n = 100){
  
  # Prior
  
  x <- seq(0, 1, length.out = n+1)
  
  pr <- data.frame(x = x,
                   y = dbeta(x, shape1 = 3.5, shape2 = 6.5),
                   category = "Prior") %>%
    mutate(y = y / sum(y))
  
  # Likelihood
  
  lh <- data.frame(x = 0:n) %>%
    mutate(y = dbinom(x = x, prob = 0.5, size = n),
           category = "Random Sample of Students",
           x = x / n)
  
  # Posterior
  
  posterior <- data.frame(x = x,
                          y = pr$y*lh$y,
                          category = "Unstandardised Posterior")
  
  st_post <- posterior %>%
    mutate(y = y / sum(y),
           category = "Standardised Posterior")
  
  p <- pr %>%
    ggplot(aes(x = x, y = y)) +
    geom_line(aes(colour = category), size = 1.25) +
    geom_line(data = lh, size = 1.25, aes(colour = category)) +
    geom_line(data = st_post, size = 1.25, aes(colour = category)) +
    labs(title = "Our prior, random sample, and posterior update",
         subtitle = paste0("N = ", n),
         x = the_xlab,
         y = "Probability Density",
         colour = NULL) +
    scale_colour_manual(values = palette) +
    theme(legend.position = "bottom",
          legend.key = element_blank())
  
  return(p)
}

p_5 <- do_bayes(n = 5)
p_10 <- do_bayes(n = 10)
p_100 <- do_bayes(n = 100)
p_1000 <- do_bayes(n = 1000)

print(p_100)
print(p_1000)

#-------------------------------------------------------------------------------
#--------------------------EXTRACT DATA FROM WEBSITE----------------------------
#-------------------------------------------------------------------------------

#' Function to automatically webscrape and parse Time Series Classification univariate two-class classification datasets
#' 
#' NOTE: The dictionary list used to identify and pass two-class problems only should be switched to a dynamic
#' webscrape table read to ensure it can scale as the dataset structure changes/is added to.
#' 
#' @return a dataframe object in tidy form
#' @author Trent Henderson
#' 

pullTSCprobs <- function(){
  
  # --------------- Set up dictionary -------------
  
  # Not all the datasets are two-class problems. Define dictionary from
  # website of two-class problems to filter downloaded dataset by
  # Source: http://www.timeseriesclassification.com/dataset.php
  
  twoclassprobs <- c("SonyAIBORobotSurface1")
  
  # --------------- Webscrape the data ------------
  
  temp <- tempfile()
  download.file("http://www.timeseriesclassification.com/Downloads/SonyAIBORobotSurface1.zip", temp, mode = "wb")
  
  # --------------- Parse into problems -----------
  
  problemStorage <- list()
  message("Parsing individual datasets...")
    
    tryCatch({
      
      # Retrieve TRAIN and TEST files
      
      train <- foreign::read.arff(unz(temp, paste0(twoclassprobs,"_TRAIN.arff"))) %>%
        mutate(id = row_number()) %>%
        mutate(set_split = "Train")
      
      themax <- max(train$id) # To add in test set to avoid duplicate IDs
      
      test <- foreign::read.arff(unz(temp, paste0(twoclassprobs,"_TEST.arff"))) %>%
        mutate(id = row_number()+themax) %>% # Adjust relative to train set to stop double-ups
        mutate(set_split = "Test")
      
      #----------------------------
      # Wrangle data to long format
      #----------------------------
      
      # Train
      
      thecolstr <- colnames(train)
      keepcolstr <- thecolstr[!thecolstr %in% c("target", "id", "set_split")]
      
      train2 <- train %>%
        mutate(problem = twoclassprobs) %>%
        tidyr::pivot_longer(cols = all_of(keepcolstr), names_to = "timepoint", values_to = "values") %>%
        mutate(timepoint = as.numeric(gsub(".*?([0-9]+).*", "\\1", timepoint)))
      
      # Test
      
      thecolste <- colnames(test)
      keepcolste <- thecolste[!thecolste %in% c("target", "id", "set_split")]
      
      test2 <- test %>%
        mutate(problem = twoclassprobs) %>%
        tidyr::pivot_longer(cols = all_of(keepcolste), names_to = "timepoint", values_to = "values") %>%
        mutate(timepoint = as.numeric(gsub(".*?([0-9]+).*", "\\1", timepoint)))
      
      #------
      # Merge
      #------
      
      problemStorage <- bind_rows(train2, test2)
    }, error = function(e){cat("ERROR :",conditionMessage(e), "\n")})

  return(problemStorage)
}

SonyAIBORobotSurface1 <- pullTSCprobs()

#-------------------------------------------------------------------------------
#--------------------------COMPUTE FEATURES-------------------------------------
#-------------------------------------------------------------------------------

feature_matrix <- calculate_features(data = SonyAIBORobotSurface1, 
                                     id_var = "id", 
                                     time_var = "timepoint", 
                                     values_var = "values", 
                                     group_var = "target",
                                     feature_set = "catch22")

# Re-join set split labels

setlabs <- SonyAIBORobotSurface1 %>%
  dplyr::select(c(id, set_split)) %>%
  distinct() %>%
  mutate(id = as.character(id))

featMat <- feature_matrix %>%
  left_join(setlabs, by = c("id" = "id"))

#-------------------------------------------------------------------------------
#--------------------------CREATE TRAIN-TEST SPLITS-----------------------------
#-------------------------------------------------------------------------------

# Separate into train-test splits

train <- featMat %>%
  filter(set_split == "Train") %>%
  mutate(group = as.character(group),
         group = factor(group, levels = c("1", "2")))
test <- featMat %>%
  filter(set_split == "Test") %>%
  mutate(group = as.character(group),
         group = factor(group, levels = c("1", "2")))

# Normalise train data, save its characteristics, and normalise test data onto its same scale

trainNormed <- train %>%
  group_by(names) %>%
  mutate(values = (values-mean(values, na.rm = TRUE))/stats::sd(values, na.rm = TRUE)) %>%
  ungroup()

trainScales <- train %>%
  group_by(names) %>%
  summarise(mean = mean(values, na.rm = TRUE),
            sd = sd(values, na.rm = TRUE)) %>%
  ungroup()

testNormed <- test %>%
  left_join(trainScales, by = c("names" = "names")) %>%
  group_by(names) %>%
  mutate(values = (values-mean)/sd) %>%
  ungroup() %>%
  dplyr::select(-c(mean, sd))

# Widen datasets for modelling

trainWide <- trainNormed %>%
  pivot_wider(id_cols = c(id, group), names_from = "names", values_from = "values") %>%
  drop_na() %>%
  mutate(group = ifelse(group == "1", 0, 1))

testWide <- testNormed %>%
  pivot_wider(id_cols = c(id, group), names_from = "names", values_from = "values") %>%
  drop_na() %>%
  mutate(group = ifelse(group == "1", 0, 1))

#-------------------------------------------------------------------------------
#--------------------------FIT UNINFORMED MODEL 1------------------------------
#-------------------------------------------------------------------------------

# Plot priors

data.frame(x = seq(from = -5, to = 5, by = 0.1)) %>%
  mutate(y = LaplacesDemon::dst(x, mu = 0, sigma = 2.5, nu = 7),
         category = "Prior") %>%
  ggplot(aes(x = x, y = y, colour = category)) +
  geom_line(size = 1.25) +
  labs(title = "Our vague coefficient prior",
       x = "x",
       y = "Probability Density") +
  scale_colour_manual(values = palette) +
  theme(legend.position = "none")

data.frame(x = seq(from = -5, to = 5, by = 0.1)) %>%
  mutate(y = dnorm(x, mean = 0, sd = 1),
         category = "Prior") %>%
  ggplot(aes(x = x, y = y, colour = category)) +
  geom_line(size = 1.25) +
  labs(title = "Our vague intercept prior",
       x = "x",
       y = "Probability Density") +
  scale_colour_manual(values = palette) +
  theme(legend.position = "none")

options(mc.cores = parallel::detectCores())

# Build model with vague and uninformative priors

stan_data <- list(N = nrow(trainWide),
                  y = trainWide$group,
                  DN_HistogramMode_5 = trainWide$DN_HistogramMode_5,
                  DN_HistogramMode_10 = trainWide$DN_HistogramMode_10,
                  CO_f1ecac = trainWide$CO_f1ecac,
                  CO_FirstMin_ac = trainWide$CO_FirstMin_ac,
                  CO_HistogramAMI_even_2_5 = trainWide$CO_HistogramAMI_even_2_5,
                  CO_trev_1_num = trainWide$CO_trev_1_num,
                  MD_hrv_classic_pnn40 = trainWide$MD_hrv_classic_pnn40,
                  SB_BinaryStats_mean_longstretch1 = trainWide$SB_BinaryStats_mean_longstretch1,
                  SB_TransitionMatrix_3ac_sumdiagcov = trainWide$SB_TransitionMatrix_3ac_sumdiagcov,
                  PD_PeriodicityWang_th0_01 = trainWide$PD_PeriodicityWang_th0_01,
                  CO_Embed2_Dist_tau_d_expfit_meandiff = trainWide$CO_Embed2_Dist_tau_d_expfit_meandiff,
                  IN_AutoMutualInfoStats_40_gaussian_fmmi = trainWide$IN_AutoMutualInfoStats_40_gaussian_fmmi,
                  FC_LocalSimple_mean1_tauresrat = trainWide$FC_LocalSimple_mean1_tauresrat,
                  DN_OutlierInclude_p_001_mdrmd = trainWide$DN_OutlierInclude_p_001_mdrmd,
                  DN_OutlierInclude_n_001_mdrmd = trainWide$DN_OutlierInclude_n_001_mdrmd,
                  SP_Summaries_welch_rect_area_5_1 = trainWide$SP_Summaries_welch_rect_area_5_1,
                  SB_BinaryStats_diff_longstretch0 = trainWide$SB_BinaryStats_diff_longstretch0,
                  SB_MotifThree_quantile_hh = trainWide$SB_MotifThree_quantile_hh,
                  SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 = trainWide$SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
                  SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 = trainWide$SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
                  SP_Summaries_welch_rect_centroid = trainWide$SP_Summaries_welch_rect_centroid,
                  FC_LocalSimple_mean3_stderr = trainWide$FC_LocalSimple_mean3_stderr)

mod <- stan(file = "scripts/base-model.stan",
            data = stan_data, iter = 4000, chains = 3, seed = 123,
            control = list(max_treedepth = 15))

# Extract posteriors

priors <- as.data.frame(mod) %>%
  clean_names() %>%
  dplyr::select(c(1:23)) %>%
  summarise(alpha_mean = mean(alpha), alpha_sd = sd(alpha),
            beta_1_mean = mean(beta_1), beta_1_sd = sd(beta_2),
            beta_2_mean = mean(beta_2), beta_2_sd = sd(beta_1),
            beta_3_mean = mean(beta_3), beta_3_sd = sd(beta_3),
            beta_4_mean = mean(beta_4), beta_4_sd = sd(beta_4),
            beta_6_mean = mean(beta_6), beta_6_sd = sd(beta_6),
            beta_5_mean = mean(beta_5), beta_5_sd = sd(beta_5),
            beta_7_mean = mean(beta_7), beta_7_sd = sd(beta_7),
            beta_8_mean = mean(beta_8), beta_8_sd = sd(beta_8),
            beta_9_mean = mean(beta_9), beta_9_sd = sd(beta_9),
            beta_10_mean = mean(beta_10), beta_10_sd = sd(beta_10),
            beta_11_mean = mean(beta_11), beta_11_sd = sd(beta_11),
            beta_12_mean = mean(beta_12), beta_12_sd = sd(beta_12),
            beta_13_mean = mean(beta_13), beta_13_sd = sd(beta_13),
            beta_14_mean = mean(beta_14), beta_14_sd = sd(beta_14),
            beta_15_mean = mean(beta_15), beta_15_sd = sd(beta_15),
            beta_16_mean = mean(beta_16), beta_16_sd = sd(beta_16),
            beta_17_mean = mean(beta_17), beta_17_sd = sd(beta_17),
            beta_18_mean = mean(beta_18), beta_18_sd = sd(beta_18),
            beta_19_mean = mean(beta_19), beta_19_sd = sd(beta_19),
            beta_20_mean = mean(beta_20), beta_20_sd = sd(beta_20),
            beta_21_mean = mean(beta_21), beta_21_sd = sd(beta_21),
            beta_22_mean = mean(beta_22), beta_22_sd = sd(beta_22))

#-------------------------------------------------------------------------------
#--------------------------FIT INFORMED MODEL 2---------------------------------
#-------------------------------------------------------------------------------

options(mc.cores = parallel::detectCores())

stan_data <- list(N = nrow(testWide),
                  y = testWide$group,
                  DN_HistogramMode_5 = testWide$DN_HistogramMode_5,
                  DN_HistogramMode_10 = testWide$DN_HistogramMode_10,
                  CO_f1ecac = testWide$CO_f1ecac,
                  CO_FirstMin_ac = testWide$CO_FirstMin_ac,
                  CO_HistogramAMI_even_2_5 = testWide$CO_HistogramAMI_even_2_5,
                  CO_trev_1_num = testWide$CO_trev_1_num,
                  MD_hrv_classic_pnn40 = testWide$MD_hrv_classic_pnn40,
                  SB_BinaryStats_mean_longstretch1 = testWide$SB_BinaryStats_mean_longstretch1,
                  SB_TransitionMatrix_3ac_sumdiagcov = testWide$SB_TransitionMatrix_3ac_sumdiagcov,
                  PD_PeriodicityWang_th0_01 = testWide$PD_PeriodicityWang_th0_01,
                  CO_Embed2_Dist_tau_d_expfit_meandiff = testWide$CO_Embed2_Dist_tau_d_expfit_meandiff,
                  IN_AutoMutualInfoStats_40_gaussian_fmmi = testWide$IN_AutoMutualInfoStats_40_gaussian_fmmi,
                  FC_LocalSimple_mean1_tauresrat = testWide$FC_LocalSimple_mean1_tauresrat,
                  DN_OutlierInclude_p_001_mdrmd = testWide$DN_OutlierInclude_p_001_mdrmd,
                  DN_OutlierInclude_n_001_mdrmd = testWide$DN_OutlierInclude_n_001_mdrmd,
                  SP_Summaries_welch_rect_area_5_1 = testWide$SP_Summaries_welch_rect_area_5_1,
                  SB_BinaryStats_diff_longstretch0 = testWide$SB_BinaryStats_diff_longstretch0,
                  SB_MotifThree_quantile_hh = testWide$SB_MotifThree_quantile_hh,
                  SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 = testWide$SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
                  SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 = testWide$SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
                  SP_Summaries_welch_rect_centroid = testWide$SP_Summaries_welch_rect_centroid,
                  FC_LocalSimple_mean3_stderr = testWide$FC_LocalSimple_mean3_stderr,
                  alpha_mean = priors$alpha_mean,
                  alpha_sd = priors$alpha_sd,
                  DN_HistogramMode_5_mean = priors$beta_1_mean,
                  DN_HistogramMode_5_sd = priors$beta_1_sd,
                  DN_HistogramMode_10_mean = priors$beta_2_mean,
                  DN_HistogramMode_10_sd = priors$beta_2_sd,
                  CO_f1ecac_mean = priors$beta_3_mean,
                  CO_f1ecac_sd = priors$beta_3_sd,
                  CO_FirstMin_ac_mean = priors$beta_4_mean,
                  CO_FirstMin_ac_sd = priors$beta_4_sd,
                  CO_HistogramAMI_even_2_5_mean = priors$beta_5_mean,
                  CO_HistogramAMI_even_2_5_sd = priors$beta_5_sd,
                  CO_trev_1_num_mean = priors$beta_6_mean,
                  CO_trev_1_num_sd = priors$beta_6_sd,
                  MD_hrv_classic_pnn40_mean = priors$beta_7_mean,
                  MD_hrv_classic_pnn40_sd = priors$beta_7_sd,
                  SB_BinaryStats_mean_longstretch1_mean = priors$beta_8_mean,
                  SB_BinaryStats_mean_longstretch1_sd = priors$beta_8_sd,
                  SB_TransitionMatrix_3ac_sumdiagcov_mean = priors$beta_9_mean,
                  SB_TransitionMatrix_3ac_sumdiagcov_sd = priors$beta_9_sd,
                  PD_PeriodicityWang_th0_01_mean = priors$beta_10_mean,
                  PD_PeriodicityWang_th0_01_sd = priors$beta_10_sd,
                  CO_Embed2_Dist_tau_d_expfit_meandiff_mean = priors$beta_11_mean,
                  CO_Embed2_Dist_tau_d_expfit_meandiff_sd = priors$beta_11_sd,
                  IN_AutoMutualInfoStats_40_gaussian_fmmi_mean = priors$beta_12_mean,
                  IN_AutoMutualInfoStats_40_gaussian_fmmi_sd = priors$beta_12_sd,
                  FC_LocalSimple_mean1_tauresrat_mean = priors$beta_13_mean,
                  FC_LocalSimple_mean1_tauresrat_sd = priors$beta_13_sd,
                  DN_OutlierInclude_p_001_mdrmd_mean = priors$beta_14_mean,
                  DN_OutlierInclude_p_001_mdrmd_sd = priors$beta_14_sd,
                  DN_OutlierInclude_n_001_mdrmd_mean = priors$beta_15_mean,
                  DN_OutlierInclude_n_001_mdrmd_sd = priors$beta_15_sd,
                  SP_Summaries_welch_rect_area_5_1_mean = priors$beta_16_mean,
                  SP_Summaries_welch_rect_area_5_1_sd = priors$beta_16_sd,
                  SB_BinaryStats_diff_longstretch0_mean = priors$beta_17_mean,
                  SB_BinaryStats_diff_longstretch0_sd = priors$beta_17_sd,
                  SB_MotifThree_quantile_hh_mean = priors$beta_18_mean,
                  SB_MotifThree_quantile_hh_sd = priors$beta_18_sd,
                  SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1_mean = priors$beta_19_mean,
                  SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1_sd = priors$beta_19_sd,
                  SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1_mean = priors$beta_20_mean,
                  SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1_sd = priors$beta_20_sd,
                  SP_Summaries_welch_rect_centroid_mean = priors$beta_21_mean,
                  SP_Summaries_welch_rect_centroid_sd = priors$beta_21_sd,
                  FC_LocalSimple_mean3_stderr_mean = priors$beta_22_mean,
                  FC_LocalSimple_mean3_stderr_sd = priors$beta_22_sd)

# Fit model

mod2 <- stan(file = "scripts/informed-model.stan",
             data = stan_data, iter = 4000, chains = 3, seed = 123,
             control = list(max_treedepth = 15))

# Plot posterior distributions

as.data.frame(mod2) %>%
  clean_names() %>%
  dplyr::select(c(2:23)) %>%
  gather(key = parameter, value = value, 1:22) %>%
  ggplot(aes(x = value)) +
  geom_histogram(aes(y = after_stat(ndensity)), binwidth = 0.1, fill = "#E494D3", alpha = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Posterior distributions for SonyAIBORobotSurface1 data with informative priors",
       subtitle = "Bayesian logistic regression model run in Stan with 3 chains and 4,000 iterations per chain.",
       x = "Coefficient Value",
       y = "Density",
       caption = "Histogram binwidth = 0.1 and density scaled to max of 1.") +
  facet_wrap(~parameter)

#----------------------------------
# Plot posteriors against prior and 
# likelihood
#----------------------------------

# Uninformed priors

features <- colnames(as.data.frame(mod2))[2:23]
features <- gsub("beta_","\\1",features)
storage <- list()

for(f in features){
  mypriors <- data.frame(x = seq(from = -10, to = 10, length.out = nrow(as.data.frame(mod2)))) %>%
    mutate(value = LaplacesDemon::dst(x, mu = 0, sigma = 2.5, nu = 7),
           category = "Uninformed Prior") %>%
    dplyr::select(-c(x)) %>%
    mutate(parameter = f)
  
  storage[[f]] <- mypriors
}

data2Prior <- rbindlist(storage, use.names = TRUE)

# Informed priors

storage2 <- list()
nums <- seq(from = 3, to = ncol(priors)-1, by = 2)

for(f in nums){
  
  mypriors2 <- data.frame(x = seq(from = -10, to = 10, length.out = nrow(as.data.frame(mod2)))) %>%
    mutate(value = dnorm(x, mean = priors[,f], sd = priors[,f+1]),
           category = "Informed Prior") %>%
    mutate(parameter = f)
  
  storage2[[f]] <- mypriors2
}

indexToFeature <- data.frame(parameter = seq(from = 3, to = ncol(priors)-1, by = 2),
                             parameterProper = features)

data2InfPrior <- rbindlist(storage2, use.names = TRUE) %>%
  left_join(indexToFeature, by = c("parameter" = "parameter")) %>%
  dplyr::select(-c(parameter)) %>%
  rename(parameter = parameterProper)

# Plot

palette2 <- c("Uninformed Prior"= "#E494D3", 
              "Informed Prior"= "#87DCC0", 
              "Posterior Estimates" = "#88BBE4")

as.data.frame(mod2) %>%
  dplyr::select(c(2:23)) %>%
  gather(key = parameter, value = value, 1:22) %>%
  mutate(category = "Posterior Estimates",
         parameter = gsub("beta_", "\\1", parameter)) %>%
  bind_rows(data2Prior, data2InfPrior) %>%
  mutate(category = factor(category, levels = c("Uninformed Prior", "Informed Prior", "Posterior Estimates"))) %>%
  ggplot(aes(x = value, fill = category)) +
  geom_histogram(aes(y = after_stat(ndensity)), binwidth = 0.2, alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Posterior distributions for SonyAIBORobotSurface1 data with informative priors",
       subtitle = "Bayesian logistic regression model run in Stan with 3 chains and 4,000 iterations per chain.",
       x = "Coefficient Value",
       y = "Density",
       fill = NULL) +
  scale_colour_manual(values = palette2) +
  theme(legend.position = "bottom",
        legend.key = element_blank()) +
  facet_wrap(~parameter, scales = "free")
