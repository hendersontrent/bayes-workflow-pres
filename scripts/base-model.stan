//
// This Stan program defines a basic logistic regression model for
// classifying two groups based on catch22 time-series features
//

//
// Author: Trent Henderson, 3 June 2021
//

data {
  int<lower=0> N; // Number of observations
  int y[N]; // Response variable (group)
  vector[N] DN_HistogramMode_5;
  vector[N] DN_HistogramMode_10;
  vector[N] CO_f1ecac;
  vector[N] CO_FirstMin_ac;
  vector[N] CO_HistogramAMI_even_2_5;
  vector[N] CO_trev_1_num;
  vector[N] MD_hrv_classic_pnn40;
  vector[N] SB_BinaryStats_mean_longstretch1;
  vector[N] SB_TransitionMatrix_3ac_sumdiagcov;
  vector[N] PD_PeriodicityWang_th0_01;
  vector[N] CO_Embed2_Dist_tau_d_expfit_meandiff;
  vector[N] IN_AutoMutualInfoStats_40_gaussian_fmmi;
  vector[N] FC_LocalSimple_mean1_tauresrat;
  vector[N] DN_OutlierInclude_p_001_mdrmd;
  vector[N] DN_OutlierInclude_n_001_mdrmd;
  vector[N] SP_Summaries_welch_rect_area_5_1;
  vector[N] SB_BinaryStats_diff_longstretch0;
  vector[N] SB_MotifThree_quantile_hh;
  vector[N] SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1;
  vector[N] SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1;
  vector[N] SP_Summaries_welch_rect_centroid;
  vector[N] FC_LocalSimple_mean3_stderr;
}

parameters {
  
  real alpha; // Intercept
  vector[22] beta; // Regression coefficients
}

transformed parameters{
  
  // Instantiate and estimate linear predictor
  
  vector[N] eta;
  
  for(n in 1:N){
    eta[n] = alpha + beta[1]*DN_HistogramMode_5[n] + beta[2]*DN_HistogramMode_10[n] + beta[3]*CO_f1ecac[n] + beta[4]*CO_FirstMin_ac[n] + beta[5]*CO_HistogramAMI_even_2_5[n] + beta[6]*CO_trev_1_num[n] + beta[7]*MD_hrv_classic_pnn40[n] + beta[8]*SB_BinaryStats_mean_longstretch1[n] + beta[9]*SB_TransitionMatrix_3ac_sumdiagcov[n] + beta[10]*PD_PeriodicityWang_th0_01[n] + beta[11]*CO_Embed2_Dist_tau_d_expfit_meandiff[n] + beta[12]*IN_AutoMutualInfoStats_40_gaussian_fmmi[n] + beta[13]*FC_LocalSimple_mean1_tauresrat[n] + beta[14]*DN_OutlierInclude_p_001_mdrmd[n] + beta[15]*DN_OutlierInclude_n_001_mdrmd[n] + beta[16]*SP_Summaries_welch_rect_area_5_1[n] + beta[17]*SB_BinaryStats_diff_longstretch0[n] + beta[18]*SB_MotifThree_quantile_hh[n] + beta[19]*SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1[n] + beta[20]*SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1[n] + beta[21]*SP_Summaries_welch_rect_centroid[n] + beta[22]*FC_LocalSimple_mean3_stderr[n];
  }
}

model {
  
  // Priors
  
  alpha ~ normal(0,3);
  beta ~ student_t(7,0,2.5); // Wide prior from Vehtari (2019)
  
  // Likelihood
  
  y ~ bernoulli_logit(eta);
}
