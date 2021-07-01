//
// This Stan program defines a logistic regression model with informed priors for
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
  
  // Prior means
  
  real alpha_mean;
  
  real DN_HistogramMode_5_mean;
  real DN_HistogramMode_10_mean;
  real CO_f1ecac_mean;
  real CO_FirstMin_ac_mean;
  real CO_HistogramAMI_even_2_5_mean;
  real CO_trev_1_num_mean;
  real MD_hrv_classic_pnn40_mean;
  real SB_BinaryStats_mean_longstretch1_mean;
  real SB_TransitionMatrix_3ac_sumdiagcov_mean;
  real PD_PeriodicityWang_th0_01_mean;
  real CO_Embed2_Dist_tau_d_expfit_meandiff_mean;
  real IN_AutoMutualInfoStats_40_gaussian_fmmi_mean;
  real FC_LocalSimple_mean1_tauresrat_mean;
  real DN_OutlierInclude_p_001_mdrmd_mean;
  real DN_OutlierInclude_n_001_mdrmd_mean;
  real SP_Summaries_welch_rect_area_5_1_mean;
  real SB_BinaryStats_diff_longstretch0_mean;
  real SB_MotifThree_quantile_hh_mean;
  real SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1_mean;
  real SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1_mean;
  real SP_Summaries_welch_rect_centroid_mean;
  real FC_LocalSimple_mean3_stderr_mean;
  
  // Prior standard deviations
  
  real alpha_sd;
  
  real DN_HistogramMode_5_sd;
  real DN_HistogramMode_10_sd;
  real CO_f1ecac_sd;
  real CO_FirstMin_ac_sd;
  real CO_HistogramAMI_even_2_5_sd;
  real CO_trev_1_num_sd;
  real MD_hrv_classic_pnn40_sd;
  real SB_BinaryStats_mean_longstretch1_sd;
  real SB_TransitionMatrix_3ac_sumdiagcov_sd;
  real PD_PeriodicityWang_th0_01_sd;
  real CO_Embed2_Dist_tau_d_expfit_meandiff_sd;
  real IN_AutoMutualInfoStats_40_gaussian_fmmi_sd;
  real FC_LocalSimple_mean1_tauresrat_sd;
  real DN_OutlierInclude_p_001_mdrmd_sd;
  real DN_OutlierInclude_n_001_mdrmd_sd;
  real SP_Summaries_welch_rect_area_5_1_sd;
  real SB_BinaryStats_diff_longstretch0_sd;
  real SB_MotifThree_quantile_hh_sd;
  real SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1_sd;
  real SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1_sd;
  real SP_Summaries_welch_rect_centroid_sd;
  real FC_LocalSimple_mean3_stderr_sd;
}

parameters {
  
  // Intercept
  
  real alpha;
  
  // Regression coefficients
  
  real beta_DN_HistogramMode_5;
  real beta_DN_HistogramMode_10;
  real beta_CO_f1ecac;
  real beta_CO_FirstMin_ac;
  real beta_CO_HistogramAMI_even_2_5;
  real beta_CO_trev_1_num;
  real beta_MD_hrv_classic_pnn40;
  real beta_SB_BinaryStats_mean_longstretch1;
  real beta_SB_TransitionMatrix_3ac_sumdiagcov;
  real beta_PD_PeriodicityWang_th0_01;
  real beta_CO_Embed2_Dist_tau_d_expfit_meandiff;
  real beta_IN_AutoMutualInfoStats_40_gaussian_fmmi;
  real beta_FC_LocalSimple_mean1_tauresrat;
  real beta_DN_OutlierInclude_p_001_mdrmd;
  real beta_DN_OutlierInclude_n_001_mdrmd;
  real beta_SP_Summaries_welch_rect_area_5_1;
  real beta_SB_BinaryStats_diff_longstretch0;
  real beta_SB_MotifThree_quantile_hh;
  real beta_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1;
  real beta_SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1;
  real beta_SP_Summaries_welch_rect_centroid;
  real beta_FC_LocalSimple_mean3_stderr;
}

transformed parameters{
  
  // Instantiate and estimate linear predictor
  
  vector[N] eta;
  
  for(n in 1:N){
    eta[n] = alpha + beta_DN_HistogramMode_5*DN_HistogramMode_5[n] + beta_DN_HistogramMode_10*DN_HistogramMode_10[n] + beta_CO_f1ecac*CO_f1ecac[n] + beta_CO_FirstMin_ac*CO_FirstMin_ac[n] + beta_CO_HistogramAMI_even_2_5*CO_HistogramAMI_even_2_5[n] + beta_CO_trev_1_num*CO_trev_1_num[n] + beta_MD_hrv_classic_pnn40*MD_hrv_classic_pnn40[n] + beta_SB_BinaryStats_mean_longstretch1*SB_BinaryStats_mean_longstretch1[n] + beta_SB_TransitionMatrix_3ac_sumdiagcov*SB_TransitionMatrix_3ac_sumdiagcov[n] + beta_PD_PeriodicityWang_th0_01*PD_PeriodicityWang_th0_01[n] + beta_CO_Embed2_Dist_tau_d_expfit_meandiff*CO_Embed2_Dist_tau_d_expfit_meandiff[n] + beta_IN_AutoMutualInfoStats_40_gaussian_fmmi*IN_AutoMutualInfoStats_40_gaussian_fmmi[n] + beta_FC_LocalSimple_mean1_tauresrat*FC_LocalSimple_mean1_tauresrat[n] + beta_DN_OutlierInclude_p_001_mdrmd*DN_OutlierInclude_p_001_mdrmd[n] + beta_DN_OutlierInclude_n_001_mdrmd*DN_OutlierInclude_n_001_mdrmd[n] + beta_SP_Summaries_welch_rect_area_5_1*SP_Summaries_welch_rect_area_5_1[n] + beta_SB_BinaryStats_diff_longstretch0*SB_BinaryStats_diff_longstretch0[n] + beta_SB_MotifThree_quantile_hh*SB_MotifThree_quantile_hh[n] + beta_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1*SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1[n] + beta_SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1*SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1[n] + beta_SP_Summaries_welch_rect_centroid*SP_Summaries_welch_rect_centroid[n] + beta_FC_LocalSimple_mean3_stderr*FC_LocalSimple_mean3_stderr[n];
  }
}

model {
  
  // Priors
  
  alpha ~ normal(alpha_mean, alpha_sd);
  beta_DN_HistogramMode_5 ~ normal(DN_HistogramMode_5_mean, DN_HistogramMode_5_sd);
  beta_DN_HistogramMode_10 ~ normal(DN_HistogramMode_10_mean, DN_HistogramMode_10_sd);
  beta_CO_f1ecac ~ normal(CO_f1ecac_mean, CO_f1ecac_sd);
  beta_CO_FirstMin_ac ~ normal(CO_FirstMin_ac_mean, CO_FirstMin_ac_sd);
  beta_CO_HistogramAMI_even_2_5 ~ normal(CO_HistogramAMI_even_2_5_mean, CO_HistogramAMI_even_2_5_sd);
  beta_CO_trev_1_num ~ normal(CO_trev_1_num_mean, CO_trev_1_num_sd);
  beta_MD_hrv_classic_pnn40 ~ normal(MD_hrv_classic_pnn40_mean, MD_hrv_classic_pnn40_sd);
  beta_SB_BinaryStats_mean_longstretch1 ~ normal(SB_BinaryStats_mean_longstretch1_mean, SB_BinaryStats_mean_longstretch1_sd);
  beta_SB_TransitionMatrix_3ac_sumdiagcov ~ normal(SB_TransitionMatrix_3ac_sumdiagcov_mean, SB_TransitionMatrix_3ac_sumdiagcov_sd);
  beta_PD_PeriodicityWang_th0_01 ~ normal(PD_PeriodicityWang_th0_01_mean, PD_PeriodicityWang_th0_01_sd);
  beta_CO_Embed2_Dist_tau_d_expfit_meandiff ~ normal(CO_Embed2_Dist_tau_d_expfit_meandiff_mean, CO_Embed2_Dist_tau_d_expfit_meandiff_sd);
  beta_IN_AutoMutualInfoStats_40_gaussian_fmmi ~ normal(IN_AutoMutualInfoStats_40_gaussian_fmmi_mean, IN_AutoMutualInfoStats_40_gaussian_fmmi_sd);
  beta_FC_LocalSimple_mean1_tauresrat ~ normal(FC_LocalSimple_mean1_tauresrat_mean, FC_LocalSimple_mean1_tauresrat_sd);
  beta_DN_OutlierInclude_p_001_mdrmd ~ normal(DN_OutlierInclude_p_001_mdrmd_mean, DN_OutlierInclude_p_001_mdrmd_sd);
  beta_DN_OutlierInclude_n_001_mdrmd ~ normal(DN_OutlierInclude_n_001_mdrmd_mean, DN_OutlierInclude_n_001_mdrmd_sd);
  beta_SP_Summaries_welch_rect_area_5_1 ~ normal(SP_Summaries_welch_rect_area_5_1_mean, SP_Summaries_welch_rect_area_5_1_sd);
  beta_SB_BinaryStats_diff_longstretch0 ~ normal(SB_BinaryStats_diff_longstretch0_mean, SB_BinaryStats_diff_longstretch0_sd);
  beta_SB_MotifThree_quantile_hh ~ normal(SB_MotifThree_quantile_hh_mean, SB_MotifThree_quantile_hh_sd);
  beta_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 ~ normal(SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1_mean, SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1_sd);
  beta_SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 ~ normal(SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1_mean, SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1_sd);
  beta_SP_Summaries_welch_rect_centroid ~ normal(SP_Summaries_welch_rect_centroid_mean, SP_Summaries_welch_rect_centroid_sd);
  beta_FC_LocalSimple_mean3_stderr ~ normal(FC_LocalSimple_mean3_stderr_mean, FC_LocalSimple_mean3_stderr_sd);
  
  // Likelihood
  
  y ~ bernoulli_logit(eta);
}

generated quantities {
  
  // Compute estimates of the likelihood for LOO-CV
  
  vector[N] log_lik; // Log-likelihood for LOO
  vector[N] y_rep; // PPC replications for model diagnostics

  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta[n]);
    y_rep[n] = bernoulli_rng(inv_logit(eta[n]));
  }
}
