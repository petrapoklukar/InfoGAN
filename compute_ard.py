#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:13:10 2020

@author: petrapoklukar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge


models = ['gan' + str(i) for i in range(1, 10)] + ['vae' + str(i) for i in range(1, 10)]

# PR data
with open('test_pr/nhoods_3_5_7_10_12/ipr_results_nhood_15000samples.pkl', 'rb') as f:
  pr_data_o = pickle.load(f)
  
  gan_configs = [
            'infogan1_l01_s2_4RL_model',
            'infogan2_l15_s2_4RL_model',
            'infogan3_l35_s2_4RL_model',
            'infogan4_l01_s3_4RL_model',
            'infogan5_l15_s3_4RL_model',
            'infogan6_l35_s3_4RL_model',
            'infogan7_l01_s6_4RL_model',
            'infogan8_l15_s6_4RL_model',
            'infogan9_l35_s6_4RL_model']
  nhoods = [3, 5, 7, 10, 12]
  nhood = nhoods.index(3)
  pr_data = {}
  for model in pr_data_o.keys():
    if 'gan' in model:
      model_name = model.split('_')[0][4:]
      precision = pr_data_o[model]['res5_gan']['precision'][nhood]
      recall = pr_data_o[model]['res5_gan']['recall'][nhood]
      pr_data[model_name] = (precision, recall)
    else:
      precision = pr_data_o[model]['res0_vae']['precision'][nhood]
      recall = pr_data_o[model]['res0_vae']['recall'][nhood]
      pr_data[model] = (precision, recall)

# Disentanglement data      
with open('disentanglement_test/disentanglement_scores_MMDalpha15_pval0p001_dict.pkl', 'rb') as g:
  dis_data_o = pickle.load(g)
  
  dis_data = {}
  for model in dis_data_o.keys():
    dis_data[model] = dis_data_o[model]['mmd']
    
    
# Linearity data      
with open('linearity_test/linearity_npoints20_var0.1_ranks.pkl', 'rb') as k:
  lin_data_o = pickle.load(k)
  
  lin_data = {}
  for model in lin_data_o.keys():
    lin_data[model] = [lin_data_o[model]['mse_test'], lin_data_o[model]['reg_score_test']]
 


# Policy training perfomance
with open('dataset/policy_performance/policy_training_performance.pkl', 'rb') as h:
  policy_data_o = pickle.load(h)
  
  policy_data_mean = []
  policy_data_max = []
  for model in sorted(models):
    policy_data_mean.append(policy_data_o[model + '_mean_reward'])
    policy_data_max.append(policy_data_o[model + '_max_reward'])
  policy_data_mean_np = np.array(policy_data_mean)
  policy_data_max_np = np.array(policy_data_max)
    

ard_data = []  
for model in sorted(models):
  features = list(dis_data[model]) + list(pr_data[model]) \
    + list(lin_data[model]) 
  ard_data.append(features)
ard_data_np = np.array(ard_data)

clf_mean = ARDRegression(compute_score=True, normalize=True)
clf_mean.fit(ard_data_np, policy_data_mean_np)
br_mean = BayesianRidge().fit(ard_data_np, policy_data_mean_np)

clf_max = ARDRegression(compute_score=True, normalize=True)
clf_max.fit(ard_data_np, policy_data_max_np)
  
print(clf_mean.coef_)
print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))

print(clf_max.coef_)
print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))

