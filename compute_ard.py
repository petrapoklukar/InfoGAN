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
import sklearn
import copy
import scipy.stats


gan_models = ['gan' + str(i) for i in range(1, 10)]
vae_models = ['vae' + str(i) for i in range(1, 10)]
models = gan_models + vae_models


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
  for model in sorted(pr_data_o.keys()):
    if 'gan' in model:
      model_name = model.split('_')[0][4:]
      precision = pr_data_o[model]['res5_gan']['precision'][nhood]
      recall = pr_data_o[model]['res5_gan']['recall'][nhood]
      pr_data[model_name] = [precision, recall]
#      pr_data[model_name] = [precision]
#      pr_data[model_name] = [recall]
    else:
      precision = pr_data_o[model]['res0_vae']['precision'][nhood]
      recall = pr_data_o[model]['res0_vae']['recall'][nhood]
      pr_data[model] = [precision, recall]
#      pr_data[model] = [precision]
#      pr_data[model] = [recall]


# Disentanglement data      
with open('disentanglement_test/disentanglement_scores_MMDalpha15_pval0p001_dict.pkl', 'rb') as g:
  dis_data_o = pickle.load(g)
  
  dis_data = {}
  for model in sorted(dis_data_o.keys()):
    dis_data[model] = dis_data_o[model]['mmd']
    
    
# Linearity data      
with open('linearity_test/linearity_npoints50_var0.2_ranks.pkl', 'rb') as k:
  lin_data_o = pickle.load(k)
  
  lin_data = {}
  for model in sorted(lin_data_o.keys()):
    lin_data[model] = [lin_data_o[model]['mse_test']]#, lin_data_o[model]['reg_score_test']]
 

# VAE training parameters
vae_params = {
    'vae1': [2, 1.6e-2], #, 1.5, 2.1e-2],
    'vae2': [2, 6.4e-3], #, 2.5, 1.1e-2],
    'vae3': [2, 3.2e-3], #, 3.4, 6.7e-3],
    'vae4': [3, 1.6e-2], #, 1.5, 2.1e-2],
    'vae5': [3, 7.2e-3], #, 2.4, 1.1e-2],
    'vae6': [3, 3.2e-3], #, 3.5, 5.5e-3],
    'vae7': [6, 1.6e-2], #, 1.5, 2.1e-2],
    'vae8': [6, 7.2e-3], #, 2.4, 1.1e-2],
    'vae9': [6, 3.6e-3] #, ]
    }

# GAN training parameters
gan_params = {
    'gan1': [2, 0.1, 2.81, 0.10, 0.54],
    'gan2': [2, 1.5, 2.28, 0.59, 0.78],
    'gan3': [2, 3.5, 2.43, 1.07, 0.63],
    'gan4': [3, 0.1, 2.52, 0.16, 0.61],
    'gan5': [3, 1.5, 2.27, 1.45, 0.75],
    'gan6': [3, 3.5, 2.23, 3.16, 0.68],
    'gan7': [6, 0.1, 2.50, 0.44, 0.63],
    'gan8': [6, 1.5, 2.23, 4.78, 0.73],
    'gan9': [6, 3.5, 2.27, 10.32, 0.67],
    
    }


# Policy training perfomance
with open('dataset/policy_performance/policy_training_performance.pkl', 'rb') as h:
  policy_data_o = pickle.load(h)  
#  policy_data_mean = []
#  policy_data_max = []
#  for model in sorted(models):
#    policy_data_mean.append(policy_data_o[model + '_mean_reward'])
#    policy_data_max.append(policy_data_o[model + '_max_reward'])
#  policy_data_mean_np = np.array(policy_data_mean)
#  policy_data_max_np = np.array(policy_data_max)

def get_policy_data(model_list, policy_data_o):
  policy_data_mean = {}
  policy_data_max = {}
  for model in sorted(model_list):
    policy_data_mean[model] = policy_data_o[model + '_mean_reward']
    policy_data_max[model] =  policy_data_o[model + '_max_reward']
  return policy_data_mean, policy_data_max
  
def get_eval_data_np(model_list):
  eval_metrics_ard_data = []  
  for model in sorted(model_list):
      features = list(dis_data[model]) + list(pr_data[model]) \
        + list(lin_data[model])
      eval_metrics_ard_data.append(features)
  eval_metrics_ard_data_np = np.array(eval_metrics_ard_data)
  return eval_metrics_ard_data_np

def get_eval_data_dict(model_list):
  eval_metrics_ard_data = {model: [] for model in model_list}
  for model in sorted(model_list):
      features = list(dis_data[model]) + list(pr_data[model]) \
        + list(lin_data[model])
      eval_metrics_ard_data[model] += features
  feature_names = ['dip', 'dir', 'p', 'r', 'loc_lin',]
  return eval_metrics_ard_data, feature_names
  
def add_training_params(eval_d, params):
  for model in eval_d.keys():
    eval_d[model] += params[model]
  return eval_d


# -------------------------------------------- #
# - Evaluation metrics
# -------------------------------------------- #
# Get data from evaluation metrics
eval_metrics_ard_data_d, f_names = get_eval_data_dict(models)
policy_data_mean, policy_data_max = get_policy_data(models, policy_data_o)
policy_data_mean_np = np.array(list(policy_data_mean.values()))
policy_data_max_np = np.array(list(policy_data_max.values()))

# Evaluate ard
clf_mean = ARDRegression(compute_score=True, normalize=True)
clf_mean.fit(np.array(list(eval_metrics_ard_data_d.values())), 
             policy_data_mean_np)

clf_max = ARDRegression(compute_score=True, normalize=True)
clf_max.fit(np.array(list(eval_metrics_ard_data_d.values())), 
            policy_data_max_np)

print('ALL eval inputs,   ', f_names)
print('Mean reward: ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(clf_mean.lambda_)))))

#print('Mean reward; ', list(map(lambda x: round(x, 3), clf_mean.lambda_)))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))

print('Max reward: ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(clf_max.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))


# -------------------------------------------- #
# - VAE Evaluation metrics and training params 
# -------------------------------------------- #
# Get VAE data from evaluation metrics and training params
vae_eval_metrics_ard_data_d, vae_eval_f_names = get_eval_data_dict(vae_models)
vae_all_ard_data_d = add_training_params(copy.deepcopy(vae_eval_metrics_ard_data_d), 
                                         vae_params)
vae_all_f_names = vae_eval_f_names + ['n_alpha', 'beta']
vae_policy_data_mean, vae_policy_data_max = get_policy_data(vae_models, policy_data_o)
vae_policy_data_mean_np = np.array(list(vae_policy_data_mean.values()))
vae_policy_data_max_np = np.array(list(vae_policy_data_max.values()))


vae_eval_clf_mean = ARDRegression(compute_score=True, normalize=True)
vae_eval_clf_mean.fit(np.array(list(vae_eval_metrics_ard_data_d.values())), 
             vae_policy_data_mean_np)

vae_eval_clf_max = ARDRegression(compute_score=True, normalize=True)
vae_eval_clf_max.fit(np.array(list(vae_eval_metrics_ard_data_d.values())), 
            vae_policy_data_max_np)

print('\n\nVAE eval inputs,   ', vae_eval_f_names)
print('Mean reward; ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(vae_eval_clf_mean.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))

print('Max reward; ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(vae_eval_clf_max.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))


vae_clf_mean = ARDRegression(compute_score=True, normalize=True)
vae_clf_mean.fit(np.array(list(vae_all_ard_data_d.values())), 
             vae_policy_data_mean_np)

vae_clf_max = ARDRegression(compute_score=True, normalize=True)
vae_clf_max.fit(np.array(list(vae_all_ard_data_d.values())), 
            vae_policy_data_max_np)

print('\nVAE all inputs,   ', vae_all_f_names)
print('Mean reward; ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(vae_clf_mean.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))

print('Max reward; ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(vae_clf_max.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))


# Calculating Pearsons coeff
vae_all_data = np.array(list(vae_all_ard_data_d.values()))

vae_pearsons_mean, vae_pearsons_max = [], []
for column in range(vae_all_data.shape[1]):
  pear_mean = scipy.stats.pearsonr(vae_all_data[:, column], vae_policy_data_mean_np)
  vae_pearsons_mean.append(list(pear_mean))
#  pearsons_mean.append([round(pear_mean[0], 3), round(pear_mean[1], 3)])
  pear_max = scipy.stats.pearsonr(vae_all_data[:, column], vae_policy_data_max_np)
  vae_pearsons_max.append(list(pear_max))
#  pearsons_max.append([round(pear_max[0], 3), round(pear_max[1], 3)])

vae_pearsons_mean_np = np.round(np.array(vae_pearsons_mean), 3)
vae_pearsons_max_np = np.round(np.array(vae_pearsons_max), 3)

print('\nVAE Pearsons R:   ', vae_all_f_names)
print('Mean reward:\n')
print('R coeff; ', 
      ' '.join(list(map(lambda z: '${0:.3f}$ & '.format(z), list(vae_pearsons_mean_np[:, 0])))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))
print('p value; ', 
      ' '.join(list(map(lambda z: '${0:.3f}$ & '.format(z), list(vae_pearsons_mean_np[:, 1])))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))

print('Max reward:\n')
print('R coeff; ', 
      ' '.join(list(map(lambda z: '${0:.3f}$ & '.format(z), list(vae_pearsons_max_np[:, 0])))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))
print('p value; ', 
      ' '.join(list(map(lambda z: '${0:.3f}$ & '.format(z), list(vae_pearsons_max_np[:, 1])))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))





# -------------------------------------------- #
# - GAN Evaluation metrics and training params 
# -------------------------------------------- #
# Get GAN data from evaluation metrics and training params
gan_eval_metrics_ard_data_d, gan_eval_f_names = get_eval_data_dict(gan_models)
gan_all_ard_data_d = add_training_params(copy.deepcopy(gan_eval_metrics_ard_data_d), 
                                         gan_params)
gan_all_f_names = gan_eval_f_names + ['n_alpha', 'lambda', 'Gloss', 'Iloss', 'totalloss']

gan_policy_data_mean, gan_policy_data_max = get_policy_data(gan_models, policy_data_o)
gan_policy_data_mean_np = np.array(list(gan_policy_data_mean.values()))
gan_policy_data_max_np = np.array(list(gan_policy_data_max.values()))

gan_eval_clf_mean = ARDRegression(compute_score=True, normalize=True)
gan_eval_clf_mean.fit(np.array(list(gan_eval_metrics_ard_data_d.values())), 
             gan_policy_data_mean_np)

gan_eval_clf_max = ARDRegression(compute_score=True, normalize=True)
gan_eval_clf_max.fit(np.array(list(gan_eval_metrics_ard_data_d.values())), 
            gan_policy_data_max_np)

print('\n\nGAN eval inputs,   ', gan_eval_f_names)
print('Mean reward; ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(gan_eval_clf_mean.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))

print('Max reward; ',
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(gan_eval_clf_max.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))


gan_clf_mean = ARDRegression(compute_score=True, normalize=True)
gan_clf_mean.fit(np.array(list(gan_all_ard_data_d.values())), 
             gan_policy_data_mean_np)

gan_clf_max = ARDRegression(compute_score=True, normalize=True)
gan_clf_max.fit(np.array(list(gan_all_ard_data_d.values())), 
            gan_policy_data_max_np)

print('\nGAN all inputs,   ', gan_eval_f_names)
print('Mean reward; ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(gan_clf_mean.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))

print('Max reward; ', 
      ' '.join(list(map(lambda z: '$\scnum{0}{1:.2e}{2}$ & '.format('{', z, '}'), list(gan_clf_max.lambda_)))))
#print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))


# Calculating Pearsons coeff
gan_all_data = np.array(list(gan_all_ard_data_d.values()))

pearsons_mean, pearsons_max = [], []
for column in range(gan_all_data.shape[1]):
  pear_mean = scipy.stats.pearsonr(gan_all_data[:, column], gan_policy_data_mean_np)
  pearsons_mean.append(list(pear_mean))
#  pearsons_mean.append([round(pear_mean[0], 3), round(pear_mean[1], 3)])
  pear_max = scipy.stats.pearsonr(gan_all_data[:, column], gan_policy_data_max_np)
  pearsons_max.append(list(pear_max))
#  pearsons_max.append([round(pear_max[0], 3), round(pear_max[1], 3)])

pearsons_mean_np = np.round(np.array(pearsons_mean), 3)
pearsons_max_np = np.round(np.array(pearsons_max), 3)

print('\nGAN Pearsons R:   ', gan_all_f_names)
print('Mean reward:\n')
print('R coeff; ', 
      ' '.join(list(map(lambda z: '${0:.3f}$ & '.format(z), list(pearsons_mean_np[:, 0])))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))
print('p value; ', 
      ' '.join(list(map(lambda z: '${0:.3f}$ & '.format(z), list(pearsons_mean_np[:, 1])))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))

print('Max reward:\n')
print('R coeff; ', 
      ' '.join(list(map(lambda z: '${0:.3f}$ & '.format(z), list(pearsons_max_np[:, 0])))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))
print('p value; ', 
      ' '.join(list(map(lambda z: '${0:.3f}$ & '.format(z), list(pearsons_max_np[:, 1])))))
#print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))


if False:  
      
  ard_data = []  
  agg_ard_data = []  
  for model in sorted(models):
    features = list(dis_data[model]) + list(pr_data[model]) \
      + list(lin_data[model])
  #  features_agg = list(map(lambda x: [x[0] + x[1], x[2] + x[3], x[4]], [features]))
    ard_data.append(features)
  #  agg_ard_data.append(features_agg[0])
  ard_data_np = np.array(ard_data)
  #agg_ard_data_np = np.array(agg_ard_data)
  #agg_clf_max = ARDRegression(compute_score=True, normalize=True)
  #agg_clf_max.fit(agg_ard_data_np, policy_data_max_np)
  
  clf_mean = ARDRegression(compute_score=True, normalize=True)
  clf_mean.fit(ard_data_np, policy_data_mean_np)
  br_mean = BayesianRidge().fit(ard_data_np, policy_data_mean_np)
  
  clf_max = ARDRegression(compute_score=True, normalize=True)
  clf_max.fit(ard_data_np, policy_data_max_np)
  
  print('Raw inputs')
  print('Max reward; ', np.round(clf_max.coef_  * 100, 3))
  #print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))
  
  print('Mean reward; ',np.round(clf_mean.coef_ * 100, 3))
  #print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))
  
  
  # Scale features t [0,1]
  normalizer = sklearn.preprocessing.MinMaxScaler()
  ard_data_np_unit = normalizer.fit_transform(ard_data_np)
  
  clf_mean_unit = ARDRegression(compute_score=True, normalize=False)
  clf_mean_unit.fit(ard_data_np_unit, policy_data_mean_np)
  
  clf_max_unit = ARDRegression(compute_score=True, normalize=False)
  clf_max_unit.fit(ard_data_np_unit, policy_data_max_np)
    
  print('Feature scaling')
  print('Max reward; ', np.round(clf_max_unit.coef_  * 100, 3))
  #print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))
  
  print('Mean reward; ',np.round(clf_mean_unit.coef_ * 100, 3))
  #print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))
  
  
  # Standardinze with mean and variance
  standardizer = sklearn.preprocessing.StandardScaler()
  ard_data_np_standard = standardizer.fit_transform(ard_data_np)
  
  results_std = []
  for i in range(100):
      clf_mean_standard = ARDRegression(compute_score=True, normalize=False)
      clf_mean_standard.fit(ard_data_np_standard, policy_data_mean_np)
      
      clf_max_standard = ARDRegression(compute_score=True, normalize=False)
      clf_max_standard.fit(ard_data_np_standard, policy_data_max_np)
      results_std.append(clf_max_standard.coef_ )
    
  print('Feature scaling')
  print('Max reward; ', np.round(clf_max_standard.coef_  * 100, 3))
  #print(sorted(list(zip(sorted(models), clf_max.predict(ard_data_np))), key=lambda x: x[1]))
  
  print('Mean reward; ',np.round(clf_mean_standard.coef_ * 100, 3))
  #print(sorted(list(zip(sorted(models), clf_mean.predict(ard_data_np))), key=lambda x: x[1]))
  
  
  
