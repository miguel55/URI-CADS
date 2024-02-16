#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:41:34 2019

@author: mmolina
"""


import os
import numpy as np
from config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Results 
results_kidney = np.zeros((0,5),dtype='float')
results_lesions = np.zeros((0,9),dtype='float')
results_global = np.zeros((0,cfg.num_classes2),dtype='float')
results_final = np.zeros((0,cfg.num_classes2),dtype='float')
results_true = np.zeros((0,cfg.num_classes2),dtype='float')
results_local = np.zeros((0,cfg.num_classes2),dtype='float')
# Alpha/beta parameters
alpha=np.zeros((0,cfg.num_classes2),dtype='float')
beta=np.zeros((0,cfg.num_classes2),dtype='float')
    
for fold in cfg.folds:
    test=sio.loadmat(os.path.join(cfg.data_dir,'splits','idxTestM'+str(fold)+'.mat'))['idxTest'][0]

    # Save results
    aux=sio.loadmat(os.path.join(cfg.result_dir,'results_kidney'+str(fold)+'.mat'))['results_kidney']
    aux[:,0]=test[aux[:,0].astype('int')]
    results_kidney=np.concatenate((results_kidney,aux),axis=0)
    aux=sio.loadmat(os.path.join(cfg.result_dir,'results_lesions'+str(fold)+'.mat'))['results_lesions']
    aux[:,0]=test[aux[:,0].astype('int')]
    results_lesions=np.concatenate((results_lesions,aux),axis=0)
    aux=sio.loadmat(os.path.join(cfg.result_dir,'results_global'+str(fold)+'.mat'))['results_global']
    results_global=np.concatenate((results_global,aux[:,1:]),axis=0)
    aux=sio.loadmat(os.path.join(cfg.result_dir,'results_final'+str(fold)+'.mat'))['results_final']
    results_final=np.concatenate((results_final,aux[:,1:]),axis=0)
    aux=sio.loadmat(os.path.join(cfg.result_dir,'results_local'+str(fold)+'.mat'))['results_local']
    results_local=np.concatenate((results_local,aux[:,1:]),axis=0)
    aux=sio.loadmat(os.path.join(cfg.result_dir,'results_true'+str(fold)+'.mat'))['results_true']
    results_true=np.concatenate((results_true,aux[:,1:]),axis=0)
    alpha=np.concatenate((alpha,sio.loadmat(os.path.join(cfg.result_dir,'alpha'+str(fold)+'.mat'))['alpha']),axis=0)
    beta=np.concatenate((beta,sio.loadmat(os.path.join(cfg.result_dir,'beta'+str(fold)+'.mat'))['beta']),axis=0)
    
        
# KIDNEYS
# Detection statistics: precision and recall per class
ious_kidney=[]
for p in np.unique(results_kidney[:,0]):
    kidney_det=results_kidney[np.where(results_kidney[:,0]==p)[0],:]
    ious_kidney.append(kidney_det[0,4])
print('Kidney Segmentation Evaluation')
print('Number of detected kidneys:{:d}'.format(np.size(ious_kidney)))
for i in range(alpha.shape[0]-np.size(ious_kidney)):
    ious_kidney.append(0)
print('IoU:{:4f}'.format(np.mean(ious_kidney)))
# LESION LOCALS
# Classification statistics: precision and recall
print('FINAL PATHOLOGY EVALUATION')
final_auc=np.zeros((len(cfg.class_names2)),dtype='float')
for i in range(len(cfg.class_names2)):
    if (len(np.unique(results_true[:,i]))>1):
        final_auc[i]=roc_auc_score(results_true[:,i], results_final[:,i], average='micro')#,multi_class='ovr'
    else:
        final_auc[i]=0.0
        
sp_sens95=np.zeros((cfg.num_classes2,),dtype='float')
plt.figure()
# Roc curves for SP-95
for i in range(cfg.num_classes2):
    fpr, tpr, thresholds=roc_curve(results_true[:,i], results_final[:,i])
    plt.plot(1-fpr,tpr)
    sp_sens95[i]=1-fpr[np.where(tpr>0.95)[0][0]]
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.title('Sensitivity-Specificity curve for every class')
plt.legend(cfg.class_names2)
plt.show()

for i in range(len(cfg.class_names2)):
    print('Class: ' + cfg.class_names2[i]+'. AUC: {:4f}'.format(final_auc[i])+'. SP-SENS95: {:4f}'.format(sp_sens95[i]))
    
print('ALPHA')
print('Mean alpha_')
print(np.mean(alpha,axis=0))
