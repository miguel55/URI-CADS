#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:41:34 2019

@author: mmolina
"""


import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import torch
from collections import OrderedDict
import torchvision_05
from torchvision_05.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision_05.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead, GeneralizedRCNNTransform
from torchvision_05.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision_05.models.detection.roi_heads import RoIHeads
from torchvision_05.ops import MultiScaleRoIAlign
import torchvision_05.transforms.functional as F
from torchvision_05.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import scipy.io as sio
import cv2
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import natsort

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def test_detection_model_full(model, dataloader, num_classes1, class_names1, num_classes2, class_names2, num_global_classes, th_score1, th_score2, th_iou1, th_iou2, th_mask, result_dir, GLOBAL_PAT, VERBOSE, train_val_test, batch_size=1):
    since = time.time()
    model.eval()   # Set model to evaluate mode

    # Detection measurements
    ret1 = 0
    rel1 = np.zeros((num_classes1-1,),dtype=int)
    ret_rel1 = np.zeros((num_classes1-1),dtype=int)
    ret2 = 0
    if (GLOBAL_PAT):
        rel2 = np.zeros((num_classes2-(num_global_classes+1),),dtype=int)
        ret_rel2 = np.zeros((num_classes2-(num_global_classes+1)),dtype=int)
    else:
        rel2 = np.zeros((num_classes2-1,),dtype=int)
        ret_rel2 = np.zeros((num_classes2-1),dtype=int)
        
    # Classification measurements
    y_true1=np.zeros((0,),dtype='int')
    y_pred1=np.zeros((0,),dtype='int')
    y_true2=np.zeros((0,),dtype='int')
    y_pred2=np.zeros((0,),dtype='int')
    # Global pathologies measurements
    global_true=np.zeros((0,num_classes2),dtype='int')
    global_pred=np.zeros((0,num_classes2),dtype='float')
    local_pred=np.zeros((0,num_classes2),dtype='float')
    # Final pathologies measurements
    final_pred=np.zeros((0,num_classes2),dtype='float')
    # Results 
    class_results_kidney = np.zeros((0,5),dtype='float')
    class_results_lesions = np.zeros((0,9),dtype='float')
    class_results_final = np.zeros((0,num_classes2+1),dtype='float')
    class_results_true = np.zeros((0,num_classes2+1),dtype='float')

    ious_kidney= np.zeros((0,1),dtype='float')
    # Alpha parameter
    alpha=np.zeros((0,num_classes2),dtype='float')
    batch_counter = 0
    print('Evaluating...')
    with torch.no_grad():
        # Iterate over data.
        for inputs, targets, paths,img_orig in dataloader:
            batch_counter = batch_counter + 1
            inputs = list(image.to(device) for image in inputs)
            labels = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Let's create the ground-truth boxes and labels for kidneys
            if (np.array(labels[0]['boxes1'].cpu()).shape[0]==0):
                gt_boxes1=np.zeros((0,4),dtype='float32')
                gt_masks1=np.zeros((0,inputs[0].size(1),inputs[0].size(2)))
            else:
                gt_boxes1=np.array(labels[0]['boxes1'].detach().cpu())
                gt_labels1=np.array(labels[0]['labels1'].detach().cpu())
                # Relevant objects per class
                for j in range(1,len(class_names1)):
                    rel1[j-1]=rel1[j-1]+np.sum(gt_labels1==j)
                gt_masks1=np.array(labels[0]['masks1'].detach().cpu())

            # Let's create the ground-truth boxes and labels for local lesions
            if (np.array(labels[0]['boxes2'].cpu()).shape[0]==0):
                gt_boxes2=np.zeros((0,4),dtype='float32')
                gt_labels2=np.zeros((0,),dtype=int)
            else:
                gt_boxes2=np.array(labels[0]['boxes2'].detach().cpu())
                gt_labels2=np.array(labels[0]['labels2'].detach().cpu())
                if (GLOBAL_PAT):
                    # Relevant objects per class
                    for j in range(1,len(class_names2)-2):
                        rel2[j-1]=rel2[j-1]+np.sum(gt_labels2==j)
                else:
                    # Relevant objects per class
                    for j in range(1,len(class_names2)):
                        rel2[j-1]=rel2[j-1]+np.sum(gt_labels2==j)
            
            # Prediction: returns boxes, scores (objectness), labels (class) and anchors
            pred = model(inputs)
            pred['detections1'][0]['scores']=pred['detections1'][0]['scores'].detach().cpu()
            pred['detections1'][0]['boxes']=pred['detections1'][0]['boxes'].detach().cpu()
            pred['detections1'][0]['labels']=pred['detections1'][0]['labels'].detach().cpu()
            pred['detections2'][0]['scores']=pred['detections2'][0]['scores'].detach().cpu()
            pred['detections2'][0]['boxes']=pred['detections2'][0]['boxes'].detach().cpu()
            pred['detections2'][0]['labels']=pred['detections2'][0]['labels'].detach().cpu()
            pred['global_logits'][0]=pred['global_logits'][0].detach().cpu()
            pred['final_logits'][0]=pred['final_logits'][0].detach().cpu()
            pred['local_logits'][0]=pred['local_logits'][0].detach().cpu()

            # GLOBAL: LESIONS AND GLOBAL PATHOLOGIES
            global_targets = torch.zeros((1,7),dtype=int)
            if (not labels[0]['labels2'].numel()==0):
                for i in range(labels[0]['labels2'].numel()):
                    global_targets[0,labels[0]['labels2'][i]]=1
                    
            global_pat=labels[0]['global_pat'].detach().cpu().numpy()
            if (global_pat[0]==1):
                global_targets[0,-2]=1
            if (global_pat[1]==1):
                global_targets[0,-1]=1
            if (torch.sum(global_targets)==0):
                global_targets[0,0]=1
            global_true=np.concatenate((global_true,global_targets),axis=0)
            global_pred=np.concatenate((global_pred,pred['global_logits'][0].cpu().numpy()[np.newaxis,:]),axis=0)
            # Final classification
            final_pred=np.concatenate((final_pred,pred['final_logits'][0].cpu().numpy()[np.newaxis,:]),axis=0)
            local_pred=np.concatenate((local_pred,pred['local_logits'][0].cpu().numpy()[np.newaxis,:]),axis=0)
            # alpha=np.concatenate((alpha,pred['alpha'][0].detach().cpu()[np.newaxis,:]),axis=0)
            # beta=np.concatenate((beta,pred['beta'][0].detach().cpu()[np.newaxis,:]),axis=0)
            
            # KIDNEYS
            # We use only those with scores>th_score and convert them to numpy arrays: kidneys
            if (len(pred['detections1'][0]['scores'].numpy())==0):
                pred_boxes1=np.zeros((0,4),dtype='float32')
                pred_labels1=np.zeros((0,),dtype=int)
                pred_masks1=np.zeros((0,inputs[0].size(1),inputs[0].size(2)),dtype='bool')
            else:
                pred_score1 = list(pred['detections1'][0]['scores'].numpy())
                if (pred_score1[0]>th_score1):
                    pred_t = [pred_score1.index(x) for x in pred_score1 if x>th_score1][-1]
                    pred_class1 = [class_names1[i] for i in list(pred['detections1'][0]['labels'].numpy())]
                    pred_labels1 = pred['detections1'][0]['labels'].numpy()
                    pred_masks1 = pred['detections1'][0]['masks'].cpu().numpy()
                    pred_boxes1 = [[i[0], i[1], i[2], i[3]] for i in list(pred['detections1'][0]['boxes'].numpy())]
                    pred_boxes1 = np.array(pred_boxes1[:pred_t+1])
                    pred_class1 = pred_class1[:pred_t+1]
                    pred_labels1 = pred_labels1[:pred_t+1]
                    pred_masks1 = pred_masks1[:pred_t+1]>th_mask
                    pred_scores1 = pred_score1[:pred_t+1]
                else:
                    pred_boxes1=np.zeros((0,4),dtype='float32')
                    pred_labels1=np.zeros((0,),dtype=int)
                    pred_masks1=np.zeros((0,inputs[0].size(1),inputs[0].size(2)),dtype='bool')

            # Retrieved objects: kidneys
            ret1=ret1+len(pred_labels1)
            
            # Detection statistics: we compute the intersection over union between the ground-truth objects
            # and the retrieved ones, if it exceed th_iou, the detection is considered as a good one. Kidneys
            for j in range(gt_boxes1.shape[0]):
                ious=np.zeros((pred_boxes1.shape[0],),dtype='float')
                for k in range(pred_boxes1.shape[0]):
                    ious[k]=bb_intersection_over_union(gt_boxes1[j,:],pred_boxes1[k,:])
                if (len(ious)>0):
                    iou_max=np.max(ious)
                    pos=np.argmax(ious)
                    if (iou_max>th_iou1):
                        ret_rel1[gt_labels1[j]-1]+=1
                        y_true1=np.concatenate((y_true1,gt_labels1[j][np.newaxis]),axis=0)
                        y_pred1=np.concatenate((y_pred1,pred_labels1[pos][np.newaxis]),axis=0)
                        
            # LOCAL LESIONS
            # We use only those with scores>th_score and convert them to numpy arrays: local lesions
            if (len(pred['detections2'][0]['scores'].numpy())==0):
                pred_boxes2=np.zeros((0,4),dtype='float32')
                pred_labels2=np.zeros((0,),dtype=int)
            else:
                pred_score2 = list(pred['detections2'][0]['scores'].numpy())
                if (pred_score2[0]>th_score2):
                    pred_t = [pred_score2.index(x) for x in pred_score2 if x>th_score2][-1]
                    pred_class2 = [class_names2[i] for i in list(pred['detections2'][0]['labels'].numpy())]
                    pred_labels2 = pred['detections2'][0]['labels'].numpy()
                    pred_boxes2 = [[i[0], i[1], i[2], i[3]] for i in list(pred['detections2'][0]['boxes'].numpy())]
                    pred_boxes2 = np.array(pred_boxes2[:pred_t+1])
                    pred_class2 = pred_class2[:pred_t+1]
                    pred_labels2 = pred_labels2[:pred_t+1]
                    pred_scores2 = pred_score2[:pred_t+1]
                else:
                    pred_boxes2=np.zeros((0,4),dtype='float32')
                    pred_labels2=np.zeros((0,),dtype=int)
            # Retrieved objects: kidneys
            ret2=ret2+len(pred_labels2)
            
            # Detection statistics: we compute the intersection over union between the ground-truth objects
            # and the retrieved ones, if it exceed th_iou, the detection is considered as a good one. Kidneys
            for j in range(gt_boxes2.shape[0]):
                ious=np.zeros((pred_boxes2.shape[0],),dtype='float')
                for k in range(pred_boxes2.shape[0]):
                    ious[k]=bb_intersection_over_union(gt_boxes2[j,:],pred_boxes2[k,:])
                if (len(ious)>0):
                    iou_max=np.max(ious)
                    pos=np.argmax(ious)
                    if (iou_max>th_iou2):
                        ret_rel2[gt_labels2[j]-1]+=1
                        y_true2=np.concatenate((y_true2,gt_labels2[j][np.newaxis]),axis=0)
                        y_pred2=np.concatenate((y_pred2,pred_labels2[pos][np.newaxis]),axis=0)
            
            # Save kidneys
            if (train_val_test=='train'):
                aux=paths[0].replace('images','results_train_kidney'+str(fold)).split(os.path.sep)
                aux2=paths[0].replace('images','segms_train_kidney'+str(fold)).split(os.path.sep)
            elif (train_val_test=='val'):
                aux=paths[0].replace('images','results_val_kidney'+str(fold)).split(os.path.sep)
                aux2=paths[0].replace('images','segms_val_kidney'+str(fold)).split(os.path.sep)    
            elif (train_val_test=='test'):
                aux=paths[0].replace('images','results_test_kidney'+str(fold)).split(os.path.sep)
                aux2=paths[0].replace('images','segms_test_kidney'+str(fold)).split(os.path.sep) 
            folder_path=result_dir
            folder_segm=result_dir
            for i in range(1,len(aux)-1):
                folder_path=folder_path+'/'+aux[i]
                folder_segm=folder_segm+'/'+aux2[i]
            if not os.path.exists(folder_path):
                os.makedirs(folder_path) 
            # if not os.path.exists(folder_segm):
            #     os.makedirs(folder_segm)
            jc_index_array=np.zeros((pred_masks1.shape[0],1),dtype='float')
            for j in range(pred_masks1.shape[0]):
                pred_mask_curr=np.squeeze(pred_masks1[j,:,:])
                aux_res=np.zeros((1,5),dtype='float')
                jc_index_array[j]=np.sum(np.logical_and(gt_masks1,pred_mask_curr))/(np.sum(gt_masks1)+np.sum(pred_mask_curr)-np.sum(np.logical_and(gt_masks1,pred_mask_curr)))
                aux_res[0,0]=batch_counter-1
                aux_res[0,1]=pred_scores1[j]
                aux_res[0,2]=gt_labels1[0]
                aux_res[0,3]=pred_labels1[j]
                aux_res[0,4]=jc_index_array[j]
                class_results_kidney=np.concatenate((class_results_kidney,aux_res),axis=0)
                pred_path=folder_path+'/'+aux[-1][:-4]+'_'+str(j)+'.png'
                cv2.imwrite(pred_path,255*pred_mask_curr.astype(np.uint8))
                
            # IoUs kidney
            if (pred_labels1.size>0):
                ious_kidney = np.concatenate((ious_kidney,jc_index_array[0,:][None,:]),axis=0)


            # Save lesions
            if (train_val_test=='train'):
                aux=paths[0].replace('images','results_train_lesions'+str(fold)).split(os.path.sep)
            elif (train_val_test=='val'):
                aux=paths[0].replace('images','results_val_lesions'+str(fold)).split(os.path.sep)
            elif (train_val_test=='test'):
                aux=paths[0].replace('images','results_test_lesions'+str(fold)).split(os.path.sep)
            folder_path=result_dir
            folder_segm=result_dir
            for i in range(1,len(aux)-1):
                folder_path=folder_path+'/'+aux[i]
            if not os.path.exists(folder_path):
                os.makedirs(folder_path) 
            detected=np.zeros((gt_labels2.shape[0],),dtype='bool')
            for j in range(pred_labels2.shape[0]):
                pred_mask_curr=np.squeeze(pred_boxes2[j,:])
                if (gt_labels2.shape[0]>0):
                    ious=np.zeros((gt_labels2.shape[0],),dtype='float')
                    for k in range(gt_labels2.shape[0]):
                        gt_mask_curr=gt_boxes2[k,:]
                        ious[k]=bb_intersection_over_union(gt_mask_curr, pred_mask_curr)
                    iou_max=np.max(ious)
                    pos=np.argmax(ious)
                    if (iou_max>0):
                        label_lesion=gt_labels2[pos]
                        gt_mask_curr=gt_boxes2[pos,:]
                        detected[pos]=True
                    else:
                        gt_mask_curr=np.zeros((0,4),dtype='bool')
                        label_lesion=0
                else:
                    gt_mask_curr=np.zeros((0,4),dtype='bool')
                    label_lesion=0
                aux_res=np.zeros((1,class_results_lesions.shape[1]),dtype='float')
                if (gt_mask_curr.shape[0]>0):
                    jc_index=bb_intersection_over_union(gt_mask_curr, pred_mask_curr)
                else:
                    jc_index=0
                aux_res[0,0]=batch_counter-1
                aux_res[0,1]=pred_scores2[j]
                aux_res[0,2]=label_lesion
                aux_res[0,3]=pred_labels2[j]
                aux_res[0,4]=jc_index
                aux_res[0,5:]=pred_mask_curr[np.newaxis,:]
                class_results_lesions=np.concatenate((class_results_lesions,aux_res),axis=0)
                pred_path=folder_path+'/'+aux[-1][:-4]+'_'+str(j)+'.png'
                pred_mask=np.zeros((inputs[0].size(1),inputs[0].size(2)),dtype='float')
                pred_mask[int(round(pred_mask_curr[1])):int(round(pred_mask_curr[1]+pred_mask_curr[3])),int(round(pred_mask_curr[0])):int(round(pred_mask_curr[0]+pred_mask_curr[2]))]=1
                cv2.imwrite(pred_path,255*pred_mask.astype(np.uint8)) 
            for j in range(detected.shape[0]):
                if (not detected[j]):
                    aux_res=np.zeros((1,class_results_lesions.shape[1]),dtype='float')
                    aux_res[0,0]=batch_counter-1
                    aux_res[0,1]=0
                    aux_res[0,2]=gt_labels2[j]
                    aux_res[0,3]=0
                    aux_res[0,4:]=0
                    class_results_lesions=np.concatenate((class_results_lesions,aux_res),axis=0)
            # Save global
            aux_res=np.zeros((1,num_classes2+1),dtype='float')
            aux_res[0,0]=batch_counter-1
            aux_res[0,1:]=pred['final_logits'][0].cpu().numpy()[np.newaxis,:]
            class_results_final=np.concatenate((class_results_final,aux_res),axis=0)
            aux_res[0,1:]=global_true[-1,:]
            class_results_true=np.concatenate((class_results_true,aux_res),axis=0)
            alpha=np.concatenate((alpha,pred['alpha'][0].detach().cpu()[np.newaxis,:]),axis=0)
            torch.cuda.empty_cache()

    # KIDNEYS
    # Detection statistics: precision and recall per class
    if (ret1>0):
        precision_RPN1=np.sum(ret_rel1)/ret1
    else:
        precision_RPN1=0
    recall_RPN1=np.zeros((rel1.size,),dtype='float32')
    for j in range(rel1.size):
        if (rel1[j]>0):
            recall_RPN1[j]=ret_rel1[j]/rel1[j]
        else:
            recall_RPN1[j]=0
    # F1 score(weighted mean of precision and recall)
    if (np.mean(recall_RPN1)==0 or precision_RPN1==0):
        f1_score_RPN1=0    
    else:
        f1_score_RPN1=2*np.mean(recall_RPN1)*precision_RPN1/(np.mean(recall_RPN1)+precision_RPN1)
    # Classification statistics: precision and recall
    prec_rec_marginal1=precision_recall_fscore_support(y_true1, y_pred1, average=None,labels=[f for f in range(1,len(class_names1))])#,zero_division=0
    prec_rec_global1=precision_recall_fscore_support(y_true1, y_pred1, average='macro',labels=[f for f in range(1,len(class_names1))]) #,zero_division=0             
    # Confusion matrix
    cm_global1=confusion_matrix(y_true1, y_pred1, labels=None, sample_weight=None)# ,normalize=None
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('KIDNEY DETECTION EVALUATION')
    print('Objectness-RPN. F1: {:4f}.     Precision: {:4f}. Recall: {:4f}'.format(f1_score_RPN1,precision_RPN1,np.mean(recall_RPN1)))
    if (VERBOSE):
        for i in range(1,len(class_names1)):
            print('Class: ' + class_names1[i]+'. Recall: {:1d}/{:1d}'.format(ret_rel1[i-1],rel1[i-1]))
        print('')
    if ((prec_rec_global1[0]+prec_rec_global1[1])>0):
        f1_class=2*prec_rec_global1[0]*prec_rec_global1[1]/(prec_rec_global1[0]+prec_rec_global1[1])
    else:
        f1_class=0
    print('RoI Pooling classification: F1: {:4f}.     Precision: {:4f}. Recall: {:4f}'.format(f1_class,prec_rec_global1[0],prec_rec_global1[1]))
    if (VERBOSE):
        for i in range(1,len(class_names1)):
            if ((prec_rec_marginal1[0][i-1]+prec_rec_marginal1[1][i-1])>0):
                f1_class=2*prec_rec_marginal1[0][i-1]*prec_rec_marginal1[1][i-1]/(prec_rec_marginal1[0][i-1]+prec_rec_marginal1[1][i-1])
            else:
                f1_class=0
            print('Class: ' + class_names1[i]+'. F1: {:4f}.     Precision: {:4f}. Recall: {:4f}'.format(f1_class,prec_rec_marginal1[0][i-1],prec_rec_marginal1[1][i-1]))
    print('KIDNEY SEGMENTATION EVALUATION')
    print('IoU:{:4f}'.format(np.mean(ious_kidney)))
    # LESION LOCALS
    # Detection statistics: precision and recall per class
    if (ret2>0):
        precision_RPN2=np.sum(ret_rel2)/ret2
    else:
        precision_RPN2=0
    recall_RPN2=np.zeros((rel2.size,),dtype='float32')
    for j in range(rel2.size):
        if (rel2[j]>0):
            recall_RPN2[j]=ret_rel2[j]/rel2[j]
        else:
            recall_RPN2[j]=0
    # F1 score(weighted mean of precision and recall)
    if (np.mean(recall_RPN2)==0 or precision_RPN2==0):
        f1_score_RPN2=0    
    else:
        f1_score_RPN2=2*np.mean(recall_RPN2)*precision_RPN2/(np.mean(recall_RPN2)+precision_RPN2)
    # Classification statistics: precision and recall
    prec_rec_marginal2=precision_recall_fscore_support(y_true2, y_pred2, average=None,labels=[f for f in range(1,len(class_names2))])#,zero_division=0
    prec_rec_global2=precision_recall_fscore_support(y_true2, y_pred2, average='macro',labels=[f for f in range(1,len(class_names2))])#,zero_division=0              
    # Confusion matrix
    cm_global2=confusion_matrix(y_true2, y_pred2, labels=None, sample_weight=None)#, normalize=None
    time_elapsed = time.time() - since
    print('LOCAL PATHOLOGY EVALUATION')
    print('Objectness-RPN. F1: {:4f}.     Precision: {:4f}. Recall: {:4f}'.format(f1_score_RPN2,precision_RPN2,np.mean(recall_RPN2)))
    if (VERBOSE):
        if (GLOBAL_PAT):
            for i in range(1,len(class_names2)-2):
                print('Class: ' + class_names2[i]+'. Recall: {:1d}/{:1d}'.format(ret_rel2[i-1],rel2[i-1]))
        else:
            for i in range(1,len(class_names2)):
                print('Class: ' + class_names2[i]+'. Recall: {:1d}/{:1d}'.format(ret_rel2[i-1],rel2[i-1]))
        print('')
    if ((prec_rec_global2[0]+prec_rec_global2[1])>0):
        f1_class=2*prec_rec_global2[0]*prec_rec_global2[1]/(prec_rec_global2[0]+prec_rec_global2[1])
    else:
        f1_class=0
    print('RoI Pooling classification: F1: {:4f}.     Precision: {:4f}. Recall: {:4f}'.format(f1_class,prec_rec_global2[0],prec_rec_global2[1]))
    if (VERBOSE):
        if (GLOBAL_PAT):
            for i in range(1,len(class_names2)-2):
                if ((prec_rec_marginal2[0][i-1]+prec_rec_marginal2[1][i-1])>0):
                    f1_class=2*prec_rec_marginal2[0][i-1]*prec_rec_marginal2[1][i-1]/(prec_rec_marginal2[0][i-1]+prec_rec_marginal2[1][i-1])
                else:
                    f1_class=0
                print('Class: ' + class_names2[i]+'. F1: {:4f}.     Precision: {:4f}. Recall: {:4f}'.format(f1_class,prec_rec_marginal2[0][i-1],prec_rec_marginal2[1][i-1]))
        else:    
            for i in range(1,len(class_names2)):
                if ((prec_rec_marginal2[0][i-1]+prec_rec_marginal2[1][i-1])>0):
                    f1_class=2*prec_rec_marginal2[0][i-1]*prec_rec_marginal2[1][i-1]/(prec_rec_marginal2[0][i-1]+prec_rec_marginal2[1][i-1])
                else:
                    f1_class=0
                print('Class: ' + class_names2[i]+'. F1: {:4f}.     Precision: {:4f}. Recall: {:4f}'.format(f1_class,prec_rec_marginal2[0][i-1],prec_rec_marginal2[1][i-1]))

    print('FINAL PATHOLOGY EVALUATION')
    final_auc=np.zeros((len(class_names2)),dtype='float')
    for i in range(len(class_names2)):
        if (len(np.unique(global_true[:,i]))>1):
            final_auc[i]=roc_auc_score(global_true[:,i], final_pred[:,i], average='micro')#,multi_class='ovr'
        else:
            final_auc[i]=0.0
    for i in range(len(class_names2)):
        print('Class: ' + class_names2[i]+'. AUC: {:4f}'.format(final_auc[i]))
        
    print('ALPHA')
    print('Mean alpha_')
    print(np.mean(alpha,axis=0))
    # Save results
    sio.savemat(os.path.join(result_dir,'results_kidney'+str(fold)+'.mat'),{'results_kidney':class_results_kidney})
    sio.savemat(os.path.join(result_dir,'results_lesions'+str(fold)+'.mat'),{'results_lesions':class_results_lesions})
    sio.savemat(os.path.join(result_dir,'results_final'+str(fold)+'.mat'),{'results_final':class_results_final})
    sio.savemat(os.path.join(result_dir,'results_true'+str(fold)+'.mat'),{'results_true':class_results_true})
    sio.savemat(os.path.join(result_dir,'alpha'+str(fold)+'.mat'),{'alpha':alpha})

    return (precision_RPN1,recall_RPN1,f1_score_RPN1,cm_global1,prec_rec_global1,prec_rec_marginal1,
            precision_RPN2,recall_RPN2,f1_score_RPN2,cm_global2,prec_rec_global2,prec_rec_marginal2,
            final_auc)
    
class MyGeneralizedRCNN(torch.nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn1, roi_heads1, rpn2, roi_heads2, classifier, alpha_weight, beta_weight, agg_method, transform):
        super(MyGeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn1 = rpn1
        self.rpn2 = rpn2
        self.roi_heads1 = roi_heads1
        self.roi_heads2 = roi_heads2
        self.classifier = classifier
        self.alpha = alpha_weight
        torch.nn.init.normal_(self.alpha[0].bias,mean=1.0,std=0.125)
        self.beta = beta_weight
        torch.nn.init.normal_(self.beta[0].bias,mean=1.0,std=0.125)
        self.agg_method=agg_method
        self.shown=False

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None :
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        if (targets is not None):
            targets1=[{}]
            targets1[0]["boxes"] = targets[0]["boxes1"]
            targets1[0]["labels"] = targets[0]["labels1"]
            targets1[0]["masks"] = targets[0]["masks1"]
            targets1[0]["image_id"] = targets[0]["image_id"]
            targets1[0]["area"] = targets[0]["area1"]
            targets2= [{}]
            targets2[0]["boxes"] = targets[0]["boxes2"]
            targets2[0]["labels"] = targets[0]["labels2"]
            targets2[0]["masks"] = targets[0]["masks2"]
            targets2[0]["image_id"] = targets[0]["image_id"]
            targets2[0]["area"] = targets[0]["area2"]
        else:
            targets1=None
            targets2=None
        images1, targets1 = self.transform(images, targets1)
        _, targets2 = self.transform(images, targets2)
        features = self.backbone.body(images1.tensors)
        features_fpn = self.backbone.fpn(features)
        if isinstance(features_fpn, torch.Tensor):
            features_fpn = OrderedDict([(0, features_fpn)])
        proposals1, proposal_losses1 = self.rpn1(images1, features_fpn, targets1)
        detections1, detector_losses1 = self.roi_heads1(features_fpn, proposals1, images1.image_sizes, targets1)
        detections1 = self.transform.postprocess(detections1, images1.image_sizes, original_image_sizes)
        features_fpn2 = self.backbone.fpn2(features)
        if isinstance(features_fpn2, torch.Tensor):
            features_fpn2 = OrderedDict([(0, features_fpn2)])
        proposals2, proposal_losses2 = self.rpn2(images1, features_fpn2, targets2)
        detections2, detector_losses2 = self.roi_heads2(features_fpn2, proposals2, images1.image_sizes, targets2)
        detections2 = self.transform.postprocess(detections2, images1.image_sizes, original_image_sizes)

        # Global classifier
        global_features = self.classifier[0](features[3])
        global_logits = self.classifier[1](global_features.view(-1,  self.classifier[1].in_features))
        global_logits = torch.sigmoid(global_logits)
        
        # Aggregate local detections 
        local_logits=torch.zeros_like(global_logits)
        if (not detections2):
            local_labels=torch.Tensor(0)
            local_scores=torch.Tensor(0)
            area_kidney=original_image_sizes[0][0]*original_image_sizes[0][1]
        else:
            local_labels=detections2[0]['labels']
            local_scores=detections2[0]['scores']
            local_areas=(detections2[0]['boxes'][:,2]-detections2[0]['boxes'][:,0])*(detections2[0]['boxes'][:,3]-detections2[0]['boxes'][:,1])
            if (not detections1 or detections1[0]['masks'].shape[0]==0):
                area_kidney=original_image_sizes[0][0]*original_image_sizes[0][1]
            else:
                area_kidney=torch.min(torch.max(torch.sum(detections1[0]['masks'][0][0]>0).float(),torch.sum(local_areas[local_scores>0.5])),torch.tensor([original_image_sizes[0][0]*original_image_sizes[0][1]]).to(detections1[0]['masks'].device).float())
       
        if (self.agg_method=='max'):
            if (local_labels.size()>torch.Tensor(0).size()):
                for i in range(1,len(cfg.class_names2)-2):
                    if ((local_scores[local_labels==i]).size()>torch.Tensor(0).size()):
                        local_logits[0,i]=torch.max(local_scores[local_labels==i])
        elif (self.agg_method=='avg'):
            if (local_labels.size()>torch.Tensor(0).size()):
                for i in range(1,len(cfg.class_names2)-2):
                    if ((local_scores[local_labels==i]).size()>torch.Tensor(0).size()):
                        local_logits[0,i]=torch.mean(local_scores[local_labels==i])
        elif (self.agg_method=='lse'):
            if (local_labels.size()>torch.Tensor(0).size()):
                for i in range(1,len(cfg.class_names2)-2):
                    if ((local_scores[local_labels==i]).size()>torch.Tensor(0).size()):
                        local_logits[0,i]=torch.log(torch.sum(torch.exp(local_scores[local_labels==i])))
        elif (self.agg_method=='lme'):
            if (local_labels.size()>torch.Tensor(0).size()):
                for i in range(1,len(cfg.class_names2)-2):
                    if ((local_scores[local_labels==i]).size()>torch.Tensor(0).size()):
                        local_logits[0,i]=torch.log(torch.mean(torch.exp(local_scores[local_labels==i])))
        elif (self.agg_method=='areaK'):
            if (local_labels.size()>torch.Tensor(0).size()):
                for i in range(1,len(cfg.class_names2)-2):
                    if ((local_scores[local_labels==i]).size()>torch.Tensor(0).size()):
                        local_logits[0,i]=(1/area_kidney)*torch.sum(local_scores[local_labels==i]*local_areas[local_labels==i])
        elif (self.agg_method=='area'):
            if (local_labels.size()>torch.Tensor(0).size()):
                for i in range(1,len(cfg.class_names2)-2):
                    if ((local_scores[local_labels==i]).size()>torch.Tensor(0).size()):
                        local_logits[0,i]=1/(original_image_sizes[0][0]*original_image_sizes[0][1])*torch.sum(local_scores[local_labels==i]*local_areas[local_labels==i])
        local_logits[0,0]=1-torch.mean(local_logits[0,1:len(cfg.class_names2)-2])
        
        # Linear combination
        alpha=self.alpha[1](self.alpha[0](torch.cat((global_logits,local_logits[:,:len(cfg.class_names2)-2]),dim=1)))
        beta=self.beta[1](self.beta[0](torch.cat((global_logits,local_logits[:,:len(cfg.class_names2)-2]),dim=1)))
        final_logits=(alpha+torch.finfo().eps)/(alpha+beta+2*torch.finfo().eps)*global_logits+(beta+torch.finfo().eps)/(alpha+beta+2*torch.finfo().eps)*local_logits

        # BCE Loss
        if (targets is not None):
            global_targets = torch.zeros_like(global_logits)
            if (not targets2[0]['labels'].numel()==0):
                for i in range(targets2[0]['labels'].numel()):
                    global_targets[0,targets2[0]['labels'][i]]=1
            # Global pathologies
            global_pat=targets[0]["global_pat"]
            if (global_pat[0]==1):
                global_targets[0,-2]=1
            if (global_pat[1]==1):
                global_targets[0,-1]=1
            if (torch.sum(global_targets)==0):
                global_targets[0,0]=1
            loss_global =  torch.nn.functional.binary_cross_entropy(global_logits, global_targets, reduction='mean')

            loss_final=torch.nn.functional.binary_cross_entropy(final_logits, global_targets, reduction='mean')

        # Organize losses
        losses = {}

        if self.training:
            losses.update(detector_losses1)
            losses['loss_classifier_kidney'] = losses.pop('loss_classifier')
            losses['loss_box_reg_kidney'] = losses.pop('loss_box_reg')
            losses['loss_mask_kidney'] = losses.pop('loss_mask')
            losses.update(detector_losses2)
            losses['loss_classifier_local'] = losses.pop('loss_classifier')
            losses['loss_box_reg_local'] = losses.pop('loss_box_reg')
            losses.update(proposal_losses1)
            losses['loss_objectness_kidney'] = losses.pop('loss_objectness')
            losses['loss_rpn_box_reg_kidney'] = losses.pop('loss_rpn_box_reg')
            losses.update(proposal_losses2)
            losses['loss_objectness_local'] = losses.pop('loss_objectness')
            losses['loss_rpn_box_reg_local'] = losses.pop('loss_rpn_box_reg')
            loss_global={'loss_global_classifier':loss_global}
            losses.update(loss_global)
            loss_final={'loss_final_classifier':loss_final}
            losses.update(loss_final)
            return losses

        return {'detections1': detections1, 'detections2': detections2, 'global_logits':global_logits, 'final_logits':final_logits, 'local_logits': local_logits, 'alpha':(alpha+torch.finfo().eps)/(alpha+beta+2*torch.finfo().eps),'beta':(beta+torch.finfo().eps)/(alpha+beta+2*torch.finfo().eps)}
    
class MyMaskRCNN(MyGeneralizedRCNN):

    def __init__(self, num_classes1, num_classes2, agg_method, **kwargs):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision_05.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes1)
    
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes1)
        
        in_channels_stage2 = 256
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = 256
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        fpn.load_state_dict(model.backbone.fpn.state_dict())
        model.backbone.fpn2=fpn
        
        # Global classifier (a nivel de imagen)
        modules=[]
        modules.append(torch.nn.AdaptiveAvgPool2d(output_size=(1,1)))
        modules.append(torch.nn.Linear(in_features=2048, out_features=num_classes2, bias=True))
        global_classifier=torch.nn.ModuleList(modules)
        
        # Local Faster RCNN
        out_channels = model.backbone.out_channels
        
        # RPN Local
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n_train=2000
        rpn_pre_nms_top_n_test=1000
        rpn_post_nms_top_n_train=2000
        rpn_post_nms_top_n_test=1000
        rpn_nms_thresh=0.7
        rpn_fg_iou_thresh=0.7
        rpn_bg_iou_thresh=0.3
        rpn_batch_size_per_image=256
        rpn_positive_fraction=0.5

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn_local = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        # RoI Heads local
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=7,
            sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes2)
        
        box_score_thresh=0.05
        box_nms_thresh=0.5
        box_detections_per_img=100
        box_fg_iou_thresh=0.5
        box_bg_iou_thresh=0.5
        box_batch_size_per_image=512
        box_positive_fraction=0.25
        bbox_reg_weights=None
        
        roi_heads_local=RoIHeads(box_roi_pool,box_head, box_predictor,
                                      box_fg_iou_thresh, box_bg_iou_thresh, box_batch_size_per_image,
                                      box_positive_fraction, bbox_reg_weights,
                                      box_score_thresh, box_nms_thresh, box_detections_per_img)
    
        rpn_local.load_state_dict(model.rpn.state_dict())
        model2 = torchvision_05.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        roi_heads_local.box_head.load_state_dict(model2.roi_heads.box_head.state_dict())
        min_size=800
        max_size=1333
        transform = GeneralizedRCNNTransform(min_size, max_size, None, None)# Image mean and std not used
        # alpha-weight
        modules=[]
        modules.append(torch.nn.Linear(in_features=num_classes2*2-2, out_features=num_classes2, bias=True))
        modules.append(torch.nn.ReLU())
        alpha_weight=torch.nn.ModuleList(modules)
        # beta-weight
        modules=[]
        modules.append(torch.nn.Linear(in_features=num_classes2*2-2, out_features=num_classes2, bias=True))
        modules.append(torch.nn.ReLU())
        beta_weight=torch.nn.ModuleList(modules)
        super(MyMaskRCNN, self).__init__(model.backbone, model.rpn, model.roi_heads, rpn_local, roi_heads_local, global_classifier, alpha_weight, beta_weight, agg_method, transform)

    
class KidneyDataset(object):
    def __init__(self, root, mean, std, valids, train_or_not):#, imdb_mean
        self.root = root
        self.mean = mean.ravel()
        self.std = std.ravel()
        self.train_or_not = train_or_not
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = natsort.natsorted([os.path.join(self.root,'images',f) for f in sorted(os.listdir(os.path.join(self.root,'images'))) if (os.path.isfile(os.path.join(self.root,'images',f)) and f.endswith('.jpg'))])
        self.imgs = [self.imgs[f] for f in (valids-1)]
        self.masks = natsort.natsorted([os.path.join(self.root,'masks',f) for f in sorted(os.listdir(os.path.join(self.root,'masks'))) if (os.path.isfile(os.path.join(self.root,'masks',f)) and f.endswith('.mat'))])
        self.masks = [self.masks[f] for f in (valids-1)]
        self.labels = natsort.natsorted([os.path.join(self.root,'labels',f) for f in sorted(os.listdir(os.path.join(self.root,'labels'))) if (os.path.isfile(os.path.join(self.root,'labels',f)) and f.endswith('.txt'))])
        self.labels = [self.labels[f] for f in (valids-1)]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(img_path).convert("RGB")
        mask_path = self.masks[idx]
        mask = sio.loadmat(mask_path)['mask']
        mask = Image.fromarray(mask, 'L')
        # Check if the kidney mask is valid and generate the boxes for Faster R-CNN and Mask R-CNN
        boxes1=[]
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        # Get aspect ratio
        W=xmax-xmin
        H=ymax-ymin
        if (W==0 or H==0 or (H<20 and ymin==0)):
            print('error')
        else:
            boxes1.append([xmin, ymin, xmax, ymax])
        # Read label file and generate lesions masks and labels
        file=open(self.labels[idx],'r')
        aux=file.readline()
        labels=np.array([int(file.readline().split('Global diagnosis: ')[-1].split('\n')[0])])
        aux=file.readline().split('Global pathologies: ')[-1].split('\n')[0]
        global_pat=np.zeros((2,1),dtype=int)
        global_pat[0,0] = int(aux.split(' ')[0])
        global_pat[1,0] = int(aux.split(' ')[1])
        aux=file.readline()
        aux=file.readlines()
        boxes2=[]

        lesions=[]
        lesion_labels=np.zeros((len(aux),1),dtype=int) 
        for i in range(len(aux)):
            indexes=aux[i].split(': ')[-1].split(' ')
            # lesion_mask=np.zeros((img.size[1],img.size[0]),dtype=np.uint8)
            # rect=rectangle((int(indexes[1]),int(indexes[0])),extent=(int(indexes[3]),int(indexes[2])),shape=(img.size[1],img.size[0]))
            # lesion_mask[rect]=255
            boxes2.append([int(indexes[0]),int(indexes[1]), int(indexes[0])+int(indexes[2]), int(indexes[1])+int(indexes[3])])
            lesion_labels[i]=int(indexes[4])
            
        for j in range(len(lesions)):
            pos = np.where(lesions[j])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # Get aspect ratio
            W=xmax-xmin
            H=ymax-ymin
            if (W==0 or H==0):
                print('error')
            else:
                boxes2.append([xmin, ymin, xmax, ymax])
        
        # convert everything into a torch.Tensor
        boxes1 = torch.as_tensor(boxes1, dtype=torch.float32)
        boxes2 = torch.as_tensor(boxes2, dtype=torch.float32)

        # Change labels to group the underrepresented categories under the class Others

        for j in range(len(lesion_labels)):
            if (lesion_labels[j]==2 or lesion_labels[j]==3):
                lesion_labels[j]=1 # Cyst, complicated cyst
            elif(lesion_labels[j]==4):
                lesion_labels[j]=2 # Pyramids
            elif(lesion_labels[j]==7):
                lesion_labels[j]=3 # Hydronephrosis
            elif(lesion_labels[j]==1 or lesion_labels[j]==5 or lesion_labels[j]==8 or lesion_labels[j]==9 or lesion_labels[j]==6):
                lesion_labels[j]=4 # Others
    
        labels1 = torch.as_tensor(labels.ravel(), dtype=torch.int64)
        labels2 = torch.as_tensor(lesion_labels.ravel(), dtype=torch.int64)
        global_pat = torch.as_tensor(global_pat.ravel(), dtype=torch.int64)

        for j in range(len(lesions)):
            lesions[j]=np.asarray(lesions[j])
        lesions=np.asarray(lesions)
        
        masks1 = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        masks1 = masks1[None,:,:]
        if (lesions.size>0):
            masks2 = torch.as_tensor(lesions, dtype=torch.uint8)
        else:
            masks2 = torch.as_tensor(np.zeros((0,np.array(mask).shape[0],np.array(mask).shape[1]),dtype=np.uint8), dtype=torch.uint8)
            
        image_id = torch.tensor([idx])

        if (not boxes1.size()==torch.Size([0])):
            area1 = (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 2] - boxes1[:, 0])
        else:
            area1 = torch.tensor([0.])
        if (not boxes2.size()==torch.Size([0])):
            area2 = (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 2] - boxes2[:, 0])
        else:
            area2 = torch.tensor([0.])
        target = {}
        target["boxes1"] = boxes1
        target["labels1"] = labels1
        target["masks1"] = masks1
        target["image_id"] = image_id
        target["boxes2"] = boxes2
        target["labels2"] = labels2
        target["masks2"] = masks2
        target["area1"] = area1
        target["area2"] = area2
        target["global_pat"] = global_pat
        
        img=F.to_tensor(img)
        img2=img.clone()
        # Normalize image
        dtype, device = img.dtype, img.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)
        img = (img - mean[:, None, None]) / std[:, None, None]

        return img, target, img_path, img2

    def __len__(self):
        return len(self.imgs)
    
def collate_fn(batch):
    return tuple(zip(*batch))

for fold in cfg.folds:
    exper='URICADS_'+cfg.agg_method+'_fold'+str(fold)
    model_ft = MyMaskRCNN(cfg.num_classes1, cfg.num_classes2, cfg.agg_method)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Send the model to GPU if available
    model_ft = model_ft.to(device)
    
    # Load the training weights
    model_weights = torch.load(os.path.join(cfg.model_dir,exper+'-epoch95.pth'))['state_dict']
    model_ft.load_state_dict(model_weights)
    
    print("Initializing Datasets and Dataloaders...")
    
    # split the dataset in train and test set
    test=sio.loadmat(os.path.join(cfg.data_dir,'splits','idxTestM'+str(fold)+'.mat'))['idxTest'][0]
    
    mean=sio.loadmat(os.path.join(cfg.data_dir,'splits','rgbM'+str(fold)+'.mat'))['rgb_mean']
    std=sio.loadmat(os.path.join(cfg.data_dir,'splits','stdM'+str(fold)+'.mat'))['rgb_cov']
    std=np.sqrt(np.diag(std))
    
    # use our dataset and defined transformations
    dataset_test = KidneyDataset(cfg.data_dir, mean, std, test, False)
    
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    
    test_detection_model_full(model_ft, data_loader_test, cfg.num_classes1, cfg.class_names1, cfg.num_classes2, cfg.class_names2, cfg.num_global_classes, cfg.th_score1, cfg.th_score2, cfg.th_iou1, cfg.th_iou2, cfg.th_mask, cfg.result_dir, cfg.GLOBAL_PAT, cfg.VERBOSE, 'test')
    