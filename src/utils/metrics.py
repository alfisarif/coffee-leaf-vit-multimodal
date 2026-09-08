import math
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from ..config import LABELS

def wilson_interval(correct,total,z=1.959963984540054):
    if total==0: return (float('nan'),float('nan'))
    p=correct/total; denom=1+z*z/total; center=(p+z*z/(2*total))/denom
    half=z*math.sqrt((p*(1-p)/total)+z*z/(4*total*total))/denom
    return center-half,center+half

def compute_metrics(df,true_col='true_label',pred_col='pred_label'):
    y_true=df[true_col].tolist(); y_pred=df[pred_col].tolist(); correct=int((df[true_col]==df[pred_col]).sum()); total=len(df)
    pma,rma,fma,_=precision_recall_fscore_support(y_true,y_pred,labels=LABELS,average='macro',zero_division=0)
    pwe,rwe,fwe,_=precision_recall_fscore_support(y_true,y_pred,labels=LABELS,average='weighted',zero_division=0)
    lo,hi=wilson_interval(correct,total); cm=confusion_matrix(y_true,y_pred,labels=LABELS)
    return {'n':total,'correct':correct,'incorrect':total-correct,'accuracy':accuracy_score(y_true,y_pred),'macro_precision':pma,'macro_recall':rma,'macro_f1':fma,'weighted_precision':pwe,'weighted_recall':rwe,'weighted_f1':fwe,'balanced_accuracy':balanced_accuracy_score(y_true,y_pred),'cohen_kappa':cohen_kappa_score(y_true,y_pred,labels=LABELS),'mcc':matthews_corrcoef(y_true,y_pred),'wilson95_low':lo,'wilson95_high':hi,'confusion_matrix_labels':LABELS,'confusion_matrix':cm.tolist()}

def aggregate_source_probabilities(pred_df):
    prob_cols=[f'prob_{x.replace(" ","_")}' for x in LABELS]; rows=[]
    for source_id,g in pred_df.groupby('source_id',as_index=False):
        labs=g['true_label'].unique()
        if len(labs)!=1: raise ValueError(f'Multiple true labels in {source_id}')
        probs=g[prob_cols].mean(axis=0).values; idx=int(np.argmax(probs))
        rows.append({'source_id':source_id,'true_label':labs[0],'pred_label':LABELS[idx],'confidence':float(probs[idx]),'n_records':len(g),**{c:float(v) for c,v in zip(prob_cols,probs)}})
    import pandas as pd
    return pd.DataFrame(rows)
