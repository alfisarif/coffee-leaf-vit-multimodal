import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from .config import CLEAN_METADATA, SENSOR_COLUMNS, LABEL_TO_ID, RESULTS_DIR, SEED

def main():
    clean=pd.read_csv(CLEAN_METADATA); clean['label_id']=clean.label_canonical.map(LABEL_TO_ID); tr=clean[clean.split=='train']; va=clean[clean.split=='validation']
    out=RESULTS_DIR/'sensor_diagnostics'; out.mkdir(parents=True,exist_ok=True)
    tr.groupby('label_canonical')[SENSOR_COLUMNS].agg(['count','mean','std','min','median','max']).to_csv(out/'per_class_sensor_distribution_training.csv')
    Xtr=tr[SENSOR_COLUMNS].astype(float).values; ytr=tr.label_id.values; Xv=va[SENSOR_COLUMNS].astype(float).values; yv=va.label_id.values
    mi=mutual_info_classif(Xtr,ytr,random_state=SEED); pd.DataFrame({'feature':SENSOR_COLUMNS,'mutual_information_training':mi}).to_csv(out/'sensor_mutual_information_training.csv',index=False)
    pipe=Pipeline([('scaler',StandardScaler()),('logreg',LogisticRegression(max_iter=5000,random_state=SEED))]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xv)
    metrics={'method':'LogisticRegression sensor-only diagnostic','train_original_records':len(tr),'validation_original_records':len(va),'accuracy':accuracy_score(yv,pred),'macro_f1':f1_score(yv,pred,average='macro'),'balanced_accuracy':balanced_accuracy_score(yv,pred)}
    pd.DataFrame({'observation_id':va.observation_id,'source_id':va.source_id,'true_label':va.label_canonical,'pred_label':[list(LABEL_TO_ID.keys())[i] for i in pred]}).to_csv(out/'logistic_regression_predictions.csv',index=False)
    pd.Series(metrics).to_json(out/'logistic_regression_metrics.json',indent=2); print(metrics)
if __name__=='__main__': main()
