import argparse, json
from pathlib import Path
import pandas as pd
from .config import CLEAN_METADATA, VALIDATION, TRAIN_AUGMENTED, LABELS

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--images-dir',required=False); args=ap.parse_args()
    clean=pd.read_csv(CLEAN_METADATA); train=pd.read_csv(TRAIN_AUGMENTED); val=pd.read_csv(VALIDATION)
    report={'original_observations':len(clean),'train_original':int(((clean.split=='train')&(clean.augmentation_id==0)).sum()),'validation_original':int(len(val)),'effective_train_entries':int(len(train)),'classes':sorted(clean.label_canonical.unique().tolist()),'train_unique_sources':int(clean.loc[clean.split=='train','source_id'].nunique()),'validation_unique_sources':int(val.source_id.nunique())}
    overlap=set(clean.loc[clean.split=='train','source_id']) & set(val.source_id)
    report['source_overlap']=len(overlap)
    if overlap: raise SystemExit(f'Source leakage detected: {len(overlap)} overlapping source IDs')
    if set(report['classes']) != set(LABELS): raise SystemExit(f'Unexpected labels: {report["classes"]}')
    if args.images_dir: report['images_dir']=str(Path(args.images_dir).resolve())
    out=Path(__file__).resolve().parents[1]/'results/DATASET_INTEGRITY_REPORT.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
