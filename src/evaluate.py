import argparse, json
from pathlib import Path
import pandas as pd
from .utils.metrics import compute_metrics, aggregate_source_probabilities
from .utils.reproducibility import save_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('prediction_csv'); args=ap.parse_args(); p=Path(args.prediction_csv); df=pd.read_csv(p); rec=compute_metrics(df); src=aggregate_source_probabilities(df); save_json({'record_level':rec,'source_level_probability_mean':compute_metrics(src)},p.with_name('metrics_recomputed_from_predictions.json')); print(json.dumps(rec,indent=2,default=str))
if __name__=='__main__': main()
