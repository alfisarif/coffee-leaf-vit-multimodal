"""Train one of the six final experimental configurations.
Usage: python -m src.train --experiment vit_sensor --images-dir /path/to/images
"""
import argparse
from pathlib import Path
import pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from .config import *
from .seed import set_seed
from .dataset.preprocessing import load_metadata
from .dataset.sensor_standardization import fit_training_scaler
from .dataset.loader import CoffeeDataset
from .models import build_model
from .engine.trainer import run_epoch
from .engine.evaluator import predict
from .utils.checkpoint import save_checkpoint, sha256_file
from .utils.metrics import compute_metrics, aggregate_source_probabilities
from .utils.reproducibility import save_json

def inputs_fn(experiment):
    def f(batch,device):
        kwargs={}
        if experiment in {'vit_only','vit_sensor','shuffled_sensor','efficientnet_b0','maxvit_tiny'}: kwargs['pixel_values']=batch['pixel_values'].to(device)
        if experiment in {'sensor_only','vit_sensor','shuffled_sensor'}: kwargs['sensor']=batch['sensor'].to(device)
        return kwargs,batch['label'].to(device)
    return f

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',required=True,choices=['sensor_only','vit_only','vit_sensor','shuffled_sensor','efficientnet_b0','maxvit_tiny']); ap.add_argument('--images-dir',required=True); args=ap.parse_args()
    set_seed(SEED)
    clean=load_metadata(CLEAN_METADATA); train_aug=pd.read_csv(TRAIN_AUGMENTED); val=pd.read_csv(VALIDATION)
    for df in (train_aug,val): df['label_id']=df['label_canonical'].map(LABEL_TO_ID).astype(int)
    scaler=fit_training_scaler(clean)
    use_image=args.experiment not in {'sensor_only'}; use_sensor=args.experiment in {'sensor_only','vit_sensor','shuffled_sensor'}
    train_ds=CoffeeDataset(train_aug,args.images_dir,scaler,use_image,use_sensor,True,args.experiment=='shuffled_sensor')
    val_ds=CoffeeDataset(val,args.images_dir,scaler,use_image,use_sensor,False,False)
    g=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,generator=g)
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=build_model(args.experiment).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY); criterion=nn.CrossEntropyLoss(); fn=inputs_fn(args.experiment)
    out=RESULTS_DIR/args.experiment; ckpt=CHECKPOINT_DIR/args.experiment/'best.pt'; out.mkdir(parents=True,exist_ok=True)
    best=None; history=[]
    for epoch in range(1,EPOCHS+1):
        tl,ta,tc,tn=run_epoch(model,train_loader,opt,criterion,device,fn,True); vl,va,vc,vn=run_epoch(model,val_loader,opt,criterion,device,fn,False)
        history.append({'epoch':epoch,'train_loss':tl,'train_accuracy':ta,'train_correct':tc,'train_incorrect':tn-tc,'val_loss':vl,'val_accuracy':va,'val_correct':vc,'val_incorrect':vn-vc})
        candidate=(vl,-va,epoch)
        if best is None or candidate < best['key']:
            best={'key':candidate,'epoch':epoch,'val_loss':vl,'val_accuracy':va}; save_checkpoint(ckpt,model,opt,epoch,vl,va,{'experiment':args.experiment})
        print(f'[{args.experiment}] epoch {epoch:02d}/{EPOCHS} | train loss {tl:.6f} acc {ta:.6f} ({tc}/{tn}) | val loss {vl:.6f} acc {va:.6f} ({vc}/{vn})')
    pd.DataFrame(history).to_csv(out/'epoch_history.csv',index=False)
    state=torch.load(ckpt,map_location=device); model.load_state_dict(state['model_state_dict']); pred=predict(model,val_loader,device,fn); pred.to_csv(out/'validation_predictions.csv',index=False)
    rec=compute_metrics(pred); src=aggregate_source_probabilities(pred); src.to_csv(out/'source_level_predictions.csv',index=False); srcm=compute_metrics(src)
    save_json({'experiment':args.experiment,'selected_epoch':best['epoch'],'selection_rule':CHECKPOINT_SELECTION_RULE,'selected_val_loss':best['val_loss'],'selected_val_accuracy_during_training':best['val_accuracy'],'record_level':rec,'source_level_probability_mean':srcm,'checkpoint_sha256':sha256_file(ckpt)},out/'metrics.json')
    print('Final:',rec)

if __name__=='__main__': main()
