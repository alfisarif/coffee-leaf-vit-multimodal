from pathlib import Path
import hashlib
import torch

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def save_checkpoint(path, model, optimizer, epoch, val_loss, val_accuracy, extra=None):
    payload={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch,'val_loss':val_loss,'val_accuracy':val_accuracy}
    if extra: payload.update(extra)
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)
