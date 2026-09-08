import torch
import pandas as pd
from ..config import LABELS

def predict(model, loader, device, inputs_fn):
    model.eval(); rows=[]
    with torch.no_grad():
        for batch in loader:
            kwargs, labels=inputs_fn(batch,device); logits=model(**kwargs); probs=torch.softmax(logits,dim=1).cpu().numpy()
            for i in range(len(labels)):
                row={'observation_id':batch['observation_id'][i],'source_id':batch['source_id'][i],'true_label':LABELS[int(labels[i])],'pred_label':LABELS[int(probs[i].argmax())]}
                row.update({f'prob_{x.replace(" ","_")}':float(probs[i,j]) for j,x in enumerate(LABELS)})
                rows.append(row)
    return pd.DataFrame(rows)
