import torch

def run_epoch(model, loader, optimizer, criterion, device, inputs_fn, train=True):
    model.train(train); total_loss=0.; correct=0; total=0
    for batch in loader:
        kwargs, labels=inputs_fn(batch,device)
        if train: optimizer.zero_grad(set_to_none=True)
        logits=model(**kwargs); loss=criterion(logits,labels)
        if train: loss.backward(); optimizer.step()
        total_loss += loss.item()*labels.size(0); correct += (logits.argmax(1)==labels).sum().item(); total += labels.size(0)
    return total_loss/total, correct/total, correct, total
