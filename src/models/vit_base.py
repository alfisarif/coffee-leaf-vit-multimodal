import torch.nn as nn
from transformers import ViTModel
from ..config import PRETRAINED_VIT, CLASSIFIER_HIDDEN, CLASSIFIER_DROPOUT

class ViTOnlyModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__(); self.vit=ViTModel.from_pretrained(PRETRAINED_VIT)
        for p in self.vit.parameters(): p.requires_grad=True
        dim=self.vit.config.hidden_size
        self.classifier=nn.Sequential(nn.Linear(dim, CLASSIFIER_HIDDEN), nn.ReLU(), nn.Dropout(CLASSIFIER_DROPOUT), nn.Linear(CLASSIFIER_HIDDEN,num_classes))
    def forward(self, pixel_values=None, **kwargs):
        v=self.vit(pixel_values=pixel_values).last_hidden_state[:,0,:]
        return self.classifier(v)
