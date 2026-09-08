import torch
import torch.nn as nn
from transformers import ViTModel
from ..config import PRETRAINED_VIT, SENSOR_HIDDEN, CLASSIFIER_HIDDEN, CLASSIFIER_DROPOUT
from .sensor_mlp import SensorEncoder

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__(); self.vit=ViTModel.from_pretrained(PRETRAINED_VIT)
        for p in self.vit.parameters(): p.requires_grad=True
        self.sensor_encoder=SensorEncoder()
        self.classifier=nn.Sequential(nn.Linear(self.vit.config.hidden_size+SENSOR_HIDDEN, CLASSIFIER_HIDDEN), nn.ReLU(), nn.Dropout(CLASSIFIER_DROPOUT), nn.Linear(CLASSIFIER_HIDDEN,num_classes))
    def forward(self, pixel_values=None, sensor=None, **kwargs):
        v=self.vit(pixel_values=pixel_values).last_hidden_state[:,0,:]
        s=self.sensor_encoder(sensor); return self.classifier(torch.cat([v,s],dim=1))
