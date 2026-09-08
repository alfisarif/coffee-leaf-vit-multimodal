import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier[-1]=nn.Linear(self.backbone.classifier[-1].in_features,num_classes)
    def forward(self,pixel_values=None,**kwargs): return self.backbone(pixel_values)
