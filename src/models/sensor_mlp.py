import torch.nn as nn
from ..config import SENSOR_HIDDEN, MLP_DROPOUT, CLASSIFIER_HIDDEN, CLASSIFIER_DROPOUT

class SensorEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=SENSOR_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Dropout(MLP_DROPOUT), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class SensorOnlyModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__(); self.sensor_encoder=SensorEncoder(); self.classifier=nn.Sequential(
            nn.Linear(SENSOR_HIDDEN, CLASSIFIER_HIDDEN), nn.ReLU(), nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, num_classes))
    def forward(self, sensor=None, **kwargs): return self.classifier(self.sensor_encoder(sensor))
