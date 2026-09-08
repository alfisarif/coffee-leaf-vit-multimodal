from .sensor_mlp import SensorEncoder, SensorOnlyModel
from .vit_base import ViTOnlyModel
from .multimodal_fusion import MultimodalModel
from .efficientnet_b0 import EfficientNetB0Model
from .maxvit_tiny import MaxViTTinyModel

def build_model(name, num_classes=4):
    if name=='sensor_only': return SensorOnlyModel(num_classes)
    if name=='vit_only': return ViTOnlyModel(num_classes)
    if name in {'vit_sensor','multimodal','shuffled_sensor'}: return MultimodalModel(num_classes)
    if name=='efficientnet_b0': return EfficientNetB0Model(num_classes)
    if name=='maxvit_tiny': return MaxViTTinyModel(num_classes)
    raise ValueError(name)
