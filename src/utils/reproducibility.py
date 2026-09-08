import json, os, platform
from pathlib import Path

def save_json(obj,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def environment_summary():
    import torch, torchvision, transformers, sklearn, pandas, numpy
    return {'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'transformers':transformers.__version__,'sklearn':sklearn.__version__,'pandas':pandas.__version__,'numpy':numpy.__version__,'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda}
