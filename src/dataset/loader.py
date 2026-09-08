import os
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTImageProcessor
from ..config import PRETRAINED_VIT, SENSOR_COLUMNS, SHUFFLE_SENSOR_SEED
from .augmentation import TRAIN_IMAGE_TRANSFORM, jointly_shuffle_sensor_vectors

def build_image_index(images_dir):
    images_dir = Path(images_dir)
    index = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "__MACOSX" not in str(p):
            if p.name in index:
                raise RuntimeError(f"Duplicate image filename: {p.name}")
            index[p.name] = p
    if not index:
        raise FileNotFoundError(f"No images found under {images_dir}")
    return index

class CoffeeDataset(Dataset):
    def __init__(self, df, images_dir, scaler, need_image=True, need_sensor=True,
                 is_train=False, shuffled_sensor=False, shuffle_seed=SHUFFLE_SENSOR_SEED):
        self.df = df.copy().reset_index(drop=True)
        if shuffled_sensor:
            self.df, self.sensor_permutation = jointly_shuffle_sensor_vectors(
                self.df, SENSOR_COLUMNS, shuffle_seed)
        else:
            self.sensor_permutation = None
        self.scaler = scaler
        self.need_image = need_image
        self.need_sensor = need_sensor
        self.is_train = is_train
        self.image_index = build_image_index(images_dir) if need_image else None
        self.image_processor = ViTImageProcessor.from_pretrained(PRETRAINED_VIT) if need_image else None
        self.sensor_features = None
        if need_sensor:
            raw = self.df[SENSOR_COLUMNS].astype(float).values
            self.sensor_features = scaler.transform(raw).astype("float32")
        self.labels = self.df["label_id"].astype(int).values

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "observation_id": str(row["observation_id"]),
            "source_id": str(row["source_id"]),
            "image_path": str(row["image_path"]),
        }
        if self.need_image:
            name = os.path.basename(str(row["image_path"]))
            if name not in self.image_index:
                raise FileNotFoundError(f"Image not found: {name}")
            image = Image.open(self.image_index[name]).convert("RGB")
            if self.is_train: image = TRAIN_IMAGE_TRANSFORM(image)
            item["pixel_values"] = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        if self.need_sensor:
            item["sensor"] = torch.tensor(self.sensor_features[idx], dtype=torch.float32)
            if "sensor_donor_observation_id" in self.df.columns:
                item["sensor_donor_observation_id"] = str(row["sensor_donor_observation_id"])
        return item
