import numpy as np
from torchvision import transforms

TRAIN_IMAGE_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
])

def jointly_shuffle_sensor_vectors(df, sensor_columns, seed=4242):
    out = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(out))
    values = out[sensor_columns].astype(float).values.copy()
    out[sensor_columns] = values[perm]
    out["sensor_donor_observation_id"] = out.iloc[perm]["observation_id"].values
    return out, perm
