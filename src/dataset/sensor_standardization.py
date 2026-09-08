import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..config import SENSOR_COLUMNS

def fit_training_scaler(clean_metadata):
    train_original = clean_metadata[
        (clean_metadata["split"] == "train") &
        (clean_metadata["augmentation_id"] == 0)
    ].copy()
    scaler = StandardScaler()
    scaler.fit(train_original[SENSOR_COLUMNS].astype(float).values)
    return scaler
