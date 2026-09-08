import pandas as pd
from ..config import LABEL_TO_ID

def load_metadata(path):
    df = pd.read_csv(path)
    if "label_canonical" not in df.columns:
        raise ValueError("Expected label_canonical column.")
    unknown = set(df["label_canonical"].dropna().unique()) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"Unexpected canonical labels: {sorted(unknown)}")
    df = df.copy()
    df["label_id"] = df["label_canonical"].map(LABEL_TO_ID).astype(int)
    return df
