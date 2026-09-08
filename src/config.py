from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

CLEAN_METADATA = DATA_DIR / "metadata/dataset_metadata_reviewer_clean_v2.csv"
TRAIN_AUGMENTED = DATA_DIR / "sensors/train_sensor_augmented_reviewer_v2.csv"
VALIDATION = DATA_DIR / "sensors/validation_records_reviewer_v2.csv"
SOURCE_MANIFEST = DATA_DIR / "metadata/source_manifest_reviewer_v2.csv"
IMAGE_MANIFEST = DATA_DIR / "metadata/image_manifest_reviewer_v2.csv"

LABELS = ["Leaf spot", "Coffee rust", "Sooty mold", "Healthy"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
SENSOR_COLUMNS = ["kelembapan_udara", "kelembapan_tanah", "ph_tanah"]
PRETRAINED_VIT = "google/vit-base-patch16-224"
SEED = 42
SHUFFLE_SENSOR_SEED = 4242
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_WORKERS = 2
SENSOR_HIDDEN = 64
MLP_DROPOUT = 0.3
CLASSIFIER_HIDDEN = 256
CLASSIFIER_DROPOUT = 0.4
CHECKPOINT_SELECTION_RULE = (
    "Minimum validation cross-entropy loss; validation accuracy as a tie-breaker "
    "within 1e-12; earlier epoch if still tied."
)
