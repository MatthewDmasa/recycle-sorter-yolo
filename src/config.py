from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset configuration file
DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "warp_recycle" / "recycle.yaml"

# Base YOLO model to fine-tune
PRETRAINED_MODEL = "yolov8m.pt"

# Training settings
IMAGE_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 15

# Use 0 for first GPU, or "cpu" if no CUDA device is available
DEVICE = "cpu"

# Run naming
RUN_NAME = "yolov8m_recycle_v2"