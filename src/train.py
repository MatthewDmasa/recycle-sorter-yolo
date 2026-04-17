from ultralytics import YOLO
from src.config import (
    DATASET_YAML,
    PRETRAINED_MODEL,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    DEVICE,
    RUN_NAME
)


def main() -> None:
    """
    Train a YOLO model on the processed recyclable waste dataset.
    """
    model = YOLO(PRETRAINED_MODEL)

    model.train(
        data=str(DATASET_YAML),
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        project="runs/train",
        name=RUN_NAME,
        pretrained=True,
        exist_ok=True
    )


if __name__ == "__main__":
    main()