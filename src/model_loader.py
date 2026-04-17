from pathlib import Path
from ultralytics import YOLO


def load_model(model_path: str | Path = "models/trained/best.pt") -> YOLO:
    """
    Load a trained YOLO model from disk.

    Args:
        model_path: Path to the trained YOLO model weights (.pt)

    Returns:
        YOLO: Loaded Ultralytics YOLO model instance

    Raises:
        FileNotFoundError: If the model file does not exist
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}\n"
            "Make sure best.pt is placed inside models/trained/"
        )

    model = YOLO(str(model_path))
    return model


if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully.")
    print(f"Model path: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'Loaded YOLO model'}")