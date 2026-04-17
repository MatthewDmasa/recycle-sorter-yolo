from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics.engine.results import Results

from src.model_loader import load_model
from src.postprocess import extract_all_detections


def predict_image(
    image: Image.Image | str | Path,
    model_path: str | Path = "models/trained/best.pt",
    conf: float = 0.35,
    iou: float = 0.45
) -> tuple[list[Results], list[Any]]:
    """
    Run YOLO inference on a single image and return both raw and parsed detections.

    Args:
        image: PIL image or image path
        model_path: Path to trained model weights
        conf: Confidence threshold
        iou: IoU threshold for NMS

    Returns:
        tuple:
            - raw YOLO results
            - parsed detections from postprocess.py
    """
    model = load_model(model_path)

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False
    )

    detections = extract_all_detections(results)
    return results, detections


def predict_video_frame(
    frame,
    model_path: str | Path = "models/trained/best.pt",
    conf: float = 0.35,
    iou: float = 0.45,
    use_tracking: bool = False
) -> tuple[list[Results], list[Any]]:
    """
    Run YOLO inference on a video frame (numpy array / OpenCV frame).

    Args:
        frame: OpenCV image frame
        model_path: Path to trained model weights
        conf: Confidence threshold
        iou: IoU threshold
        use_tracking: Whether to use tracker mode

    Returns:
        tuple:
            - raw YOLO results
            - parsed detections
    """
    model = load_model(model_path)

    if use_tracking:
        results = model.track(
            source=frame,
            conf=conf,
            iou=iou,
            persist=True,
            verbose=False
        )
    else:
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            verbose=False
        )

    detections = extract_all_detections(results)
    return results, detections