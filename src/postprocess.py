from dataclasses import dataclass
from typing import Any

from src.sorting_logic import get_sorting_recommendation


@dataclass
class DetectionResult:
    """
    Standardized detection object used by the app.
    """
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[int]
    draw_label: str
    sort_category: str
    sort_message: str


def format_draw_label(class_name: str, confidence: float, track_id: int | None = None) -> str:
    """
    Create a short label for drawing on an image or video frame.

    Examples:
        'plastic | 0.87'
        'ID 3 | can | 0.91'
    """
    if track_id is not None:
        return f"ID {track_id} | {class_name} | {confidence:.2f}"
    return f"{class_name} | {confidence:.2f}"


def _safe_int_list(values: Any) -> list[int]:
    """
    Convert a tensor/list-like box into a list of ints.
    """
    return [int(v) for v in values]


def extract_detections(result: Any) -> list[DetectionResult]:
    """
    Convert one Ultralytics YOLO result object into a list of DetectionResult items.

    Args:
        result: One result from YOLO prediction output

    Returns:
        list[DetectionResult]: Parsed detections with sorting recommendations
    """
    detections: list[DetectionResult] = []

    if result.boxes is None:
        return detections

    names = result.names

    # Iterate through each detected box
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = names[class_id]

        # Bounding box in [x1, y1, x2, y2] format
        bbox_xyxy = _safe_int_list(box.xyxy[0].tolist())

        # Track ID is optional and only exists in tracking mode
        track_id = None
        if hasattr(box, "id") and box.id is not None:
            try:
                track_id = int(box.id[0].item())
            except Exception:
                track_id = None

        draw_label = format_draw_label(
            class_name=class_name,
            confidence=confidence,
            track_id=track_id
        )

        sorting = get_sorting_recommendation(class_name)

        detections.append(
            DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=bbox_xyxy,
                draw_label=draw_label,
                sort_category=sorting.sort_category,
                sort_message=sorting.sort_message
            )
        )

    return detections


def extract_all_detections(results: list[Any]) -> list[DetectionResult]:
    """
    Flatten detections across multiple YOLO results into a single list.

    Args:
        results: List of YOLO result objects

    Returns:
        list[DetectionResult]: All parsed detections
    """
    all_detections: list[DetectionResult] = []

    for result in results:
        all_detections.extend(extract_detections(result))

    return all_detections