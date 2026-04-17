from pathlib import Path
import tempfile

import cv2

from src.model_loader import load_model
from src.postprocess import extract_all_detections


def process_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    model_path: str | Path = "models/trained/best.pt",
    conf: float = 0.35,
    iou: float = 0.45
) -> dict:
    """
    Run YOLO inference on a video file and save an annotated output video.
    """
    model = load_model(model_path)

    input_video_path = str(input_video_path)
    output_video_path = str(output_video_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    # Safer codec for local Windows/OpenCV use
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create output video writer with mp4v codec.")

    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            verbose=False
        )

        annotated_frame = results[0].plot()
        detections = extract_all_detections(results)

        total_detections += len(detections)
        frame_count += 1

        writer.write(annotated_frame)

    cap.release()
    writer.release()

    avg_detections = total_detections / frame_count if frame_count > 0 else 0

    return {
        "frame_count": frame_count,
        "total_detections": total_detections,
        "avg_detections_per_frame": avg_detections,
        "output_video_path": output_video_path
    }


def save_uploaded_video_to_temp(uploaded_file) -> str:
    """
    Save a Streamlit uploaded video file to a temporary path.
    """
    suffix = Path(uploaded_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name