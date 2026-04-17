from dataclasses import dataclass
from typing import Any

import av
import cv2

from src.model_loader import load_model
from src.postprocess import extract_all_detections


@dataclass
class WebcamConfig:
    """
    Configuration for webcam inference.
    """
    model_path: str = "models/trained/best.pt"
    conf: float = 0.35
    iou: float = 0.45
    resize_width: int = 960


class WebcamProcessor:
    """
    Frame processor for live webcam inference with Streamlit WebRTC.
    """

    def __init__(self, config: WebcamConfig):
        self.config = config
        self.model = load_model(config.model_path)

    def _resize_frame(self, frame_bgr):
        """
        Resize frame to reduce inference cost while keeping aspect ratio.
        """
        h, w = frame_bgr.shape[:2]

        if w <= self.config.resize_width:
            return frame_bgr

        scale = self.config.resize_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(frame_bgr, (new_w, new_h))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process each webcam frame and return an annotated frame.
        """
        img_bgr = frame.to_ndarray(format="bgr24")
        img_bgr = self._resize_frame(img_bgr)

        results = self.model.predict(
            source=img_bgr,
            conf=self.config.conf,
            iou=self.config.iou,
            verbose=False
        )

        annotated_bgr = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")