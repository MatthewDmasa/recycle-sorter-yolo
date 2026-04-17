from pathlib import Path
import sys
from collections import Counter
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from src.model_loader import load_model
from src.postprocess import extract_all_detections
from src.video_utils import process_video, save_uploaded_video_to_temp
from src.webcam_processor import WebcamProcessor, WebcamConfig


st.set_page_config(
    page_title="Recycle Sorter YOLO",
    page_icon="♻️",
    layout="wide"
)


RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)


@st.cache_resource
def get_model(model_path: str = "models/trained/best.pt"):
    return load_model(model_path)


def run_inference(image: Image.Image, conf: float, iou: float):
    model = get_model()
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False
    )
    detections = extract_all_detections(results)
    return results, detections


def detections_to_rows(detections):
    rows = []
    for det in detections:
        rows.append(
            {
                "Detected Class": det.class_name,
                "Confidence": round(det.confidence, 3),
                "Sort Category": det.sort_category,
                "Recommendation": det.sort_message
            }
        )
    return rows


def build_summary(detections):
    class_counts = Counter(det.class_name for det in detections)
    sort_counts = Counter(det.sort_category for det in detections)
    return class_counts, sort_counts


def render_image_mode(conf_threshold: float, iou_threshold: float):
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection..."):
        results, detections = run_inference(
            image=image,
            conf=conf_threshold,
            iou=iou_threshold
        )

    annotated_bgr = results[0].plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    with right_col:
        st.subheader("Detection Result")
        st.image(annotated_rgb, use_container_width=True)

    st.markdown("---")
    st.subheader("Detected Items")

    if not detections:
        st.warning("No recyclable items detected.")
        return

    total_detections = len(detections)
    avg_conf = sum(det.confidence for det in detections) / total_detections

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Total Detections", total_detections)
    metric_col2.metric("Average Confidence", f"{avg_conf:.2f}")

    st.dataframe(detections_to_rows(detections), use_container_width=True)

    st.markdown("---")
    st.subheader("Summary")

    class_counts, sort_counts = build_summary(detections)

    sum_col1, sum_col2 = st.columns(2)

    with sum_col1:
        st.write("**Detected Classes**")
        for class_name, count in class_counts.items():
            st.write(f"- {class_name}: {count}")

    with sum_col2:
        st.write("**Suggested Sort Categories**")
        for category, count in sort_counts.items():
            st.write(f"- {category}: {count}")


def render_video_mode(conf_threshold: float, iou_threshold: float):
    uploaded_video = st.file_uploader(
        "Upload a demo video",
        type=["mp4", "mov", "avi"],
        key="video_uploader"
    )

    if uploaded_video is None:
        st.info("Upload a video to begin.")
        return

    st.subheader("Original Video")
    st.video(uploaded_video)

    if st.button("Process Video"):
        with st.spinner("Processing video frame by frame..."):
            input_video_path = save_uploaded_video_to_temp(uploaded_video)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output:
                output_video_path = tmp_output.name

            summary = process_video(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                conf=conf_threshold,
                iou=iou_threshold
            )

        st.success("Video processing complete.")

        st.subheader("Annotated Video")
        st.caption(
            "If inline playback does not appear in your browser, use the download button below to view the annotated video."
        )

        with open(summary["output_video_path"], "rb") as video_file:
            video_bytes = video_file.read()

        st.video(video_bytes)

        st.download_button(
            label="Download Annotated Video",
            data=video_bytes,
            file_name="annotated_output.mp4",
            mime="video/mp4"
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Frames Processed", summary["frame_count"])
        metric_col2.metric("Total Detections", summary["total_detections"])
        metric_col3.metric(
            "Avg Detections / Frame",
            f"{summary['avg_detections_per_frame']:.2f}"
        )


def render_webcam_mode(conf_threshold: float, iou_threshold: float):
    st.subheader("Live Webcam Detection")
    st.caption(
        "Webcam mode processes live frames directly and does not save every frame, which helps reduce lag."
    )

    resize_width = st.slider(
        "Webcam Resize Width",
        min_value=480,
        max_value=1280,
        value=960,
        step=160
    )

    processor_config = WebcamConfig(
        model_path="models/trained/best.pt",
        conf=conf_threshold,
        iou=iou_threshold,
        resize_width=resize_width
    )

    webrtc_streamer(
        key="recycle-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: WebcamProcessor(processor_config),
        async_processing=True
    )


def main():
    st.title("♻️ Recyclable Material Sorting App")
    st.write(
        "Detect recyclable items from uploaded images, demo videos, or live webcam input "
        "using a custom-trained YOLO model."
    )

    st.info(
        "Best results come from clear views where full recyclable items are visible in frame. "
        "Performance may be weaker on tightly cropped objects or backgrounds very different from training data."
    )

    with st.sidebar:
        st.header("Settings")

        input_mode = st.radio(
            "Choose Input Type",
            ["Image", "Video", "Webcam"]
        )

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            step=0.05
        )

        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.45,
            step=0.05
        )

        st.markdown("---")
        st.subheader("Model")
        st.caption("Using: models/trained/best.pt")

        st.markdown("---")
        st.subheader("Supported Classes")
        st.caption("plastic, detergent, can, cardboard, glass, canister")

    if input_mode == "Image":
        render_image_mode(conf_threshold, iou_threshold)
    elif input_mode == "Video":
        render_video_mode(conf_threshold, iou_threshold)
    else:
        render_webcam_mode(conf_threshold, iou_threshold)


if __name__ == "__main__":
    main()