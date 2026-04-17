# Recyclable Material Sorting with YOLO and Streamlit

A modular computer vision application for detecting and sorting waste materials using a custom-trained **Ultralytics YOLO** model and a **Streamlit** frontend.

This project was built as a portfolio-ready recyclable waste sorting demo. It supports **image upload**, **demo video upload**, and **live webcam detection**, and provides a simple sorting recommendation for each detected object.

---

## Project Overview

The goal of this project is to identify recyclable waste items in images or video and group them into practical sorting categories such as:

- Plastic
- Metal
- Cardboard
- Glass

The detection model was trained on a grouped version of the **WaRP (Waste Recycling Plant)** dataset using a custom YOLO pipeline. The application was designed to be clean, modular, and realistic enough to showcase on GitHub as a student or junior developer portfolio project.

---

## Features

- Custom-trained YOLO object detection model
- Streamlit frontend for interactive testing
- Image upload mode
- Demo video upload mode
- Live webcam mode
- Bounding box visualisation with class labels and confidence scores
- Sorting recommendations based on the detected class
- Modular Python codebase for training, inference, postprocessing, and UI

---

## Tech Stack

- **Python**
- **Ultralytics YOLO**
- **Streamlit**
- **OpenCV**
- **Pillow**
- **streamlit-webrtc**

---

## Dataset

This project uses the **WaRP-D** dataset and remaps the original fine-grained class labels into 6 grouped recyclable categories:

- `plastic`
- `detergent`
- `can`
- `cardboard`
- `glass`
- `canister`

### Grouped Class Strategy

The original WaRP-D dataset contains more specific labels such as bottle variants, detergent variants, and glass colours. For this project, those labels were grouped into broader categories to make the model more practical for waste-sorting use.

### Final Training Classes

```yaml
names: [plastic, detergent, can, cardboard, glass, canister]
```

---

## Model Training

Two YOLO pipelines were tested on the grouped dataset:

### Run A
- Model: `yolov8s`
- Image size: `640`

### Run B
- Model: `yolov8m`
- Image size: `640`

### Final Selected Model

The final deployment model was:

```text
yolov8m_recycle_v2
```

It was selected because it slightly improved the overall validation metrics and improved weaker categories such as `can` and `canister`.

---

## Validation Summary

### YOLOv8s (Run A)
- Precision: `0.649`
- Recall: `0.461`
- mAP50: `0.550`
- mAP50-95: `0.418`

### YOLOv8m (Run B)
- Precision: `0.603`
- Recall: `0.528`
- mAP50: `0.554`
- mAP50-95: `0.426`

The YOLOv8m model was chosen as the final deployment model because it offered the best overall balance for the application.

---

## Application Modes

### 1. Image Upload
Upload an image and view:
- detected objects
- confidence scores
- sorting recommendations
- annotated image output

### 2. Demo Video Upload
Upload a short demo video and:
- process it frame by frame
- generate an annotated output video
- download the processed result

### 3. Webcam Detection
Use a live webcam feed to:
- detect recyclable items in real time
- show bounding boxes directly on the stream
- test the model in a portable demo setting

---

## Sorting Logic

The model predicts one of the grouped detection classes, and the app maps that class into a broader sorting recommendation.

Examples:
- `plastic` → Plastic recycling
- `detergent` → Plastic recycling
- `can` → Metal recycling
- `cardboard` → Cardboard / paper recycling
- `glass` → Glass recycling
- `canister` → Plastic / container recycling, depending on local rules

---

## Project Structure

```text
recycle-sorter-yolo/
├── app/
│   └── streamlit_app.py
├── assets/
│   ├── demo_images/
│   ├── demo_videos/
│   │   └── tester_vid.mp4
│   └── screenshots/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── train.py
│   ├── model_loader.py
│   ├── sorting_logic.py
│   ├── postprocess.py
│   ├── predict.py
│   ├── video_utils.py
│   └── webcam_processor.py
├── scripts/
│   ├── remap_classes.py
│   └── split_dataset.py
├── data/
│   └── processed/
│       └── warp_recycle/
│           ├── images/
│           ├── labels/
│           └── recycle.yaml
├── models/
│   └── trained/
│       └── best.pt
├── WaRP/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Live Demo

A hosted version of the app is available here:

[https://appapppy-fizakawds68gotgxkfjjnj.streamlit.app/]

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/MatthewDmasa/recycle-sorter-yolo
cd recycle-sorter-yolo
```

### 2. Create a virtual environment

```bash
py -m venv .venv
```

### 3. Activate the environment

```bash
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
py -m streamlit run app/streamlit_app.py
```

---

## Training Workflow

### 1. Remap classes

```bash
py scripts\remap_classes.py
```

### 2. Split train/validation data

```bash
py scripts\split_dataset.py
```

### 3. Train the model

```bash
py -m src.train
```

---

## Pretrained Model

This repository includes a pretrained YOLO model so the application can be run immediately:

```text
models/trained/best.pt
```

This was the final selected deployment model for the project and is the model used by the Streamlit app.

Training scripts are still included for reference, reproducibility, and future experimentation, but they are not required to run the application.

---

## Notes on Performance

The model performs best on images or video frames that resemble the training distribution:
- multiple recyclable items in frame
- moderate object scale
- clear visibility of full objects
- viewpoints similar to sorting or conveyor-style scenes

Performance is weaker on:
- tightly cropped objects
- unusual backgrounds
- objects shown at scales very different from training

This is expected because the model was trained on a fixed-style waste-sorting dataset rather than a fully general object detection dataset.

---

## Known Limitations

- Inline playback of processed video may depend on browser and video codec compatibility
- Tightly cropped single-object images may not perform as well as full-scene images
- Webcam performance depends on device speed and frame size
- Sorting recommendations are simplified and should not replace local municipal recycling rules

---

## Future Improvements

- Add tracked object IDs for video streams
- Add downloadable annotated image output
- Improve the results UI with cards and summaries
- Add more robust evaluation and experiment tracking
- Expand class coverage or test additional datasets
- Optimise webcam inference for lower latency

---

## Demo Assets

Add screenshots or demo GIFs here:

```text
assets/screenshots/
assets/demo_images/
```

Recommended visuals:
- annotated image example
- processed video example
- webcam demo screenshot
- model comparison chart or results screenshot

---

## Demo Video

A custom video was created to test the application's video upload mode and webcam mode using a virtual webcam workflow.

Suggested test file:

```text
assets/demo_videos/tester_vid.mp4
```

The demo video was used to:
- test the uploaded video detection workflow
- validate annotated video export
- simulate webcam input using a virtual webcam tool
- provide a portable demo asset when webcam access is unavailable

### Using the Demo Video

To test the video upload mode:
1. Launch the Streamlit app
2. Select **Video** mode
3. Upload `tester_vid.mp4`
4. Click **Process Video**
5. Download the annotated output if inline playback is unavailable

To test webcam mode with a virtual webcam:
1. Route `tester_vid.mp4` into your virtual webcam software
2. Select **Webcam** mode in the app
3. Allow camera access in the browser
4. Start the webcam stream and observe live detections

---

## Why This Project Matters

This project demonstrates:
- dataset preprocessing for object detection
- YOLO training and model comparison
- modular ML pipeline design
- real-time and offline inference workflows
- frontend integration with Streamlit
- practical thinking around model limitations and deployment trade-offs

---

## Licence

MIT
