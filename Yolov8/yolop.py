import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Traffic Lane and Object Detection",
    page_icon=":camera_video:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
st.sidebar.header("Traffic Lane Detection Options")
source_type = st.sidebar.radio("Select Input Source:", ("Image", "Video"))
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
)

# --- Load YOLO Model ---
@st.cache_resource  # Cache the model for faster loading
def load_model():
    return YOLO("yolov8n.pt")  # Load a pretrained YOLOv8n model

model = load_model()

# --- Functions ---
def detect_lanes_and_objects_image(image, model, confidence_threshold):
    """Runs YOLO on a single image."""
    img_np = np.array(image)
    results = model(img_np, conf=confidence_threshold)

    for result in results:
        boxes = result.boxes
        names = result.names  # Get class names dynamically
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            if confidence > confidence_threshold:
                label = f"{names[class_id]} {confidence:.2f}"
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_np,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
    return img_np


def detect_lanes_and_objects_video(video_path, model, confidence_threshold):
    """Runs YOLO on a video file."""
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

    stframe = st.empty()  # Create an empty placeholder in Streamlit

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame, conf=confidence_threshold)

        for result in results:
            boxes = result.boxes
            names = result.names  # Get class names dynamically
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                if confidence > confidence_threshold:
                    label = f"{names[class_id]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        out.write(frame)
        stframe.image(frame, channels="BGR", use_column_width=True)  # Display the frame

    video.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_path


# --- Main Application ---
st.title("Traffic Lane and Object Detection")

if source_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Running YOLOv8..."):
                detected_image = detect_lanes_and_objects_image(
                    image, model, confidence_threshold
                )
                st.image(detected_image, caption="Detected Image", use_column_width=True)

elif source_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        tfile.close()  # Close the file before processing

        if st.button("Run Detection"):
            with st.spinner("Running YOLOv8 on video..."):
                output_video_path = detect_lanes_and_objects_video(
                    video_path, model, confidence_threshold
                )
                st.video(output_video_path)

                # Clean up temporary files after displaying
                os.unlink(video_path)
                os.unlink(output_video_path)

st.markdown("---")
st.markdown(
    """
    **Note:** This example uses YOLOv8 for object detection. Lane detection is a more complex task and requires additional image processing techniques. This is a simplified demo and may not perform well on complex or noisy video.
    """
)
