import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Weapon Detection")

uploaded_img = st.file_uploader(
    "Upload Image",
    type=['jpg', 'png', 'jpeg']
)

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")

    model = YOLO("runs/detect/train/weights/best.pt")
    result = model(image)

    annotated_frame = result[0].plot()

    st.image(annotated_frame, caption="Detection Result")