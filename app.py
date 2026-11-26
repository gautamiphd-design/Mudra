"""
Streamlit app: upload a Bharatanatyam mudra image and see the predicted
mudra plus the probability of every class.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Works with both kinds of input images: dataset-style images with the
MediaPipe landmark dots already drawn on them (same as Mudraas/), and
plain photos of a hand with nothing drawn (falls back to live MediaPipe
hand detection automatically).
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from landmark_utils import get_landmarks_from_array
from mudra_ensemble import predict_all_proba, class_labels

st.set_page_config(page_title="Bharatanatyam Mudra Classifier", layout="wide")

st.title("Bharatanatyam Mudra Classifier")
st.caption(
    "Upload a hand image to classify its mudra. Works with images that already have "
    "MediaPipe landmark dots drawn on them, or with plain hand photos (falls back to "
    "live MediaPipe hand detection automatically)."
)

with st.sidebar:
    st.header("Options")
    topk = st.slider("Show top K classes", min_value=3, max_value=len(class_labels()),
                      value=10, step=1)
    show_all_28 = st.checkbox("Show all 28 classes in the table", value=False)

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload an image to get started.")
    st.stop()


def annotate(img_bgr, pts, label, conf, source):
    vis = img_bgr.copy()
    for (x, y) in pts:
        cv2.circle(vis, (int(x), int(y)), max(2, vis.shape[1] // 150), (0, 255, 0), -1)
    return vis


for uploaded in uploaded_files:
    st.markdown("---")
    st.subheader(uploaded.name)

    file_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not read this file as an image.")
        continue

    pts, source = get_landmarks_from_array(img_bgr)

    col1, col2 = st.columns([1, 1.4])

    if pts is None:
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.error(
                "Could not find a hand / usable landmark pattern in this image. "
                "Try an image with clearly visible fingers, or a dataset-style "
                "image with the landmark dots drawn on it."
            )
        continue

    ranked = predict_all_proba(pts)
    if ranked is None:
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.warning(f"Found {len(pts)} landmark points (via {source}) but not enough to classify.")
        continue

    top_label, top_conf = ranked[0]
    vis = annotate(img_bgr, pts, top_label, top_conf, source)

    with col1:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True,
                  caption=f"Detected {len(pts)} landmark points ({source})")

    with col2:
        st.metric("Predicted mudra", top_label, f"{top_conf*100:.1f}% confidence")

        n_show = len(ranked) if show_all_28 else min(topk, len(ranked))
        df = pd.DataFrame(ranked[:n_show], columns=["Mudra", "Probability"])
        df["Probability"] = (df["Probability"] * 100).round(2)

        st.bar_chart(df.set_index("Mudra")["Probability"], horizontal=True)
        st.dataframe(
            df.rename(columns={"Probability": "Probability (%)"}),
            use_container_width=True,
            hide_index=True,
        )
