"""
Streamlit version of the real-time Bharatanatyam mudra classifier.

Same detection + ensemble pipeline as webcam_demo_v2.py (MediaPipe Hands ->
mudra_ensemble.predict_all_proba), just rendered as a Streamlit page instead
of an OpenCV window, so the full 28-class probability breakdown gets a
proper sortable bar chart / table instead of a hand-drawn text panel.

Run locally (needs a real webcam -- will NOT work inside a cloud/remote
sandbox):

    pip install -r requirements.txt
    streamlit run webcam_streamlit.py

How the live loop works: Streamlit normally reruns the whole script top to
bottom on every interaction, which doesn't suit a continuous video feed. So
this uses the common pattern of a single `while` loop driven by the "Start
camera" checkbox, updating a handful of st.empty() placeholders in place
each frame (video image, label, FPS, bar chart, table) rather than
re-rendering the page. Unchecking the box (or any other widget interaction)
makes Streamlit interrupt the loop and rerun the script fresh -- the
try/finally around the loop guarantees the camera gets released either way.
"""
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp

from mudra_ensemble import predict_all_proba, class_labels

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Same tuning knobs as webcam_demo_v2.py.
MODEL_COMPLEXITY = 1   # 0 = lite/faster, 1 = full/most accurate (default)
CAM_WIDTH = 640
CAM_HEIGHT = 480

st.set_page_config(page_title="Bharatanatyam Mudra Classifier - Live", layout="wide")
st.title("Bharatanatyam Mudra Classifier — Live Webcam")
st.caption(
    "MediaPipe hand landmarks -> SVM + Random Forest ensemble, same pipeline as "
    "webcam_demo_v2.py. Landmark coordinates are fed to the model directly (no "
    "cropping/re-detection needed -- the features are scale/rotation/translation "
    "invariant), so this keeps the same low-latency detection path."
)

with st.sidebar:
    st.header("Controls")
    run = st.checkbox("Start camera", value=False, key="run_camera")
    cam_index = st.number_input("Camera index", min_value=0, max_value=4, value=0, step=1)
    topk = st.slider("Classes to show in chart", min_value=3, max_value=len(class_labels()),
                      value=10, step=1)
    show_table = st.checkbox("Show full 28-class probability table", value=True)

col_video, col_probs = st.columns([1.3, 1])
with col_video:
    frame_slot = st.empty()
    label_slot = st.empty()
    fps_slot = st.empty()
with col_probs:
    st.subheader("Live class probabilities")
    chart_slot = st.empty()
    table_slot = st.empty()

if not run:
    frame_slot.info("Check **Start camera** in the sidebar to begin.")
else:
    cap = cv2.VideoCapture(int(cam_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        st.error(f"Could not open webcam index {cam_index}. Try a different camera index.")
    else:
        try:
            prev_t = time.time()
            fps = 0.0
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=MODEL_COMPLEXITY,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        st.warning("Lost camera feed.")
                        break

                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = hands.process(rgb)

                    label_text = "No hand detected"
                    ranked = None

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                        pts = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
                        ranked = predict_all_proba(pts)
                        if ranked:
                            top_label, top_conf = ranked[0]
                            label_text = f"{top_label}  ({top_conf * 100:.1f}%)"

                    now = time.time()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
                    prev_t = now

                    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                                      use_container_width=True)
                    label_slot.markdown(f"### {label_text}")
                    fps_slot.caption(f"{fps:.1f} FPS")

                    if ranked:
                        df = pd.DataFrame(ranked, columns=["Mudra", "Probability"])
                        df["Probability"] = (df["Probability"] * 100).round(2)
                        chart_slot.bar_chart(df.head(topk).set_index("Mudra")["Probability"],
                                              horizontal=True)
                        if show_table:
                            table_slot.dataframe(
                                df.rename(columns={"Probability": "Probability (%)"}),
                                use_container_width=True, hide_index=True, height=400,
                            )
                        else:
                            table_slot.empty()
                    else:
                        chart_slot.empty()
                        table_slot.empty()
        finally:
            cap.release()
