"""
Real-time Bharatanatyam mudra classifier using a webcam (ensemble version).

Uses the SVM + Random Forest weighted-blend ensemble (mudra_ensemble.py).
Landmarks are detected once per frame with MediaPipe and fed straight into
the classifier as coordinates -- no cropping, no re-detection, no extra
image work. mudra_features.compute_features() is translation/scale/
rotation-invariant, so a crop of the hand region would produce the exact
same feature vector as the full frame; it only ever changed what was drawn
on screen, not what the model saw. Skipping it removes dead weight without
costing any accuracy.

Where the latency actually goes, per frame:
  - hands.process(rgb)   -> MediaPipe inference. This is ~90%+ of frame
                             time and is the only step that meaningfully
                             trades off against accuracy (see MODEL_COMPLEXITY
                             below).
  - draw + predict + text -> a few hundred microseconds combined, even with
                             the full 28-class confidence panel.
So the panel below does not meaningfully affect latency; the knobs that do
are CAM_WIDTH/CAM_HEIGHT and MODEL_COMPLEXITY.

Tuning knobs (top of file):
  MODEL_COMPLEXITY: 1 = full MediaPipe hand-landmark model (default,
      highest landmark accuracy). Set to 0 for the "lite" model if you want
      to trade some landmark precision for lower latency.
  CAM_WIDTH / CAM_HEIGHT: lower capture resolution = less cvtColor/resize
      work per frame. 640x480 is a good speed/accuracy default -- the hand
      is still plenty big enough for MediaPipe to find precisely. Raising
      this won't improve accuracy once the hand is already well-resolved,
      it'll just cost latency.
  SHOW_ALL_CONFIDENCES: toggle at runtime with 'c' if you want to compare
      FPS with/without the panel yourself.

Run locally (needs a real webcam -- this will NOT work inside a
cloud/remote sandbox):

    pip install -r requirements.txt
    python webcam_demo_v2.py

Press "c" to toggle the full confidence panel, "q" to quit.
"""
import time

import numpy as np
import cv2
import mediapipe as mp

from mudra_ensemble import predict_all_proba

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

MODEL_COMPLEXITY = 1     # 0 = lite/faster, 1 = full/most accurate (default)
CAM_WIDTH = 640
CAM_HEIGHT = 480
SHOW_ALL_CONFIDENCES = True


def draw_confidence_panel(frame, ranked, top_label):
    """Right-hand sidebar listing every class's confidence, sorted
    descending. ranked: list of (label, prob) for ALL classes."""
    h, w = frame.shape[:2]
    panel_w = 230
    row_h = 18
    panel_h = min(h, 40 + row_h * len(ranked))

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 20
    for label, prob in ranked:
        is_top = label == top_label
        color = (0, 255, 0) if is_top else (200, 200, 200)
        text = f"{label:<16}{prob * 100:5.1f}%"
        cv2.putText(frame, text, (w - panel_w + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
        y += row_h
        if y > panel_h - 4:
            break


def main():
    global SHOW_ALL_CONFIDENCES

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Try a different camera index.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

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
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False  # skip mediapipe's internal copy
            results = hands.process(rgb)

            label_text = "No hand detected"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                pts = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])

                ranked = predict_all_proba(pts)
                if ranked is not None:
                    top_label, top_conf = ranked[0]
                    label_text = f"{top_label}  ({top_conf * 100:.0f}%)"
                    if SHOW_ALL_CONFIDENCES:
                        draw_confidence_panel(frame, ranked, top_label)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"{fps:4.1f} FPS", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            cv2.imshow("Bharatanatyam Mudra Classifier (Ensemble)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                SHOW_ALL_CONFIDENCES = not SHOW_ALL_CONFIDENCES

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
