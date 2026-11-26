"""
Shared landmark-detection logic used by both test_image.py and the
Streamlit app (app.py).

Works on an already-loaded image array (no need to save to disk first,
which matters for Streamlit file uploads), and handles two kinds of
input:
  1. Dataset-style images with MediaPipe landmarks already drawn as red
     dots + white lines -- landmarks are recovered via dot detection.
  2. Plain photos of a hand with nothing drawn -- landmarks are detected
     live with MediaPipe Hands instead.

get_landmarks_from_array() tries (1) first and automatically falls back
to (2) if no usable dot pattern is found.
"""
import cv2
import numpy as np

from dot_detect_v2 import get_dots_v2_from_array


def get_landmarks_from_array(img):
    """
    img: a BGR image array (as returned by cv2.imread / cv2.imdecode).

    Returns (points, source) where source is "dots" or "mediapipe", or
    (None, None) if no usable hand/landmarks were found either way.
    """
    pts = get_dots_v2_from_array(img)
    if pts is not None and 15 <= len(pts) <= 50:
        return pts, "dots"

    try:
        import mediapipe as mp
    except ImportError:
        return None, None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                   min_detection_confidence=0.3) as hands:
        results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None, None
    lm = results.multi_hand_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])
    return pts, "mediapipe"


def get_landmarks(image_path):
    """Same as get_landmarks_from_array, but reads the image from a file path."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    return get_landmarks_from_array(img)
