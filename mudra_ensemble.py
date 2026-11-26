"""
Inference helper for the SVM + Random Forest weighted-blend ensemble.

Loads the trained artifacts from model_artifacts/ once, and exposes
predict(pts) which takes a (21, 2) array of hand-landmark (x, y) points
and returns (predicted_label, confidence).
"""
import os
import joblib
import numpy as np

from mudra_features import compute_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")

_rf = joblib.load(os.path.join(ARTIFACTS_DIR, "rf_model.joblib"))
_svm = joblib.load(os.path.join(ARTIFACTS_DIR, "svm_model.joblib"))
_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
_label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
_blend_w = joblib.load(os.path.join(ARTIFACTS_DIR, "blend_weight.joblib"))  # SVM weight; RF gets (1 - w)


def class_labels():
    """All class names in the fixed order the model's probabilities correspond to."""
    return list(_label_encoder.classes_)


def predict_all_proba(pts):
    """
    pts: array-like of shape (N, 2) hand-landmark (x, y) coordinates
         (pixel or normalized 0-1 coords both work).

    Returns a list of (label, probability) for ALL classes, sorted by
    probability descending. Returns None if the point set is unusable.
    """
    feat = compute_features(pts)
    if feat is None:
        return None
    X = feat.reshape(1, -1)
    X_s = _scaler.transform(X)

    proba_rf = _rf.predict_proba(X)[0]
    proba_svm = _svm.predict_proba(X_s)[0]
    proba = _blend_w * proba_svm + (1 - _blend_w) * proba_rf

    labels = _label_encoder.classes_
    results = sorted(zip(labels, proba), key=lambda t: t[1], reverse=True)
    return [(label, float(p)) for label, p in results]


def predict(pts):
    """
    pts: array-like of shape (N, 2) hand-landmark (x, y) coordinates
         (pixel or normalized 0-1 coords both work).

    Returns (label: str, confidence: float) or (None, None) if the
    point set is unusable.
    """
    all_proba = predict_all_proba(pts)
    if all_proba is None:
        return None, None
    return all_proba[0]
