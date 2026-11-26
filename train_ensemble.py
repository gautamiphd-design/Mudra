"""
Trains the final SVM + Random Forest weighted-blend ensemble on the v2
(improved-detection) feature set, and saves all artifacts needed for
inference (mudra_ensemble.py / webcam_demo_v2.py).

Usage:
    python extract_features_v2.py <Class1> <Class2> ...   # for each of the 28 classes
    # then concatenate the per-class CSVs in features_v2/ into all_features_v2.csv
    python train_ensemble.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "all_features_v2.csv")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("filename", "label")]
    X = df[feature_cols].values
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("Fitting Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print("Fitting SVM...")
    svm = SVC(kernel="rbf", C=5, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_s, y_train)

    proba_rf_val = rf.predict_proba(X_val)
    proba_svm_val = svm.predict_proba(X_val_s)

    rf_val_acc = accuracy_score(y_val, proba_rf_val.argmax(axis=1))
    svm_val_acc = accuracy_score(y_val, proba_svm_val.argmax(axis=1))
    print(f"RF val_acc={rf_val_acc:.4f}   SVM val_acc={svm_val_acc:.4f}")

    # find the best SVM/RF blend weight on the validation set
    best_w, best_acc = 0.5, -1
    for w in np.arange(0, 1.01, 0.05):
        blend = w * proba_svm_val + (1 - w) * proba_rf_val
        acc = accuracy_score(y_val, blend.argmax(axis=1))
        if acc > best_acc:
            best_acc, best_w = acc, w
    print(f"Best blend weight (SVM weight={best_w:.2f}) val_acc={best_acc:.4f}")

    proba_rf_test = rf.predict_proba(X_test)
    proba_svm_test = svm.predict_proba(X_test_s)
    blend_test = best_w * proba_svm_test + (1 - best_w) * proba_rf_test
    test_pred = blend_test.argmax(axis=1)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Ensemble TEST accuracy: {test_acc:.4f}")

    print("\nClassification report (test set):")
    print(classification_report(y_test, test_pred, target_names=le.classes_))

    joblib.dump(rf, os.path.join(ARTIFACTS_DIR, "rf_model.joblib"), compress=3)
    joblib.dump(svm, os.path.join(ARTIFACTS_DIR, "svm_model.joblib"))
    joblib.dump(float(best_w), os.path.join(ARTIFACTS_DIR, "blend_weight.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_cols.joblib"))

    cm = confusion_matrix(y_test, test_pred)
    np.save(os.path.join(ARTIFACTS_DIR, "confusion_matrix.npy"), cm)

    with open(os.path.join(ARTIFACTS_DIR, "meta.txt"), "w") as f:
        f.write("model=SVM+RandomForest weighted blend\n")
        f.write(f"blend_svm_weight={best_w:.2f}\n")
        f.write(f"rf_val_acc={rf_val_acc:.4f}\n")
        f.write(f"svm_val_acc={svm_val_acc:.4f}\n")
        f.write(f"blend_val_acc={best_acc:.4f}\n")
        f.write(f"blend_test_acc={test_acc:.4f}\n")

    print("\nSaved model artifacts to", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
