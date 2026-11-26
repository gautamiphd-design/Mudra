import os, sys, glob, csv
from dot_detect_v2 import get_dots_v2
from mudra_features import compute_features, FEATURE_NAMES

BASE = "/sessions/dazzling-charming-keller/mnt/bharatnatyam_mediapipe/Mudraas"
OUT_DIR = "/sessions/dazzling-charming-keller/mnt/outputs/work/features_v2"
os.makedirs(OUT_DIR, exist_ok=True)


def process_class(cls):
    files = sorted(glob.glob(os.path.join(BASE, cls, "*.jpg")))
    out_path = os.path.join(OUT_DIR, f"{cls}.csv")
    n_ok, n_fail = 0, 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"] + FEATURE_NAMES)
        for fp in files:
            pts = get_dots_v2(fp)
            feat = compute_features(pts) if pts is not None and len(pts) else None
            if feat is None:
                n_fail += 1
                continue
            writer.writerow([os.path.basename(fp), cls] + list(feat))
            n_ok += 1
    return n_ok, n_fail


if __name__ == "__main__":
    classes = sys.argv[1:]
    for cls in classes:
        n_ok, n_fail = process_class(cls)
        print(f"{cls}: ok={n_ok} fail={n_fail}")
