"""
Test script: classify one or more hand-mudra images and print the
predicted mudra plus the probability of every class.

Works with two kinds of input images:
  1. Images from your original dataset style (hand crop with MediaPipe
     landmarks already drawn as red dots + white lines) -- landmarks are
     recovered via the same dot-detector used for training.
  2. Plain/clean photos of a hand with no markers drawn -- landmarks are
     detected live with MediaPipe Hands instead.

The script tries (1) first and automatically falls back to (2) if no
usable dot pattern is found, so you can just point it at any image.

Usage:
    python test_image.py path/to/image.jpg
    python test_image.py path/to/folder_of_images/
    python test_image.py path/to/image.jpg --topk 5
    python test_image.py path/to/image.jpg --annotate            # also saves a visualization
    python test_image.py path/to/image.jpg --annotate --out-dir results/
"""
import argparse
import os
import sys

import cv2

from landmark_utils import get_landmarks
from mudra_ensemble import predict_all_proba

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def annotate_image(image_path, pts, source, ranked, out_path):
    img = cv2.imread(image_path)
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    label, conf = ranked[0]
    text = f"{label} ({conf*100:.1f}%) [{source}]"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)


def classify_image(image_path, topk, annotate, out_dir):
    pts, source = get_landmarks(image_path)
    print(f"\n=== {os.path.basename(image_path)} ===")
    if pts is None:
        print("  Could not find a hand / usable landmark pattern in this image.")
        return

    ranked = predict_all_proba(pts)
    if ranked is None:
        print(f"  Landmarks found ({len(pts)} points, via {source}) but not enough to classify.")
        return

    print(f"  Landmark source: {source}  ({len(pts)} points)")
    print(f"  Predicted mudra: {ranked[0][0]}  ({ranked[0][1]*100:.1f}% confidence)\n")
    print(f"  {'Rank':<5}{'Mudra':<16}{'Probability':>12}")
    n_show = len(ranked) if topk is None else min(topk, len(ranked))
    for i, (label, prob) in enumerate(ranked[:n_show], start=1):
        marker = " <--" if i == 1 else ""
        print(f"  {i:<5}{label:<16}{prob*100:>10.2f}%{marker}")

    if annotate:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"pred_{os.path.basename(image_path)}")
        annotate_image(image_path, pts, source, ranked, out_path)
        print(f"\n  Saved annotated image -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Classify Bharatanatyam mudra image(s).")
    parser.add_argument("path", help="Path to an image file or a folder of images.")
    parser.add_argument("--topk", type=int, default=None,
                         help="Show only the top K class probabilities (default: show all 28).")
    parser.add_argument("--annotate", action="store_true",
                         help="Save an annotated copy of each image with detected points + prediction.")
    parser.add_argument("--out-dir", default="test_results",
                         help="Where to save annotated images (default: ./test_results)")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Path not found: {args.path}")
        sys.exit(1)

    if os.path.isdir(args.path):
        files = sorted(
            os.path.join(args.path, f) for f in os.listdir(args.path)
            if f.lower().endswith(IMAGE_EXTS)
        )
        if not files:
            print(f"No image files found in {args.path}")
            sys.exit(1)
        for f in files:
            classify_image(f, args.topk, args.annotate, args.out_dir)
    else:
        classify_image(args.path, args.topk, args.annotate, args.out_dir)


if __name__ == "__main__":
    main()
