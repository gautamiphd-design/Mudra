"""
Shared feature-engineering code for the Bharatanatyam mudra classifier.

Turns a set of 21 hand-landmark (x, y) points into a fixed-length,
scale/rotation/translation-invariant "shape descriptor" embedding.

Used both to build the training dataset (from landmark points recovered
from the annotated dataset images) and at inference time (from real
MediaPipe Hands landmarks off a live camera feed) -- so the SAME function
must be used in both places for the model to work correctly.
"""
import numpy as np
from scipy.spatial import ConvexHull

N_DIST_BINS = 20
N_CENTROID_BINS = 12
DIST_MAX = 3.5
CENTROID_MAX = 2.5

FEATURE_NAMES = (
    ["n_points"]
    + [f"pdist_bin{i}" for i in range(N_DIST_BINS)]
    + [f"cdist_bin{i}" for i in range(N_CENTROID_BINS)]
    + ["elongation", "hull_area", "hull_perimeter", "density"]
)


def compute_features(pts):
    """
    pts: array-like of shape (N, 2) -- (x, y) coordinates of the hand
         landmark points (any consistent unit -- pixels or normalized
         0-1 coords both work, since everything is scale-normalized
         internally).

    Returns a 1D numpy array of length len(FEATURE_NAMES), or None if
    the point set is not usable (too few / too many points, or points
    are degenerate).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 15 or n > 50:
        return None

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    scale = np.sqrt((centered ** 2).sum(axis=1)).mean()
    if scale < 1e-9:
        return None
    norm = centered / scale

    # pairwise distances -> histogram (rotation/translation/permutation invariant)
    diffs = norm[:, None, :] - norm[None, :, :]
    pdist = np.sqrt((diffs ** 2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    pdist_vals = pdist[iu]
    pdist_hist, _ = np.histogram(pdist_vals, bins=N_DIST_BINS, range=(0, DIST_MAX))
    pdist_hist = pdist_hist / max(pdist_hist.sum(), 1)

    # centroid distances -> histogram
    cdist_vals = np.sqrt((norm ** 2).sum(axis=1))
    cdist_hist, _ = np.histogram(cdist_vals, bins=N_CENTROID_BINS, range=(0, CENTROID_MAX))
    cdist_hist = cdist_hist / max(cdist_hist.sum(), 1)

    # PCA elongation ratio
    cov = np.cov(norm.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    elongation = np.sqrt(eigvals[-1] / eigvals[0])

    # convex hull area / perimeter
    try:
        hull = ConvexHull(norm)
        hull_area = hull.volume       # 2D "volume" == area
        hull_perimeter = hull.area    # 2D "area" == perimeter
    except Exception:
        hull_area = 0.0
        hull_perimeter = 0.0

    density = n / max(hull_area, 1e-6)

    feat = np.concatenate([
        [n],
        pdist_hist,
        cdist_hist,
        [elongation, hull_area, hull_perimeter, density],
    ])
    return feat
