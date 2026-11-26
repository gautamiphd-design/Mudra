import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def base_red_mask(img):
    b, g, r = cv2.split(img.astype(np.int16))
    mask = ((r > 170) & (g < 90) & (b < 90) & (r - g > 90) & (r - b > 90)).astype(np.uint8) * 255
    mask[:6, :] = 0; mask[-6:, :] = 0; mask[:, :6] = 0; mask[:, -6:] = 0
    return mask


def loose_red_mask(img):
    # more permissive, to be intersected/used for a secondary recall pass
    b, g, r = cv2.split(img.astype(np.int16))
    mask = ((r > 140) & (g < 120) & (b < 120) & (r - g > 55) & (r - b > 55)).astype(np.uint8) * 255
    mask[:6, :] = 0; mask[-6:, :] = 0; mask[:, :6] = 0; mask[:, -6:] = 0
    return mask


def split_component(comp_mask, expected_k, min_dot_radius):
    """Split a merged blob into expected_k points via distance-transform watershed."""
    dist = ndi.distance_transform_edt(comp_mask)
    coords = peak_local_max(dist, min_distance=max(int(min_dot_radius * 1.2), 2),
                             labels=comp_mask, num_peaks=expected_k)
    if len(coords) < 2:
        # fall back: can't confidently split, just return centroid
        ys, xs = np.where(comp_mask)
        return [(xs.mean(), ys.mean())]
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    labels_ws = watershed(-dist, markers, mask=comp_mask)
    pts = []
    for lbl in range(1, markers.max() + 1):
        ys, xs = np.where(labels_ws == lbl)
        if len(xs) == 0:
            continue
        pts.append((xs.mean(), ys.mean()))
    return pts


def get_dots_v2(path, min_area=8):
    """Detect landmark dots in an image file on disk."""
    img = cv2.imread(path)
    if img is None:
        return None
    return get_dots_v2_from_array(img, min_area=min_area)


def get_dots_v2_from_array(img, min_area=8):
    """Detect landmark dots in an already-loaded BGR image array (e.g. from
    cv2.imdecode on uploaded bytes, no need to save to disk first)."""
    mask = base_red_mask(img)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return np.zeros((0, 2))

    areas = np.array([a for _, a in comps])
    unit_area = np.median(areas[areas <= np.percentile(areas, 70)]) if len(areas) > 3 else np.median(areas)
    unit_area = max(unit_area, 10)
    min_dot_radius = np.sqrt(unit_area / np.pi)

    pts = []
    for i, area in comps:
        k = int(round(area / unit_area))
        k = max(k, 1)
        if k <= 1:
            pts.append(tuple(centroids[i]))
        else:
            comp_mask = (labels == i).astype(np.uint8)
            split_pts = split_component(comp_mask, k, min_dot_radius)
            pts.extend(split_pts)

    pts = np.array(pts)

    # secondary recall pass: try to recover missed dots using an adaptive
    # color range derived from the dots we already found confidently
    if 0 < len(pts) < 21:
        sample_colors = []
        h, w = img.shape[:2]
        for (x, y) in pts:
            xi, yi = int(round(x)), int(round(y))
            xi = np.clip(xi, 0, w - 1); yi = np.clip(yi, 0, h - 1)
            sample_colors.append(img[yi, xi].astype(np.float64))
        mean_color = np.mean(sample_colors, axis=0)  # BGR

        loose = loose_red_mask(img)
        diff = np.abs(img.astype(np.float64) - mean_color[None, None, :]).sum(axis=2)
        color_close = (diff < 90).astype(np.uint8) * 255
        candidate_mask = cv2.bitwise_and(loose, color_close)

        n2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
        for i in range(1, n2):
            area2 = stats2[i, cv2.CC_STAT_AREA]
            if area2 < min_area * 0.5 or area2 > unit_area * 1.6:
                continue
            cx, cy = centroids2[i]
            # skip if too close to an already-found point (avoid duplicates)
            if len(pts) and np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) < min_dot_radius:
                continue
            pts = np.vstack([pts, [cx, cy]]) if len(pts) else np.array([[cx, cy]])

    return pts
