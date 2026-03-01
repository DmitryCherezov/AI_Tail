
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image

import data_utils  # expects paths defined as you described


###############################################################################
# CONFIG
###############################################################################

SEED = 42
CLASS_ID = getattr(data_utils, "CLASS_ID", 0)  # default 0
TARGET_CLASS_ID = CLASS_ID  # we will only paste class 0 tails

NUM_TO_CREATE = 120

SCALE_RANGE = (0.75, 1.25)          # scale of tail cutout
ROTATION_DEG_RANGE = (-25, 25)      # random rotation degrees
MAX_TRIES_PER_SAMPLE = 30           # retry if placement fails
MIN_VISIBLE_PIXELS = 150            # discard if tail becomes too tiny after transform
ALLOW_PARTIAL_OUTSIDE = False       # if True, allow cutout to go outside background



###############################################################################
# CODE
###############################################################################

def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def read_yolo_seg_lines(lbl_path: Path) -> List[str]:
    if not lbl_path.exists():
        return []
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse_yolo_seg_line(line: str) -> Tuple[int, np.ndarray]:
    """
    Returns (class_id, coords_norm) where coords_norm is shape (N,2) float32 in [0,1].
    """
    parts = line.strip().split()
    if len(parts) < 1 + 6 or (len(parts) - 1) % 2 != 0:
        raise ValueError(f"Bad YOLO-seg line: {line[:80]}...")
    cls = int(float(parts[0]))
    nums = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    pts = nums.reshape(-1, 2)
    return cls, pts


def norm_to_abs_points(pts_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.empty_like(pts_norm, dtype=np.float32)
    pts[:, 0] = pts_norm[:, 0] * w
    pts[:, 1] = pts_norm[:, 1] * h
    return pts


def abs_to_norm_points(pts_abs: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.empty_like(pts_abs, dtype=np.float32)
    pts[:, 0] = pts_abs[:, 0] / w
    pts[:, 1] = pts_abs[:, 1] / h
    # clamp to [0,1] to be safe
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def polygon_mask(h: int, w: int, pts_abs: np.ndarray) -> np.ndarray:
    """
    pts_abs: (N,2) float or int in pixel coords (x,y).
    Returns uint8 mask {0,255}.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(pts_abs).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 255)
    return mask


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def rotate_scale_rgba_and_points(
    rgba: np.ndarray, pts: np.ndarray, angle_deg: float, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    rgba: (H,W,4) uint8
    pts: (N,2) float32 points in rgba coordinate system
    returns transformed_rgba, transformed_pts (in new RGBA coords)
    """
    h, w = rgba.shape[:2]

    # Center
    cx, cy = w / 2.0, h / 2.0

    # Rotation+scale matrix for cv2.warpAffine
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    # Compute new bounds to keep whole object
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])  # (4,3)
    new_corners = (M @ corners_h.T).T  # (4,2)

    min_xy = new_corners.min(axis=0)
    max_xy = new_corners.max(axis=0)
    new_w = int(np.ceil(max_xy[0] - min_xy[0]))
    new_h = int(np.ceil(max_xy[1] - min_xy[1]))

    # Shift to make all coords positive
    M[0, 2] -= min_xy[0]
    M[1, 2] -= min_xy[1]

    transformed = cv2.warpAffine(
        rgba,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Transform points
    pts_h = np.hstack([pts.astype(np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)])
    new_pts = (M @ pts_h.T).T  # (N,2)

    return transformed, new_pts


def alpha_paste(bg_rgb: np.ndarray, fg_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Paste fg_rgba onto bg_rgb at top-left (x,y).
    Returns updated bg_rgb.
    """
    bh, bw = bg_rgb.shape[:2]
    fh, fw = fg_rgba.shape[:2]

    x1, y1 = x, y
    x2, y2 = x + fw, y + fh

    if x2 <= 0 or y2 <= 0 or x1 >= bw or y1 >= bh:
        return bg_rgb  # no overlap

    # Clip to background
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(bw, x2)
    cy2 = min(bh, y2)

    # Corresponding region in fg
    fx1 = cx1 - x1
    fy1 = cy1 - y1
    fx2 = fx1 + (cx2 - cx1)
    fy2 = fy1 + (cy2 - cy1)

    fg = fg_rgba[fy1:fy2, fx1:fx2, :].astype(np.float32)
    bg = bg_rgb[cy1:cy2, cx1:cx2, :].astype(np.float32)

    alpha = fg[:, :, 3:4] / 255.0
    out = fg[:, :, :3] * alpha + bg * (1.0 - alpha)
    bg_rgb[cy1:cy2, cx1:cx2, :] = np.clip(out, 0, 255).astype(np.uint8)
    return bg_rgb


@dataclass
class TailCutout:
    rgba: np.ndarray          # (H,W,4)
    poly_pts: np.ndarray      # (N,2) in cutout coords


def extract_tail_cutout(donor_img_path: Path, donor_lbl_path: Path) -> TailCutout | None:
    """
    Picks a random tail polygon from donor label file and extracts tight RGBA patch.
    """
    lines = read_yolo_seg_lines(donor_lbl_path)
    if not lines:
        return None

    # Parse all tail objects
    candidates = []
    with Image.open(donor_img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        rgb = np.array(im, dtype=np.uint8)

    for ln in lines:
        try:
            cls, pts_norm = parse_yolo_seg_line(ln)
        except Exception:
            continue
        if cls != TARGET_CLASS_ID:
            continue
        pts_abs = norm_to_abs_points(pts_norm, w, h)
        candidates.append(pts_abs)

    if not candidates:
        return None

    pts_abs = random.choice(candidates)
    mask = polygon_mask(h, w, pts_abs)
    bb = bbox_from_mask(mask)
    if bb is None:
        return None

    x1, y1, x2, y2 = bb
    # Add small padding
    pad = 4
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    crop_rgb = rgb[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

    # Build RGBA cutout
    rgba = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_rgb
    rgba[:, :, 3] = crop_mask  # 0 or 255

    # Shift polygon into cutout coords
    pts_cutout = pts_abs.copy().astype(np.float32)
    pts_cutout[:, 0] -= x1
    pts_cutout[:, 1] -= y1

    return TailCutout(rgba=rgba, poly_pts=pts_cutout)


def choose_random_paste_location(bg_w: int, bg_h: int, fg_w: int, fg_h: int) -> Tuple[int, int] | None:
    """
    Returns (x,y) top-left for placing fg onto bg.
    If ALLOW_PARTIAL_OUTSIDE is False, ensure full fg fits.
    """
    if not ALLOW_PARTIAL_OUTSIDE:
        if fg_w > bg_w or fg_h > bg_h:
            return None
        x = random.randint(0, bg_w - fg_w)
        y = random.randint(0, bg_h - fg_h)
        return x, y

    # Allow partial outside: choose x,y so that some overlap exists
    x = random.randint(-fg_w // 2, bg_w - fg_w // 2)
    y = random.randint(-fg_h // 2, bg_h - fg_h // 2)
    return x, y


def yolo_line_from_points(cls: int, pts_abs: np.ndarray, img_w: int, img_h: int) -> str:
    pts_norm = abs_to_norm_points(pts_abs, img_w, img_h)
    coords = []
    for x, y in pts_norm:
        coords.append(f"{float(x):.6f}")
        coords.append(f"{float(y):.6f}")
    return f"{cls} " + " ".join(coords)


def next_aug_index(out_dir: Path, prefix: str = "augmented_") -> int:
    """
    Finds next integer index based on existing files.
    """
    existing = list(out_dir.glob(f"{prefix}*.png"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        stem = p.stem  # augmented_000001
        try:
            n = int(stem.replace(prefix, ""))
            nums.append(n)
        except Exception:
            continue
    return (max(nums) + 1) if nums else 1



###############################################################################
# MAIN
###############################################################################

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    # Input sources
    donors = list_images(data_utils.PNG_IMAGES)
    bgs = list_images(data_utils.PNG_AUGMENTATION_IMAGES)

    if not donors:
        raise RuntimeError(f"No donor images in {data_utils.PNG_IMAGES}")
    if not bgs:
        raise RuntimeError(f"No background images in {data_utils.PNG_AUGMENTATION_IMAGES}")

    start_idx = next_aug_index(data_utils.DATASET_TRAIN_IMG)
    created = 0
    attempts = 0

    while created < NUM_TO_CREATE:
        attempts += 1

        donor_img = random.choice(donors)
        donor_lbl = Path(data_utils.LABELS) / f"{donor_img.stem}.txt"
        if not donor_lbl.exists():
            continue

        cut = extract_tail_cutout(donor_img, donor_lbl)
        if cut is None:
            continue

        # Apply random rotation/scale
        angle = random.uniform(*ROTATION_DEG_RANGE)
        scale = random.uniform(*SCALE_RANGE)

        fg_rgba, fg_pts = rotate_scale_rgba_and_points(cut.rgba, cut.poly_pts, angle, scale)

        # Check fg visible pixels
        if int((fg_rgba[:, :, 3] > 0).sum()) < MIN_VISIBLE_PIXELS:
            continue

        # Pick background image
        bg_img = random.choice(bgs)
        with Image.open(bg_img) as im_bg:
            im_bg = im_bg.convert("RGB")
            bg_w, bg_h = im_bg.size
            bg_rgb = np.array(im_bg, dtype=np.uint8)

        fg_h, fg_w = fg_rgba.shape[:2]
        loc = choose_random_paste_location(bg_w, bg_h, fg_w, fg_h)
        if loc is None:
            continue
        x0, y0 = loc

        # If not allowing partial outside, we already ensured fit.
        # If allowing partial outside, label polygon could go out-of-bounds; we clamp later.

        # Paste
        bg_rgb2 = alpha_paste(bg_rgb, fg_rgba, x0, y0)

        # New polygon in background coords
        new_pts_abs = fg_pts.copy()
        new_pts_abs[:, 0] += x0
        new_pts_abs[:, 1] += y0

        # If disallow partial outside, ensure all points are inside
        if not ALLOW_PARTIAL_OUTSIDE:
            if (new_pts_abs[:, 0].min() < 0 or new_pts_abs[:, 1].min() < 0 or
                new_pts_abs[:, 0].max() >= bg_w or new_pts_abs[:, 1].max() >= bg_h):
                continue

        # Produce YOLO line
        yolo_line = yolo_line_from_points(CLASS_ID, new_pts_abs, bg_w, bg_h)

        # Save outputs
        out_idx = start_idx + created
        out_name = f"augmented_{out_idx:06d}"

        out_img_path = Path(data_utils.DATASET_TRAIN_IMG) / f"{out_name}.png"
        out_lbl_path = Path(data_utils.DATASET_TRAIN_LBL) / f"{out_name}.txt"

        # Avoid accidental overwrite
        if out_img_path.exists() or out_lbl_path.exists():
            # bump index and retry
            start_idx += 1
            continue

        Image.fromarray(bg_rgb2).save(out_img_path, format="PNG")
        out_lbl_path.write_text(yolo_line + "\n", encoding="utf-8")

        created += 1

        if created % 25 == 0:
            print(f"[OK] Created {created}/{NUM_TO_CREATE} (attempts={attempts})")

        # Avoid infinite loops if data is problematic
        if attempts > NUM_TO_CREATE * MAX_TRIES_PER_SAMPLE:
            print("[WARN] Too many attempts; stopping early.")
            break

    print(f"[DONE] Created: {created}")
    print(f"[OUT] images: {Path(data_utils.DATASET_TRAIN_IMG).resolve()}")
    print(f"[OUT] labels: {Path(data_utils.DATASET_TRAIN_LBL).resolve()}")


if __name__ == "__main__":
    main()
    











