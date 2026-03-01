'''
streamlit run yolo_seg_viewer.py
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image
import streamlit as st


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class DatasetSplit:
    name: str
    images_dir: Path
    labels_dir: Path


def find_splits(dataset_root: Path) -> List[DatasetSplit]:
    """
    Expects YOLO structure:
      dataset_root/
        images/train|val|test
        labels/train|val|test
    Returns splits that exist.
    """
    splits = []
    for split in ["train", "val", "test"]:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        if images_dir.exists() and labels_dir.exists():
            splits.append(DatasetSplit(split, images_dir, labels_dir))
    return splits


def list_images(images_dir: Path) -> List[Path]:
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def parse_yolo_seg_file(lbl_path: Path) -> List[Tuple[int, np.ndarray]]:
    """
    YOLOv8-seg label format per line:
      class x1 y1 x2 y2 ... (normalized 0..1)
    Returns list of (class_id, pts_norm Nx2).
    """
    if not lbl_path.exists():
        return []
    txt = lbl_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    objects = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 1 + 6:
            continue
        if (len(parts) - 1) % 2 != 0:
            continue

        try:
            cls = int(float(parts[0]))
            nums = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            pts = nums.reshape(-1, 2)
            objects.append((cls, pts))
        except Exception:
            continue
    return objects


def norm_to_abs_points(pts_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.empty_like(pts_norm, dtype=np.float32)
    pts[:, 0] = pts_norm[:, 0] * w
    pts[:, 1] = pts_norm[:, 1] * h
    return pts


def draw_polygons(
    img_rgb: np.ndarray,
    objects: List[Tuple[int, np.ndarray]],
    class_names: Optional[Dict[int, str]] = None,
    fill_alpha: float = 0.25,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw polygons on RGB image. Uses OpenCV.
    - fill_alpha: 0..1
    """
    out = img_rgb.copy()
    overlay = img_rgb.copy()
    h, w = out.shape[:2]

    for cls, pts_norm in objects:
        pts_abs = norm_to_abs_points(pts_norm, w, h)
        poly = np.round(pts_abs).astype(np.int32).reshape((-1, 1, 2))

        # deterministic color from class id
        rng = np.random.default_rng(seed=cls + 12345)
        color = tuple(int(x) for x in rng.integers(40, 220, size=3))  # (B,G,R)? We'll convert below.

        # OpenCV uses BGR; our image is RGB, but we are drawing directly on RGB array.
        # So we still pass color as RGB tuple.
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=line_thickness)

        # label text near first point
        name = class_names.get(cls, str(cls)) if class_names else str(cls)
        x0, y0 = int(poly[0, 0, 0]), int(poly[0, 0, 1])
        cv2.putText(out, name, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # alpha blend fill
    out = cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0.0)
    return out


def main():
    st.set_page_config(page_title="YOLOv8-seg Viewer", layout="wide")
    st.title("YOLOv8-seg dataset viewer (polygons overlay)")

    with st.sidebar:
        st.header("Dataset")
        dataset_path_str = st.text_input(
            "Dataset root path",
            value="",
            placeholder=r"D:\000.Projects\InTexRos\data\dataset",
        )
        class_names_str = st.text_input(
            "Class names (optional)",
            value="0:tail",
            help="Format: 0:tail,1:other",
        )

        fill_alpha = st.slider("Fill alpha", 0.0, 0.9, 0.25, 0.05)
        thickness = st.slider("Line thickness", 1, 8, 2, 1)

    if not dataset_path_str.strip():
        st.info("Enter dataset root path in the sidebar.")
        return

    dataset_root = Path(dataset_path_str).expanduser().resolve()
    if not dataset_root.exists():
        st.error(f"Path does not exist: {dataset_root}")
        return

    # parse class names
    class_names: Dict[int, str] = {}
    if class_names_str.strip():
        for part in class_names_str.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            try:
                class_names[int(k.strip())] = v.strip()
            except Exception:
                pass

    splits = find_splits(dataset_root)
    if not splits:
        st.error("No splits found. Expected folders: images/train|val|test and labels/train|val|test")
        return

    split_names = [s.name for s in splits]
    split_choice = st.selectbox("Split", split_names, index=0)
    split = next(s for s in splits if s.name == split_choice)

    images = list_images(split.images_dir)
    if not images:
        st.warning(f"No images found in: {split.images_dir}")
        return

    file_names = [p.name for p in images]
    selected_name = st.selectbox("Image file", file_names, index=0)
    img_path = split.images_dir / selected_name

    # label file with same stem
    lbl_path = split.labels_dir / (img_path.stem + ".txt")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        # load image
        im = Image.open(img_path).convert("RGB")
        img_rgb = np.array(im, dtype=np.uint8)

        # load objects
        objects = parse_yolo_seg_file(lbl_path)

        drawn = draw_polygons(
            img_rgb=img_rgb,
            objects=objects,
            class_names=class_names,
            fill_alpha=float(fill_alpha),
            line_thickness=int(thickness),
        )

        st.image(drawn, caption=f"{split_choice}: {img_path.name}", use_container_width=True)

    with col2:
        st.subheader("Paths")
        st.code(str(img_path), language="text")
        st.code(str(lbl_path), language="text")

        st.subheader("YOLO label content")
        if lbl_path.exists():
            st.code(lbl_path.read_text(encoding="utf-8"), language="text")
        else:
            st.warning("No label file found for this image.")

        st.subheader("Objects parsed")
        if objects:
            for i, (cls, pts) in enumerate(objects, start=1):
                st.write(f"#{i}: class={cls} ({class_names.get(cls, cls)}) points={len(pts)}")
        else:
            st.write("No objects.")

    st.caption("Tip: if polygons look shifted, verify image/label stem match and that labels are normalized 0..1.")


if __name__ == "__main__":
    main()