from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image
import streamlit as st


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ViewerConfig:
    images_dir: Path
    labels_dir: Path
    class_names: Dict[int, str]


def parse_args() -> ViewerConfig:
    """
    Streamlit passes args after `--`:
      streamlit run app.py -- --images-dir ... --labels-dir ...
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--images-dir", required=True, help="Folder with images (jpg/png/...)")
    ap.add_argument("--labels-dir", required=True, help="Folder with YOLOv8-seg .txt labels/predictions")
    ap.add_argument(
        "--classes",
        default="",
        help='Optional class names mapping, e.g. "0:tail,1:other"',
    )

    args, _unknown = ap.parse_known_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    labels_dir = Path(args.labels_dir).expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels-dir not found: {labels_dir}")

    class_names: Dict[int, str] = {}
    if args.classes.strip():
        for part in args.classes.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            k, v = part.split(":", 1)
            try:
                class_names[int(k.strip())] = v.strip()
            except Exception:
                pass

    return ViewerConfig(images_dir=images_dir, labels_dir=labels_dir, class_names=class_names)


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

    objects: List[Tuple[int, np.ndarray]] = []
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
    out = img_rgb.copy()
    overlay = img_rgb.copy()
    h, w = out.shape[:2]

    for cls, pts_norm in objects:
        pts_abs = norm_to_abs_points(pts_norm, w, h)
        poly = np.round(pts_abs).astype(np.int32).reshape((-1, 1, 2))

        # deterministic color from class id
        rng = np.random.default_rng(seed=cls + 12345)
        color = tuple(int(x) for x in rng.integers(40, 220, size=3))  # RGB

        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=line_thickness)

        name = class_names.get(cls, str(cls)) if class_names else str(cls)
        x0, y0 = int(poly[0, 0, 0]), int(poly[0, 0, 1])
        cv2.putText(out, name, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    out = cv2.addWeighted(overlay, float(fill_alpha), out, 1.0 - float(fill_alpha), 0.0)
    return out


def main():
    st.set_page_config(page_title="YOLO-seg Viewer", layout="wide")

    try:
        cfg = parse_args()
    except Exception as e:
        st.error(
            "Failed to parse CLI args.\n\n"
            "Run like:\n"
            "streamlit run yolo_seg_viewer.py -- --images-dir <...> --labels-dir <...> [--classes \"0:tail\"]\n\n"
            f"Error: {e}"
        )
        return

    st.title("YOLO segmentation viewer (polygons overlay)")
    st.caption(f"Images: {cfg.images_dir} • Labels: {cfg.labels_dir}")

    images = list_images(cfg.images_dir)
    if not images:
        st.warning(f"No images found in: {cfg.images_dir}")
        return

    with st.sidebar:
        st.header("View options")
        fill_alpha = st.slider("Fill alpha", 0.0, 0.9, 0.25, 0.05)
        thickness = st.slider("Line thickness", 1, 8, 2, 1)
        show_missing_only = st.checkbox("Show only images with missing label file", value=False)

        # filter list if needed
        if show_missing_only:
            images_filtered = []
            for p in images:
                lbl = cfg.labels_dir / (p.stem + ".txt")
                if not lbl.exists():
                    images_filtered.append(p)
            images = images_filtered or images

    file_names = [p.name for p in images]
    selected_name = st.selectbox("Image file", file_names, index=0)
    img_path = cfg.images_dir / selected_name
    lbl_path = cfg.labels_dir / (img_path.stem + ".txt")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        im = Image.open(img_path).convert("RGB")
        img_rgb = np.array(im, dtype=np.uint8)

        objects = parse_yolo_seg_file(lbl_path)

        drawn = draw_polygons(
            img_rgb=img_rgb,
            objects=objects,
            class_names=cfg.class_names,
            fill_alpha=float(fill_alpha),
            line_thickness=int(thickness),
        )

        st.image(drawn, caption=img_path.name, use_container_width=True)

    with col2:
        st.subheader("Paths")
        st.code(str(img_path), language="text")
        st.code(str(lbl_path), language="text")

        st.subheader("YOLO label content")
        if lbl_path.exists():
            content = lbl_path.read_text(encoding="utf-8")
            st.code(content if content.strip() else "(empty file)", language="text")
        else:
            st.warning("No label file found for this image.")

        st.subheader("Objects parsed")
        if objects:
            for i, (cls, pts) in enumerate(objects, start=1):
                name = cfg.class_names.get(cls, str(cls))
                st.write(f"#{i}: class={cls} ({name}) points={len(pts)}")
        else:
            st.write("No objects.")

    st.caption("Tip: labels must be normalized (0..1). Image and label filenames must share the same stem.")


if __name__ == "__main__":
    main()