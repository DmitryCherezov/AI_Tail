from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import numpy as np
#from PIL import Image
from ultralytics import YOLO


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    )


def save_yolo_seg_txt(
    txt_path: Path,
    masks_xy: List[np.ndarray],
    classes: np.ndarray,
    w: int,
    h: int,
):
    """
    Saves YOLOv8 segmentation format:
    class x1 y1 x2 y2 ...
    coordinates normalized to [0,1]
    """
    lines = []

    for cls, poly in zip(classes, masks_xy):
        # poly: (N,2) absolute pixel coords
        coords = []
        for x, y in poly:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")

        line = f"{int(cls)} " + " ".join(coords)
        lines.append(line)

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="YOLOv11n-seg inference → YOLO txt output")
    ap.add_argument("--images", required=True, help="Folder with input images")
    ap.add_argument("--weights", required=True, help="Path to saved project")
    ap.add_argument("--out", required=True, help="Output folder for YOLO txt files")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default='cuda')

    args = ap.parse_args()

    images_dir = Path(args.images).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    weights_path = weights_path / 'train' / 'weights' / 'best.pt'

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))  # loads architecture

    #state_dict = torch.load(weights_path, map_location="cpu")
    #model.model.load_state_dict(state_dict)

    image_paths = list_images(images_dir)
    if not image_paths:
        raise RuntimeError("No images found.")

    print(f"Found {len(image_paths)} images")

    results = model.predict(
        source=str(images_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=False,
        stream=True,
        verbose=False,
    )

    for result in results:
        img_path = Path(result.path)
        txt_path = out_dir / (img_path.stem + ".txt")

        if result.masks is None:
            txt_path.write_text("", encoding="utf-8")
            continue

        masks_xy = result.masks.xy  # list of polygons (absolute coords)
        classes = result.boxes.cls.cpu().numpy()

        # image size
        h, w = result.orig_shape

        save_yolo_seg_txt(txt_path, masks_xy, classes, w, h)

        print(f"[OK] {img_path.name}")

    print("Inference completed.")


if __name__ == "__main__":
    main()














