from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO


DEFAULT_WEIGHTS_NAME = "yolo11n-seg.pt"


def ensure_source_weights(
        models_dir: Path, 
        weights_name: str = DEFAULT_WEIGHTS_NAME
        ) -> Path:
    """
    Ensures pretrained weights exist in:
        models/source/<weights_name>

    If missing, Ultralytics will download them on first load (to its cache),
    then we copy the resolved ckpt_path into models/source/.
    """
    source_dir = models_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    dst = source_dir / weights_name
    if dst.exists():
        return dst

    # Trigger Ultralytics download (to SETTINGS["weights_dir"] or similar cache)
    model = YOLO(weights_name)  # e.g. "yolo11n-seg.pt"
    ckpt_path = Path(getattr(model, "ckpt_path", ""))

    if not ckpt_path.exists():
        raise RuntimeError(
            f"Ultralytics loaded model but ckpt_path not found: {ckpt_path}. "
            f"Try placing {weights_name} manually into: {dst}"
        )

    shutil.copy2(ckpt_path, dst)
    return dst


def resolve_data_arg(data_arg: str) -> str:
    """
    Accepts either:
      - path to data.yaml
      - path to dataset root (then uses <root>/data.yaml)
    """
    p = Path(data_arg).expanduser().resolve()
    if p.is_dir():
        yaml_path = p / "data.yaml"
    else:
        yaml_path = p

    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    return str(yaml_path)


def extract_map5095(metrics_obj) -> float:
    """
    Tries to extract mAP50-95 for segmentation.
    Fallbacks for different Ultralytics versions.
    """
    # Preferred for segmentation
    if hasattr(metrics_obj, "seg") and metrics_obj.seg is not None:
        if hasattr(metrics_obj.seg, "map"):
            return float(metrics_obj.seg.map)

    # Sometimes people use box metrics; keep as fallback
    if hasattr(metrics_obj, "box") and metrics_obj.box is not None:
        if hasattr(metrics_obj.box, "map"):
            return float(metrics_obj.box.map)

    # Last resort: try common dict-like interfaces
    if hasattr(metrics_obj, "results_dict"):
        d = metrics_obj.results_dict
        # keys may differ by version; try typical ones
        for k in [
            "metrics/mAP50-95(M)",  # masks
            "metrics/mAP50-95(mask)",
            "metrics/mAP50-95",
            "metrics/mAP50-95(B)",  # boxes
        ]:
            if k in d:
                return float(d[k])

    raise RuntimeError("Could not extract mAP50-95 from metrics object.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train YOLOv11n-seg (tails) and save ONLY state_dict.")
    ap.add_argument("--data", required=True, help="Path to data.yaml OR dataset root containing data.yaml")
    ap.add_argument("--imgsz", type=int, default=640, help="Train/val image size (square).")
    ap.add_argument("--batch", type=int, default=16, help="Batch size.")
    ap.add_argument("--epochs", type=int, default=5, help="Epochs.")
    ap.add_argument("--degrees", type=float, default=10.0, help="Random rotation degrees.")
    ap.add_argument("--translate", type=float, default=0.05, help="Random translation fraction.")
    ap.add_argument("--scale", type=float, default=0.20, help="Random scale fraction.")
    ap.add_argument("--shear", type=float, default=0.0, help="Random shear degrees.")
    ap.add_argument("--perspective", type=float, default=0.0, help="Random perspective fraction.")
    
    ap.add_argument("--hsv_h", type=float, default=0.015, help="HSV-H augmentation.")
    ap.add_argument("--hsv_s", type=float, default=0.7, help="HSV-S augmentation.")
    ap.add_argument("--hsv_v", type=float, default=0.4, help="HSV-V augmentation.")
    
    ap.add_argument("--fliplr", type=float, default=0.5, help="Left-right flip probability.")
    ap.add_argument("--flipud", type=float, default=0.0, help="Up-down flip probability.")
    
    # Для маленького датасета сегментации часто лучше почти выключить mosaic/mixup
    ap.add_argument("--mosaic", type=float, default=0.0, help="Mosaic probability.")
    ap.add_argument("--mixup", type=float, default=0.0, help="MixUp probability.")
    ap.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze first N layers ",
    )
    ap.add_argument(
        "--save",
        required=True,
        help="Directory where Ultralytics will write runs (segment/train/weights/best.pt, last.pt)",
    )
    ap.add_argument(
        "--models-dir",
        default="models",
        help="Project models directory containing 'source/'. Default: ./models",
    )
    ap.add_argument("--device", default=None, help="e.g. '0' or 'cpu'. Default: auto.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = ap.parse_args()

    data_yaml = resolve_data_arg(args.data)

    models_dir = Path(args.models_dir).expanduser().resolve()
    weights_path = ensure_source_weights(models_dir=models_dir, weights_name=DEFAULT_WEIGHTS_NAME)

    project_dir = Path(args.save).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # Keep Ultralytics outputs out of your project: use a temporary project directory
    #tmp_project = Path(tempfile.mkdtemp(prefix="ultra_train_"))

    try:
        model = YOLO(str(weights_path))

        # Train (no saving of checkpoints/plots)
        model.train(
        data=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        freeze=args.freeze,
        seed=args.seed,
        device=args.device,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        save=True,      # ВАЖНО
        plots=False,
        verbose=True,
    
        # augmentations
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=args.mixup,
        )

        # Validate to report mAP50-95 on val
        metrics = model.val(
            data=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            split="val",
            device=args.device,
            project=str(project_dir),
            name="val",
            exist_ok=True,
            save=False,
            plots=False,
            verbose=False,
        )

        map5095 = extract_map5095(metrics)
        print(f"[METRIC] val mAP50-95 = {map5095:.6f}")

        # Save ONLY state_dict (direct state)
        # model.model is the underlying torch.nn.Module
        #torch.save(model.model.state_dict(), save_path)
        #print(f"[SAVED] state_dict -> {save_path}")

    finally:
        # Remove all Ultralytics run artifacts
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())