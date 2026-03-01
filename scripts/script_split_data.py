import random
import shutil
from pathlib import Path

import data_utils


SEED = 42
TRAIN_SIZE = 120
VAL_SIZE = 30

CLASS_NAMES = {0: "tail"}  # class_id -> name
YAML_FILENAME = "data.yaml"


def ensure_dirs() -> None:
    for p in [
        data_utils.DATASET_TRAIN_IMG,
        data_utils.DATASET_TRAIN_LBL,
        data_utils.DATASET_VAL_IMG,
        data_utils.DATASET_VAL_LBL,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def write_data_yaml(dataset_root: Path) -> Path:
    """
    Generates Ultralytics YOLO data.yaml under dataset_root.

    Expected dataset structure:
      dataset_root/
        images/train
        images/val
        labels/train
        labels/val
    """
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Ultralytics data.yaml expects either:
    # - path + relative train/val
    # or absolute train/val paths.
    # We'll use: path: <abs>, train: images/train, val: images/val
    names_lines = "\n".join([f"  {k}: {v}" for k, v in sorted(CLASS_NAMES.items())])

    yaml_text = (
        f"path: {dataset_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"{names_lines}\n"
    )

    yaml_path = dataset_root / YAML_FILENAME
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main() -> None:
    random.seed(SEED)
    ensure_dirs()

    png_files = sorted(data_utils.PNG_IMAGES.glob("*.png"))
    if not png_files:
        raise RuntimeError(f"No PNG images found in: {data_utils.PNG_IMAGES}")

    random.shuffle(png_files)

    train_files = png_files[:TRAIN_SIZE]
    val_files = png_files[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    print(f"Total images found: {len(png_files)}")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Ignored: {max(0, len(png_files) - (TRAIN_SIZE + VAL_SIZE))}")

    # --- TRAIN ---
    for img_path in train_files:
        label_path = data_utils.LABELS / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"[WARN] Missing label for train image: {img_path.name}")
            continue

        shutil.copy2(img_path, data_utils.DATASET_TRAIN_IMG / img_path.name)
        shutil.copy2(label_path, data_utils.DATASET_TRAIN_LBL / label_path.name)

    # --- VAL ---
    for img_path in val_files:
        label_path = data_utils.LABELS / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"[WARN] Missing label for val image: {img_path.name}")
            continue

        shutil.copy2(img_path, data_utils.DATASET_VAL_IMG / img_path.name)
        shutil.copy2(label_path, data_utils.DATASET_VAL_LBL / label_path.name)

    yaml_path = write_data_yaml(data_utils.DATASET_PATH)

    print("Dataset split completed.")
    print(f"[OUT] data.yaml: {yaml_path.resolve()}")


if __name__ == "__main__":
    main()