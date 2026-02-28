from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

import data_utils




IMAGES_DIR = data_utils.PNG_IMAGES
ANN_DIR    = data_utils.ANNOTATIONS
LABELS_OUT = data_utils.LABELS

TARGET_LABEL = data_utils.TARGET_LABEL   # как ты называл хвост в LabelMe
CLASS_ID = data_utils.CLASS_ID



def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def convert_one(
        json_path: Path, 
        img_path: Path
        ) -> list[str]:
    

    with json_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    with Image.open(img_path) as im:
        w, h = im.size

    lines: list[str] = []
    shapes = ann.get("shapes", [])

    for sh in shapes:
        label = sh.get("label", "")
        shape_type = sh.get("shape_type", "")
    
        if label != TARGET_LABEL:
            continue
        if shape_type != "polygon":
            continue
    
        points = sh.get("points", [])
        if len(points) < 3:
            continue
    
        coords = []
        for x, y in points:
            xn = clamp01(float(x) / w)
            yn = clamp01(float(y) / h)
            coords.append(f"{xn:.6f}")
            coords.append(f"{yn:.6f}")
    
        lines.append(f"{CLASS_ID} " + " ".join(coords))

    return lines


def main():

    pngs = sorted(IMAGES_DIR.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No PNG files found in: {IMAGES_DIR}")

    missing_json = 0
    written = 0

    for img_path in pngs:
        json_path = ANN_DIR / (img_path.stem + ".json")
        if not json_path.exists():
            missing_json += 1
            print(f"[WARN] No JSON for {img_path.name}: expected {json_path.name}")
            continue

        lines = convert_one(json_path, img_path)
        out_txt = LABELS_OUT / (img_path.stem + ".txt")

        # В YOLO обычно если нет объектов — кладут пустой .txt (это нормально)
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        written += 1

    print(f"[DONE] Processed images: {len(pngs)}")
    print(f"[DONE] Written label files: {written}")
    print(f"[DONE] Missing JSON files: {missing_json}")
    print(f"[OUT] Labels folder: {LABELS_OUT.resolve()}")


if __name__ == "__main__":
    main()



