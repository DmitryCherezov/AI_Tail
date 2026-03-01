import os
from pathlib import Path
from PIL import Image

import data_utils

def convert_images_to_png(
        input_dir: str, 
        output_dir: str
        ):

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp", ".avif"
    }

    for file_path in input_path.rglob("*"):
        if file_path.suffix.lower() in supported_extensions:
            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGBA")

                    relative_path = file_path.relative_to(input_path)
                    output_file = output_path / relative_path.with_suffix(".png")

                    
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    img.save(output_file, format="PNG")

                    #print(f"[OK] {file_path.name} -> {output_file}")

            except Exception as e:
                print(f"[ERROR] {file_path} : {e}")


def main():
    input_directory = data_utils.RAW_AUGMENTATION_IMAGES
    output_directory = data_utils.PNG_AUGMENTATION_IMAGES

    convert_images_to_png(input_directory, output_directory)


if __name__ == "__main__":
    main()







