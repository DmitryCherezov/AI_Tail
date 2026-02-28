from pathlib import Path

###############################################################################
#   PATHS
###############################################################################

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# path to data folder
DATA_ROOT = PROJECT_ROOT / "data"

RAW_IMAGES = DATA_ROOT / "raw_images"
PNG_IMAGES = DATA_ROOT / "png_images"
ANNOTATIONS = DATA_ROOT / "annotations"
LABELS = DATA_ROOT / "validation_prediction"

DATASET_PATH = DATA_ROOT / 'dataset'


DATASET_TRAIN_IMG = DATASET_PATH / 'images' / 'train'
DATASET_TRAIN_LBL = DATASET_PATH / 'labels' / 'train'

DATASET_VAL_IMG = DATASET_PATH / 'images' / 'val'
DATASET_VAL_LBL = DATASET_PATH / 'labels' / 'val'

RAW_AUGMENTATION_IMAGES = DATA_ROOT / "raw_augmentations"
PNG_AUGMENTATION_IMAGES = DATA_ROOT / "png_augmentations"




###############################################################################
#   LABELS
###############################################################################

TARGET_LABEL = "tail"  
CLASS_ID = 0           


###############################################################################
#   INITIALIZATION
###############################################################################


for path in [
        PNG_IMAGES, ANNOTATIONS, LABELS,
        DATASET_TRAIN_IMG, DATASET_TRAIN_LBL, 
        DATASET_VAL_IMG, DATASET_VAL_LBL,
        RAW_AUGMENTATION_IMAGES, PNG_AUGMENTATION_IMAGES
        ]:
    path.mkdir(parents=True, exist_ok=True)
