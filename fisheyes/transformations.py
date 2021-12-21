import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

def train_transform():
    print("####################################")
    print("Apply transformation on train dataset")
    print("####################################")
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    return transform


def val_transform():
    print("####################################")
    print("Apply transformation on valid dataset")
    print("####################################")
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    return transform

