import pandas as pd
from torch.utils.data import Dataset
import glob
from PIL import Image, ImageDraw
import numpy as np
import torch

stickers = ["structure", "person", "grouped_pedestrian_and_animals", "construction"]
classes = [0, 1, 2, 3]

class MyDataset(Dataset):  
    def __init__(self, train=True, transform=None):
            print("####################################")
            print("Initalization")
            print("####################################")
            if train == True:
                self.img_labels = sorted(glob.glob("targets/*.jpg"))[:6587]
                self.img_dir = sorted(glob.glob("rgb_images/*.png"))[:6587]
                self.transform = transform
            else:
                self.img_labels = sorted(glob.glob("targets/*.jpg"))[6587:]
                self.img_dir = sorted(glob.glob("rgb_images/*.png"))[6587:]
                self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        mask_path = self.img_labels[idx]

        image = Image.open(img_path)

        targets = {}
        mask = pd.read_json(mask_path)
        detections = mask[mask_path[12:]]["annotation"]

        boxes = []
        labels = []
        masks = []

        for detection in detections:
            if detection["tags"][0] in stickers:
                x_min, x_max = 1500, 0
                y_min, y_max = 1500, 0
                for coords in detection["segmentation"]:
                    if coords[0] > x_max:
                        x_max = coords[0]
                    elif coords[0] < x_min:
                        x_min = coords[0]
                    elif coords[1] > y_max:
                        y_max = coords[1]
                    elif coords[1] < y_min:
                        y_min = coords[1]
                    
                masks.append(detection["segmentation"])
                labels.append(stickers.index(detection["tags"]))
                boxes.append([x_min, y_min, x_max, y_max])

        image_id = torch.tensor([idx])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.tensor(labels, dtype=torch.uint8)

        targets["boxes"] = boxes
        targets["labels"] = labels
        targets["masks"] = masks
        targets["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(image, targets)
                
            
        return image, targets

    def __len__(self):
        return len(self.img_labels)
