import pandas
from torch.utils.data import Dataset
import glob
from PIL import Image, ImageDraw
import numpy as np


colors = {
    "road_surface": "#696969",
    "curb": "#a9a9a9",
    "car": "#556b2f",
    "train/tram": "#8b4513",
    "truck": "#2e8b57",
    "other_wheeled_transport": "#228b22",
    "trailer": "#191970",
    "van": "#8b0000",
    "caravan": "#b8860b",
    "bus": "#008b8b",
    "bicycle": "#4682b4",
    "motorcycle": "#d2691e",
    "person": "#9acd32",
    "rider": "#00008b",
    "grouped_botts_dots": "#32cd32",
    "cats_eyes_and_botts_dots": "#7f007f",
    "parking_marking": "#8fbc8f",
    "lane_marking": "#9932cc",
    "parking_line": "#ff4500",
    "other_ground_marking": "#00ced1",
    "zebra_crossing": "#ffa500",
    "trafficsign_indistingushable": "#6a5acd",
    "sky": "#0000cd",
    "fence": "#00ff00",
    "traffic_light_yellow": "#00fa9a",
    "ego_vehicle": "#dc143c",
    "pole": "#00bfff",
    "structure": "#f4a460",
    "traffic_sign": "#adff2f",
    "animal": "#da70d6",
    "free_space": "#ff00ff",
    "traffic_light_red": "#1e90ff",
    "unknown_traffic_light": "#db7093",
    "movable_object": "#f0e68c",
    "traffic_light_green": "#fa8072",
    "void": "#ffff54",
    "grouped_vehicles": "#dda0dd",
    "grouped_pedestrian_and_animals": "#90ee90",
    "grouped_animals": "#ff1493",
    "green_strip": "#afeeee",
    "nature": "#7fffd4",
    "construction": "#ffe4c4",
    "Other_NoSight": "#ffc0cb"
}

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
        mask = Image.open(mask_path)
            
        image = np.asarray(image)
        image = image[0:-6]
        mask = np.asarray(mask)
        mask = mask[0:-6]
        print("a", mask.shape)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["image"]
            
        return image, mask
