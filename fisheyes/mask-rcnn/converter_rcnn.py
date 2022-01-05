from PIL import Image, ImageDraw
import glob
import pandas
import numpy as np
import random

images = sorted(glob.glob("rgb_images/*.png"))
labels = sorted(glob.glob("annotations/*.json"))
print(images[0], labels[0])

stickers = ["void", "structure", "person", "grouped_pedestrian_and_animals", "construction"]




for i in range(len(images)):
    image = Image.open(images[i])
    img_mask = image.copy()
    draw = ImageDraw.Draw(img_mask)


    pd = pandas.read_json(labels[i])
    filename = labels[i].split('/')[1][:-5] + ".json"
    annotations = pd[filename]["annotation"]

    for detection in annotations:
        print(detection["tags"])
        if detection["tags"][0] in stickers:
            color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])  
            mask = detection["segmentation"]
            for j in range(len(mask)):
                mask[j] = tuple(mask[j])
            draw.polygon(mask, fill = color)
        else:
            print(filename, " : ", detection["tags"][0])
            color = "black"
            mask = detection["segmentation"]
            for j in range(len(mask)):
                mask[j] = tuple(mask[j])
            draw.polygon(mask, fill = color)

    
    fp = "targets_rcnn/" + images[i][10:-4] + ".jpg"
    img_mask.save(fp, "JPEG")
