from init import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os
import glob

# load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)

cap = jetson.utils.videoSource("csi://0")

while cap.isOpened():
    ret, image = cap.read()

    # make bounding box predictions on the input image
    if model.predict(image):
        
        preds = model.predict(image)[0]
        (startX, startY, endX, endY) = preds

        #image = imutils.resize(image, width=600)
        h = image.shape[0]
        w = image.shape[1]

        # scale the predicted bounding box coordinates based on the image
        # dimensions
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        # draw the predicted bounding box on the image
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", image)
    key = cv2.waitKey(0)
    if key%256 == 27:
        break

cap.release()
cv2.destroyAllWindows()
