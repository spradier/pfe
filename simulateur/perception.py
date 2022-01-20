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
import time
import tensorflow as tf
import base64
import setup_path
import airsim
import tempfile
import pprint


#############################################
#############################################
##### Setup Airsim Drone #######
#############################################
#############################################

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)


#############################################
#############################################
##### Classificaton, Detection, Depth #######
#############################################
#############################################

# load our trained bounding box regressor
bb_model = load_model("detector.h5")
# load our depth model
depth_model = cv2.dnn.readNet("depth.onnx")
# load our classification model
class_model = load_model("classfication.h5")

# Set backend and target to CUDA to use GPU
depth_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
depth_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#############################################
#############################################
##### Streaming #######
#############################################
#############################################


airsim.wait_key('Press any key to start simulation')
while 1:

    # get camera images from the car
    img = client.simGetImages([
        airsim.ImageRequest("1", airsim.ImageType.Scene)
        ])
    
    jpg_as_np = np.frombuffer(img[0].image_data_uint8, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    print(img)

    # Read in the image
    imgHeight, imgWidth, channels = img.shape
    img_depth = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resize = cv2.resize(img,(224,224))
    img_tensor = img_resize.reshape(224,224,3)
    image_array = img_to_array(img_tensor) / 255.0
    image_dims = np.expand_dims(image_array, axis=0)

    #################################
    image = cv2.resize(img,(160,160))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    #prediction to know if they are a drone or not
    predictions = class_model.predict(image).flatten()
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    print('Predictions:\n', predictions.numpy())
    
    #if a drone is detected the prediction is 0
    #so we can call our regression model
    if predictions.numpy()==0:
        preds = bb_model.predict(image_dims)[0]
        (startX, startY, endX, endY) = preds

        # load the input image (in OpenCV format), resize it such that it
        # fits on our screen, and grab its dimensions
        image = imutils.resize(img, width=600)
        h = image.shape[0]
        w = image.shape[1]

        # scale the predicted bounding box coordinates based on the image
        # dimensions
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)
        
        # draw the predicted bounding box on the image
        cv2.rectangle(img, (startX, startY), (endX, endY),
            (0,255,0), 2)

    ######### for depth ############

    blob = cv2.dnn.blobFromImage(img_depth, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)
    # Set input to the model
    depth_model.setInput(blob)
    output = depth_model.forward()
    
    output = output[0,:,:]
    output = cv2.resize(output, (imgWidth, imgHeight))

    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    end = time.time()

    cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('depth', 300,300)
    cv2.imshow('depth', output)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 300,300)
    cv2.imshow('image', img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()