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
import time
import tensorflow as tf

# load our trained bounding box regressor
model_drone = load_model("output/prediction.h5")
# load our depth model
model = cv2.dnn.readNet("output/model-f6b98070.onnx")
# load our classification model
model_class = load_model("output/classfication.h5")

# Set backend and target to CUDA to use GPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#############################################
#############################################
###### Classificaton, Detection, Depth ######
#############################################
#############################################

#to open your webcam video 
cap = cv2.VideoCapture(0)

while cap.isOpened():

    # Read in the image
    success, img = cap.read()

    imgHeight, imgWidth, channels = img.shape
    img_depth = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resize = cv2.resize(img,(224,224))
    img_tensor = img_resize.reshape(224,224,3)
    image_array = img_to_array(img_tensor) / 255.0
    image_dims = np.expand_dims(image_array, axis=0)

    ######## for detection #########
    image = cv2.resize(img,(160,160))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    #prediction to know if they are a drone or not
    predictions = model_class.predict(image).flatten()
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    print('Predictions:\n', predictions.numpy())
    
    #if a drone is detected the prediction is 0
    #so we can call our regression model
    if predictions.numpy()==0:
        preds = model_drone.predict(image_dims)[0]
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
    model.setInput(blob)
    output = model.forward()
    
    output = output[0,:,:]
    output = cv2.resize(output, (imgWidth, imgHeight))

    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    end = time.time()

    cv2.imshow('Depth Map', output)
    cv2.imshow('image', img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()