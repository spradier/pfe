# import the necessary packages
import os

# image drone directory
# annotation directory 
BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "drone"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotation.csv"])

# output directory
BASE_OUTPUT = "output"

# model path 
# plot path 
# 
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32