#Determine the distance to people using the SSD Mobilenet model
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras
import jetson.inference
import jetson.utils

# Depth map default preset
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 130
TTH = 100
UR = 10
SR = 15
SPWS = 100

#Distance preset
distance = 0

def load_map_settings(file):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings, sbm
    print('Loading parameters from file...')
    f = open(file, 'r')
    data = json.load(f)
    #loading data from the json file and assigning it to the Variables
    SWS = data['SADWindowSize']
    PFS = data['preFilterSize']
    PFC = data['preFilterCap']
    MDS = data['minDisparity']
    NOD = data['numberOfDisparities']
    TTH = data['textureThreshold']
    UR = data['uniquenessRatio']
    SR = data['speckleRange']
    SPWS = data['speckleWindowSize']
    
    #changing the actual values of the variables
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=SWS) 
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print('Parameters loaded from file ' + file)

def stereo_depth_map(rectified_pair):
    #blockSize is the SAD Window Size

    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    image = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return disparity_color, disparity_normalized

#Determine distance to pixel by clicking mouse
def onMouse(event, x, y, flag, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        print("Distance in centimeters {}".format(distance))
        return distance

#Distance to person through object detection
def objectDetection(item):
    item_class = item.ClassID
    item_coords = item.Center
    x_coord = int(item_coords[0])
    y_coord = int(item_coords[1])
    distance = disparity_normalized[y_coord][x_coord]

    #to avoid detection of different objects, we only focus on people which have a ClassID of 1
    if item_class == 1:
        print("Person is: {}cm away".format(distance))

#Object Detection model 
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

#net = jetson.inference.detectNet(argv=["--model=/home/aryan/StereoVision/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff",
#"--labels=/home/aryan/StereoVision/SSD-Mobilenet-v2/ssd_coco_labels.txt", 
#"--input-blob=Input", "--output-cvg=NMS", "--output-bbox=NMS_1"], threshold=0.5)



if __name__ == "__main__":
    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()
    load_map_settings("../3dmap_set.txt")

    cv2.namedWindow("DepthMap")

    while True:
        left_grabbed, left_frame = left_camera.read()
        right_grabbed, right_frame = right_camera.read()

        if left_grabbed and right_grabbed:  
            #Convert BGR to Grayscale     
            left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            #calling all calibration results
            calibration = StereoCalibration(input_folder='calib_result')
            rectified_pair = calibration.rectify((left_gray_frame, right_gray_frame))
            disparity_color, disparity_normalized = stereo_depth_map(rectified_pair)

            #Mouse clicked function
            cv2.setMouseCallback("DepthMap", onMouse, disparity_normalized)
           
            #Object detection & distance
            left_cuda_frame = jetson.utils.cudaFromNumpy(left_frame)
            detections = net.Detect(left_cuda_frame)
            if len(detections):
                for item in detections:
                    objectDetection(item)


            
            left_stacked = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
            cv2.imshow("DepthMap", np.hstack((disparity_color, left_stacked)))


            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

            else:
                continue

    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    cv2.destroyAllWindows()
                


    


