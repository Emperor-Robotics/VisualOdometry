#!/bin/env python
import cv2
import sys
import numpy as np
from IMU import IMU

# https://learnopencv.com/depth-perception-using-stereo-camera-python-c/

if len(sys.argv) < 5:
    print(
        "Please run with arguments: [left_cam index] [left_cam calibration] [cam_right index] [cam_right calibration]")
    exit(1)
cam_left_index = sys.argv[1]
cam_left_calibration_file = sys.argv[2]
cam_right_index = sys.argv[3]
cam_right_calibration_file = sys.argv[4]

print(f"Capturing cameras: L:{cam_left_index} R:{cam_right_index}")

camL = cv2.VideoCapture(int(cam_left_index))
camR = cv2.VideoCapture(int(cam_right_index))
# maps for stereo image rectification?


# stero = cv2.StereoBM_create()
while True:
    retL, imgL = camL.read()
    retR, imgR = camR.read()
    # print(f"L: {retL} R:{retR}")
    if not retL and not retR:
        print(
            f"Issue recovering both frames. RL: {retL} RR: {retR} Might be a bandwith problem...")
        exit(1)

    # Post process to correct orientation for display given current rig
    imgL_corrected = np.rot90(imgL, k=-1)
    imgL_corrected = np.flipud(imgL_corrected)
    imgR_corrected = np.rot90(imgR, k=-1)
    imgR_corrected = np.flipud(imgR_corrected)

    full_img = np.hstack((imgL_corrected, imgR_corrected))
    cv2.waitKey(30)
    cv2.imshow('disp', full_img)

# if __name__ == '__main__':
