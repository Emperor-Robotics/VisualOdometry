#!/bin/env python
import cv2
import sys
import numpy as np
from threading import Thread
from ..common.IMU import IMU
from ..common.ThreadedCamera import ThreadedCamera
import time
# https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
# https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(
            "Please run with arguments: [left_cam index] [left_cam calibration] [cam_right index] [cam_right calibration]")
        exit(1)
    cam_left_index = sys.argv[1]
    cam_left_calibration_file = sys.argv[2]
    cam_right_index = sys.argv[3]
    cam_right_calibration_file = sys.argv[4]

    print(f"Capturing on cameras: L:{cam_left_index} R:{cam_right_index}")

    # If you want 720P, they need to go into different USB ports
    width_val = 420  # minimum for laptop hub
    height_val = 240  # minimum for laptop hub
    camL = ThreadedCamera(int(cam_left_index), width_val, height_val)
    camR = ThreadedCamera(int(cam_right_index), width_val, height_val)
    # maps for stereo image rectification?

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)

    # Sleep so that the threads have time to spin up.
    time.sleep(1)
    while True:
        retL, imgL = camL.get_current_frame()
        retR, imgR = camR.get_current_frame()
        if not retL and not retR:
            print(
                f"Issue recovering both frames. RL: {retL} RR: {retR} Might be a bandwith problem... Try different USB buses")
            exit(1)

        # Post process to correct orientation for display given current rig
        imgL_corrected = np.rot90(imgL, k=-1)
        imgL_corrected = np.flipud(imgL_corrected)
        imgR_corrected = np.rot90(imgR, k=-1)
        imgR_corrected = np.flipud(imgR_corrected)

        imgL_corrected_gray = cv2.cvtColor(imgL_corrected, cv2.COLOR_BGR2GRAY)
        imgR_corrected_gray = cv2.cvtColor(imgR_corrected, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(imgL_corrected_gray, imgR_corrected_gray)
        full_img = np.hstack((imgL_corrected, imgR_corrected))
        cv2.waitKey(30)
        cv2.imshow('disp', full_img)
        cv2.imshow('disparity', disparity)
