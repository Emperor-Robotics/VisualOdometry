#!/bin/env python
import cv2
import sys
import numpy as np
from threading import Thread
from emperorviopy.common.IMU import IMU
from emperorviopy.common.CameraModel import PinholeCamera
from emperorviopy.common.ThreadedCamera import ThreadedCamera
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
    # width_val = 420  # minimum for same laptop hub
    # height_val = 240  # minimum for same laptop hub
    # PersonalRig: Plug B into right side, hub into left
    width_val = 800
    height_val = 600
    camL = ThreadedCamera(int(cam_left_index), width_val, height_val)
    camLP = PinholeCamera.loadFromOST(cam_left_calibration_file)
    camR = ThreadedCamera(int(cam_right_index), width_val, height_val)
    camRP = PinholeCamera.loadFromOST(cam_right_calibration_file)
    # maps for stereo image rectification?

    # These were obtained from dual_gui.py, make this into something that can be serailized and deserialized.
    stereo = cv2.StereoBMGM_create()
    numDisparities = 5 * 16
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(1 * 2 + 5)
    # stereo.setPreFilterType(1)
    # stereo.setPreFilterSize(2 * 2 + 5)
    # stereo.setPreFilterCap(5)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(0)
    stereo.setSpeckleRange(5)
    stereo.setSpeckleWindowSize(0 * 2)
    stereo.setDisp12MaxDiff(25)
    minDisparity = 9
    stereo.setMinDisparity(minDisparity)

    # Sleep so that the threads have time to spin up.
    time.sleep(2)
    while True:
        retL, imgL = camL.get_current_frame()
        retR, imgR = camR.get_current_frame()
        if not retL and not retR:
            print(
                f"Issue recovering both frames. RL: {retL} RR: {retR} Might be a bandwith problem... Try different USB buses")
            exit(1)

        # undisort
        imgL = cv2.undistort(imgL, camLP.camera_matrix, camLP.d)
        imgR = cv2.undistort(imgR, camRP.camera_matrix, camRP.d)

        # Post process to correct orientation for display given current rig
        imgL_corrected = np.rot90(imgL, k=-1)
        imgL_corrected = np.flipud(imgL_corrected)
        imgL_corrected = np.fliplr(imgL_corrected)
        imgR_corrected = np.rot90(imgR, k=-1)
        imgR_corrected = np.flipud(imgR_corrected)
        imgR_corrected = np.fliplr(imgR_corrected)

        imgL_corrected_gray = cv2.cvtColor(imgL_corrected, cv2.COLOR_BGR2GRAY)
        imgR_corrected_gray = cv2.cvtColor(imgR_corrected, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(imgL_corrected_gray, imgR_corrected_gray)

        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity/16.0 - minDisparity)/numDisparities
        full_img = np.hstack((imgL_corrected_gray, imgR_corrected_gray))
        print(full_img.shape)
        full_img_w_disparity = np.hstack((full_img, disparity))
        cv2.waitKey(30)
        cv2.imshow('disp', full_img)
        cv2.imshow('disparity', disparity)
        # cv2.imshow('disp', full_img_w_disparity)
