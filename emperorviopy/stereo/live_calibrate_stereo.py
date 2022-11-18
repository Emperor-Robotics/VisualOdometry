#!/bin/env python
import cv2
import sys
import numpy as np
from threading import Thread
from emperorviopy.common.IMU import IMU
from emperorviopy.common.CameraModel import PinholeCamera
from emperorviopy.common.ThreadedCamera import ThreadedCamera
import time

numBoards = 30  # number of images for calibration
board = (7, 10)  # row cols
squareSize = 2.5  # standard = 1 small chessboard = 2.5 large chessboard = 4.4


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
    width_val = None
    height_val = None
    camLP = PinholeCamera.loadFromOST(cam_left_calibration_file)
    camL = ThreadedCamera(int(cam_left_index), width_val,
                          height_val, camera_model=camLP)
    camRP = PinholeCamera.loadFromOST(cam_right_calibration_file)
    camR = ThreadedCamera(int(cam_right_index), width_val,
                          height_val, camera_model=camRP)
    # maps for stereo image rectification?
    # Sleep so that the threads have time to spin up.
    COOLDOWN = 0
    time.sleep(2)
    while numBoards > 0:
        retL, imgL = camL.get_current_frame_undistorted()
        retR, imgR = camR.get_current_frame_undistorted()
        if not retL and not retR:
            print(
                f"Issue recovering both frames. RL: {retL} RR: {retR} Might be a bandwith problem... Try different USB buses")
            exit(1)
        # Post process to correct orientation for display given current rig
        imgL_corrected = np.rot90(imgL, k=-1)
        imgL_corrected = np.flipud(imgL_corrected)
        imgL_corrected = np.fliplr(imgL_corrected)
        imgR_corrected = np.rot90(imgR, k=-1)
        imgR_corrected = np.flipud(imgR_corrected)
        imgR_corrected = np.fliplr(imgR_corrected)

        imgL_corrected_gray = cv2.cvtColor(imgL_corrected, cv2.COLOR_BGR2GRAY)
        imgR_corrected_gray = cv2.cvtColor(imgR_corrected, cv2.COLOR_BGR2GRAY)

        # Get chessboards
        foundL = False
        foundR = False

        foundL, cornersL = cv2.findChessboardCorners(
            imgL_corrected_gray, board, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        foundR, cornersR = cv2.findChessboardCorners(
            imgR_corrected_gray, board, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if foundL:
            imgL_corrected = cv2.drawChessboardCorners(
                np.ascontiguousarray(imgL_corrected, dtype=np.uint8), board, cornersL, foundL)
        if foundR:
            imgR_corrected = cv2.drawChessboardCorners(
                np.ascontiguousarray(imgR_corrected, dtype=np.uint8), board, cornersR, foundR)

        if foundL and foundR and time.time() > COOLDOWN:
            # save chessboards
            numBoards -= 1
            COOLDOWN = time.time() + 5
            print("L-R Image saved saved! Cooldown set!")

            # decrement numBoards

        full_img = np.hstack((imgL_corrected, imgR_corrected))
        cv2.waitKey(30)
        cv2.imshow('disp', full_img)
