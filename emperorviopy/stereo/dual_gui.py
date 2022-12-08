#!/bin/env python
import cv2
import sys
import numpy as np
from threading import Thread
from emperorviopy.common.IMU import IMU
from emperorviopy.common.CameraModel import PinholeCamera
from emperorviopy.common.ThreadedCamera import ThreadedCamera, ThreadedStereo
import time
import os
from stereovision.calibration import StereoCalibrator, StereoCalibration
from glob import glob
import yaml

translation_of_cam2 = np.array([-74.9112, -1.1862, -3.7268])
rotation_of_cam2 = np.array([
    [.9989, -0.0040, -0.0467],
    [0.0041, 1.0000, 0.0015],
    [0.0467, -0.0017, 0.9989]])


def nothing(x):
    pass


def runGui(stereo_cameras: ThreadedStereo) -> None:
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 600)

    cv2.namedWindow('disp filtered', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp filtered', 600, 600)

    cv2.createTrackbar('numDisparities', 'disp', 16, 17, nothing)
    cv2.createTrackbar('blockSize', 'disp', 1, 50, nothing)
    # cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
    # cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
    # cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
    # cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 0, 100, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 5, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 0, 25, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 25, 25, nothing)
    cv2.createTrackbar('minDisparity', 'disp', 9, 25, nothing)

    cv2.createTrackbar('sigma', 'disp filtered', 3, 10, nothing)
    cv2.createTrackbar('lambda', 'disp filtered', 80, 100, nothing)

    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoSGBM_create()

    # DisparityFilter
    wsize = 31
    max_disp = 128
    sigma = 3
    lmbda = 8000.0
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)

    # Remaps
    flags = cv2.CALIB_ZERO_DISPARITY
    image_size = (600, 800)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(stereo_cameras.camera_left.camera_model.camera_matrix, stereo_cameras.camera_left.camera_model.d,
                                                      stereo_cameras.camera_right.camera_model.camera_matrix, stereo_cameras.camera_right.camera_model.d, image_size, rotation_of_cam2, translation_of_cam2, flags=flags, alpha=0)

    # init rectificationmap
    leftmapX, leftmapY = cv2.initUndistortRectifyMap(
        stereo_cameras.camera_left.camera_model.camera_matrix, stereo_cameras.camera_left.camera_model.d, R1, P1, image_size, cv2.CV_32FC1)
    rightmapX, rightmapY = cv2.initUndistortRectifyMap(
        stereo_cameras.camera_right.camera_model.camera_matrix, stereo_cameras.camera_right.camera_model.d, R2, P2, image_size, cv2.CV_32FC1)

    while True:
        # ret, imgL, imgR = stereo_cameras.get_current_rectified_frame()
        ret, imgL, imgR = stereo_cameras.get_current_frame_pair_pre_processed()
        if not ret:
            print("Cannot open stereo cameras.. continuing to try")
            continue
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        imgL_gray = cv2.remap(imgL_gray, leftmapX, leftmapY,
                              cv2.INTER_LINEAR)
        imgR_gray = cv2.remap(imgR_gray, rightmapX, rightmapY,
                              cv2.INTER_LINEAR)

        Left_nice = imgL_gray
        Right_nice = imgR_gray

        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp')*16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp')*2 + 5
        # preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        # preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp')*2 + 5
        # preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        # textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        sig = cv2.getTrackbarPos('sigma', 'disp filtered')
        lam = cv2.getTrackbarPos('lambda', 'disp filtered') * 100

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        # stereo.setPreFilterType(preFilterType)
        # stereo.setPreFilterSize(preFilterSize)
        # stereo.setPreFilterCap(preFilterCap)
        # stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # wls_filter.setLambda(lam)
        # wls_filter.setSigmaColor(sig)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(2.6)

        # # Calculating disparity using the StereoBM algorithm
        # left_remap = cv2.remap(Left_nice, leftmapX, leftmapY,
        #                        cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        # right_remap = cv2.remap(
        #     Right_nice, rightmapX, rightmapY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # left_disp = stereo.compute(left_remap, right_remap)
        # right_disp = right_matcher.compute(right_remap, left_remap)
        # filtered_disp = wls_filter.filter(
        #     left_disp, left_remap, disparity_map_right=right_disp)

        # # filtered_disp = filtered_disp.astype(np.float32)
        # filtered_disp = (filtered_disp/256).astype(np.uint8)
        # filtered_disp = cv2.applyColorMap(filtered_disp, cv2.COLORMAP_JET)

        # Converting to float32
        disparity = left_disp.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity/16.0 - minDisparity)/numDisparities
        # Displaying the disparity map
        # full_img = np.hstack((Left_nice, Right_nice))
        cv2.imshow("disp", disparity)
        # cv2.imshow("disp filtered", filtered_disp)
        # cv2.imshow('full', full_img)
        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break


def main():
    cam_left_index = sys.argv[1]
    cam_left_calibration_file = sys.argv[2]
    cam_right_index = sys.argv[3]
    cam_right_calibration_file = sys.argv[4]
    stereo_calibration_folder = sys.argv[5]
    print(f"Capturing on cameras: L:{cam_left_index} R:{cam_right_index}")
    # If you want 720P, they need to go into different USB ports
    # width_val = 420  # minimum for same laptop hub
    # height_val = 240  # minimum for same laptop hub
    width_val = 800
    height_val = 600
    camLP = PinholeCamera.loadFromOST(cam_left_calibration_file)
    camL = ThreadedCamera(int(cam_left_index), width_val,
                          height_val, camera_model=camLP)
    camRP = PinholeCamera.loadFromOST(cam_right_calibration_file)
    camR = ThreadedCamera(int(cam_right_index), width_val,
                          height_val, camera_model=camRP)
    stereo_cameras = ThreadedStereo(camL, camR, stereo_calibration_folder)

    time.sleep(2)
    runGui(stereo_cameras)


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print(
            "Please run with arguments: [left_cam index] [left_cam calibration] [cam_right index] [cam_right calibration] [stereo calibration folder]")
        exit(1)
    main()
