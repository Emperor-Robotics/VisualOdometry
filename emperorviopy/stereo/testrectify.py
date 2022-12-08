#!/bin/env python
import cv2
import sys
import numpy as np
from threading import Thread
from emperorviopy.common.IMU import IMU
from emperorviopy.common.CameraModel import PinholeCamera
from emperorviopy.common.ThreadedCamera import ThreadedCamera
import time
import os
from stereovision.calibration import StereoCalibrator, StereoCalibration
from glob import glob
import yaml

if len(sys.argv) < 5:
    print(
        "Please run with arguments: [left_cam index] [left_cam calibration] [cam_right index] [cam_right calibration]")
    exit(1)
cam_left_index = sys.argv[1]
cam_left_calibration_file = sys.argv[2]
cam_right_index = sys.argv[3]
cam_right_calibration_file = sys.argv[4]
print(f"Capturing on cameras: L:{cam_left_index} R:{cam_right_index}")

FOLDER = 'OUTPUT/'
left_images = sorted(glob(os.path.join(FOLDER, 'left', '*_left.png')))
right_images = sorted(glob(os.path.join(FOLDER, 'right', '*_right.png')))
