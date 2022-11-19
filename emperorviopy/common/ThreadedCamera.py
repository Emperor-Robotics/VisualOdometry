#!/bin/env python
import cv2
import sys
import numpy as np
from threading import Thread
import time
import os
from glob import glob
from typing import Dict, Tuple
from emperorviopy.common.CameraModel import PinholeCamera
# Optimizations to keep cameras relatively sync'd and together.


class ThreadedCamera(object):
    def __init__(self, src: int = 0, width: int = None, height: int = None,
                 buffersize: int = 2, camera_model: PinholeCamera = None, exposure=-1) -> None:
        self.capture = cv2.VideoCapture(src)
        # Limit the buffer to maintain high FPS and sync
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)

        # Turn off auto exposure to prevent lighting changes producing noise.
        # If anything, should have it so the two cameras are sync'd in exposure changes
        # handled through python
        if exposure != -1:
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual exposure
            self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
        else:
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto exposure

        if width is not None and height is not None:
            self.capture.set(3, width)  # width set
            self.capture.set(4, height)  # height set
        self.camera_model = camera_model
        self.thread = Thread(target=self.update, args=())
        self.frame = None
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # read the next frame, should be done in a different thread
        while True:
            self.status, self.frame = self.capture.read()
            time.sleep(.01)  # needed?

    def get_current_frame(self):
        return self.status, self.frame

    def get_current_frame_undistorted(self):
        if self.camera_model is None:
            print("Camera model not set but was asked for an undistorted image.")
            return False, None
        return self.status, cv2.undistort(self.frame, self.camera_model.camera_matrix, self.camera_model.d)

    def release(self):
        self.capture.release()


class ThreadedStereo(object):
    def __init__(self, camera_left: ThreadedCamera, camera_right: ThreadedCamera,
                 calibration_folder: str = '') -> None:
        self.camera_left: ThreadedCamera = camera_left
        self.camera_right: ThreadedCamera = camera_right
        self.undistortion_map: Dict[str, np.ndarray] = {}
        self.rectify_map: Dict[str, np.ndarray] = {}
        self.calibration_folder = calibration_folder
        self._load_calibration()

    def _load_calibration(self):
        if self.calibration_folder == None:
            return
        if os.path.exists(os.path.join(self.calibration_folder, 'rectification_map_left.npy')):
            self.rectify_map['left'] = np.load(os.path.join(
                self.calibration_folder, 'rectification_map_left.npy'))

        if os.path.exists(os.path.join(self.calibration_folder, 'rectification_map_right.npy')):
            self.rectify_map['right'] = np.load(os.path.join(
                self.calibration_folder, 'rectification_map_right.npy'))

        if os.path.exists(os.path.join(self.calibration_folder, 'undistortion_map_left.npy')):
            self.undistortion_map['left'] = np.load(os.path.join(
                self.calibration_folder, 'undistortion_map_left.npy'))

        if os.path.exists(os.path.join(self.calibration_folder, 'undistortion_map_right.npy')):
            self.undistortion_map['right'] = np.load(os.path.join(
                self.calibration_folder, 'undistortion_map_right.npy'))

    def get_current_frame_pair_raw(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        leftret, leftframe = self.camera_left.get_current_frame()
        rightret, rightframe = self.camera_left.get_current_frame()
        return leftret and rightret, leftframe, rightframe

    def get_current_frame_pair_pre_processed(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Returns unrectified rotated and flipped images according to custom rig,
        if using this with a different stereo camera that does not require this, 
        override this function in a child class.
        """
        ret, imgL, imgR = self.get_current_frame_pair_raw()
        imgL = np.rot90(imgL, k=-1)
        imgL = np.flipud(imgL)
        imgL = np.fliplr(imgL)
        imgR = np.rot90(imgR, k=-1)
        imgR = np.flipud(imgR)
        imgR = np.fliplr(imgR)
        return ret, imgL, imgR

    def get_current_rectified_frame(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        ret, leftframe, rightframe = self.get_current_frame_pair_pre_processed()
        new_frames = []
        new_frames.append(cv2.remap(
            leftframe, self.undistortion_map['left'], self.rectify_map['left'], cv2.INTER_NEAREST))
        new_frames.append(cv2.remap(
            rightframe, self.undistortion_map['right'], self.rectify_map['right'], cv2.INTER_NEAREST))
        return ret, new_frames[0], new_frames[1]
