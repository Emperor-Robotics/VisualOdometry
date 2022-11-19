from __future__ import annotations
import numpy as np
import yaml
from typing import List
# https://answers.opencv.org/question/189506/understanding-the-result-of-camera-calibration-and-its-units/


class PinholeCamera:
    def __init__(self, width: float, height: float, fx: float, fy: float, cx: float, cy: float,
                 k1: float = 0.0, k2: float = 0.0, p1: float = 0.0, p2: float = 0.0, k3: float = 0.0,
                 raw_dat: dict = None) -> None:
        self.width: float = width
        self.height: float = height
        self.fx: float = fx
        self.fy: float = fy
        self.cx: float = cx
        self.cy: float = cy
        self.distortion: bool = (abs(k1) > 0.0000001)
        self.d: np.ndarray = np.array([k1, k2, p1, p2, k3])
        self.raw_calibration_data: dict = raw_dat
        self.camera_matrix: np.ndarray = np.array([[self.fx, 0, self.cx],
                                                   [0, self.fx, self.cy],
                                                   [0, 0, 1]])

    @staticmethod
    def loadFromOST(filename: str) -> PinholeCamera:
        """
            Generate a PinholeCamera class from the yaml calibration output.
            yaml output obtained from ros2 image_pipeline calibration tools. 

            distortion_map and rectification_map obtained from stereo calibration.

            TODO: Can ros2 image_pipeline produce stereo rectification?
        """
        with open(filename, 'r') as f:
            dat = yaml.safe_load(f)

        return PinholeCamera(
            width=float(dat['image_width']),
            height=float(dat['image_height']),
            fx=float(dat['camera_matrix']['data'][0]),
            fy=float(dat['camera_matrix']['data'][4]),
            cx=float(dat['camera_matrix']['data'][2]),
            cy=float(dat['camera_matrix']['data'][5]),
            k1=float(dat['distortion_coefficients']['data'][0]),
            k2=float(dat['distortion_coefficients']['data'][1]),
            p1=float(dat['distortion_coefficients']['data'][2]),
            p2=float(dat['distortion_coefficients']['data'][3]),
            k3=float(dat['distortion_coefficients']['data'][4]),
            raw_dat=dat
        )


if __name__ == '__main__':
    import sys
    PinholeCamera.loadFromOST(sys.argv[1])
