#!/bin/env python
# https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/
import sys
import numpy as np
import math
import cv2
from CameraModel import PinholeCamera


class CVPlotter:
    """
    Used to plot trajectory of odometry.
    """

    def __init__(self) -> None:
        self.DRAWOFFSET_X = 300
        self.DRAWOFFSET_Z = 300
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)

    def update_plot(self, x, y, z, frame_id):
        draw_x, draw_y = int(x)+self.DRAWOFFSET_X, int(z)+self.DRAWOFFSET_Z
        cv2.circle(self.traj, (draw_x, draw_y), 1, (frame_id *
                   255/4540, 255-frame_id*255/4540, 0), 1)
        cv2.rectangle(self.traj, (10, 20), (600, 60), (0, 0, 0), -1)
        text = f"coordinates: x={x}, y={y}, z={z}"
        cv2.putText(self.traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), 1, 8)


class Pose:
    """
        world coordinate system is: x-axis pointing to right, y-axis pointing forward, z-axix pointing up.
        Note: https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
        Note: https://stackoverflow.com/questions/32175286/strange-issue-with-stereo-triangulation-two-valid-solutions/36213818#36213818
        The above posts says that these transforms are pixel transforms instead of camera relative.

        So in a way it gives you the transform of the previous frame?

    """

    def __init__(self) -> None:
        self.translation: np.ndarray = np.zeros((3, 1), dtype=np.float32)
        self.rotation: np.ndarray = np.ones((3, 3), dtype=np.float32)

    def update_translation(self, new_translation: np.ndarray, absolute_scale: float = 1.0) -> None:
        self.translation = self.translation + \
            absolute_scale * self.rotation.dot(new_translation)

    def update_rotation(self, new_rotation: np.ndarray) -> None:
        self.rotation = new_rotation.dot(self.rotation)

    def resetPose(self):
        self.translation: np.ndarray = np.zeros((3, 1), dtype=np.float32)
        self.rotation: np.ndarray = np.ones((3, 3), dtype=np.float32)

    def getXYZ(self):
        return (self.translation[0][0], self.translation[1][0], self.translation[2][0])

    def __str__(self) -> str:
        return f"\nX: {self.translation[0][0]} Y: {self.translation[1][0]} Z: {self.translation[2][0]}\n\nRotation: {self.rotationMatrixToEulerAngles(self.rotation)}\n" + "="*30

    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array([[1,         0,                  0],
                        [0,         math.cos(theta[0]), -math.sin(theta[0])],
                        [0,         math.sin(theta[0]), math.cos(theta[0])]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                        [0,                     1,      0],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])]
                        ])
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    # Checks if a matrix is a valid rotation matrix. Untested.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ). Untested
    def rotationMatrixToEulerAngles(self, R):
        if not self.isRotationMatrix(R):
            return [-1, -1, -1]
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])


# Camera Calibration file
cam = PinholeCamera.loadFromOST(sys.argv[1])


# Video Capture or Video File
START_FRAME = 0
if sys.argv[2].isnumeric():
    cap = cv2.VideoCapture(int(sys.argv[2]))
    # START FRAME IN CASE VIDEO FILE
    START_FRAME = 0
else:
    cap = cv2.VideoCapture(sys.argv[2])

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# for i in range(START_FRAME):
# _, _r = cap.read()

# Take first frame and find corners in it
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame,
                         cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(first_frame)

pose = Pose()
plot = CVPlotter()

# cv2.namedWindow("Optical Flow", cv2.WINDOW_KEEPRATIO)
# cv2.resizeWindow("Optical Flow", 1280, 720)

while (cap.isOpened()):

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)
    if (len(p0) < 10):
        # print("Retracking!")
        mask = np.zeros_like(first_frame)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if (p0 is not None):

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, frame_gray, p0, None, **lk_params)

        # Check this..
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)),
                            (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)),
                               5, color[i].tolist(), -1)

        # Essential matrix, needs calibrated camera.
        if (len(good_new) > 5):
            # TODO: Confirm essential matrix is correct
            E, _ = cv2.findEssentialMat(good_new, good_old, focal=cam.fx, pp=(
                cam.cx, cam.cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if (E is not None and sum(E.shape) >= 6):
                """
                Note: 
                Recover pose chooses between the 4 diffent solutions of translation and rotation given 
                an essential matrix estimate. Roughly following the steps..
                    - Estimating the 4 possible solutions
                    - 3D triangulation of all inlier points
                    - Cheirality Check: Checking if the trangulated point is in front of both cameras ( Positive Z )
                    - Choose a solution and flag all inliers that fail the cheirality check as outliers.
                """
                inliers, currentRotation, currentTransform, _mask = cv2.recoverPose(
                    E, good_new, good_old, focal=cam.fx, pp=(cam.cx, cam.cy))
                # need imu to combine with this to get translation scale.
                pose.update_rotation(currentRotation)
                pose.update_translation(currentTransform, 1)
                print(pose)
                plot.update_plot(*pose.getXYZ(), 300)
    img = cv2.add(frame, mask)

    # HUD
    color_g = (0, 255, 0)
    color_b = (255, 0, 0)
    color_r = (0, 0, 255)
    selcol = color_g
    hudtext = f'Locked on: {len(good_old)}'
    if (p0 is None or len(good_old) < 10):
        selcol = color_r
        hudtext = f'Retracking'
    img_txt = cv2.putText(
        img, hudtext, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, selcol, 2, cv2.LINE_AA)
    cv2.imshow("Trajectory", plot.traj)
    cv2.imshow("Optical Flow", img_txt)

    k = cv2.waitKey(25)
    if k == 27:
        break
    if k == ord('r'):
        pose.resetPose()
        p0 = []
        continue
    prev_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)


cv2.destroyAllWindows()
cap.release()
