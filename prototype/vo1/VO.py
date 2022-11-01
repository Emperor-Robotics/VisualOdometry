#!/bin/env python
import cv2
from cv2 import KeyPoint
from cv2 import KeyPoint_convert
from cv2 import TermCriteria_EPS
import numpy as np
import sys

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

kMinNumFeature = 1500

lk_params = dict(winSize=(21, 21),
                 #  maxLevel=3,
                 criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 30, 0.01))


def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(
        image_ref, image_cur, px_ref, None, **lk_params)
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]
    return kp1, kp2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(
            threshold=25, nonmaxSuppression=True)
        # with open(annotations) as f:
        #     self.annotations = f.readlines()

    def getAbsoluteScale(self, frame_id):  # specialized for kitti odometry dataset?
        ss = self.anotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev))

    def SafeEssentialMat(self, PixelCurrent, PixelReference):
        if len(PixelCurrent) == 0 or len(PixelReference) == 0:
            print(
                "Cannot Compute essential Mat, No lockon for pixels. [Resetting Frame Stage to 0]")
            self.frame_stage = STAGE_FIRST_FRAME
            return False, -1, -1
        E, mask = cv2.findEssentialMat(
            PixelCurrent, PixelReference, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        return True, E, mask

    def SafePoseRecovery(self, EssentialMatrix, PixelCurrent, PixelReference):
        if (EssentialMatrix is None):
            print(
                "Lost Tracking, no essential matrix. [Resetting Frame Stage to 0]")
            self.frame_stage = STAGE_FIRST_FRAME
            return False, -1, -1
        _, currentRotation, currentTransformation, mask = cv2.recoverPose(
            EssentialMatrix, PixelCurrent, PixelReference, focal=self.focal, pp=self.pp)
        return True, currentRotation, currentTransformation

    def processFirstFrame(self):
        print("First Frame!")
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        if (len(self.px_ref) < 100):
            print(f"No good keypoints FIRSTFRAME: {len(self.px_ref)}")
            return
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(
            self.last_frame, self.new_frame, self.px_ref)
        success, E, mask = self.SafeEssentialMat(self.px_cur, self.px_ref)
        if not success:
            return
        success, currentRot, currentTrans = self.SafePoseRecovery(
            E, self.px_cur, self.px_ref)
        if not success:
            return
        if self.cur_t is None:
            self.cur_t = currentTrans
        else:
            self.cur_t = self.cur_t + 1*self.cur_R.dot(currentTrans)

        if self.cur_R is None:
            self.cur_R = currentRot
        else:
            self.cur_R = currentRot.dot(self.cur_R)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(
            self.last_frame, self.new_frame, self.px_ref)
        success, E, mask = self.SafeEssentialMat(self.px_cur, self.px_ref)
        if not success:
            return
        success, currentRot, currentTrans = self.SafePoseRecovery(
            E, self.px_cur, self.px_ref)
        if not success:
            return
        R, t = currentRot, currentTrans
        # absolute_scale = self.getAbsoluteScale(frame_id)
        absolute_scale = 1
        if (absolute_scale > 0.1):
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t)
            print(self.cur_t)
            self.cur_R = R.dot(self.cur_R)
        if (self.px_ref.shape[0] < kMinNumFeature):
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array(
                [x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] ==
                self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if (self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
        elif (self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif (self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("pass in file.")
        exit(1)
    print(f"Streaming file {sys.argv[1]}")
    cap = cv2.VideoCapture(sys.argv[1])
    if (cap.isOpened() == False):
        print("Cant open file")
        exit(1)
