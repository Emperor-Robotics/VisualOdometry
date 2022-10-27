#!/bin/env python
import cv2
from cv2 import KeyPoint
from cv2 import KeyPoint_convert
from cv2 import TermCriteria_EPS
import numpy as np
import sys


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# def featureTracking(frame1, frame2, points1, points2, status):
#     global lk_params


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("pass in file.")
        exit(1)
    print(f"Streaming file {sys.argv[1]}")
    cap = cv2.VideoCapture(sys.argv[1])
    fast = cv2.FastFeatureDetector_create(20, True)
    # fast.setNonmaxSuppression(1)
    if (cap.isOpened() == False):
        print("Cant open file")
        exit(1)

    # Grab first frame and find corners
    ret, old_frame = cap.read()
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = fast.detect(old_frame, None)

    while (cap.isOpened()):
        ret, frame = cap.read()
        # CVT to gray and undistort
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features
        # p1 = fast.detect(frame, None)
        # point2fs = KeyPoint_convert(p1)
        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_frame, frame, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)),
                            (int(c), int(d)), [5, 20, 100], 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, [5, 20, 100], -1)
        img = cv2.add(frame, mask)

        # img2 = cv2.drawKeypoints(frame, p1, None, color=(255, 0, 0))

        if ret == True:
            cv2.imshow('Frame', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

        old_frame = frame.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()
