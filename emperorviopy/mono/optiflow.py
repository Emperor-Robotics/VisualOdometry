#!/bin/env python
# https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/
import numpy as np
import sys
import cv2


cap = cv2.VideoCapture(sys.argv[1])
# cap = cv2.VideoCapture(2)

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
START_FRAME = 0
for i in range(START_FRAME):
    _, _r = cap.read()

# Take first frame and find corners in it
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame,
                         cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(first_frame)

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

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)),
                            (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)),
                               5, color[i].tolist(), -1)

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
    cv2.imshow("Optical flow", img_txt)

    k = cv2.waitKey(25)
    if k == 27:
        break
    prev_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)


cv2.destroyAllWindows()
cap.release()
