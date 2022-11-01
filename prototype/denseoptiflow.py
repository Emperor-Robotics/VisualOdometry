#!/bin/env python
# https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
import numpy as np
import sys
import cv2

cap = cv2.VideoCapture(sys.argv[1])


# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame,
                         cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(first_frame)

mask[..., 1] = 255

while (cap.isOpened()):

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)
    cv2.imshow('input', frame)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mask[..., 0] = angle * 180 / np.pi / 2

    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    cv2.imshow("Dense Optical flow", rgb)

    prev_gray = frame_gray

    k = cv2.waitKey(25)
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()
