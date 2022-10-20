#!/bin/env python
import cv2
import numpy as np
import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("pass in file.")
        exit(1)
    print(f"Streaming file {sys.argv[1]}")
    cap = cv2.VideoCapture(sys.argv[1])

    if (cap.isOpened() == False):
        print("Cant open file")
        exit(1)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
