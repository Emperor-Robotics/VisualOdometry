#!/bin/env python
import cv2
import sys
import numpy as np
from threading import Thread
import time
# Optimizations to keep cameras relatively sync'd and together.


class ThreadedCamera(object):
    def __init__(self, src: int = 0, width: int = None, height: int = None, buffersize: int = 2) -> None:
        self.capture = cv2.VideoCapture(src)
        # Limit the buffer to maintain high FPS and sync
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
        if width is not None and height is not None:
            self.capture.set(3, width)  # width set
            self.capture.set(4, height)  # height set
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
