#!/bin/env python

# Much of this is pulled from: https://github.com/uoip/monoVO-python
import sys
import numpy as np
import cv2

from VO import PinholeCamera, VisualOdometry


cam = PinholeCamera(1280.0, 720.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, '/home/xxx/datasets/KITTI_odometry_poses/00.txt')

traj = np.zeros((600, 600, 3), dtype=np.uint8)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("pass in file.")
        exit(1)
    print(f"Streaming file {sys.argv[1]}")
    cap = cv2.VideoCapture(sys.argv[1])

    if (cap.isOpened() == False):
        print("Cant open file")
        exit(1)

    START_FRAME = 300
    DRAWOFFSET_X = 300
    DRAWOFFSET_Z = 300

    frame_id = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (frame_id < START_FRAME):
            print(frame_id)
            frame_id += 1
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(gray_frame.shape)

        vo.update(gray_frame, frame_id)

        cur_t = vo.cur_t
        if (frame_id > START_FRAME+3):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0, 0, 0

        draw_x, draw_y = int(x)+DRAWOFFSET_X, int(z)+DRAWOFFSET_Z
        true_x, true_y = int(vo.trueX) + \
            DRAWOFFSET_X, int(vo.trueZ) + DRAWOFFSET_Z

        cv2.circle(traj, (draw_x, draw_y), 1, (frame_id *
                   255/4540, 255-frame_id*255/4540, 0), 1)
        cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        text = f"coordinates: x={x}, y={y}, z={z}"
        cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), 1, 8)

        if ret == True:
            cv2.imshow('Camera Frame', frame)
            cv2.imshow('Trajectory', traj)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
