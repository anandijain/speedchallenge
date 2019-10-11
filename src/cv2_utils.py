import numpy as np
import cv2

import utils as u
import macros as m

def play_video(fn=m.TRAIN_FN):
  speeds = u.frame_speeds().values
  cap = cv2.VideoCapture(fn)
  i = 0
  while(cap.isOpened()):
    ret, frame = cap.read()
    speed = str(speeds[i][0])
    print(speed)
    cv2.imshow('f', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.033)
    i += 1
  cap.release()
  cv2.destroyAllWindows()
