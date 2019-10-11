import os
import time

import cv2
import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd

FRAME_LEN = 512
TRAIN_FN = './data/train.mp4'
TEST_FN = './data/test.mp4'
LABELS_FN = './data/train.txt'



def frame_speeds():
  df = pd.read_csv(LABELS_FN, sep='\n', header=None)
  df.columns = ['mph']
  return df

def cv2read():
  speeds = frame_speeds().values
  cap = cv2.VideoCapture(TRAIN_FN)
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



if __name__ == '__main__':
  # vid = read_video()
  cv2read()
