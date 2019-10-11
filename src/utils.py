import os
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch_run as rn
import macros as m

def frame_speeds():
  df = pd.read_csv(m.LABELS_FN, sep='\n', header=None)
  df.columns = ['mph']
  return df

def plot_speeds():
  # timestamps = rn.read_timestamps()
  speeds = frame_speeds()
  plt.plot(speeds)
  plt.xlabel = 'position in video'
  plt.ylabel = 'mph'
  plt.show()

if __name__ == '__main__':
  pass
