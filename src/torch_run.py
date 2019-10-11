import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

import numpy as np

import utils as u
import dsutils as ds
import macros as m

def read_video(start_frame, stop_frame):
  vid = tv.io.read_video(m.TRAIN_FN, start_pts=m.FRAME_LEN*start_frame, end_pts=m.FRAME_LEN*stop_frame)
  return vid

def read_timestamps():
	timestamps = tv.io.read_video_timestamps(m.TRAIN_FN)
	return timestamps

class Frames(torch.utils.data.Dataset):
	def __init__(self):
		# self.timestamps = tv.io.read_video_timestamps(TRAIN_FN)
		self.num_frames = 20400  # len(timestamps[0])
		self.speeds = u.frame_speeds().values

	def __getitem__(self, index):
		vid = read_video(index, index)
		frame = vid[0].flatten().float()
		speed = torch.tensor(self.speeds[index], dtype=torch.float)
		return (frame, speed)

	def __len__(self):
		return self.num_frames

class MLP(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Net, self).__init__()
		self.l1 = nn.Linear(input_size, hidden_size)
		self.l2 = nn.Linear(hidden_size, 100)
		self.l3 = nn.Linear(100, output_size)
	def forward(self, x):
		x = torch.tanh(self.l1(x))
		x = self.l2(x)
		x = self.l3(x)
		return x


if __name__ == '__main__':
	input_size = 640 * 480 * 3  # W, H, C
	hidden_size = 1000
	output_size = 1
	batch_size = 128

	log_interval = 1

	# net = Net(input_size, hidden_size, output_size)
	net = ds.auto.mlp.MLP(input_size, output_size, factor=1000, classify=False)
	print(net)
	train_set = Frames()
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)

	criterion = nn.MSELoss()
	optim = torch.optim.Adam(net.parameters())

	for i, (x, y) in enumerate(train_loader):
		optim.zero_grad()
		output = net(x)
		loss = criterion(output, y)
		loss.backward()
		optim.step()
		with torch.no_grad():
			if i % log_interval == 0:
				print(f'loss: {loss}')
				print(f'input: {x[0]}')
				print(f'out: {output[0]}, y: {y[0]}\n')
