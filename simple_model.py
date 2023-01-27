import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset


class SimplePatternModel(nn.Module):
	def __init__(self, in_channels, target_device, out_channels=20, final_activation=None, pattern_attr=False):
		super().__init__()
		self.name = 'SimplePatternModel'
		n = 32
		self.n = n
		self.final = nn.Sequential(
			nn.Linear(in_channels, n, device=target_device),
			nn.ReLU(),
			nn.Linear(n, n, device=target_device),
			nn.ReLU(),
			nn.Linear(n, out_channels, device=target_device),
		)
		if final_activation:
			self.final_activation_fn = final_activation
		else:
			self.final_activation_fn = nn.Softmax(dim=-1)

	def forward(self, x):
		return self.final_activation_fn(self.final(x))


def prepare_pattern_dataset(dataframe, target_device):
	pattern_attr = ['Stream', 'Voltage', 'Air', 'Freeze', 'Chaos',
	                'n_jump', 'v_jump', 'n_stair', 'v_stair', 'n_candle', 'v_candle', 'n_cross', 'v_cross', 'n_drill',
	                'v_drill', 'n_jack', 'v_jack', 'n_step_j', 'v_step_j'
	                ]
	X = torch.from_numpy(dataframe[pattern_attr].to_numpy()).to(device=target_device, dtype=torch.float)
	y = torch.from_numpy(dataframe['Difficulty'].to_numpy()).to(device=target_device, dtype=torch.long)
	return TensorDataset(X, y)
