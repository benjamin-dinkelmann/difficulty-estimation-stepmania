import torch
from torch import nn
from torch.utils.data import Dataset
import os
from random import random
from math import floor, ceil, e, pi, log, gcd
from matplotlib import pyplot as plt


def print_model_parameter_overview(model, only_total=False):
	tensor_list = list(model.state_dict().items())
	total = 0
	for layer_tensor_name, tensor in tensor_list:
		k = torch.numel(tensor)
		total += k
		if not only_total:
			print('Layer {}: {} elements'.format(layer_tensor_name, k))
	print("Total number of parameters: {}".format(total))
	return total


def get_total_number_of_parameters(state_dict):
	tensor_list = list(state_dict.items())
	total = 0
	for _, t in tensor_list:
		total += torch.numel(t)
	return total


def get_slice(tensor, dim):
	tensor_dim = tensor.dim()
	d = dim if dim >= 0 else tensor_dim + dim
	return [slice(None)] * d


class TimeSeriesDataset(Dataset):
	def __init__(self, label_frame, data_input_dir, transform=None, target_transform=None, target_device=None):
		if not target_device:
			if torch.cuda.is_available():
				target_device = 'cuda'
			else:
				target_device = 'cpu'

		self.label_frame = label_frame
		self.target_device = target_device
		self.time_series_dir = data_input_dir
		self.transform = transform
		self.target_transform = target_transform
		self.buffer = {}

	def __len__(self):
		return len(self.label_frame)

	def __getitem__(self, idx):
		if idx in self.buffer:
			time_series, label = self.buffer[idx]
		else:
			label = self.label_frame.iloc[idx, 2]
			ts_path = os.path.join(self.time_series_dir, self.label_frame.iloc[idx, 4])
			time_series = torch.load(ts_path, map_location=self.target_device)

			label = torch.tensor(label, dtype=torch.long, device=self.target_device, requires_grad=False)

			self.buffer[idx] = (time_series, label)

		if self.transform:
			time_series = self.transform(time_series)
		if self.target_transform:
			label = self.target_transform(label)
		return time_series, label


class RandomSubSampleTransform(object):
	def __init__(self, sample_size, sample_level=48, subsamples=2, target_device='cuda', multisample_mode=True, seed=None):
		self.sample_size = sample_size
		self.sub_samples = subsamples
		self.sample_level = max(sample_level, 1)
		self.rng = torch.Generator(device=target_device)
		if seed is not None:
			self.rng.manual_seed(seed)

		self.mask = torch.zeros(self.sub_samples * self.sample_size, dtype=torch.long, device=target_device)
		# self.random_indices = torch.zeros([self.sub_samples], dtype=torch.long, device=target_device)
		self.random_numbers = torch.empty([self.sub_samples], dtype=torch.float, device=target_device)
		self.multi_mode = multisample_mode
		self.cached_sample = None
		self.cached_sequence_length = None
		self.target_device = target_device

	def __call__(self, sample):
		sequence_length = sample.shape[1]
		channels = sample.shape[0]
		torch.rand([self.sub_samples], out=self.random_numbers, generator=self.rng)

		if sequence_length < self.sample_size:
			out = torch.zeros([channels, self.sub_samples*self.sample_size], dtype=torch.float, device=self.target_device)
			self.random_indices = (self.random_numbers*(self.sample_size - sequence_length) - 1e-8).int()
			for j in range(self.sub_samples):
				start = j*self.sample_size+self.random_indices[j]
				out[:, start:start+sequence_length] = sample
		else:
			m = (sequence_length - self.sample_size) // self.sample_level + 1
			sub_slot_width = m/self.sub_samples
			self.random_indices = (self.random_numbers*ceil(sub_slot_width) - 1e-8).int()

			for j in range(self.sub_samples):
				start = (self.random_indices[j] + floor(j*sub_slot_width)) * self.sample_level
				torch.arange(start, start+self.sample_size, out=self.mask[j*self.sample_size:(j+1)*self.sample_size])

			out = sample[:, self.mask]
		if self.multi_mode:
			out = out.reshape(channels, self.sub_samples, self.sample_size).transpose(0, 1)
		return out


class FixedSizeInputSampleTS(Dataset):
	def __init__(self, dataset, sample_size, k=6, sub_samples=2, multisample_mode=True, seed=None, target_device='cuda'):
		self.subset = dataset
		self.sample_level = k  # only samples all kth entries
		self.sub_samples = sub_samples
		self.transform = RandomSubSampleTransform(sample_size, sample_level=k, subsamples=self.sub_samples, multisample_mode=multisample_mode, seed=seed, target_device=target_device)

	def __len__(self):
		return len(self.subset)

	def __getitem__(self, idx):
		X, y = self.subset[idx]
		X = self.transform(X)
		return X, y


class GeneralTSModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.name = "GeneralTSModel"
		self.min_size = 5


class MultiWindowModelWrapper(GeneralTSModel):
	def __init__(self, model, target_device='cuda', output_classes=20, final_activation=None, variant=None):
		super().__init__()
		if isinstance(model, GeneralTSModel):
			self.name = 'MultiSample'+model.name
			self.min_size = model.min_size
			self.model = model
			self.final = self.model.final
			self.classes = torch.arange(1, output_classes + 1, dtype=torch.float, device=target_device)
			self.target_device = target_device
			# self.attention = nn.MultiheadAttention(8, 1, batch_first=True, device=target_device)
			# self.encoder = nn.TransformerEncoderLayer(output_classes, 1, dim_feedforward=128, dropout=0.05, device=target_device)
			# self.rnn = nn.GRU(output_classes, output_classes, num_layers=1, batch_first=True, bidirectional=False, device=target_device)
			if final_activation:
				self.final_act = final_activation
			else:
				self.final_act = nn.Softmax(dim=-1)
			# self.rootTwoPi = (2*pi)**.5
			# Not used atm
			if variant is None:
				variant = 1.2
			if output_classes == 1 and not (variant < 2 or abs(variant % 1 - 0.1) < 1e-5):
				variant = 1.2
			self.variant = variant
			# print(variant)
			assert (0 < self.variant < 8)
		else:
			raise AttributeError

	def forward(self, x):
		if len(x.shape) == 3:
			x = x.unsqueeze(0)
		else:
			x = x.transpose(0, 1)
		k = x.shape[0]

		res = self.model(x[0])

		# Variant 7: SoftMax
		if int(self.variant) == 7:
			if self.variant == 7.1:  # regression variant
				weight_f = lambda a, b: a
			elif self.variant == 7.2:
				weight_f = lambda a, b: (a >= 0.5).to(torch.float).sum(dim=-1, keepdim=True)
			else:
				weight_f = lambda a, b: (a*b).sum(dim=-1, keepdim=True)
			res = self.final_act(self.final(res))
			with torch.no_grad():
				factor = weight_f(res, self.classes).exp()
			factor_sum = factor
			res = res * factor
			for i in range(1, k):
				tmp = self.final_act(self.final(self.model(x[i])))
				with torch.no_grad():
					factor = weight_f(tmp, self.classes).exp()
					factor_sum = factor_sum + factor
				res = res + tmp * factor
			res = res / factor_sum

		# Variant 6: RNN
		elif self.variant == 6:
			out = torch.empty([k, *res.shape], dtype=torch.float, device=self.target_device)
			out[0] = res
			for i in range(1, k):
				out[i] = self.model(x[i])
			out = out.transpose(0, -2)
			res = self.rnn(out)[1][0]
			res = self.final_act(res)

	# Variant 5: Max + Background
		elif self.variant == 5:
			background = 0.1*res
			avg_classes = (res * self.classes).sum(-1, keepdim=True)
			for i in range(1, k):
				tmp = self.model(x[i])
				tmp2 = (tmp * self.classes).sum(-1, keepdim=True)
				res = res.where(tmp2 < avg_classes, tmp)
				background = background + 0.1*tmp
				avg_classes = avg_classes.where(tmp2 < avg_classes, tmp2)
			res = 0.9*res + background/k
	# Variant 4: Max
		elif self.variant == 4:
			avg_classes = (res * self.classes).sum(-1, keepdim=True)
			for i in range(1, k):
				tmp = self.model(x[i])
				tmp2 = (tmp * self.classes).sum(-1, keepdim=True)
				res = res.where(tmp2 < avg_classes, tmp)
				avg_classes = avg_classes.where(tmp2 < avg_classes, tmp2)
	# Variant 3: Attention - needs raw model outputs
		# Possible to allocate this once?
		elif int(self.variant) == 3:
			out = torch.empty([k, *res.shape], dtype=torch.float, device=self.target_device)
			out[0] = res
			for i in range(1, k):
				out[i] = self.model(x[i])
			out = out.transpose(0, -2)   # 0,1
			res = self.attention(out, out, out, need_weights=False)[0].sum(-2)
			# res = self.encoder(out).sum(-2)
			if self.variant == 3.2:
				res = self.final(res)
			res = self.final_act(res)

	# Variant 2: Weighted Sum
		elif self.variant == 2:
			with torch.no_grad():
				avg_classes_sum = (res * self.classes).sum(-1, keepdim=True)
			res = res * avg_classes_sum
			for i in range(1, k):
				tmp = self.model(x[i])
				with torch.no_grad():
					tmp2 = (tmp*self.classes).sum(-1, keepdim=True)
				res = res + tmp * tmp2
				avg_classes_sum = avg_classes_sum + tmp2
			res = res/avg_classes_sum

		elif self.variant == 2.1: #regression version
			with torch.no_grad():
				avg_classes_sum = res.abs()
			res = res * avg_classes_sum
			for i in range(1, k):
				tmp = self.model(x[i])
				with torch.no_grad():
					tmp2 = tmp.abs()
				res = res + tmp * tmp2
				avg_classes_sum = avg_classes_sum + tmp2
			res = res/avg_classes_sum

	# Variant 1: Simple Mean
		elif int(self.variant) == 1:
			for i in range(1, k):
				res = res + self.model(x[i])
			res = res/k
			# print(res.shape)
			if self.variant == 1.2:
				res = self.final(res)
			if self.variant > 1:
				res = self.final_act(res)

		return res


class TimeSeriesTransformerModel(GeneralTSModel):
	def __init__(self, in_channels, target_device, out_channels=20, sequence_length=60, final_activation=None):
		super().__init__()
		if out_channels > 1:
			self.name = 'TimeSeriesTransformerModel'
		else:
			self.name = 'TimeSeriesTransformerRegressionModel'
		n = 64
		self.n = n
		self.out_channels = out_channels
		self.min_size = 16
		self.initial = nn.Sequential(
			nn.Conv1d(in_channels, n, 2, 1, 0, device=target_device),  #  small conv
			nn.ReLU(),
		)
		encoderLayer = nn.TransformerEncoderLayer(n, nhead=4, dim_feedforward=n, batch_first=True, device=target_device)
		self.encoder = nn.TransformerEncoder(encoderLayer, num_layers=3)

		pe = torch.empty([n, sequence_length + 1], dtype=torch.float, device=target_device)
		pos = torch.arange(sequence_length + 1, device=target_device).unsqueeze(0)
		dim = torch.arange(n // 2, device=target_device).unsqueeze(1)
		vals = (pos * (-log(10000) * 2 / in_channels * dim).exp())
		# print(vals.shape)
		pe[0::2] = vals.sin()
		pe[1::2] = vals.cos()
		self.positional_encoding = pe

		if final_activation:
			self.final_activation_fn = final_activation
		else:
			self.final_activation_fn = nn.Softmax(dim=-1)
		self.final = nn.Sequential(
			nn.Identity(),
			nn.Identity(),
			nn.Linear(n, out_channels, device=target_device)
		)
		initial_weights = torch.ones(in_channels, device=target_device, requires_grad=False)
		initial_weights[:11] = torch.tensor([1.5, 1., 1, 1, 1, 1, 1, 1, 4., 4., .8], dtype=torch.float, device=target_device, requires_grad=False)
		self.initial_weights = initial_weights.unsqueeze(-1)

	def forward(self, x):
		x = x*self.initial_weights
		x2 = self.initial(x)
		l = x2.shape[-1]
		x2 = x2 + self.positional_encoding[:, :l]
		x2 = x2.transpose(-1, -2)
		x2 = self.encoder(x2)
		x2 = x2.mean(dim=-2) # GAP
		# x2 = self.final_activation_fn(self.final(x2))
		return x2