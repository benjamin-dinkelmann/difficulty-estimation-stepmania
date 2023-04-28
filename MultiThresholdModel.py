import os.path

import torch

from run_model import *
from torch.utils.data import Dataset


# Todo: Improve Performance
class RAMTSDataset(TimeSeriesDataset):
	"""
	Time Series dataset that is supposed to be loaded into RAM regardless of the model's device.
	At the moment, significantly slower than dataset in VRAM for cuda.
	"""
	def __init__(self, label_frame, data_input_dir, transform=None, target_transform=None):
		super().__init__(label_frame, data_input_dir, transform, target_transform, None)

	def __len__(self):
		return len(self.label_frame)

	def __getitem__(self, idx):
		if idx in self.buffer:
			time_series, label = self.buffer[idx]
		else:
			label = self.label_frame.iloc[idx, 2]
			ts_path = os.path.join(self.time_series_dir, self.label_frame.iloc[idx, 4])
			time_series = torch.load(ts_path, map_location='cpu')

			label = torch.tensor(label, dtype=torch.long, device='cpu', requires_grad=False)

			self.buffer[idx] = (time_series, label)

		if self.transform:
			time_series = self.transform(time_series)
		if self.target_transform:
			label = self.target_transform(label)
		return time_series.to(self.target_device), label.to(self.target_device)


class CombinedDataset(Dataset):
	"""
	Combines multiple datasets into one, that behaves almost exactly like a single dataset.
	When retrieving an item, this dataset also provides the id of the dataset where that item originated.
	"""
	def __init__(self, datasets, dataset_names=None):
		assert len(datasets) > 0
		if hasattr(datasets[0], 'target_device'):
			target_device = datasets[0].target_device
		else:
			target_device = default_device

		self.datasets = datasets
		dataset_sizes = []
		for dataset in datasets:
			dataset_sizes.append(len(dataset))
		dataset_ids = []
		for i, s in enumerate(dataset_sizes):
			tmp = np.empty([s], dtype=int)
			tmp.fill(i)
			dataset_ids.append(tmp)
		self.dataset_ids = np.hstack(dataset_ids)
		dataset_sizes.insert(0, 0)
		self.cum_sizes = np.cumsum(dataset_sizes)
		self.dataset_names = dataset_names

		print(self.cum_sizes)

	def __len__(self):
		return self.cum_sizes[-1]

	def __getitem__(self, index):
		# Assume in range
		# error if index < 0 or index > len(self)
		# idx = np.argmin(index >= self.cum_sizes) - 1
		idx = self.dataset_ids[index]
		X, y = self.datasets[idx][index - self.cum_sizes[idx]]
		return (X, torch.tensor([idx], device=X.device, dtype=torch.long)), y


def combined_dataset_collate(batch):
	A, B, C = [], [], []
	for (a, b), c in batch:
		A.append(a[None, :])
		B.append(b)
		C.append(c)
	return (torch.vstack(A), torch.hstack(B)), torch.hstack(C)


class MultiThresholdREDWrapper(REDWrapper):
	"""
	Transforms a regression model into an REDSVM style classification model by introducing learnable thresholds.
	Learns a set of thresholds per individual dataset and expects to be provided the dataset ids for prediction.
	"""
	def __init__(self, sub_model, target_device=default_device, output_classes=20, thresholds=1):
		super().__init__(sub_model, target_device, output_classes)
		self.name = "MTRED_"+self.name
		require_grad = True
		if isinstance(thresholds, torch.Tensor):
			self.theta = nn.Parameter(torch.clone(thresholds), requires_grad=require_grad)
		elif type(thresholds) == int:
			standard_thresholds = torch.arange(output_classes, device=target_device, dtype=torch.float)
			self.theta = nn.Parameter(torch.tile(standard_thresholds, (thresholds, 1)), requires_grad=require_grad)
		else:
			raise TypeError("Given thresholds of incorrect type. Either provide a tensor containing intial thresholds or an int specifying the number of sets of thresholds.")
		self.alpha = nn.Parameter(torch.tensor([0.1], dtype=torch.float, device=target_device), requires_grad=True)
		self.gamma = nn.Parameter(torch.tensor([1], dtype=torch.float, device=target_device), requires_grad=True)

	def forward(self, x):
		x, dataset_ids = x
		return self.sigmoid((self.model(x) - self.theta[dataset_ids]*self.gamma)*self.alpha)


"""
# Code for optimal setting of thresholds given stored regression outputs and datasets ids together with correct labels
def adjust_thresholds(self, y):

		target_device = y.device
		# joining stored data
		y = y.flatten()
		stored_outputs = torch.hstack(self.stored_outputs).flatten()
		stored_dataset_ids = torch.hstack(self.stored_dataset_ids).flatten()
		assert len(y) == len(stored_dataset_ids)

		# Split by dataset
		stored_dataset_ids, sort_indices = torch.sort(stored_dataset_ids)
		stored_outputs = stored_outputs[sort_indices]
		y = y[sort_indices]
		# unknown behaviour if a dataset is missing entirely
		dataset_id_counts = torch.bincount(stored_dataset_ids)
		split_sizes = [i.item() for i in dataset_id_counts if i> 0]
		per_dataset_outputs = torch.split(stored_outputs, split_sizes)
		per_dataset_y = torch.split(y, split_sizes)
		eps = 2e-2

		# Split by class
		i = -1
		for c_dataset in dataset_id_counts:
			if c_dataset == 0:
				continue
			i += 1
			outputs = per_dataset_outputs[i]
			ys = per_dataset_y[i]
			# ys, sort_indices = torch.sort(ys)
			# outputs = outputs[sort_indices]
			outputs, sort_indices = torch.sort(outputs)
			ys = ys[sort_indices]
			# print(torch.unique(ys))
			y_counts = torch.bincount(ys)
			# split_sizes = [i.item() for i in y_counts if i > 0]
			# per_class_outputs = torch.split(outputs, split_sizes)

			upper = []
			lower = []
			n = len(ys)
			# todo: might create too many thresholds? fix filled thresholds?
			for j in range(len(y_counts)-1):
				binary_labels = (ys > j).to(dtype=float)
				sum_inc = (1-binary_labels).cumsum(0)
				sum_dec = (-binary_labels).cumsum(0)+binary_labels.sum()
				threshold_idx = ((sum_inc+sum_dec)/n).argmax()
				# if threshold_idx == 0:
				# 	lower.append(outputs[threshold_idx]-0.5)
				# 	upper.append(lower[-1])
				if threshold_idx == n-1:
					lower.append(outputs[threshold_idx] + 0.5)
					upper.append(lower[-1])
				else:
					lower.append(outputs[threshold_idx])
					upper.append(outputs[threshold_idx + 1])

			# print("Dataset", i)
			# print("lower", lower)
			# print("upper", upper)
			for j in range(self.n_classes - len(upper)):
				lower.append(lower[-1]+0.5)
				upper.append(upper[-1]+0.5)
			new_thresholds = (torch.tensor(lower, dtype=torch.float, device=target_device, requires_grad=False) + torch.tensor(upper, dtype=torch.float, device=target_device, requires_grad=False))/2
			self.theta[i] = self.gamma*((1-self.beta)*self.theta[i] + self.beta*new_thresholds)

		self.stored_outputs = []
		self.stored_dataset_ids = []
"""


def compute_optimal_thresholds(predictions, labels):
	outputs, sort_indices = torch.sort(predictions)
	ys = labels[sort_indices]
	# print(torch.unique(ys))
	y_counts = torch.bincount(ys)
	# split_sizes = [i.item() for i in y_counts if i > 0]
	# per_class_outputs = torch.split(outputs, split_sizes)

	upper = []
	lower = []
	n = len(ys)

	for j in range(len(y_counts) - 1):
		binary_labels = (ys > j).to(dtype=float)
		sum_inc = (1 - binary_labels).cumsum(0)
		sum_dec = (-binary_labels).cumsum(0) + binary_labels.sum()
		threshold_idx = ((sum_inc + sum_dec) / n).argmax()
		# if threshold_idx == 0:
		# 	lower.append(outputs[threshold_idx]-0.5)
		# 	upper.append(lower[-1])
		if threshold_idx == n - 1:
			lower.append(outputs[threshold_idx] + 0.5)  # constant problematic? -> not for eval
			upper.append(lower[-1])
		else:
			lower.append(outputs[threshold_idx])
			upper.append(outputs[threshold_idx + 1])
	target_device = outputs.device
	optimal_thresholds = (torch.tensor(lower, dtype=torch.float, device=target_device, requires_grad=False) +
	                      torch.tensor(upper, dtype=torch.float, device=target_device, requires_grad=False)) / 2
	return optimal_thresholds


def compute_optimal_threshold_labels(outputs, targets, label_selection, pred_fn: MultiThresholdREDWrapper):
	optimal_thresholds = compute_optimal_thresholds(outputs, targets)
	# Avoids full prediction by re-using previous computed outputs and model thresholding strategy
	return label_selection(pred_fn.sigmoid((outputs.unsqueeze(-1) - optimal_thresholds) * pred_fn.alpha))


def compute_raw_labels(outputs, *args):
	return outputs


def compute_eval_predictions_combined_dataset(dataloader, pred_fn: MultiThresholdREDWrapper, label_selection, y_transform=None, label_computation=compute_optimal_threshold_labels):
	pred_fn.eval()
	stored_raw_results = []
	stored_dataset_ids = []
	stored_ys = []
	names_available = hasattr(dataloader.dataset, 'dataset_names') and dataloader.dataset.dataset_names is not None
	name = ''
	with torch.no_grad():
		for X, y in dataloader:
			X, dataset_ids = X
			y = y.squeeze()
			if y_transform:
				y = y_transform(y)
			stored_ys.append(y.flatten())
			stored_dataset_ids.append(dataset_ids.flatten())
			stored_raw_results.append(pred_fn.model(X).flatten())

		stored_outputs = torch.hstack(stored_raw_results).flatten()
		stored_dataset_ids = torch.hstack(stored_dataset_ids).flatten()
		y = torch.hstack(stored_ys).flatten()
		stored_dataset_ids, sort_indices = torch.sort(stored_dataset_ids)
		stored_outputs = stored_outputs[sort_indices]
		y = y[sort_indices]

		dataset_id_counts = torch.bincount(stored_dataset_ids)
		split_sizes = [i.item() for i in dataset_id_counts if i > 0]
		per_dataset_outputs = torch.split(stored_outputs, split_sizes)
		per_dataset_y = torch.split(y, split_sizes)

		# Split by class
		i = -1  # all ids positive?
		for c_dataset in dataset_id_counts:
			if c_dataset == 0:
				continue
			i += 1
			if names_available:
				name = dataloader.dataset.dataset_names[i]

			outputs = per_dataset_outputs[i]
			ys = per_dataset_y[i]
			# ys, sort_indices = torch.sort(ys)
			# outputs = outputs[sort_indices]
			pred_labels = label_computation(outputs, ys, label_selection, pred_fn)
			# print('Test Dataset', i, optimal_thresholds)
			yield pred_labels.flatten(), ys, name


def altered_test_loop(dataloader, pred_fn: MultiThresholdREDWrapper, loss_function, label_selection=default_label_selection, y_transform=None, print_metrics=True, eval_metric=None, metric_name="Accuracy", metric_mean=False):
	"""
	Computes optimal thresholds for the test datasets, to enable evaluating datasets without training on parts thereof.
	"""
	# Does not compute loss atm
	# pred_fn.eval()
	size = len(dataloader.dataset)
	test_loss, eval_value = 0, 0
	predictions = []
	ground_truth = []
	if not eval_metric:
		eval_metric = lambda a, b: a.isclose(b).type(torch.float).sum()
	for (pred_labels, ys, _) in compute_eval_predictions_combined_dataset(dataloader, pred_fn, label_selection, y_transform):
		predictions.append(pred_labels)
		eval_value += eval_metric(pred_labels, ys)
		ground_truth.append(ys)

	test_loss = -1
	if metric_mean:
		eval_value = eval_value / size
	if print_metrics:
		print(f"{metric_name}: {eval_value:>6f}")
	if isinstance(eval_value, torch.Tensor):
		eval_value = eval_value.item()
	if isinstance(test_loss, torch.Tensor):
		test_loss = test_loss.item()
	return eval_value, test_loss, torch.hstack(predictions), torch.hstack(ground_truth)


def train_loop(dataloader, pred_fn, loss_function, optimizer, label_selection=default_label_selection, epochs=20, eval_dataloader=None, scheduler=None, target_device='cuda', y_transform=None, reporting_freq=1, test_freq=2, train_eval_dataloader=None):
	"""
	Slightly altered train loop from run_model.py, using the altered_test_loop with optimized thresholds.
	"""
	pred_fn.train()
	size = len(dataloader.dataset)
	number_batches = len(dataloader)
	train_accs = []
	train_losses = []
	test_accs = []
	test_losses = []
	print("Train start time:", get_time_string())
	if eval_dataloader:
		WAE = WeightedAE(getClassCounts(eval_dataloader))
	else:
		WAE = nn.L1Loss()
	if train_eval_dataloader:
		WAE_train = WeightedAE(getClassCounts(train_eval_dataloader))
	else:
		WAE_train = WAE
		train_eval_dataloader = dataloader

	metric_name = "Weighted AE"

	# possible future work, simply ignore
	multi_threshold_mode = False
	if hasattr(pred_fn, 'adjust_thresholds'):
		multi_threshold_mode = False
	stored_y = []
	for epoch in range(epochs):
		running_loss = torch.zeros([1], device=target_device, requires_grad=False)
		loss = torch.zeros([1], device=target_device)

		for batch, (X, y) in enumerate(dataloader):
			# Compute prediction and loss
			if multi_threshold_mode:
				pred = pred_fn(X, store_raw_pred=True)
				stored_y.append(y.flatten())
			else:
				pred = pred_fn(X)

			if y_transform:
				y = y_transform(y)
			loss = loss_function(pred, y)

			# Backpropagation
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				running_loss += loss.detach()
		running_loss = running_loss.item()/number_batches
		train_losses.append(running_loss)
		if scheduler:
			scheduler.step()
		print(f"active training loss: {running_loss:>7f}")

		if epoch%10==9:
			if multi_threshold_mode:
				ys = torch.hstack(stored_y)
				pred_fn.adjust_thresholds(ys)
				stored_y = []
		if epoch%10==9:
			if hasattr(pred_fn, 'theta'):
				print(pred_fn.theta)
			if hasattr(pred_fn, 'alpha'):
				if hasattr(pred_fn, 'gamma'):
					print(pred_fn.alpha, pred_fn.gamma)
				else:
					print(pred_fn.alpha)

		if epoch % reporting_freq == reporting_freq-1 or epoch == epochs-1:
			eval_result, loss_train, _, _ = test_loop(train_eval_dataloader, pred_fn, loss_function, label_selection=label_selection, y_transform=y_transform, print_metrics=False, eval_metric=WAE_train, metric_name=metric_name)
			print(f"Base train evaluation: loss: {loss_train:>7f} in epoch {epoch + 1},  {metric_name} {eval_result:>6f}")
			eval_result, loss_train, _, _ = altered_test_loop(train_eval_dataloader, pred_fn, loss_function, label_selection=label_selection, y_transform=y_transform, print_metrics=False, eval_metric=WAE_train, metric_name=metric_name)
			print(f"Optimized train evaluation: loss: {loss_train:>7f} in epoch {epoch + 1},  {metric_name} {eval_result:>6f}")
			train_accs.extend([eval_result]*reporting_freq)

		break_cond = False
		if eval_dataloader:
			if epoch % test_freq == test_freq - 1:
				# print(loss.item(), y[0], pred[0])
				eval_result, test_loss, _, _ = altered_test_loop(eval_dataloader, pred_fn, loss_function, label_selection=label_selection, y_transform=y_transform, eval_metric=WAE, metric_name=metric_name)
				test_accs.extend([eval_result] * test_freq)
				test_losses.extend([test_loss] * test_freq)
			elif epoch == epochs-1 or break_cond:
				# print(loss.item(), y[0], pred[0])
				eval_result, test_loss, _, _ = altered_test_loop(eval_dataloader, pred_fn, loss_function, label_selection=label_selection, y_transform=y_transform, eval_metric=WAE, metric_name=metric_name)
				test_accs.extend([eval_result] * (epochs % test_freq))
				test_losses.extend([test_loss] * (epochs % test_freq))
		if break_cond:
			break
	return train_accs, test_accs, train_losses, test_losses


def load_all_dataset_indices(dataset_names, df_dir, pooling_threshold=0.02):
	"""
	Attempts to load the dataframes for each dataset in the dataframe directory df_dir.
	Also applies class pooling per dataset.
	(Set pooling_threshold = 0 to avoid any pooling)
	"""
	dataset_indices = []
	num_classes = 2
	for name in dataset_names:
		name += ts_name_ext
		file_path = os.path.join(df_dir, name + '.txt')
		try:
			dataframe = pd.read_csv(filepath_or_buffer=file_path, index_col=0)
		except FileNotFoundError:
			print('Dataset not found', name)
			continue
		dataframe['Difficulty'] -= 1
		cm = getClassPoolMap(np.bincount(dataframe['Difficulty'].to_numpy()), threshold=pooling_threshold)
		num_classes = max(num_classes, max(cm.values()))
		dataset_indices.append((name, dataframe, cm))
		print('Found dataset', name)
	return dataset_indices, num_classes


def construct_combined_dataset(dataframes, sequence_dir, dataset_constructor):
	datasets = []
	dataset_names = []
	for name, df, cm in dataframes:
		dataset_names.append(name)
		datasets.append(dataset_constructor(df, sequence_dir, cm))
	return CombinedDataset(datasets, dataset_names)


def LODatasetOCV_Strategy(datasets, sequence_dir, dataset_constructor):
	"""
	Leave one dataset out CrossValidation.
	One of multiple strategies to construct train and test datasets.
	Provides a finite generator
	"""

	for current_fold in range(len(datasets)):
		train_set = construct_combined_dataset([datasets[i] for i in range(len(datasets)) if i != current_fold], sequence_dir, dataset_constructor)
		test_set = construct_combined_dataset([datasets[current_fold]], sequence_dir, dataset_constructor)
		print('Current Test set', datasets[current_fold][0])

		yield train_set, test_set, [datasets[current_fold]]


def prepare_groups(datasets, group_dict: dict):
	inv_group_dict = {}
	for k, v in group_dict.items():
		if v in inv_group_dict:
			inv_group_dict[v].append(k)
		else:
			inv_group_dict[v] = [k]
	dataset_names = [ds[0] for ds in datasets]
	groups_present = [dataset for dataset in datasets if dataset[0] in inv_group_dict.keys()]
	groups_complete = []
	groups_complete_individual_dataset_dict = {}
	for grouped_dataset in groups_present:
		complete = True
		individual_datasets = []
		expected_datasets = inv_group_dict[grouped_dataset[0]]
		for j in expected_datasets:
			try:
				idx = dataset_names.index(j)
			except ValueError:
				complete = False
				break
			individual_datasets.append(datasets[idx])
		if complete:
			groups_complete.append(grouped_dataset)
			groups_complete_individual_dataset_dict[grouped_dataset[0]] = individual_datasets
	return groups_complete, groups_complete_individual_dataset_dict


def groupedLOOCV_Strategy(datasets, sequence_dir, dataset_constructor, group_dict: dict):
	"""
	Handles sets of datasets as indivisible groups and applies LOOCV on these groups.
	For the MultiThresholdModel, groups are resolved on the training side into individual datasets.

	:param datasets: Assumes both grouped datasets and all corresponding individual datasets included.
	:param group_dict: denotes individual -> group relationship.
	"""
	groups_complete, groups_complete_individual_dataset_dict = prepare_groups(datasets, group_dict)

	# assert len(groups_complete) > 0  # 1?
	for current_fold in range(len(groups_complete)):
		test_set = construct_combined_dataset([groups_complete[current_fold]], sequence_dir, dataset_constructor)
		# print('Current Test set', groups_complete[current_fold][0])
		grouped_train_datasets = [groups_complete[i] for i in range(len(groups_complete)) if i != current_fold]
		train_datasets = []
		for grouped_dataset in grouped_train_datasets:
			train_datasets.extend(groups_complete_individual_dataset_dict[grouped_dataset[0]])
		# print("Train sets:", [ds[0] for ds in train_datasets])
		train_set = construct_combined_dataset(train_datasets, sequence_dir, dataset_constructor)
		yield train_set, test_set, [groups_complete[current_fold]]


def approxMCCV_fillTrain(datasets, indices, threshold):
	train_datasets = [datasets[indices[0]]]

	# datasets: list of (name, df as index, class pool map)
	total_size = sum([len(dataset) for (_, dataset, _) in datasets])
	current_size = len(train_datasets[0][1])
	for i in indices[1:-1]:
		new_dataset = datasets[i]
		k = len(new_dataset[1])
		current_size += k
		if current_size / total_size > threshold:
			# add dataset crossing threshold only if dataset exceeds by less than 50%
			if (current_size - k / 2) / total_size < threshold:
				train_datasets.append(new_dataset)
			break
		train_datasets.append(new_dataset)
	return train_datasets


def approximateMCCV_Strategy(datasets, sequence_dir, dataset_constructor, threshold=0.8, rng=None, seed=None):
	"""
	Constructs a train/test split by randomly adding datasets to the train set until the threshold is reached (or only one element remains).
	One of multiple strategies to construct train and test datasets.
	Provides an infinite generator
	"""
	if rng is None:
		rng = np.random.default_rng(seed=seed)
	indices = np.arange(len(datasets))
	while True:
		rng.shuffle(indices)
		train_datasets = approxMCCV_fillTrain(datasets, indices, threshold)

		test_datasets = [dataset for dataset in datasets if dataset not in train_datasets]
		train_set = construct_combined_dataset(train_datasets, sequence_dir, dataset_constructor)
		test_set = construct_combined_dataset(test_datasets, sequence_dir, dataset_constructor)
		# print('Current Test Set', [obj[0] for obj in test_datasets])
		yield train_set, test_set, test_datasets


def grouped_aMCCV_strategy(datasets, sequence_dir, dataset_constructor, group_dict: dict, threshold=0.8, dropout=0.1, rng=None, seed=None):
	if dropout >= 1:
		raise ValueError("Can't train with over 100% dropout")
	groups_complete, groups_complete_individual_dataset_dict = prepare_groups(datasets, group_dict)
	if rng is None:
		rng = np.random.default_rng(seed=seed)
	indices = np.arange(len(groups_complete))
	while True:
		rng.shuffle(indices)
		train_datasets = approxMCCV_fillTrain(groups_complete, indices, threshold)

		test_datasets = [dataset for dataset in groups_complete if dataset not in train_datasets]
		individual_train_datasets = []
		for grouped_dataset in train_datasets:
			individual_train_datasets.extend(groups_complete_individual_dataset_dict[grouped_dataset[0]])

		if dropout > 0:
			level2_indices = np.arange(len(individual_train_datasets))
			rng.shuffle(indices)
			cutoff_idx = floor(len(individual_train_datasets)*dropout)
			remaining_train_datasets = []
			for i in level2_indices[cutoff_idx:]:
				remaining_train_datasets.append(individual_train_datasets[i])
			individual_train_datasets = remaining_train_datasets

		train_set = construct_combined_dataset(individual_train_datasets, sequence_dir, dataset_constructor)
		test_set = construct_combined_dataset(test_datasets, sequence_dir, dataset_constructor)
		# print('Current Test Set', [obj[0] for obj in test_datasets])
		yield train_set, test_set, test_datasets


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-root', type=str, help='Root directory. Simplifies definition of other directories',
	                    required=False)
	parser.add_argument('-input_dir', type=str, help='Time series input directory', required=False)
	parser.add_argument('-model_dir', type=str, help='Output directory', required=False)
	parser.add_argument('-device', type=str, help='Model Training device', required=False)
	parser.add_argument('-run_id', type=str, help='Opt. ID associated with this execution', required=False)

	parser.set_defaults(
		root='',
		input_dir='data',
		output_dir='model_artifacts/',
		device='cuda',
		run_id='',
	)
	torch.backends.cudnn.benchmark = True
	cmd_args = parser.parse_args()

	all_datasets = ["CinderellaGirlsStarlightDancefloor", "CinderellaGirlsStarlightRemix", "FraxtilsArrowArrangements", "FraxtilsBeastBeats", "Galaxy", "GpopsPackofOriginalPadSims", "GpopsPackofOriginalPadSimsII", "GpopsPackofOriginalPadSimsIII", "GullsArrows", "InTheGroove", "InTheGroove2", "InTheGroove3", "InTheGrooveRebirth", "KantaiCollectionPadColle", "KantaiCollectionPadColleKai", "TouhouGouyoukyousokuTouhouPadPackRevival", "TouhouKousaikaiScarletFestivalGatheringvideoless", "TouhouOumukanSakuraDreamSensation", "TsunamixIII", "VocaloidProjectPadPack4thVideoless", "VocaloidProjectPadPack5th", "ITG", "fraxtil", "Gpop"]
	# all_datasets = ["FraxtilsArrowArrangements", "FraxtilsBeastBeats", "TsunamixIII", "fraxtil", "GpopsPackofOriginalPadSimsIII", "GpopsPackofOriginalPadSimsII", "GpopsPackofOriginalPadSims", "GullsArrows", "InTheGroove2", "InTheGroove3", "InTheGrooveRebirth", "InTheGroove", "ITG", "KantaiCollectionPadColleKai", "KantaiCollectionPadColle", ]
	group_relationship_dict = {"CinderellaGirlsStarlightDancefloor":'Gpop', "CinderellaGirlsStarlightRemix":'Gpop', "FraxtilsArrowArrangements":"fraxtil", "FraxtilsBeastBeats":"fraxtil", "Galaxy":"Galaxy", "GpopsPackofOriginalPadSims":'Gpop', "GpopsPackofOriginalPadSimsII":'Gpop', "GpopsPackofOriginalPadSimsIII":'Gpop', "GullsArrows":'GullsArrows', "InTheGroove":"ITG", "InTheGroove2":"ITG", "InTheGroove3":"ITG", "InTheGrooveRebirth":"ITG", "KantaiCollectionPadColle":'Gpop', "KantaiCollectionPadColleKai":'Gpop', "TouhouGouyoukyousokuTouhouPadPackRevival":'Gpop', "TouhouKousaikaiScarletFestivalGatheringvideoless":'Gpop', "TouhouOumukanSakuraDreamSensation":'Gpop', "TsunamixIII":"fraxtil", "VocaloidProjectPadPack4thVideoless":'Gpop', "VocaloidProjectPadPack5th":'Gpop'}
	root = cmd_args.root
	input_dir = os.path.join(root, cmd_args.input_dir)
	output_dir = os.path.join(root, cmd_args.output_dir)
	input_dir = os.path.join(input_dir, "time_series")
	eval_dir = os.path.join(output_dir, "evaluations")

	time_series_dir = os.path.join(input_dir, 'repository')
	if not os.path.isdir(time_series_dir):
		raise IOError('Data repository not found.')
	start_time = get_time_string()
	print('Start time', start_time)
	run_id = cmd_args.run_id
	execution_id = start_time + ("_{}".format(run_id) if len(run_id) > 0 else '')

	learning_rate = 1e-4
	ts_sample_freq = 0
	b_variant = False
	weight_decay = 5e-2
	ts_name_ext = ""
	train = True
	reset = True
	# save_predictions = False
	store_raw_predictions = True
	CV_dir_name = '20230421-1550_6992764'
	eval_CV = len(CV_dir_name) > 0

	FinalActivation = nn.Identity()
	minLossSelection = minLossSelectionRED
	n_classes = 1
	multi_agg_variant = 7.1
	ytransform = lambda x: x.type(torch.long)

	batch_size = 128
	multi_sample = True
	chart_channels = 31
	ts_in_channels = chart_channels
	num_epochs = 100
	sub_samples = 8
	sample_start_interval = 1
	model_sample_size = 60
	cross_validation = 25

	if eval_CV:
		execution_id = CV_dir_name
		start_time = CV_dir_name.split('_')[0]
		run_id = CV_dir_name.split('_')[1]

	if store_raw_predictions:
		label_computation_for_storing = compute_raw_labels
		prediction_type = torch.float
	else:
		label_computation_for_storing = compute_optimal_threshold_labels
		prediction_type = torch.int

	if not torch.cuda.is_available():
		device = 'cpu'
	else:
		device = cmd_args.device

	loaded_dataset_indices, all_classes = load_all_dataset_indices(all_datasets, input_dir, pooling_threshold=0)
	all_classes = all_classes - 1
	ClassificationLoss = RelativeEntropy(number_classes=all_classes, target_device=device)

	general_dataset_constructor = lambda dataframe, ts_dir, cm, seed=None: FixedSizeInputSampleTS(ClassPoolingWrapper(
		TimeSeriesDataset(dataframe, ts_dir),
		cm), sample_size=model_sample_size, k=sample_start_interval, sub_samples=sub_samples, multisample_mode=multi_sample, seed=seed)

	train_dataloader_constructor = lambda train_dataset, seed_dl=None: getWeightedDataLoader(
		train_dataset, target_device=device, batch_size=batch_size, seed=seed_dl, collate_fn=combined_dataset_collate
	)

	unweighted_dataloader_constructor = lambda train_dataset: DataLoader(
		train_dataset, batch_size=batch_size, collate_fn=combined_dataset_collate,
		# train_dataset
	)

	modelBase_constructor = lambda: TimeSeriesTransformerModel(ts_in_channels, device, out_channels=n_classes,
	                                                           sequence_length=model_sample_size,
	                                                           final_activation=FinalActivation)
	if multi_sample:
		model_constructor = lambda: MultiWindowModelWrapper(modelBase_constructor(), output_classes=n_classes,
		                                                    final_activation=FinalActivation, variant=multi_agg_variant)
	else:
		model_constructor = modelBase_constructor

	if all_classes > 0:
		model_constructor_old = model_constructor
		# todo: len(loaded dataset indices) > len(train dataset indices), but simplifies construction.
		model_constructor = lambda n_thresholds=len(loaded_dataset_indices): MultiThresholdREDWrapper(model_constructor_old(), target_device=device,
		                                                     output_classes=all_classes,
		                                                     thresholds=n_thresholds)

	optim_constructor = lambda model: torch.optim.AdamW([{'params': model.model.parameters()}, {'params': [model.alpha], 'lr': 100*learning_rate, 'weight_decay': weight_decay}, {'params': [model.gamma], 'lr': 20*learning_rate, 'weight_decay': 0.2*weight_decay}, {'params': [model.theta], 'lr': 20*learning_rate, 'weight_decay': 0}], lr=learning_rate, weight_decay=weight_decay)
	#                                                    {'params': [model.alpha], 'lr': 20*learning_rate, 'weight_decay': weight_decay},
	rep_f = 20
	test_f = 10

	if cross_validation > 0:

		eval_developments = []
		final_performances = [[], []]
		prediction_frame_dict = {}
		start = 0

		print('Execution ID: {}'.format(execution_id))
		model_dir = os.path.join(output_dir, "saved_models", execution_id)
		if not os.path.isdir(model_dir):
			os.mkdir(model_dir)

		seed_data = 12345
		# strategy = approximateMCCV_Strategy(loaded_dataset_indices, time_series_dir, general_dataset_constructor, seed=seed_data)
		# strategy = LODatasetOCV_Strategy(loaded_dataset_indices, time_series_dir, general_dataset_constructor)
		# strategy = groupedLOOCV_Strategy(loaded_dataset_indices, time_series_dir, general_dataset_constructor, group_relationship_dict)
		strategy = grouped_aMCCV_strategy(loaded_dataset_indices, time_series_dir, general_dataset_constructor, group_relationship_dict, seed=seed_data)
		for fold, (constructed_train_dataset, constructed_test_dataset, chosen_test_datasets) in enumerate(strategy):
			if fold >= cross_validation:
				break

			# seed_data = fold
			torch.manual_seed(fold)

			model = model_constructor(len(loaded_dataset_indices)-len(chosen_test_datasets))
			optimizer = optim_constructor(model)

			if fold == 0:
				print_model_parameter_overview(model)

			# Takes very long -> Todo: Speedup possible?
			weighted_train_dataloader = train_dataloader_constructor(constructed_train_dataset, seed_dl=seed_data)
			unweighted_train_data = unweighted_dataloader_constructor(constructed_train_dataset)
			eval_dataloader = unweighted_dataloader_constructor(constructed_test_dataset)

			lr_scheduler = None     # torch.optim.lr_scheduler.StepLR(optimizer, max(80, 3 * num_epochs // 4), gamma=0.5)

			print("############")
			print("CV Run {}: ".format(fold + 1))
			print("############")

			print('Datasets')
			print('Train:', constructed_train_dataset.dataset_names if constructed_train_dataset.dataset_names is not None else '')
			print('Test:', constructed_test_dataset.dataset_names if constructed_test_dataset.dataset_names is not None else '')


			model_save_path = os.path.join(model_dir, '{}_{}_weights_Run_{}.pth'.format(start_time, model.name, fold))
			if eval_CV and os.path.isfile(model_save_path):
				model.load_state_dict(torch.load(model_save_path))
			else:
				print(model_save_path)
				train_results = train_loop(weighted_train_dataloader, model, ClassificationLoss, optimizer, scheduler=lr_scheduler,
				                        eval_dataloader=eval_dataloader, epochs=num_epochs,
				                        y_transform=ytransform, reporting_freq=rep_f, test_freq=test_f,
				                        train_eval_dataloader=unweighted_train_data, label_selection=minLossSelection)
				optimizer.zero_grad()
				torch.save(model.state_dict(), model_save_path)
				eval_developments.append(train_results[1])
				final_performances[0].append(train_results[0][-1])
				final_performances[1].append(train_results[1][-1])

			dataset_number = -1
			for (dataset_predictions, dataset_labels, dataset_name) in compute_eval_predictions_combined_dataset(eval_dataloader, model, minLossSelection, ytransform, label_computation=label_computation_for_storing):
				dataset_number += 1
				if chosen_test_datasets[dataset_number][0] != dataset_name:
					print("!!!")
					print("Mismatch between predicted dataset and dataset index! {} != {}".format(chosen_test_datasets[dataset_number][0], dataset_name))
					print("!!!")
				if dataset_name not in prediction_frame_dict:
					prediction_frame_dict[dataset_name] = []
				# print(chosen_test_datasets[dataset_number])
				prediction_frame = chosen_test_datasets[dataset_number][1]
				prediction_frame = prediction_frame.loc[:, ['Name', 'Difficulty', 'Permutation', 'sm_fp']].copy()
				prediction_frame['Run'] = fold
				prediction_frame['Predicted Difficulty'] = dataset_predictions.detach().to(device='cpu', dtype=prediction_type).numpy()
				prediction_frame['Pooled Difficulty'] = dataset_labels.detach().to(device='cpu', dtype=prediction_type).numpy()
				prediction_frame_dict[dataset_name].append(prediction_frame.copy())

		for i in range(len(eval_developments)):
			plt.plot(eval_developments[i], label='Eval Run {}'.format(i))
		img_path = os.path.join(output_dir, '{}_Loss_{}.png'.format(get_time_string(), "CV_{}".format(model.name)))
		plt.savefig(img_path)
		plt.show()
		final_performances = np.array(final_performances)
		print("Final results:", final_performances)
		if final_performances.shape[0] > 0: #????
			print(final_performances.shape)
			mean, std = final_performances.mean(axis=1), final_performances.std(axis=1)
			print("Results Train - Mean: {}  Std: {}".format(mean[0], std[0]))
			print("Results Eval - Mean: {}  Std: {}".format(mean[1], std[1]))

		for frame_key in prediction_frame_dict.keys():
			eval_CV_dir = os.path.join(eval_dir, execution_id)
			if not os.path.isdir(eval_CV_dir):
				os.mkdir(eval_CV_dir)
			full_prediction_frame = pd.concat(prediction_frame_dict[frame_key], axis=0, ignore_index=True)
			full_prediction_frame.to_csv(os.path.join(eval_CV_dir, "{}_predicted.txt".format(frame_key)))

	else:
		strategy = approximateMCCV_Strategy(loaded_dataset_indices, time_series_dir, general_dataset_constructor)
		for (constructed_train_dataset, constructed_test_dataset, chosen_test_datasets) in strategy:
			# Hope it has at least one element?
			break

		# print(actual_datasets)
		train_dataloader = train_dataloader_constructor(constructed_train_dataset)
		# print('train loader done')
		unweighted_train_dataloader = unweighted_dataloader_constructor(constructed_train_dataset)
		evaluation_dataloader = unweighted_dataloader_constructor(constructed_test_dataset)

		model1 = model_constructor(len(loaded_dataset_indices)-len(chosen_test_datasets))
		optim = optim_constructor(model1)

		dataset_name = 'CombinedDatasets'

		model_path = os.path.join(output_dir, "saved_models",
		                          '{}_weights_{}.pth'.format(model1.name, dataset_name))
		optim_path = os.path.join(output_dir, "saved_models", "{}_optimizer_{}".format(model1.name, dataset_name))

		t1 = print_model_parameter_overview(model1)
		if not reset and os.path.isfile(model_path):
			st_dict = torch.load(model_path)

			if t1 == get_total_number_of_parameters(st_dict):
				model1.load_state_dict(st_dict)
				print("model loaded")
				if train and os.path.isfile(optim_path):
					optim.load_state_dict(torch.load(optim_path))

		model1.name = model1.name + "_{}".format(t1)
		configuration_name = "{}_{}_{}".format(model1.name, dataset_name, num_epochs)

		print('ready for training')
		if train:
			loss_fn = ClassificationLoss
			lr_scheduler = None
			accuracies = train_loop(train_dataloader, model1, loss_fn, optim, scheduler=lr_scheduler,
			                        eval_dataloader=evaluation_dataloader, epochs=num_epochs,
			                        y_transform=ytransform, reporting_freq=rep_f, test_freq=test_f,
			                        train_eval_dataloader=unweighted_train_dataloader, label_selection=minLossSelection)
			torch.save(model1.state_dict(), model_path)
			torch.save(optim.state_dict(), optim_path)
			plt.plot(accuracies[2])
			plt.plot(accuracies[0])
			plt.plot(accuracies[1])
			img_path = os.path.join(eval_dir, '{}_Loss_{}.png'.format(get_time_string(), configuration_name))
			plt.savefig(img_path)
			plt.show()

		# print(class_pool_map)

		const_loss = lambda a, b: 0
		evaluate_other = False
		y_shift = 0
		if not evaluate_other:

			# evaluation_dataloader = test_dataloader_constructor(test_dataframe, class_pool_map)
			evaluate_trained_model_on_dataset(model1, evaluation_dataloader, minLossSelection, y_transform=ytransform,
			                                  output_path=eval_dir, config_name=configuration_name, plot_results=True,
			                                  label_shift=y_shift)
			# save_predictions(test_dataframe, model1, evaluation_dataloader, os.path.join(input_dir, "{}_predicted.txt".format(dataset_name)), selection_function=minLossSelection)






