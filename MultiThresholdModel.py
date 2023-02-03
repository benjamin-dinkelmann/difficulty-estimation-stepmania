import torch

from run_model import *
from torch.utils.data import Dataset


class RAMTSDataset(TimeSeriesDataset):
	def __init__(self, label_frame, data_input_dir, transform=None, target_transform=None):
		super().__init__(label_frame, data_input_dir, transform, target_transform, None)

	def __len__(self):
		return len(self.label_frame)

	def __getitem__(self, idx):
		if idx in self.buffer:
			time_series, label = self.buffer[idx]
		else:
			label = self.label_frame.iloc[idx, 1]
			variant = self.label_frame.iloc[idx, 2]
			ts_path = os.path.join(self.time_series_dir, self.label_frame.iloc[idx, 0] + '-{}_{}.pt'.format(label + 1, variant))
			time_series = torch.load(ts_path, map_location='cpu')

			label = torch.tensor(label, dtype=torch.long, device='cpu', requires_grad=False)

			self.buffer[idx] = (time_series, label)

		if self.transform:
			time_series = self.transform(time_series)
		if self.target_transform:
			label = self.target_transform(label)
		return time_series.to(self.target_device), label.to(self.target_device)


class CombinedDataset(Dataset):
	def __init__(self, datasets):
		assert len(datasets) > 0

		self.datasets = datasets
		dataset_sizes = []
		for dataset in datasets:
			dataset_sizes.append(len(dataset))
		self.dataset_sizes = np.array(dataset_sizes)
		dataset_sizes.insert(0, 0)
		self.cum_sizes = np.cumsum(dataset_sizes)
		print(self.cum_sizes)

	def __len__(self):
		return self.cum_sizes[-1]

	def __getitem__(self, index):
		# Assume idx in range
		idx = np.argmin(index >= self.cum_sizes) - 1
		# error if idx == -1 since either index < 0 or index > len(self)
		error_message = "Illegal index: {}".format(index)
		assert idx != -1, error_message

		X, y = self.datasets[idx][index - self.cum_sizes[idx]]
		return (X, torch.tensor([idx], device=X.device)), y


def combined_dataset_collate(batch):
	A, B, C = [], [], []
	for (a, b), c in batch:
		A.append(a[None, :])
		B.append(b)
		C.append(c)
	return (torch.vstack(A), torch.hstack(B)), torch.hstack(C)


class MultiThresholdREDWrapper(REDWrapper):
	def __init__(self, sub_model, target_device=default_device, output_classes=20, thresholds=1):
		super().__init__(sub_model, target_device, output_classes)
		self.name = "MTRED_"+self.name
		require_grad = False
		if isinstance(thresholds, torch.Tensor):
			self.theta = nn.Parameter(torch.clone(thresholds), requires_grad=require_grad)
		elif type(thresholds) == int:
			self.theta = nn.Parameter(torch.tile(torch.arange(output_classes, device=target_device, dtype=torch.float), (thresholds, 1)), requires_grad=require_grad)
		else:
			raise TypeError("Given thresholds of incorrect type. Either provide a tensor containing intial thresholds or an int specifying the number of sets of thresholds.")
		self.stored_outputs = []
		self.stored_dataset_ids = []
		self.beta = 0.2
		self.gamma = 1  # 1/(max(1-self.beta, 1e-2))

	def forward(self, x, store_raw_pred=False):
		x, dataset_ids = x
		model_output = self.model(x)
		if store_raw_pred:
			self.stored_outputs.append(model_output.detach().flatten())
			self.stored_dataset_ids.append(dataset_ids.detach().flatten())

		return self.sigmoid(model_output - self.theta[dataset_ids])

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

			"""lower = [-1]
			upper = [-1, -1]
			# c = -1
			for j, c_class in enumerate(y_counts):
				# empty classes basically get no threshold at all
				last_threshold = upper[-1]
				if c_class == 0:
					lower.append(last_threshold)
					upper.append(last_threshold)
				else:
					# c += 1
					# outputs = per_class_outputs[c]
					# compute upper and lower edge of per class distribution, but enfore ascending order
					lower.append(max(torch.quantile(outputs, 0.2).item(), lower[-1]+eps))
					upper.append(max(torch.quantile(outputs, 0.8).item(), upper[-1]+eps))
			# print(lower)
			# print(upper)
			lower.pop(0)
			lower.pop(0)
			upper.pop(0)
			upper.pop(0)
			upper.pop(-1)"""

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


def getWeightedDataLoader(dataset, target_device=default_device, batch_size=1, seed=None, collate_fn=combined_dataset_collate):
	unweighted_dataloader = DataLoader(dataset)
	class_weights = getClassCounts(unweighted_dataloader)
	weights = torch.tensor([class_weights[int(i)] for _, i in unweighted_dataloader], requires_grad=False)
	rng = torch.Generator()
	if seed is not None:
		rng.manual_seed(seed)
	sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=rng)
	return DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-root', type=str, help='Root directory. Simplifies definition of other directories',
	                    required=False)
	parser.add_argument('-input_dir', type=str, help='Time series input directory', required=False)
	parser.add_argument('-model_dir', type=str, help='Output directory', required=False)
	parser.add_argument('-device', type=str, help='Model Training device', required=False)

	parser.set_defaults(
		root='',
		input_dir='data',
		output_dir='model_artifacts/',
		device='cuda',
	)
	torch.backends.cudnn.benchmark = True
	args = parser.parse_args()

	# all_datasets = ["CinderellaGirlsStarlightDancefloor", "CinderellaGirlsStarlightRemix", "FraxtilsArrowArrangements", "FraxtilsBeastBeats", "Galaxy", "GpopsPackofOriginalPadSimsIII", "GpopsPackofOriginalPadSimsII", "GpopsPackofOriginalPadSims", "GullsArrows1", "GullsArrows2", "GullsArrows3", "GullsArrows4", "GullsArrows5", "GullsArrows6", "GullsArrows7", "InTheGroove2", "InTheGroove3", "InTheGrooveRebirth", "InTheGroove", "KantaiCollectionPadColleKai", "KantaiCollectionPadColle", "TouhouGouyoukyousokuTouhouPadPackRevival", "TouhouKousaikaiScarletFestivalGathering_videoless_", "TouhouOumukanSakuraDreamSensation", "TsunamixIII", "VocaloidProjectPadPack4th_Videoless_", "VocaloidProjectPadPack5th"]
	all_datasets = ["FraxtilsArrowArrangements", "FraxtilsBeastBeats", "GpopsPackofOriginalPadSimsIII", "GpopsPackofOriginalPadSimsII", "GpopsPackofOriginalPadSims", "GullsArrows", "InTheGroove2", "InTheGroove3", "InTheGrooveRebirth", "InTheGroove", "KantaiCollectionPadColleKai", "KantaiCollectionPadColle", ]

	root = args.root
	input_dir = os.path.join(root, args.input_dir)
	output_dir = os.path.join(root, args.output_dir)
	input_dir = os.path.join(input_dir, "time_series")
	eval_dir = os.path.join(output_dir, "evaluations")

	if not torch.cuda.is_available():
		device = 'cpu'
	else:
		device = args.device

	n_classes = 16   # Todo: Max of all
	learning_rate = 1e-4
	ts_sample_freq = 0
	b_variant = False
	weight_decay = 5e-2
	ts_name_ext = "_{}".format(ts_sample_freq) + ("b" if b_variant else "")
	train = True
	reset = True

	FinalActivation = nn.Identity()
	minLossSelection = minLossSelectionRED
	all_classes = n_classes - 1
	ClassificationLoss = RelativeEntropy(number_classes=all_classes)
	n_classes = 1
	multi_agg_variant = 7.1
	ytransform = lambda x: x.type(torch.long)

	batch_size = 128
	multi_sample = True
	chart_channels = 19
	ts_in_channels = chart_channels
	num_epochs = 100
	sub_samples = 8
	sample_start_interval = 1
	model_sample_size = 60
	seed = 1234

	train_dataset_constructor = lambda train_df, time_series_dir, class_pool, seed_rtss=None: FixedSizeInputSampleTS(ClassPoolingWrapper(
		TimeSeriesDataset(train_df, time_series_dir),
		class_pool), sample_size=model_sample_size, k=sample_start_interval, sub_samples=sub_samples, multisample_mode=multi_sample, seed=seed_rtss)
	test_dataset_constructor = lambda test_df,  time_series_dir, class_pool, seed_rtss=None: FixedSizeInputSampleTS(ClassPoolingWrapper(
		TimeSeriesDataset(test_df, time_series_dir),
		class_pool),
		sample_size=model_sample_size, k=sample_start_interval, sub_samples=sub_samples, multisample_mode=multi_sample, seed=seed_rtss)


	train_dataloader_constructor = lambda train_dataset, seed_dl=None: getWeightedDataLoader(
		train_dataset, target_device=device, batch_size=batch_size, seed=seed_dl
	)

	unweighted_train_dataloader_constructor = lambda train_dataset: DataLoader(
		train_dataset, batch_size=batch_size, collate_fn=combined_dataset_collate,
		# train_dataset
	)

	test_dataloader_constructor = lambda test_dataset: DataLoader(
		test_dataset, batch_size=batch_size, collate_fn=combined_dataset_collate,
		# test_dataset
	)

	modelBase_constructor = lambda: TimeSeriesTransformerModel(ts_in_channels, device, out_channels=n_classes,
	                                                           sequence_length=model_sample_size,
	                                                           final_activation=FinalActivation)
	if multi_sample:
		model_constructor = lambda: MultiWindowModelWrapper(modelBase_constructor(), output_classes=n_classes,
		                                                    final_activation=FinalActivation, variant=multi_agg_variant)
	else:
		model_constructor = modelBase_constructor

	optim_constructor = lambda model: torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

	train_test_datasets = []
	actual_datasets = []
	for dataset_name in all_datasets:
		# raw_dataset_name = dataset_name
		dataset_name += ts_name_ext
		time_series_dir = os.path.join(input_dir, dataset_name)
		if not os.path.isdir(time_series_dir):
			continue
		actual_datasets.append(dataset_name)
		print(dataset_name)
		input_file = os.path.join(input_dir, dataset_name + '.txt')
		cm, train_df, test_df, _ = create_data_split(input_file, .8, accept_fail=True, seed=seed) #, seed=seed_data_split
		single_train_dataset = train_dataset_constructor(train_df, time_series_dir, cm)
		single_test_dataset = test_dataset_constructor(test_df, time_series_dir, cm)
		train_test_datasets.append((single_train_dataset, single_test_dataset))
	# print(actual_datasets)
	constructed_train_dataset = CombinedDataset([dataset[0] for dataset in train_test_datasets])
	train_dataloader = train_dataloader_constructor(constructed_train_dataset)
	print('train loader done')
	unweighted_train_dataloader = unweighted_train_dataloader_constructor(constructed_train_dataset)
	evaluation_dataloader = test_dataloader_constructor(CombinedDataset([dataset[1] for dataset in train_test_datasets]))

	if all_classes > 0:
		model_constructor_old = model_constructor
		model_constructor = lambda: MultiThresholdREDWrapper(model_constructor_old(), target_device=device,
		                                       output_classes=all_classes, thresholds=len(actual_datasets))

	model1 = model_constructor()
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
		rep_f = 20
		test_f = 10
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






