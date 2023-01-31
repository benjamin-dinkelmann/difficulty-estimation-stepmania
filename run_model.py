import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from util import get_time_string, split_containing_all_classes, beep
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from time_series_model import *
from simple_model import SimplePatternModel, prepare_pattern_dataset
from matplotlib.ticker import PercentFormatter

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeightedAE(nn.Module):
	def __init__(self, class_weights):
		super().__init__()
		self.class_weights = class_weights
		m = len(class_weights[class_weights > 0])
		self.C = 1/m if m > 0 else 1

	def forward(self, inputs, targets):
		if not targets.dtype == torch.long:
			targets = targets.long()
		errors = self.C * self.class_weights[targets] * (inputs-targets).abs()
		return errors.sum()


class BalancedMetric(nn.Module):  # compute a version of a metric normalized by class
	# raw metric = without summation or normalization
	def __init__(self, class_weights, raw_metric, final_op=None):
		super().__init__()
		self.class_weights = class_weights
		m = len(class_weights[class_weights > 0])
		self.C = 1 / m if m > 0 else 1
		self.metric = raw_metric
		if final_op is None:
			self.do_final = False
			self.final_f = None
		else:
			self.do_final = True
			self.final_f = final_op

	def forward(self, inputs, targets):
		if not targets.dtype == torch.long:
			targets = targets.long()
		errors = self.C * self.class_weights[targets] * self.metric(inputs, targets)
		if self.do_final:
			return self.final_f(errors.sum())
		else:
			return errors.sum()


# Classification Loss
class LaplaceLoss(nn.Module):
	# Assumes classes 0 - (#classes - 1)
	def __init__(self, number_classes=20, target_device=default_device):
		super().__init__()
		self.nClasses = number_classes
		self.v = torch.arange(number_classes, dtype=torch.float, device=target_device, requires_grad=False).reshape(1, -1)
		self.machine_eps = 1e-16

	def forward(self, inputs, targets):
		weights = (self.v - targets.unsqueeze(-1)).abs()
		inv_weights = (-weights).exp()
		inv_weights = inv_weights/(inv_weights.sum(dim=-1, keepdim=True))  # Normalization
		return -(((inputs + self.machine_eps).log() * inv_weights).sum(dim=-1)).mean()


class BinomialTargetCE(nn.Module):
	def __init__(self, number_classes=20, target_device=default_device, variance=1):
		super().__init__()
		self.nClasses = number_classes - 1
		self.machine_eps = 1e-16
		if variance < 1e-10: # close to zero or negative
			self.variant = 0
		else:
			self.variant = 1

		n = torch.tensor([self.nClasses], dtype=torch.float, device=target_device, requires_grad=False)
		ks = torch.arange(number_classes, dtype=torch.float, device=target_device, requires_grad=False)
		self.ps = ks/n

		one = torch.ones([1], dtype=torch.float, device=target_device, requires_grad=False)
		if self.variant == 0:  # Standard Binomial
			self.binomials = ((n+1).lgamma() - (ks+1).lgamma() - ((n - ks) + 1).lgamma()).exp()
			ks_prime = ks[None, :]
			ps = self.ps[:, None]
			self.soft_weights = self.binomials[None, :] * ps.pow(ks_prime) * (1-ps).pow(n-ks_prime)
		else:  # Poisson-Binomial
			self.mu = ks
			ks_prime = ks[None, :, None]
			i_prime = ks[None, None, :]
			eps = 1e-5 * one
			assert torch.logical_and(self.mu >= 0, self.mu <= n).all()
			alpha = ((self.mu * (1 - self.ps) - variance).maximum(one-one) / (self.mu.maximum(eps) * (1 + self.mu / (n - self.mu).maximum(eps)))).sqrt()
			mu_prime = self.mu[:, None, None]
			ps = torch.vstack([self.ps + alpha, self.ps - self.mu*alpha/(n - self.mu).maximum(eps)])
			assert torch.logical_and(ps <= 1, ps >= 0).all()
			valid = torch.logical_and(i_prime <= mu_prime, i_prime >= (mu_prime + ks_prime - n))  # 2mu - n?
			self.binomials = (((n - mu_prime + 1).lgamma() + (mu_prime + 1).lgamma()) - (ks_prime - i_prime + 1).maximum(one).lgamma()
								- (i_prime+1).lgamma() - (mu_prime - i_prime + 1).maximum(one).lgamma() - (n - mu_prime - ks_prime + i_prime + 1).maximum(one).lgamma()).exp() * valid.to(torch.float)
			p = ps[:, :, None, None]
			self.ps = ps
			stable_results = torch.logical_not(torch.logical_or(p.isclose(0*one), p.isclose(one)))
			stable_results_num = stable_results.to(torch.float)
			p = p.where(stable_results, 0.5*one)
			products = (((p[0].log() * i_prime + (1 - p[0]).log() * (mu_prime - i_prime)
						 + (p[1].log() * (ks_prime - i_prime))*stable_results_num[0]
						 + (1 - p[1]).log() * (n - mu_prime - ks_prime + i_prime)) * stable_results_num[1]) * valid).exp()
			self.soft_weights = (self.binomials * products).sum(dim=-1)

	def forward(self, inputs, targets):
		long_targets = targets.long()
		return -(((inputs + self.machine_eps).log() * self.soft_weights[long_targets]).sum(dim=-1)).mean() - 1


class LogNLLLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.nllloss = nn.NLLLoss()
		self.machine_eps = 1e-16

	def forward(self, inputs, targets):
		return self.nllloss((inputs+self.machine_eps).log(), targets)


class RelativeEntropy(nn.Module): # Series of Binary Classifiers
	def __init__(self, number_classes=20, target_device=default_device):
		super().__init__()
		self.class_labels = torch.arange(number_classes, device=target_device)
		self.machine_eps = 1e-10

	def forward(self, inputs, targets):
		smooth_targets = (self.class_labels < targets.unsqueeze(-1)).to(torch.float)
		return -(smooth_targets*(inputs + self.machine_eps).log() + (1-smooth_targets)*((1-inputs) + self.machine_eps).log()).mean()


# selects the labels that produce the smallest loss using laplace soft labels
class MinLossSelectionLaplace(nn.Module):
	def __init__(self, number_classes=20, target_device=default_device):
		super().__init__()
		v = torch.arange(number_classes, dtype=torch.float, device=target_device, requires_grad=False)
		self.gaussian_weights = (-(v.unsqueeze(0) - v.unsqueeze(1)).abs()).exp()
		self.gaussian_weights = self.gaussian_weights/self.gaussian_weights.mean(dim=0, keepdim=True)
		self.machine_eps = 1e-16

	def forward(self, inputs, islog=False):
		if islog:
			return (-inputs).matmul(self.gaussian_weights).argmin(-1)
		else:
			return (-(inputs + self.machine_eps).log()).matmul(self.gaussian_weights).argmin(-1)


class MinLossSelectionBinomial(nn.Module):
	def __init__(self, number_classes=20, p0=0.5, variance=0, target_device=default_device):
		super().__init__()
		tmp = BinomialTargetCE(number_classes=number_classes, target_device=target_device, p0=p0, variance=variance)
		self.soft_weights = tmp.soft_weights
		self.machine_eps = 1e-16
		self.soft_weights /= self.soft_weights.sum(dim=-1, keepdim=True)

	def forward(self, inputs, islog=False):
		if islog:
			return (-inputs).matmul(self.soft_weights.T).argmin(-1)
		else:
			return (-(inputs + self.machine_eps).log()).matmul(self.soft_weights.T).argmin(-1)


def minLossSelectionNNRank(inputs):
	tmp = (inputs >= 0.5).to(torch.float)
	return tmp.argmin(dim=-1).where((tmp != 1).any(dim=-1), tmp.sum(dim=-1))


def minLossSelectionRED(inputs):
	return (inputs >= 0.5).to(torch.float).sum(dim=-1)


default_label_selection = lambda x: x.argmax(-1)


def plot_evaluations(pred: np.array, actual: np.array, evaluation_dir, config_name, divide_counts_by=1, normalized_conf=True, image_id=None):
	pred = np.maximum(pred, 1)  # no negative classes

	errors = (pred-actual)
	lowest = errors.min()
	lowest = lowest if lowest < 0 else 0
	error_counts = np.bincount(errors-lowest)//divide_counts_by
	if image_id is None:
		identifier = get_time_string()
	else:
		identifier = image_id

	if len(identifier) > 0:
		identifier = identifier + '_'

	plt.bar(np.arange(len(error_counts))+lowest, error_counts)
	save_results = output_dir is not None and len(str(config_name)) > 0
	if save_results:
		plot_save_path = os.path.join(evaluation_dir, '{}Histogram_{}.png'.format(identifier, config_name))
		plt.savefig(plot_save_path)
	plt.show()
	if normalized_conf:
		counts = np.bincount(actual)/100
		pred_n = []
		actual_n = []
		for i, c in enumerate(counts):
			if c > 0:
				pred_counts_in_class = (np.bincount(pred[actual == i])/c).round().astype(np.int_)
				actual_n.extend([i]*pred_counts_in_class.sum())
				for j, l in enumerate(pred_counts_in_class):
					if l > 0:
						pred_n.extend([j]*l)

		ConfusionMatrixDisplay.from_predictions(actual_n, pred_n)
	else:
		class_counts = np.bincount(actual - 1)
		classes = [c+1 for c in range(len(class_counts)) if class_counts[c] > 0]
		conf_mat = confusion_matrix(actual, pred)
		conf_mat_normed = conf_mat / conf_mat.sum(axis=1, keepdims=True)
		fig, ax = plt.subplots(figsize=(10, 8))
		cb = ax.imshow(conf_mat_normed, cmap='viridis', vmax=1.0, vmin=0.0)
		if conf_mat_normed.shape[0] != len(classes):
			print('{} out of {} classes found'.format(conf_mat_normed.shape[0], len(classes)))
			classes = [c+1 for c in range(conf_mat_normed.shape[0])]

		plt.xticks(range(len(classes)), classes)  # rotation=90
		plt.yticks(range(len(classes)), classes)

		conf_mat_counts = conf_mat/divide_counts_by
		for i in range(len(classes)):
			for j in range(len(classes)):
				color = 'gold' if conf_mat_normed[i, j] < 0.5 else 'indigo'
				ax.annotate(f'{conf_mat_counts[i, j]:.0f}', (j, i),
				            color=color, va='center', ha='center')
		plt.colorbar(cb, ax=ax)

	plt.show()
	if save_results:
		plot_save_path = os.path.join(evaluation_dir,
							 '{}ConfusionMatrix_normalized_{}.png'.format(identifier, config_name))
		plt.savefig(plot_save_path)
	plt.show()


def getClassCounts(dataloader, target_device=default_device, inverse_counts=True):
	class_counts = {}
	for _, ys in dataloader:
		if len(ys.shape) > 0:
			for i in range(len(ys)):
				y = int(ys[i].item())
				if y in class_counts:
					class_counts[y] += 1
				else:
					class_counts[y] = 1
		else:
			y = int(ys.item())
			if y in class_counts:
				class_counts[y] += 1
			else:
				class_counts[y] = 1
	max_class = -1
	for k in class_counts.keys():
		max_class = max(max_class, k)
	assert max_class > -1
	inv_class_counts = torch.zeros([max_class+1], dtype=torch.float, device=target_device, requires_grad=False)
	if inverse_counts:
		for k, v in class_counts.items():
			inv_class_counts[k] = 1/v
	else:
		for k, v in class_counts.items():
			inv_class_counts[k] = v
	return inv_class_counts


def getClassPoolMap(class_counts, threshold=0.02):
	class_counts = class_counts/class_counts.sum()

	k = len(class_counts)
	class_pool_lists = [[[i], class_counts[i]] for i in range(k)]
	i = 0
	while i < len(class_pool_lists):
		fraction = class_pool_lists[i][1]
		if fraction < threshold:
			minimum = 1
			choice = 0
			if i > 0:
				minimum = class_pool_lists[i-1][1]
				choice = -1
			if i < len(class_pool_lists)-1 and class_pool_lists[i+1][1] <= minimum:
				choice = 1
			class_pool_lists[i + choice][0].extend(class_pool_lists[i][0])
			class_pool_lists[i + choice][1] += class_pool_lists[i][1]
			class_pool_lists.pop(i)
			if choice == -1:
				i += 1  # since then the next is also > threshold
		else:
			i += 1
	class_map = {}
	for i, class_pool in enumerate(class_pool_lists):
		for j in class_pool[0]:
			class_map[j] = i
	return class_map


class ClassPoolingWrapper(Dataset):
	def __init__(self, dataset, pooling_map, target_device=None):
		self.subset = dataset
		if not target_device:
			target_device = default_device
		self.pooling_map = torch.tensor([pooling_map[i] for i in range(len(pooling_map))], device=target_device, requires_grad=False)
		self.max_class = torch.tensor([max(pooling_map.keys())], dtype=torch.long, device=target_device)

	def __len__(self):
		return len(self.subset)

	def __getitem__(self, idx):
		X, y = self.subset[idx]
		return X, self.pooling_map[self.max_class.minimum(y)].squeeze()


class REDWrapper(GeneralTSModel):
	def __init__(self, model, target_device=default_device, output_classes=20):
		super().__init__()
		if isinstance(model, GeneralTSModel):
			self.name = model.name
			self.min_size = model.min_size
		elif hasattr(model, 'name'):
			self.name = model.name
		self.model = model
		self.final = model.final
		self.theta = nn.Parameter(data=torch.arange(output_classes, device=target_device, dtype=torch.float)[None, :])
		self.sigmoid = nn.Sigmoid()
		self.n_classes = output_classes

	def forward(self, x):
		model_output = self.model(x)
		return self.sigmoid(self.n_classes*model_output - self.n_classes*self.theta)


def getWeightedDataLoader(dataset, target_device=default_device, batch_size=1, seed=None):
	unweighted_dataloader = DataLoader(dataset)
	class_weights = getClassCounts(unweighted_dataloader)
	weights = torch.tensor([class_weights[int(i)] for _, i in unweighted_dataloader], device=target_device, requires_grad=False)
	rng = torch.Generator(device=target_device)
	if seed is not None:
		rng.manual_seed(seed)
	sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
	return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def train_loop(dataloader, pred_fn, loss_function, optimizer, label_selection=default_label_selection, epochs=20, eval_dataloader=None, scheduler=None, target_device='cuda', y_transform=None, reporting_freq=1, test_freq=2, train_eval_dataloader=None):
	pred_fn.train()
	size = len(dataloader.dataset)
	number_batches = len(dataloader)
	train_accs = []
	train_losses = []
	test_accs = []
	test_losses = []
	print("Start time:", get_time_string())
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

	for epoch in range(epochs):
		running_loss = torch.zeros([1], device=target_device, requires_grad=False)
		loss = torch.zeros([1], device=target_device)
		for batch, (X, y) in enumerate(dataloader):
			# Compute prediction and loss
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
		if epoch % reporting_freq == reporting_freq-1 or epoch == epochs-1:
			eval_result, loss_train, _, _ = test_loop(train_eval_dataloader, pred_fn, loss_function, label_selection=label_selection, y_transform=y_transform, print_metrics=False, eval_metric=WAE_train, metric_name=metric_name)
			print(f"train evaluation: loss: {loss_train:>7f} in epoch {epoch + 1},  {metric_name} {eval_result:>6f}")
			train_accs.extend([eval_result]*reporting_freq)

		break_cond = False
		if eval_dataloader:
			if epoch % test_freq == test_freq - 1:
				print(loss.item(), y[0], pred[0])
				eval_result, test_loss, _, _ = test_loop(eval_dataloader, pred_fn, loss_function, label_selection=label_selection, y_transform=y_transform, eval_metric=WAE, metric_name=metric_name)
				test_accs.extend([eval_result] * test_freq)
				test_losses.extend([test_loss] * test_freq)
			elif epoch == epochs-1 or break_cond:
				print(loss.item(), y[0], pred[0])
				eval_result, test_loss, _, _ = test_loop(eval_dataloader, pred_fn, loss_function, label_selection=label_selection, y_transform=y_transform, eval_metric=WAE, metric_name=metric_name)
				test_accs.extend([eval_result] * (epochs % test_freq))
				test_losses.extend([test_loss] * (epochs % test_freq))
		if break_cond:
			break
	return train_accs, test_accs, train_losses, test_losses


def test_loop(dataloader, pred_fn, loss_function, label_selection=default_label_selection, y_transform=None, print_metrics=True, eval_metric=None, metric_name="Accuracy", metric_mean=False):
	pred_fn.eval()
	size = len(dataloader.dataset)
	test_loss, eval_value = 0, 0
	predictions = []
	ground_truth = []
	if not eval_metric:
		eval_metric = lambda a, b: a.isclose(b).type(torch.float).sum()

	with torch.no_grad():
		for X, y in dataloader:
			y = y.squeeze()
			if y_transform:
				y = y_transform(y)
			pred = pred_fn(X)
			pred_labels = label_selection(pred)
			predictions.append(pred_labels.flatten())
			eval_value += eval_metric(pred_labels, y)
			ground_truth.append(y.flatten())
			test_loss += loss_function(pred, y)*y.shape[-1]

	test_loss /= size
	if metric_mean:
		eval_value = eval_value / size
	if print_metrics:
		print(f"Test loss: {test_loss:>7f} and {metric_name}: {eval_value:>6f}")
	if isinstance(eval_value, torch.Tensor):
		eval_value = eval_value.item()
	if isinstance(test_loss, torch.Tensor):
		test_loss = test_loss.item()
	return eval_value, test_loss, torch.hstack(predictions), torch.hstack(ground_truth)

def visualize_dataset(dataframe, print_stats=False, save_path=None, permutations=1):
	counts = np.bincount(dataframe['Difficulty'].to_numpy()) / permutations
	counts = pd.Series(data=counts, index=range(1, len(counts)+1))
	if print_stats:
		print("Number of Datapoints", counts.sum())
		print("Difficulty Distribution", counts)
	counts = counts/counts.sum()
	counts *= 100  # to percent
	ax = counts.plot(kind="bar")
	ax.yaxis.set_major_formatter(PercentFormatter(decimals=None))
	plt.yticks(fontsize=13)
	plt.xticks(range(1, len(counts)+1, 2), rotation=0, fontsize=13)
	ax.set_xlabel('Difficulties', fontsize=14)
	ax.set_ylabel('Proportion of levels [%]', fontsize=14)
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path)
	plt.show()


def create_data_split(dataframe_path, train_split_size=0.8, seed=None):
	assert os.path.isfile(dataframe_path)
	dataframe = pd.read_csv(filepath_or_buffer=dataframe_path, index_col=0)

	# Combine severely underrepresented classes
	dataframe['Difficulty'] -= 1
	class_frequencies = np.bincount(dataframe['Difficulty'].to_numpy())
	class_map = getClassPoolMap(class_frequencies, threshold=0.02)
	train_df, test_df = split_containing_all_classes(dataframe, split_sizes=train_split_size,
	                                                 class_map=class_map, seed=seed)
	return class_map, train_df, test_df, dataframe


def create_persistent_data_split(dataframe_path, train_split_size=0.8, output_dir=None, diagram_dir=None, dataset_name=None):
	seed = 101
	class_map, train_df, test_df, dataframe = create_data_split(dataframe_path, train_split_size, seed=seed)

	if diagram_dir and dataset_name:
		diagram_path = os.path.join(diagram_dir,
									"DifficultyDistribution_{}_{}".format(dataset_name, len(dataframe)))
		visualize_dataset(dataframe, permutations=4, print_stats=True, save_path=diagram_path)

	if output_dir:
		out_dir = output_dir
	else:
		out_dir = os.path.dirname(dataframe_path)
	name = os.path.split(dataframe_path)[-1].split(".")[0]
	ext = ".txt"
	for m, df in [("train", train_df), ("test", test_df),("class_map", pd.Series(class_map))]:
		out_path = os.path.join(out_dir, "{}_{}{}".format(name, m, ext))
		df.to_csv(path_or_buf=out_path)


def evaluate_trained_model_on_dataset(model, dataloader, label_selection, y_transform=id, plot_results=False, output_path=None, config_name='', label_shift=0):
	const_l = lambda a, b: 0
	metrics = get_eval_metrics(dataloader)
	_, _, predictions, ground_truth = test_loop(dataloader, model, const_l, label_selection=label_selection, y_transform=y_transform,
	                                     eval_metric=const_loss, metric_name='', print_metrics=False)
	results = evaluate_metrics(predictions, ground_truth, metrics)
	print(results)
	if plot_results:
		assert os.path.isdir(output_path)
		plot_evaluations(predictions.to(device='cpu', dtype=torch.int).detach().numpy() + 1 - label_shift,
						 ground_truth.to(device='cpu', dtype=torch.int).detach().numpy() + 1 - label_shift, output_path, config_name)
	return results


def evaluate_metrics(predictions, ground_truth, metrics: dict):
	eval_results = {}
	for metric_name in metrics.keys():
		value = metrics[metric_name](predictions, ground_truth)
		if isinstance(value, torch.Tensor):
			value = value.item()
		eval_results[metric_name] = value
	return eval_results


def get_eval_metrics(dataloader):
	total_set_size = len(dataloader.dataset)
	class_counts = getClassCounts(dataloader)
	acc_f = lambda a, b: (a == b).type(torch.float).sum() / total_set_size
	mae = lambda a, b: (a - b).abs().sum() / total_set_size
	tpr = BalancedMetric(class_counts, lambda a, b: (a == b).type(torch.float))  # == Avg TPR
	rmse = lambda a,b: (a-b).float().square().mean().sqrt()   # Only works if all samples are added at once
	balanced_rmse = BalancedMetric(class_counts, lambda a, b: (a-b).square(), final_op=lambda a: a.sqrt())
	eval_metrics = {'WAE': WeightedAE(class_counts), 'MAE': mae, 'Accuracy': acc_f,
	             "TPR": tpr, 'RMSE': rmse, 'Balanced_RMSE': balanced_rmse}
	return eval_metrics

def save_predictions(eval_dataframe, model, eval_dataloader, save_path, selection_function=default_label_selection):
	# assumes same ordering of eval_frame and dataloader
	eval_dataframe.loc[:, 'Predicted Difficulty'] = eval_dataframe['Difficulty'].copy()
	eval_dataframe.loc[:, 'Pooled Difficulty'] = eval_dataframe['Difficulty'].copy()
	index_counter = 0
	diff_column_idx = eval_dataframe.columns.get_loc('Pooled Difficulty')
	pred_diff_colum_idx = eval_dataframe.columns.get_loc('Predicted Difficulty')
	with torch.no_grad():
		for X, y in eval_dataloader:
			m = X.shape[0]
			pred = selection_function(model(X))
			pred = pred.to(dtype=int, device='cpu').numpy()
			eval_dataframe.iloc[index_counter:index_counter+m, pred_diff_colum_idx] = pred

			eval_dataframe.iloc[index_counter:index_counter+m, diff_column_idx] = y.to(dtype=int, device='cpu').numpy()
			index_counter += m
	eval_dataframe.to_csv(save_path)


def MonteCarloCrossValidation(input_path, model_constructor, optim_constructor, loss_function, train_dataloader_constructor, unweighted_train_dataloader_constructor, train_dataset_constructor, eval_dataloader_constructor, tries=5, out_dir=None, data_set_name=None, class_map=None, config_data=None, continue_id=None, rid='', **kwargs):
	worst_perf = -100
	worst_model = None
	assert tries > 0
	eval_developments = []
	final_performances = [[], []]
	start = 0
	if continue_id is None:
		start_time = get_time_string() + ("_{}".format(rid) if len(rid) > 0 else '')
	else:
		start_time = continue_id
	print('Execution ID: {}'.format(start_time))
	model_dir = os.path.join(output_dir, "saved_models", start_time)
	test_set_dir = os.path.join(output_dir, "CV_test_datasets", start_time)
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)
	elif continue_id is not None:
		start = len([file for file in os.scandir(model_dir) if file.is_file() and str(file.name).endswith('.pth')])
		print("Starting at ", start)
	if not os.path.isdir(test_set_dir):
		os.mkdir(test_set_dir)
	if config_data is not None:
		config_data.to_csv(os.path.join(model_dir, 'config.txt'))

	for fold in range(start, tries):
		seed_data_split = (fold+17)*877 % 289469
		seed_data_loader = (fold+29)*821 % 634237
		seed_rtss_train = (fold+83)*677 % 656423
		seed_rtss_test = seed_rtss_train
		torch.manual_seed(seed_rtss_test)

		cm, train_df, test_df, _ = create_data_split(input_path, .8, seed=seed_data_split)
		if class_map is None:
			class_map = cm
		model = model_constructor()
		if fold == 0:
			print_model_parameter_overview(model)
		optimizer = optim_constructor(model)
		train_set = train_dataset_constructor(train_df, class_map)
		weighted_train_dataloader = train_dataloader_constructor(train_set, seed_rtss=seed_rtss_train, seed_dl=seed_data_loader)
		unweighted_train_data = unweighted_train_dataloader_constructor(train_set, seed_rtss=seed_rtss_train)
		eval_dataloader = eval_dataloader_constructor(test_df, class_map, seed_rtss=seed_rtss_test)
		train_epochs = kwargs['epochs'] if "epochs" in kwargs else 100
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(80, 3*train_epochs//4), gamma=0.5)
		test_set_path = os.path.join(test_set_dir, '{}_{}_test_dataset_{}_Run_{}.txt'.format(start_time, model.name, data_set_name, fold))
		test_df.to_csv(test_set_path)

		print("############")
		print("CV Run {}: ".format(fold+1))
		print("############")
		print("Seeds: DataSplit {}, DataLoader {}, RTSS Train {}, RTSS Test {}".format(seed_data_split, seed_data_loader, seed_rtss_train, seed_rtss_test))
		train_results = train_loop(weighted_train_dataloader, model, loss_function, optimizer, scheduler=scheduler,
								   eval_dataloader=eval_dataloader, train_eval_dataloader=unweighted_train_data, **kwargs)
		optimizer.zero_grad()
		model_save_path = os.path.join(model_dir, '{}_{}_weights_{}_Run_{}.pth'.format(start_time, model.name, data_set_name, fold))
		torch.save(model.state_dict(), model_save_path)
		eval_developments.append(train_results[1])
		eval_result = train_results[1][-1]
		final_performances[0].append(train_results[0][-1])
		final_performances[1].append(eval_result)
		if eval_result > worst_perf:
			worst_model = model
	if start < tries:
		for i in range(len(eval_developments)):
			plt.plot(eval_developments[i], label='Eval Run {}'.format(i))
		img_path = os.path.join(out_dir, '{}_Loss_{}.png'.format(get_time_string(), "CV_{}_{}".format(model.name, data_set_name)))
		plt.savefig(img_path)
		plt.show()
		final_performances = np.array(final_performances)
		print("Final results:", final_performances)
		mean, std = final_performances.mean(axis=1), final_performances.std(axis=1)
		print("Results Train - Mean: {}  Std: {}".format(mean[0], std[0]))
		print("Results Eval - Mean: {}  Std: {}".format(mean[1], std[1]))
		return worst_model
	else:
		print('Run already complete. Nothing to continue.')


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-root', type=str, help='Root directory. Simplifies definition of other directories', required=False)
	parser.add_argument('-input_dir', type=str, help='Time series input directory', required=False)
	parser.add_argument('-model_dir', type=str, help='Output directory', required=False)
	parser.add_argument('-device', type=str, help='Model Training device', required=False)
	parser.add_argument('-learning_rate', type=float, help='Model Learning Rate', required=False)
	parser.add_argument('-decay', type=float, help='Model Weight Decay', required=False)
	parser.add_argument('-dataset', type=str, help='Name of Dataset to be evaluated', required=False)
	parser.add_argument('-cv_repeats', type=int, help='Number of Cross Validation Repeats', required=False)
	parser.add_argument('-loss_variant', type=int, help='Id for the loss function to be used', required=False)
	parser.add_argument('-multi_agg_variant', type=float, help='Variant for aggregating all samples per chart', required=False)
	parser.add_argument('-ts_freq', type=str, help='TimeSeries sampling frequency. Baselines for < 0', required=False)
	parser.add_argument('-run_id', type=str, help='Opt. ID associated with this execution', required=False)
	parser.add_argument('-msg', type=str, help='Opt. log message added at the beginning', required=False)
	parser.add_argument('-regen_split', action='store_true', help='Force to re-generate split. Only useful when not using CV', required=False)
	parser.add_argument('-eval_cv_id', type=str, help='ID of the cross validation that should be evaluated', required=False)
	parser.add_argument('-eval_cv_test', action='store_true', help='Run evaluation on CV test datasets', required=False)
	parser.add_argument('-continue_cv', type=str, help='Run evaluation on CV test datasets', required=False)

	parser.set_defaults(
		root='',
		input_dir='data',
		output_dir='model_artifacts/',
		device='cuda',
		learning_rate=None,
		dataset='GullsArrows',
		cv_repeats=None,
		loss_variant=None,
		multi_agg_variant=None,
		ts_freq=None,
		run_id='',
		msg=None,
		decay=5e-2,
		regen_split=False,
		eval_cv_id=None,
		eval_cv_test=False,
		continue_CV=None,
	)

	args = parser.parse_args()

	root = args.root
	input_dir = os.path.join(root, args.input_dir)
	output_dir = os.path.join(root, args.output_dir)
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	CV_dir_name = ''
	if args.eval_cv_id is not None:
		CV_dir_name = args.eval_cv_id
		print(CV_dir_name)
	eval_CV = len(CV_dir_name) > 0

	if eval_CV:
		config_path = os.path.join(output_dir, 'saved_models', CV_dir_name, 'config.txt')
		if os.path.isfile(config_path):
			config = pd.read_csv(config_path, index_col=0).squeeze('index').to_dict()
			print(config)
			for key in config.keys():
				if hasattr(args, key):
					setattr(args, key, config[key])

	continue_CV = None
	if args.continue_cv is not None:
		continue_CV = args.continue_cv

	train = True and not eval_CV
	reset = True and train
	evaluate_other = False or eval_CV
	CV_repeats = args.cv_repeats
	if CV_repeats is None:
		CV_repeats = 1
	cross_validation = CV_repeats > 0 and train
	eval_CV_test = (args.eval_cv_test or False) and eval_CV

	loss_mode = 4
	loss_modes = ['NLL', 'Ordinal_Regression', 'Poisson-Binomial', 'RED-SVM', 'Gaussian', 'Regression', 'Binomial', ]
	force_new_split = False or args.regen_split

	args_ts_freq = args.ts_freq
	if args_ts_freq is None:
		ts_sample_freq = 0
		b_variant = False
	else:
		if type(args_ts_freq) == str and args_ts_freq[-1] == 'b':
			b_variant = True
			ts_sample_freq = int(args_ts_freq[:-1])
		else:
			ts_sample_freq = int(args_ts_freq)
			b_variant = False

	is_time_series = ts_sample_freq >= 0
	pattern_attr = ts_sample_freq < -1

	torch.backends.cudnn.benchmark = True

	dataset_name = args.dataset
	other_dataset_names = ['Gpop', 'itg', 'Speirmix', 'fraxtil', 'GullsArrows']  #, 'Gpop', 'itg', 'GullsArrows', 'fraxtil'
	if dataset_name in other_dataset_names:
		other_dataset_names.pop(other_dataset_names.index(dataset_name))

	weight_decay = args.decay

	raw_dataset_name = dataset_name
	ts_name_ext = "_{}".format(ts_sample_freq) + ("b" if b_variant else "")
	if is_time_series:
		dataset_name += ts_name_ext
		other_dataset_names = [name + ts_name_ext for name in other_dataset_names]

	if args.msg is not None:
		print(args.msg)

	# Not general but convenient
	audio_input_dir = os.path.join(input_dir, "audio")
	if is_time_series:
		input_dir = os.path.join(input_dir, "time_series")
	elif pattern_attr:
		input_dir = os.path.join(input_dir, "pattern_attr")
	else:
		input_dir = os.path.join(input_dir, "characteristics")

	if not torch.cuda.is_available():
		device = 'cpu'
	else:
		device = args.device

	train_file = os.path.join(input_dir, dataset_name + "_train" + '.txt')
	test_file = os.path.join(input_dir, dataset_name + "_test" + '.txt')
	class_pool_file = os.path.join(input_dir, dataset_name + "_class_map" + '.txt')
	eval_dir = os.path.join(output_dir, "evaluations")
	input_file = os.path.join(input_dir, dataset_name + '.txt')
	if cross_validation:
		if not os.path.isfile(input_file):
			raise AssertionError("Inputfile not found at: {}".format(input_file))
		class_pool_map, train_dataframe, test_dataframe, _ = create_data_split(input_file)
	else:
		regen_split_cond = force_new_split
		if not(os.path.isfile(train_file) and os.path.isfile(test_file)) or regen_split_cond:
			if not os.path.isfile(input_file):
				raise AssertionError("Inputfile not found at: {}".format(input_file))
			create_persistent_data_split(input_file, dataset_name=dataset_name, diagram_dir=(eval_dir if not regen_split_cond else None))
		train_dataframe = pd.read_csv(filepath_or_buffer=train_file, index_col=0)
		test_dataframe = pd.read_csv(filepath_or_buffer=test_file, index_col=0)
		class_pool_map = pd.read_csv(filepath_or_buffer=class_pool_file, index_col=0).squeeze("columns").to_dict()  # storage as pandas series is readable and avoids further packages
	
	dataset_lr = {'GullsArrows': 5e-4, 'Gpop': 5e-5, 'itg': 1e-4, 'fraxtil': 1e-4}
	if args.learning_rate is not None:
		learning_rate = args.learning_rate
	else:
		learning_rate = len(train_dataframe)/1500 * 0.8 * 1e-4   # 0.8 is the percentage of the training set 

	run_id = args.run_id
	
	print("===========================")
	print("Configuration")
	if ts_sample_freq < 0:
		print("Baseline Evaluation: {} Model".format("Pattern" if pattern_attr else "Characteristics"))
	print("Dataset:", dataset_name)
	if is_time_series:
		print("Learning Rate:", learning_rate)
		print("Weight Decay:", weight_decay)

	if args.cv_repeats is not None:
		CV_repeats = args.cv_repeats
	if cross_validation:
		print("CrossValidation:", CV_repeats, "times")

	lv = args.loss_variant
	if lv is not None:
		loss_mode = lv
	if not 0 <= loss_mode < len(loss_modes):
		raise AttributeError('Unknown Loss Variant')

	if loss_modes[loss_mode] in ["Gaussian", 'Binomial', 'Poisson-Binomial']:
		y_shift = 4
	else:
		y_shift = 0
	n_classes = max(class_pool_map.values()) + 1 + 2 * y_shift
	if y_shift > 0:
		for key in class_pool_map.keys():
			class_pool_map[key] += y_shift
	print("Number of Classes: {}".format(n_classes) + (" ({})".format(n_classes - 2 * y_shift) if y_shift > 0 else ""))

	ytransform = lambda x: x.type(torch.long)
	all_classes = 0  # RED-SVM Parameter

	print("Chosen Loss Variant:", loss_modes[loss_mode])
	if loss_modes[loss_mode] == 'Regression':
		n_classes = 1
		ClassificationLoss = nn.L1Loss()
		FinalActivation = nn.Identity()
		minLossSelection = lambda x: x.round()
		ytransform = lambda x: x.type(torch.float)[:, None]
	elif loss_modes[loss_mode] == 'NLL':
		ClassificationLoss = LogNLLLoss()
		FinalActivation = nn.Softmax(dim=-1)
		minLossSelection = lambda x: x.argmax(dim=-1)
	elif loss_modes[loss_mode] == 'Ordinal_Regression':
		FinalActivation = nn.Sigmoid()
		ClassificationLoss = RelativeEntropy(number_classes=n_classes)
		minLossSelection = minLossSelectionRED  
	elif loss_modes[loss_mode] == 'RED-SVM':
		FinalActivation = nn.Identity()
		minLossSelection = minLossSelectionRED
		all_classes = n_classes - 1
		ClassificationLoss = RelativeEntropy(number_classes=all_classes)
		n_classes = 1
	else:
		FinalActivation = nn.Softmax(dim=-1)
		if loss_modes[loss_mode] == 'Gaussian':
			ClassificationLoss = LaplaceLoss(number_classes=n_classes)
			minLossSelection = MinLossSelectionLaplace(number_classes=n_classes)
		elif loss_modes[loss_mode] in ['Binomial', 'Poisson-Binomial']:

			var, first_p = 0.5, 0.99

			if loss_modes[loss_mode] == 'Binomial':
				var = 0
			ClassificationLoss = BinomialTargetCE(number_classes=n_classes, variance=var, p0=first_p)
			minLossSelection = MinLossSelectionBinomial(number_classes=n_classes, variance=var, p0=first_p)
		else:
			raise AttributeError('Unknown Loss Variant')
	loss_fn = ClassificationLoss

	multi_agg_variant = args.multi_agg_variant
	if multi_agg_variant is None:
		multi_agg_variant = 7
	if multi_agg_variant == 7 and loss_modes[loss_mode] == 'Ordinal_Regression':
		multi_agg_variant = 7.2
	multi_agg_variant = (multi_agg_variant//1 + 0.1 if multi_agg_variant >= 2 and n_classes == 1 else multi_agg_variant)
	print("MultiSample Aggregation Variant: ", multi_agg_variant)

	if is_time_series:
		prelim_sample_size = int(60*max(1, ts_sample_freq))
		batch_size = 128
		multi_sample = True
		chart_channels = 19
		ts_in_channels = chart_channels
		num_epochs = 200
		time_series_dir = os.path.join(input_dir, dataset_name)
		assert os.path.isdir(time_series_dir)

		modelBase_constructor = lambda: TimeSeriesTransformerModel(ts_in_channels, device, out_channels=n_classes, sequence_length=prelim_sample_size, final_activation=FinalActivation)
		if multi_sample:
			model_constructor = lambda: MultiWindowModelWrapper(modelBase_constructor(), output_classes=n_classes, final_activation=FinalActivation, variant=multi_agg_variant)
		else:
			model_constructor = modelBase_constructor

		optim_constructor = lambda model: torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

		model_sample_size = prelim_sample_size
		sub_samples = 8
		sample_start_interval = max(ts_sample_freq, 1)
		print("Samples:", model_sample_size, "x", sub_samples)

		train_dataset_constructor = lambda train_df, class_pool: ClassPoolingWrapper(
			TimeSeriesDataset(train_df, time_series_dir, target_device=device),
			class_pool)
		test_dataset_constructor = lambda test_df, class_pool: ClassPoolingWrapper(
			TimeSeriesDataset(test_df, time_series_dir, target_device=device),
			class_pool)

		train_dataloader_constructor = lambda train_dataset, seed_rtss=None, seed_dl=None: getWeightedDataLoader(
				FixedSizeInputSampleTS(train_dataset, sample_size=model_sample_size, k=sample_start_interval, sub_samples=sub_samples, multisample_mode=multi_sample, seed=seed_rtss)
			, target_device=device, batch_size=batch_size, seed=seed_dl
		)
		unweighted_train_dataloader_constructor = lambda train_dataset, seed_rtss=None: DataLoader(
			FixedSizeInputSampleTS(train_dataset, sample_size=model_sample_size, k=sample_start_interval, sub_samples=sub_samples, multisample_mode=multi_sample, seed=seed_rtss), batch_size=batch_size
			# train_dataset
		)

		test_dataloader_constructor = lambda test_df, class_pool, seed_rtss=None: DataLoader(
			FixedSizeInputSampleTS(test_dataset_constructor(test_df, class_pool), sample_size=model_sample_size, k=sample_start_interval, sub_samples=sub_samples, multisample_mode=multi_sample, seed=seed_rtss), batch_size=batch_size
			# test_dataset
		)

		rep_f, test_f = 20, 10
	else:
		batch_size = 128
		num_epochs = 600
		
		n_input_dim = 19
		train_dataset_constructor = lambda train_df, class_pool: ClassPoolingWrapper(prepare_pattern_dataset(train_df, target_device=device),
																					 class_pool)
		test_dataloader_constructor = lambda test_df, class_pool, seed_rtss=None: DataLoader(
			ClassPoolingWrapper(prepare_pattern_dataset(test_df, target_device=device),
								class_pool), batch_size=batch_size)
		train_dataloader_constructor = lambda train_dataset, seed_rtss=None, seed_dl=None: getWeightedDataLoader(train_dataset, target_device=device, batch_size=batch_size, seed=seed_dl)
		unweighted_train_dataloader_constructor = lambda train_dataset, seed_rtss=None: DataLoader(train_dataset, batch_size=batch_size)
		
		model_constructor = lambda: SimplePatternModel(n_input_dim, device, out_channels=n_classes, final_activation=FinalActivation, pattern_attr=pattern_attr)
		optim_constructor = lambda model: torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
		rep_f, test_f = 20, 20
		embed_model = None

	if all_classes > 0:
		model_constructor_old = model_constructor
		model_constructor = lambda: REDWrapper(model_constructor_old(), target_device=device,
		                                       output_classes=all_classes)

	print("====================================")

	if cross_validation:
		assert os.path.isfile(input_file)
		config = {'learning_rate': learning_rate, 'dataset': raw_dataset_name, 'loss_variant': loss_mode, 'multi_agg_variant': multi_agg_variant,  'ts_freq': str(ts_sample_freq),  'decay': weight_decay}
		model1 = MonteCarloCrossValidation(input_file, model_constructor, optim_constructor, loss_fn, train_dataloader_constructor, unweighted_train_dataloader_constructor, train_dataset_constructor,
										   test_dataloader_constructor, tries=CV_repeats, out_dir=eval_dir, data_set_name=dataset_name, continue_id=continue_CV, epochs=num_epochs, config_data=pd.DataFrame(data=config, index=[0]),
										   y_transform=ytransform, reporting_freq=rep_f, test_freq=test_f, class_map=class_pool_map, rid=run_id, label_selection=minLossSelection)  #, exp=nllloss

		t1 = print_model_parameter_overview(model1, only_total=True)
		model1.name = model1.name + "_{}".format(t1)
		configuration_name = "{}_{}_{}".format(model1.name, dataset_name, num_epochs)
	else:
		model1 = model_constructor()
		optim = optim_constructor(model1)
		constructed_train_dataset = train_dataset_constructor(train_dataframe, class_pool_map)
		train_dataloader = train_dataloader_constructor(constructed_train_dataset)
		unweighted_train_dataloader = unweighted_train_dataloader_constructor(constructed_train_dataset)
		evaluation_dataloader = test_dataloader_constructor(test_dataframe, class_pool_map)

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

		if train:
			lr_scheduler = None
			accuracies = train_loop(train_dataloader, model1, loss_fn, optim, scheduler=lr_scheduler, eval_dataloader=evaluation_dataloader, epochs=num_epochs,
									y_transform=ytransform, reporting_freq=rep_f, test_freq=test_f, train_eval_dataloader=unweighted_train_dataloader, label_selection=minLossSelection)
			torch.save(model1.state_dict(), model_path)
			torch.save(optim.state_dict(), optim_path)
			plt.plot(accuracies[2])
			plt.plot(accuracies[0])
			plt.plot(accuracies[1])
			img_path = os.path.join(eval_dir, '{}_Loss_{}.png'.format(get_time_string(), configuration_name))
			plt.savefig(img_path)
			plt.show()

		print(class_pool_map)

	const_loss = lambda a, b: 0
	if not evaluate_other:
		print(class_pool_map)
		evaluation_dataloader = test_dataloader_constructor(test_dataframe, class_pool_map)
		evaluate_trained_model_on_dataset(model1, evaluation_dataloader, minLossSelection, y_transform=ytransform, output_path=eval_dir, config_name=configuration_name, plot_results=True, label_shift=y_shift)
		save_predictions(test_dataframe, model1, evaluation_dataloader, os.path.join(input_dir, "{}_predicted.txt".format(dataset_name)), selection_function=minLossSelection)
	else:
		if eval_CV_test:
			other_dataset_names = [file.path for file in os.scandir(os.path.join(output_dir, "CV_test_datasets", CV_dir_name)) if file.is_file()]
		overall_results = {}
		other_dataset_number = -1
		all_preds = []
		all_y = []
		prediction_frames = []
		if eval_CV:
			eval_CV_dir = os.path.join(eval_dir, CV_dir_name)
			if not os.path.isdir(eval_CV_dir):
				os.mkdir(eval_CV_dir)
		other_dataset_names.sort()
		for other_dataset_name in other_dataset_names:
			other_dataset_number += 1
			if eval_CV_test:
				other_df_path = other_dataset_name
				other_dataset_name = os.path.basename(other_dataset_name).split('.')[0]
			else:
				other_df_path = os.path.join(input_dir, other_dataset_name + '.txt')

			print('---------------------------------')
			print(other_dataset_name)

			# Dataset Preparation
			other_dataset_frame = pd.read_csv(filepath_or_buffer=other_df_path, index_col=0)
			if not eval_CV_test:
				other_dataset_frame['Difficulty'] -= 1
			if 'Permutation' not in other_dataset_frame.columns:
				other_dataset_frame['Permutation'] = 1
			prediction_frame = other_dataset_frame.loc[:, ['Name', 'Difficulty', 'Permutation', 'sm_fp']].copy()
			if is_time_series:
				if eval_CV_test:
					other_dataset_ts_path = os.path.join(input_dir, dataset_name)
				else:
					other_dataset_ts_path = os.path.join(input_dir, other_dataset_name)
				assert os.path.isdir(other_dataset_ts_path)
				random_seed = None  # Seeds for that?
				evaluation_dataset = FixedSizeInputSampleTS(TimeSeriesDataset(other_dataset_frame, other_dataset_ts_path, target_device=device), sample_size=model_sample_size, k=sample_start_interval, sub_samples=sub_samples, multisample_mode=multi_sample, seed=random_seed)
			else:   # pattern_attr
				evaluation_dataset = prepare_pattern_dataset(other_dataset_frame, target_device=device)
			evaluation_dataloader = DataLoader(ClassPoolingWrapper(evaluation_dataset, class_pool_map), batch_size=batch_size)

			# Evaluation of all models for this dataset
			if eval_CV:
				model_paths = [file.path for file in os.scandir(os.path.join(output_dir, "saved_models", CV_dir_name))
							   if file.is_file() and str(file.name).endswith('.pth')]
				model_paths.sort()
				eval_losses = get_eval_metrics(evaluation_dataloader)
				full_results = {key: [] for key in eval_losses.keys()}
				if not eval_CV_test:
					all_preds = []
					all_y = []
					prediction_frames = []
				for i in range(len(model_paths)):
					if eval_CV_test and i != other_dataset_number:
						continue
					prediction_frame['Run'] = i
					print(model_paths[i])
					st_dict = torch.load(model_paths[i])
					model1.load_state_dict(st_dict)
					_, _, test_preds, test_y = test_loop(evaluation_dataloader, model1, const_loss, label_selection=minLossSelection,
					                                     y_transform=ytransform, eval_metric=const_loss, metric_name='',
					                                     print_metrics=False)
					model_results = evaluate_metrics(test_preds, test_y, eval_losses)
					all_y.append(test_y)
					all_preds.append(test_preds)
					prediction_frame['Predicted Difficulty'] = test_preds.detach().to(device='cpu', dtype=torch.int).numpy()
					prediction_frame['Pooled Difficulty'] = test_y.detach().to(device='cpu', dtype=torch.int).numpy()
					prediction_frames.append(prediction_frame.copy())
					for key in eval_losses.keys():
						full_results[key].append(model_results[key])
				print(full_results)
				if not eval_CV_test:
					for key in eval_losses.keys():
						print(key, sum(full_results[key])/len(model_paths))
					plot_evaluations(torch.hstack(all_preds).to(device='cpu', dtype=torch.int).detach().numpy() + 1 - y_shift, torch.hstack(all_y).to(device='cpu', dtype=torch.int).detach().numpy() + 1 - y_shift,
						eval_CV_dir, "CV_Results_{}_trained_{}_eval_{}".format(model1.name, dataset_name, other_dataset_name), divide_counts_by=len(model_paths), image_id=CV_dir_name)
					full_prediction_frame = pd.concat(prediction_frames, axis=0, ignore_index=True)
					cleaned_other_dataset_name = ('_'.join(other_dataset_name.split('_')[:-1]) if is_time_series else other_dataset_name)
					full_prediction_frame.to_csv(os.path.join(eval_CV_dir, "{}_predicted.txt".format(cleaned_other_dataset_name)))

				else:
					for key in eval_losses.keys():
						if key in overall_results:
							overall_results[key].append(sum(full_results[key]))
						else:
							overall_results[key] = [sum(full_results[key])]

			else:
				evaluate_trained_model_on_dataset(model1, evaluation_dataloader, minLossSelection, y_transform=ytransform, output_path=eval_dir, config_name="{}_trained_{}_eval_{}".format(model1.name, dataset_name, other_dataset_name), plot_results=True, label_shift=y_shift)
				if other_dataset_name.endswith("_test"):
					save_predictions(test_dataframe, model1, evaluation_dataloader,
									 os.path.join(input_dir, "{}_predicted.txt".format(other_dataset_name)), selection_function=minLossSelection)
		if eval_CV_test:
			print('------------------------------')
			print("Overall Dataset Results:", overall_results)
			for key in overall_results.keys():
				print(key, sum(overall_results[key]) / len(other_dataset_names))
			plot_evaluations(torch.hstack(all_preds).to(device='cpu', dtype=torch.int).detach().numpy() + 1 - y_shift,
			torch.hstack(all_y).to(device='cpu', dtype=torch.int).detach().numpy() + 1 - y_shift,
			eval_CV_dir, "CV_Results_{}_on_{}".format(model1.name, dataset_name), divide_counts_by=len(other_dataset_names), image_id=CV_dir_name)
			full_prediction_frame = pd.concat(prediction_frames, axis=0, ignore_index=True)
			full_prediction_frame.to_csv(os.path.join(eval_CV_dir, "{}_predicted.txt".format(raw_dataset_name)))

	beep()
