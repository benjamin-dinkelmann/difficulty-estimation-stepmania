import re
import numpy as np
from time import strftime
try:
	from winsound import Beep
except ImportError:
	def Beep(a, b):
		return


def filter_name(name):
	name = re.sub(r'[\s-]+', '_', name.strip())
	name = re.sub(r'\W', '', name)
	name = name.encode('ascii', errors='backslashreplace').decode('ascii').replace('\\', '')
	return name


def get_random_split_indices(num_data_points, split_sizes=0.8, rng=None):
	"""Generates multiple random lists, partitioning a number of data points randomly in splits of the given sizes.
	Split_sizes are normalized to 1.0 if they cover more than 100%, if the sum of splits is smaller than 1, an additional split covering the remaining datapoints is assumed."""
	if rng is None:
		rng = np.random.default_rng()
	if type(split_sizes) == float:
		split_szs = [split_sizes]
	else:
		split_szs = split_sizes.copy()
	split_sum = sum(split_szs)
	norm_splits = [0]
	if split_sum < 1:
		split_sum = 1
		split_szs.append(-1)  # just acts a filler
	for x in split_szs[:-1]:
		norm_splits.append(norm_splits[-1] + x / split_sum)
	norm_splits.append(1)

	split_idx = np.arange(num_data_points)
	rng.shuffle(split_idx)

	out = []
	right = 0
	for i in range(len(norm_splits)-1):
		left = right
		right = int(norm_splits[i+1]*num_data_points)
		out.append(split_idx[left:right])
	return out


def split_grouped_by_column(dataframe, column='Name', split_sizes=0.8, rng=None):
	"""Generates a split of a dataframe grouped by a column.
	Assumes that each group has approximately the same number of members."""
	grouped_frame = dataframe.groupby(column)
	number_songs = grouped_frame.ngroups
	splits = get_random_split_indices(number_songs, split_sizes, rng=rng)
	split_frames = []
	for split in splits:
		split_frames.append(dataframe[grouped_frame.ngroup().isin(split)])
	return split_frames


class SamplingError(Exception):
	pass


def split_containing_all_classes(dataframe, group_column='Name', class_column='Difficulty', split_sizes=0.8, class_map=None, seed=None, accept_fail=False):
	"""Rejection Sampling approach to generate a split.
	 The split both keeps items grouped based on group_column and ensures all values of class_column are present in each split."""
	rng = np.random.default_rng(seed=seed)

	if class_map:
		n_classes_present = len(set(class_map.values()))
	else:
		n_classes_present = dataframe[class_column].nunique()

	tries = 100
	for i in range(tries):
		splits = split_grouped_by_column(dataframe, column=group_column, split_sizes=split_sizes, rng=rng)
		valid_split = True
		for j in range(len(splits)):
			if class_map:
				n_classes_in_test = splits[j][class_column].apply(lambda x: class_map[x]).nunique()
			else:
				n_classes_in_test = splits[j][class_column].nunique()
			if n_classes_in_test != n_classes_present:
				valid_split = False

		if valid_split:
			return splits
	if accept_fail:
		print("A split satisfying all constraints could not be found in {} tries".format(tries), "Returning imperfect split")
		return splits
	else:
		raise SamplingError("A split satisfying all constraints could not be found in {} tries".format(tries))


def get_time_string():
	return strftime('%Y%m%d-%H%M')


def remove_ext(string, symbol='_', up_to=-1):
	"""Removes last up_to substrings after splitting by symbol.
	Generally useful for removing extensions."""
	substrings = string.split(symbol)
	if up_to < 0:
		up_to = max(1, len(substrings)+up_to)
	else:
		up_to = max(1, up_to)
	return symbol.join(substrings[:up_to])


def beep():
	"""Play simple notification sound"""
	duration = 700
	freq = 490
	Beep(freq, duration)

