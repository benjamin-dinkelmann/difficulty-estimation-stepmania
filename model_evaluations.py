
import numpy as np
import pandas as pd
from util import remove_ext
from time_series_model import *
from random import random
import json


def score_model(frame, validated_pairs, shift_difficulty=1):
	"""Scores a models quality based on a set of (human) validated pairs of levels (with an associated correctness weights)"""
	multi_run_mode = 'Run' in frame.columns
	full_frame = frame
	# The Version deal with the different versions to handle equality in a ranking
	# Either equally ranked levels are fully correct (1), one ordering is correct (0.5) or neither (0). Alternatively, simply ignore equally ranked levels.
	scoring_versions = [1, .5, 0, 'ignore']
	result_names = ['Correct fraction', 'Eq factor', 'Best', 'Worst', 'Expected', 'Average']

	# If the results are created through cross validation, evaluate each fold separately.
	if not multi_run_mode:
		full_frame['Run'] = 0
	full_frame = full_frame.groupby(by='Run')
	n_runs = len(full_frame.groups)

	results = {v: {res_n: [] for res_n in result_names} for v in scoring_versions}
	for (name, frame) in full_frame:
		# Correct for different mapping of difficulty, e.g. 1 to K (originally) or 0 to K-1 (for models)
		frame['Predicted Difficulty'] += shift_difficulty

		frame = frame.drop('Run', axis=1)
		cross = frame.merge(frame, how='cross')

		# every (non-reflexive) pair once
		cross_ne = cross[(cross["Predicted Difficulty_x"] < cross["Predicted Difficulty_y"])]
		cross_eq = cross[(cross["Predicted Difficulty_x"] == cross["Predicted Difficulty_y"]) & ~((cross["Difficulty_x"] == cross["Difficulty_y"]) & (cross["Name_x"] == cross["Name_y"]))]
		# full_base = cross[(cross["Predicted Difficulty_x"] <= cross["Predicted Difficulty_y"]) & ~((cross["Difficulty_x"] == cross["Difficulty_y"]) & (cross["Name_x"] == cross["Name_y"]))]
		cross_ne = cross_ne.drop(["Predicted Difficulty_x", "Predicted Difficulty_y"], axis=1)

		validated_weights_ne = pd.merge(cross_ne, validated_pairs)
		validated_score_ne = validated_weights_ne['Weight'].sum()

		validation_count_ne = len(validated_weights_ne)
		count_non_validated = len(cross_ne)-len(validated_weights_ne)
		validated_score = validated_score_ne
		full_validation_count = validation_count_ne
		if validation_count_ne == 0:
			n_runs -= 1
			continue

		for scoring_version in scoring_versions:
			if scoring_version == 'ignore':
				base_count = len(cross_ne)
			else:
				base_count = len(cross_ne) + len(cross_eq)

			eq_score = len(cross_eq) * (0 if scoring_version == 'ignore' else scoring_version)

			correct_fraction = validated_score / full_validation_count
			eq_factor = eq_score/base_count
			best = (eq_score+validated_score+count_non_validated)/base_count
			worst = (eq_score+validated_score)/base_count
			expected = (eq_score+validated_score+(validated_score/full_validation_count)*count_non_validated)/base_count
			average = (eq_score+validated_score+0.5*count_non_validated)/base_count
			individual_results = [correct_fraction, eq_factor, best, worst, expected, average]
			scoring_version_dict = results[scoring_version]
			for i, res in enumerate(individual_results):
				scoring_version_dict[result_names[i]].append(res)

	print(n_runs)
	return_dict = {}
	for scoring_version in scoring_versions:
		print('-----------------------')
		print("For {} Equality".format(scoring_version))
		scoring_version_dict = results[scoring_version]
		return_dict[scoring_version] = {}
		for res_n in result_names:
			result_arr = np.array(scoring_version_dict[res_n])
			res_mean = np.mean(result_arr)
			res_std = np.std(result_arr)
			print(res_n+':', res_mean, res_std)
			return_dict[scoring_version][res_n+'_Mean'] = res_mean
			return_dict[scoring_version][res_n + '_Std'] = res_std
	return return_dict


def prepare_prediction_dataframe(import_dir=None, dataset_name=None, frame_path=None, load_base=False, use_pooled_diff=True):
	"""Loads and clean a dataframe containing the difficulties predicted for one dataset
	Loads the original difficulties inplace of the model predicted difficulties when load_base is True"""
	if frame_path is not None:
		try:
			assert os.path.isfile(frame_path)
		except AssertionError:
			print(frame_path)
			raise AssertionError
		predicted_frame_path = frame_path
	else:
		assert os.path.isdir(import_dir)
		if 'time_series' in import_dir:
			dataset_a = '_0'
		else:
			dataset_a = ''
		name_ext = '_predicted.txt'
		predicted_frame_path = os.path.join(import_dir, dataset_name + dataset_a + name_ext)

	dataset_frame = pd.read_csv(filepath_or_buffer=predicted_frame_path, index_col=0)
	if 'Permutation' in dataset_frame.columns:
		dataset_frame = dataset_frame.loc[dataset_frame["Permutation"] == 1].drop(["Permutation", "sm_fp"],
		                                                                          axis=1)
	cols = ['Name', 'Difficulty']
	if 'Run' in dataset_frame.columns:
		cols.append('Run')
	if 'Pooled Difficulty' in dataset_frame.columns:
		cols.append('Pooled Difficulty')
	dataset_a_frame_2 = dataset_frame.drop(
		list(dataset_frame.columns[~dataset_frame.columns.isin(cols)]), axis=1)
	dataset_a_frame_2.drop_duplicates(inplace=True)
	if 'Predicted Difficulty' in dataset_frame.columns:
		cols.append('Predicted Difficulty')
	dataset_frame.drop(list(
		dataset_frame.columns[~dataset_frame.columns.isin(cols)]),
		axis=1, inplace=True)
	dataset_frame = dataset_frame[dataset_frame.index.isin(dataset_a_frame_2.index)]

	if load_base:
		if 'Predicted Difficulty' in dataset_frame.columns:
			dataset_frame = dataset_frame.drop("Predicted Difficulty", axis=1)
		if 'Pooled Difficulty' in dataset_frame.columns and use_pooled_diff:
			dataset_frame["Predicted Difficulty"] = dataset_frame["Pooled Difficulty"]
		else:
			dataset_frame["Predicted Difficulty"] = dataset_frame["Difficulty"]
	return dataset_frame


def prepare_model_scoring(import_dir, dataset_name, experiment_frame_dir, use_base=False):
	"""Loads and cleans dataframe with the predicted difficulties as well as the corresponding validated pairs"""
	dataset_frame = prepare_prediction_dataframe(import_dir, dataset_name, load_base=use_base, use_pooled_diff=False)

	experiment_frame_path = os.path.join(experiment_frame_dir, dataset_name + '_experiment_validated.txt')
	if os.path.isfile(experiment_frame_path):
		experiment_frame = pd.read_csv(filepath_or_buffer=experiment_frame_path, index_col=0)
	else:
		print('No experiment data found. Returning without results')
		return None

	if special_case == 'random':
		rng = np.random.default_rng()
		n_classes = len(dataset_frame['Difficulty'].unique())
		random_numbers_a = rng.integers(0, n_classes, len(dataset_frame))
		df_a = pd.DataFrame(random_numbers_a, columns=['Difficulty'])
		df_a['Predicted Difficulty'] = df_a['Difficulty']
		df_a['Name'] = df_a.index.astype('string')
		experiment_entries = pd.concat([experiment_frame.loc[:, ['Name_x', 'Difficulty_x']],
		                                experiment_frame.loc[:, ['Name_y', 'Difficulty_y']].rename(
			                                columns={'Name_y': 'Name_x', 'Difficulty_y': 'Difficulty_x'})],
		                               axis=0).drop_duplicates()
		m = len(experiment_entries)
		df_a.iloc[0:m, 2] = experiment_entries['Name_x']
		df_a.iloc[0:m, 0] = experiment_entries['Difficulty_x']
		dataset_frame = df_a
	return score_model(dataset_frame, experiment_frame)


def select_df_run(df, run, n_runs, index=None):
	if n_runs > 1:
		reduced_frame = df.get_group(run)
	else:
		reduced_frame = df.get_group(0)
	reduced_frame = reduced_frame.loc[:, ['Name', 'Difficulty', 'Predicted Difficulty']].set_index(['Name', 'Difficulty'])
	if index is not None:
		reduced_frame = reduced_frame.reindex(index, fill_value=-100)
	return reduced_frame


def prepare_grouped_df(df, col_name='Run'):
	if col_name not in df.columns:
		df[col_name] = 0
	return df.groupby(by=col_name)


def compute_ranking_agreement(pred_a, pred_b):
	"""Computes the average agreement between the ranking defined by two frames of predicted difficulties on the same dataset
	Either levelA < levelB, levelA > levelB or levelA == levelB for each ranking.
	disagreement_eq quantifies the number of disagreements where one of the two defined the levels as equal.
	Note: This agreement is only valid if both frames contain the same songs in each fold/run."""
	full_frame_a = prepare_grouped_df(pred_a)
	full_frame_b = prepare_grouped_df(pred_b)
	n_runs_a = len(full_frame_a.groups)
	n_runs_b = len(full_frame_b.groups)
	if not (n_runs_a == n_runs_b or n_runs_a == 1 or n_runs_b == 1):
		print('Agreement not computable')
		return None
	disagreement = []
	disagreement_eq = []
	for run in range(max(n_runs_a, n_runs_b)):
		pred_a = select_df_run(full_frame_a, run, n_runs_a)
		pred_b = select_df_run(full_frame_b, run, n_runs_b)

		array_a = pred_a['Predicted Difficulty'].to_numpy().reshape(-1, 1)
		array_b = pred_b['Predicted Difficulty'].to_numpy().reshape(-1, 1)
		n = array_a.shape[0]
		assert n == array_b.shape[0]
		M_a = np.sign(array_a - array_a.T)
		M_b = np.sign(array_b - array_b.T)
		abs_diff_matrix = np.abs(M_a - M_b).flatten()
		norm_factor = (n*(n-1))
		disagreement_eq.append(np.sum(abs_diff_matrix[np.isclose(abs_diff_matrix, 1)]) / norm_factor)
		disagreement.append(np.sum(np.sign(abs_diff_matrix)) / norm_factor)
	disagreement = np.array(disagreement)
	disagreement_eq = np.array(disagreement_eq)
	return 1 - np.mean(disagreement), np.std(disagreement), np.mean(disagreement_eq), np.std(disagreement_eq), np.mean(1-disagreement+disagreement_eq), np.std(1-disagreement+disagreement_eq)


def generate_difference_matrix(pred_frame, n, index=None):
	"""Computes an upper triangular matrix denoting the average ranking over all runs/folds.
	A resulting entry of the matrix a_ij (-1 < a_ij < 1) represents the following:
	diff(Level_i) < diff(Level_j) if a_ij < 0,  diff(Level_i) > diff(Level_j) if a_ij > 0 and equal otherwise."""
	full_frame = prepare_grouped_df(pred_frame)
	n_runs = len(full_frame.groups)
	diff_matrix = np.zeros([n, n], dtype=int)
	invalid_entries = 0
	for run in range(n_runs):
		not_full = len(full_frame.get_group(run).index) < n
		pred_frame = select_df_run(full_frame, run, n_runs, index=index)
		pred_array = pred_frame['Predicted Difficulty'].to_numpy().reshape(-1, 1)
		assert pred_array.shape[0] == n
		invalid_vector = pred_array == -100
		neg_count = np.count_nonzero(invalid_vector)
		if not_full:
			invalid_matrix = np.logical_or(invalid_vector, invalid_vector.T).astype(int)
			invalid_entries = invalid_matrix + invalid_entries
			diff_matrix += np.sign(pred_array - pred_array.T) * (1-invalid_matrix.astype(int))
		else:
			if neg_count > 0:
				print(neg_count)
			diff_matrix += np.sign(pred_array - pred_array.T)
	return np.triu(diff_matrix/np.maximum(n_runs - invalid_entries, 1))


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-root', type=str, help='Root directory', required=False)
	parser.add_argument('-input_dir', type=str, help='Time series input directory', required=False)
	parser.add_argument('-model_dir', type=str, help='Output directory', required=False)
	parser.add_argument('-device', type=str, help='Model Training device', required=False)
	parser.add_argument('-dataset', type=str, help='Name of Dataset to be evaluated', required=False)
	parser.add_argument('-eval_cv_id', type=str, help='ID of the cross validation that should be evaluated', required=False)
	parser.set_defaults(
		input_dir='data/time_series/',
		output_dir='model_artifacts/evaluations/',
		device='cuda',
		dataset=None,
		root=None,
		eval_cv_id='20221218-2118_6244004',
	)
	args = parser.parse_args()
	root = ""
	if args.root is not None:
		root = args.root
	eval_cv_dir = args.eval_cv_id
	if len(eval_cv_dir) > 0:
		print(eval_cv_dir)

	swap = False
	gen_swapped_pack = False
	model_comparison = True
	ranking_eval = False

	input_dir = os.path.join(root, args.input_dir)
	output_dir = os.path.join(root, args.output_dir)
	if not torch.cuda.is_available():
		device = 'cpu'
	else:
		device = args.device

	data_name = 'fraxtil'
	if args.dataset is not None:
		data_name = args.dataset

	if swap:
		swap_save_dir = os.path.join(input_dir, 'swaps')
		if not os.path.isdir(swap_save_dir):
			os.mkdir(swap_save_dir)

		index_file_paths = [file for file in os.scandir(input_dir) if
		                    file.is_file() and str(file.name).endswith('_0.txt')]
		indices = {}
		index_frames = {}
		for file in index_file_paths:
			index_frame = pd.read_csv(filepath_or_buffer=file.path, index_col=0)
			file_name = remove_ext(file.name)
			if not ('Name' in index_frame.columns and 'Difficulty' in index_frame.columns):
				continue
			index_frame['Difficulty'] -= 1
			if 'Permutation' in index_frame.columns:
				index_frame = index_frame.loc[index_frame["Permutation"] == 1].drop(["Permutation"], axis=1)
			index_frames[file_name] = index_frame
			index_frame = index_frame.set_index(['Name', 'Difficulty'])
			indices[file_name] = index_frame.index
		dataset_names = list(indices.keys())

		generate = False
		if generate:
			cv_dirs = ['20221218-1135_6243977', '20221218-1136_6243978', '20221218-1136_6243979', '20221218-1136_6243981', '20221218-1137_6243982', '20221218-1137_6243983', '20221218-1138_6243984', '20221218-1138_6243985', '20221218-1138_6243987', '20221218-1138_6243988', '20221218-1139_6243989', '20221218-1140_6243992', '20221218-1140_6243993', '20221218-1140_6243994', '20221218-1141_6243995', '20221218-1141_6243996', '20221218-1141_6243997', '20221218-1142_6243998', '20221218-1143_6243999', '20221218-1143_6244000', '20221218-1143_6244001', '20221218-1624_6244002', '20221218-2112_6244003', '20221218-2118_6244004', '20221218-2131_6244005', '20221218-2146_6244006', '20221219-0624_6244007', '20221219-0636_6244008', '20221219-0636_6244009', '20221219-0636_6244010', '20221222-0916_6248259', '20221222-0916_6248260', '20221222-0922_6248269', '20221222-0922_6248270', '20221231-1210_6263777']
			print(len(cv_dirs))

			configurations = {}
			for i, cv_dir in enumerate(cv_dirs):
				# print(i, ':', cv_dir)
				comparison_dir = os.path.join(output_dir, cv_dir)
				config_path = os.path.join(comparison_dir, 'config.txt')
				if os.path.isfile(config_path):
					config = pd.read_csv(config_path, index_col=0).loc[:, ['dataset', 'loss_variant', 'ts_freq']].squeeze('index').to_dict()
					if config['ts_freq'] < 0:
						config['loss_variant'] = config['ts_freq']
					config.pop('ts_freq')
					# if config['loss_variant'] >= 0:  # Don't factor in baselines
					configurations[cv_dir] = config
			for dataset_name in dataset_names:
				configurations[dataset_name] = {'dataset': dataset_name, 'loss_variant': 'base'}

			configuration_frame = pd.DataFrame.from_dict(configurations, orient='index')
			print("Configurations", configuration_frame)
			configurations = configuration_frame.groupby(by='loss_variant')
			group_ids = configurations.groups.keys()
			print("Loss names", group_ids)
			loss_modes = ['NLL', 'NNRank', 'Poisson-Binomial', 'RED-SVM', 'Laplace', 'Regression', 'Pattern', 'Characteristics']

			# Preparation of matrices
			for dataset_name in dataset_names:
				print('Starting pre-computation for', dataset_name)
				dataset_index = indices[dataset_name]
				matrix_length = len(dataset_index)
				for group_id in group_ids:
					group_name = (loss_modes[group_id] if type(group_id) == int and -len(loss_modes)+1 <= group_id < len(loss_modes) else group_id)
					print('Handling group', group_name)
					group = configurations.get_group(group_id)
					swap_importance_matrix = np.zeros([matrix_length, matrix_length], dtype=float)
					c = 0
					for cv_dir in group.index:
						if group_id == 'base':
							if cv_dir != dataset_name:
								continue
							comparison_dir = input_dir
							file_names = [remove_ext(file.name) for file in os.scandir(comparison_dir) if file.is_file() and str(file.name).endswith('_0.txt')]
							if dataset_name not in file_names:
								continue
							use_original_difficulties = True
							frame_path = os.path.join(comparison_dir, cv_dir+'_0.txt')
						else:
							comparison_dir = os.path.join(output_dir, cv_dir)
							file_names = [remove_ext(file.name) for file in os.scandir(comparison_dir) if file.is_file() and str(file.name).endswith('_predicted.txt')]
							use_original_difficulties = False
							if dataset_name not in file_names:
								continue
							frame_path = os.path.join(comparison_dir, dataset_name+'_predicted.txt')
						prediction_frame = prepare_prediction_dataframe(frame_path=frame_path, load_base=use_original_difficulties)
						if group_id == 'base':
							prediction_frame.loc[:, 'Difficulty'] -= 1
						swap_importance_matrix += generate_difference_matrix(prediction_frame, matrix_length, index=dataset_index)
						c += 1
					file_name = dataset_name + '_' + group_name + '_swap_importance.npy'
					group_save_path = os.path.join(swap_save_dir, file_name)
					np.save(group_save_path, swap_importance_matrix/max(c, 1))
			print('Pre-computation complete')

		all_swap_groups = [file for file in os.scandir(swap_save_dir) if file.is_file() and str(file.name).endswith('_swap_importance.npy')]
		swap_group_datasets = np.array([file.name.split('_')[0] for file in all_swap_groups])
		print('Collecting all swaps')
		for dataset_name in dataset_names:
			print(dataset_name)
			reduced_swap_group_indices = np.arange(len(swap_group_datasets))[swap_group_datasets == dataset_name]
			if len(reduced_swap_group_indices) < 2:
				print('Not enough entries found for', dataset_name)
				continue
			swap_difference_matrix = 0
			c = 0
			for i, idx1 in enumerate(reduced_swap_group_indices):
				# print(i)
				swap_group_a = all_swap_groups[idx1]
				dataset_name_a, loss_name_a = swap_group_a.name.split('_')[:2]
				ranking_matrix_a = np.load(swap_group_a.path)

				for idx2 in reduced_swap_group_indices[i+1:]:
					swap_group_b = all_swap_groups[idx2]
					dataset_name_b, loss_name_b = swap_group_b.name.split('_')[:2]
					if dataset_name_a != dataset_name_b:
						print(dataset_name_a, dataset_name_b)
						continue
					ranking_matrix_b = np.load(swap_group_b.path)
					temp = ranking_matrix_a * ranking_matrix_b
					swap_difference_matrix = swap_difference_matrix - temp * (temp < 0).astype(float)
					c += 1
			swap_difference_matrix = swap_difference_matrix/max(c, 1)
			print(dataset_name, ' - Possible Swaps:', np.count_nonzero(swap_difference_matrix > 0))
			upper_threshold = 0.5
			swap_difference_matrix = np.where(swap_difference_matrix > upper_threshold, 0, swap_difference_matrix)
			k = 1000
			top_k_threshold = np.partition(swap_difference_matrix, axis=None, kth=-k)[-k]
			quantile_thresholds = [0.95, .99]
			quantiles = np.quantile(swap_difference_matrix, quantile_thresholds)
			quantile_thresholds.append('top_n threshold')
			quantiles = list(quantiles)
			quantiles.append(top_k_threshold)
			print('Count Quantiles', '; '.join(
				[str(quantile_thresholds[i]) + ': ' + str(quantiles[i]) for i in range(len(quantiles))]))
			multi_index_frame = index_frames[dataset_name]
			save_path = os.path.join(input_dir, dataset_name + '_swaps.txt')
			importance_threshold = max(quantiles[-1], 1e-4)
			relevant_indices = np.nonzero(swap_difference_matrix >= importance_threshold)
			print(dataset_name, ' - Chosen Swaps:', np.count_nonzero(swap_difference_matrix >= importance_threshold))
			n_swaps = relevant_indices[0].shape[0]
			if len(relevant_indices) < 2:
				print(swap_difference_matrix.shape)
			new_range_index = pd.RangeIndex(stop=n_swaps)
			swap_frame_a = multi_index_frame.iloc[relevant_indices[0]]
			swap_frame_a.index = new_range_index
			swap_frame_b = multi_index_frame.iloc[relevant_indices[1]]
			swap_frame_b.index = new_range_index
			swap_frame = swap_frame_a.merge(swap_frame_b, left_index=True, right_index=True)
			swap_frame['swap_importance'] = swap_difference_matrix[relevant_indices[0], relevant_indices[1]]
			swap_frame.sort_values(by='swap_importance', ascending=False, inplace=True)
			swap_frame.to_csv(save_path)

	if model_comparison:
		if len(eval_cv_dir) > 0:
			comparison_dir = os.path.join(output_dir, eval_cv_dir)
			all_dataset_scores = {}
			for file in os.scandir(comparison_dir):
				if file.is_file() and str(file.name).endswith('_predicted.txt'):
					file_name = remove_ext(file.name)
					print('########################################################')
					print(file_name)
					score_results = prepare_model_scoring(comparison_dir, file_name, input_dir)
					if score_results is not None and len(score_results) > 0:
						all_dataset_scores[file_name] = score_results
			with open(os.path.join(comparison_dir, 'scoring_results.json'), 'w') as f:
				json.dump(all_dataset_scores, f)
		else:
			for dataset_name in ['itg', 'fraxtil', 'Gpop', 'GullsArrows', 'Speirmix']:
				print(dataset_name)
				special_case ='base'
				comparison_dir = os.path.join(root, 'data/'+'time_series')
				# prepare_model_scoring(comparison_dir, dataset_name, input_dir, special_case=special_case)
				name_ext = '_0.txt'
				original_frame_path = os.path.join(input_dir, dataset_name+name_ext)
				original_frame = pd.read_csv(original_frame_path, index_col=0)
				original_frame["Difficulty"] -= 1
				original_frame["Predicted Difficulty"] = original_frame["Difficulty"]
				original_frame.drop(list(
					original_frame.columns[~original_frame.columns.isin(['Name', 'Difficulty', 'Predicted Difficulty'])]),
					axis=1, inplace=True)
				original_frame.drop_duplicates(inplace=True)
				experiment_frame_path = os.path.join(comparison_dir, dataset_name + '_experiment_validated.txt')
				if os.path.isfile(experiment_frame_path):
					experiment_frame = pd.read_csv(filepath_or_buffer=experiment_frame_path, index_col=0)
					score_model(original_frame, experiment_frame)

	if ranking_eval:
		if len(eval_cv_dir) > 0:
			comparison_dir = os.path.join(output_dir, eval_cv_dir)
		else:
			comparison_dir = os.path.join(root, 'data/'+'time_series')
		comparison_dir2 = comparison_dir
		use_original_difficulties = False
		if comparison_dir2 == comparison_dir:
			use_original_difficulties = True
		data_set_names_a = [remove_ext(file.name, up_to=-2) for file in os.scandir(comparison_dir) if file.is_file() and str(file.name).endswith('_predicted.txt')]
		data_set_names_b = [remove_ext(file.name, up_to=-2) for file in os.scandir(comparison_dir2) if file.is_file() and str(file.name).endswith('_predicted.txt')]
		agreement_result_cols = ['Agreement_Mean', 'Agreement_Std', 'Disagreement_Eq_Mean', 'Disagreement_Eq_Std', 'MostlyAgreement_Mean', 'MostlyAgreement_Std']
		agreement_result_dict = {}
		for col in agreement_result_cols:
			agreement_result_dict[col] = {}
		for file_name in data_set_names_a:
			if file_name in data_set_names_b:
				print(file_name)
				frame_a = prepare_prediction_dataframe(comparison_dir, file_name)
				frame_b = prepare_prediction_dataframe(comparison_dir2, file_name, load_base=use_original_difficulties)
				agreement_results = compute_ranking_agreement(frame_a, frame_b)
				if agreement_results is not None:
					result_string = ''
					for i, result in enumerate(agreement_results):
						result_string += ' '+str(result)
						agreement_result_dict[agreement_result_cols[i]][file_name] = result
					print('Ranking Acc:' + result_string)
		with open(os.path.join(comparison_dir, 'ranking_results.json'), 'w') as f:
			json.dump(agreement_result_dict, f)

	if gen_swapped_pack:
		import shutil
		import re
		import glob

		index_file_names = [remove_ext(file.name) for file in os.scandir(input_dir) if
		                    file.is_file() and str(file.name).endswith('_0.txt')]
		name_ext = '_swaps.txt'
		output_folder = '../experiment_packs'
		pair_id = 0
		k = 50
		for dataset_name in index_file_names:
			print('Generating packs for', dataset_name)
			output_folder_path = os.path.join(input_dir, output_folder, dataset_name)
			swap_frame_path = os.path.join(input_dir, dataset_name + name_ext)
			assert os.path.isfile(swap_frame_path)
			swap_frame = pd.read_csv(filepath_or_buffer=swap_frame_path, index_col=0)

			# visualize distribution
			"""difficulties = np.maximum(swap_frame['Difficulty_x'].to_numpy(), swap_frame['Difficulty_y'].to_numpy())
			difficulty_counts = np.bincount(difficulties)
			plt.bar(np.arange(len(difficulty_counts))+1, difficulty_counts)
			plt.show()"""

			# clear previous entries
			if os.path.isdir(output_folder_path):
				shutil.rmtree(output_folder_path)
				os.mkdir(output_folder_path)

			extensions = ['x', 'y']
			completed_name_set = {}
			waiting_name_queue = set()
			variants = ["a", "b"]
			assigned_ids = []
			c = 0
			for j, row_tuple in enumerate(swap_frame.itertuples()):
				assigned_ids.append(-1)
				if c >= k:
					continue

				row = row_tuple._asdict()
				names = row['Name_x'] + "-{}".format(row['Difficulty_x']), row['Name_y'] + "-{}".format(row['Difficulty_y'])

				if names[0] in completed_name_set and names[1] in completed_name_set:
					continue
				elif names[0] in completed_name_set or names[1] in completed_name_set:
					idx = (1 if names[0] in completed_name_set else 0)
					if not names[idx] in waiting_name_queue or completed_name_set[names[1-idx]] > 1:
						waiting_name_queue.add(names[idx])
						continue
				else:
					idx = -1
					completed_name_set[names[0]] = 1
					completed_name_set[names[1]] = 1

				if idx >= 0:
					waiting_name_queue.remove(names[idx])
					completed_name_set[names[idx]] = 1
					completed_name_set[names[1-idx]] += 1
				else:
					for idx in [0, 1]:
						if names[idx] in waiting_name_queue:
							waiting_name_queue.remove(names[idx])

				difficulty = max(row['Difficulty_x'], row['Difficulty_y'])+1
				pair_id += 1
				print(j)
				c += 1
				assigned_ids[-1] = pair_id
				if random() > 0.5:
					variants.reverse()

				for i in range(2):
					# Todo: Add info AB for easier reconstruction
					ext = extensions[i]
					sm_fp = row['sm_fp_'+ext]
					sm_dir, sm_filename = os.path.split(sm_fp)
					dest_path = os.path.join(output_folder_path, row['Name_' + ext]+"_{}".format(pair_id))
					shutil.copytree(sm_dir, dest_path, dirs_exist_ok=True)
					for file in glob.glob(dest_path+'/*.ssc'):
						os.remove(file)
					new_sm_fp = os.path.join(dest_path, sm_filename)
					original_difficulty = row['Difficulty_'+ext] + 1
					title = "#TITLE:{:02d}_{:03d}{}_{};".format(difficulty, pair_id, variants[i], names[i].split('-')[-2])
					with open(new_sm_fp, 'r', encoding='utf8') as f:
						text = f.read()
					reg_select_chart = re.compile(r'#TITLE:[^;]*;(?P<before>.*?)(?:#NOTES.*)?(?P<active>#NOTES:\W+dance-single:[^:]+:)[^:]+:(?P<space>[^:0-9]+)'+str(original_difficulty)+r'(?P<after>[^#]+).*', flags=re.DOTALL)
					new_text = reg_select_chart.sub(title+r'\g<before>\g<active>\g<space>Edit:\g<space>'+str(difficulty)+r'\g<after>', text)
					with open(new_sm_fp, 'w', encoding='utf8') as f:
						f.write(new_text)
			id_array = np.array(assigned_ids)
			swap_frame['Experiment_IDs'] = id_array
			print('Number of pairs accepted', np.max(id_array) - np.min(id_array[id_array >= 0]) + 1)
			swap_frame.to_csv(swap_frame_path)
			swap_frame = swap_frame[swap_frame['Experiment_IDs'] >= 0]
			difficulties = np.maximum(swap_frame['Difficulty_x'].to_numpy(), swap_frame['Difficulty_y'].to_numpy())
			difficulty_counts = np.bincount(difficulties)
			plt.bar(np.arange(len(difficulty_counts)) + 1, difficulty_counts)
			plt.show()

