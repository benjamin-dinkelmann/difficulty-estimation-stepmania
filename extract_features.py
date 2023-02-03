import numpy as np
import pandas as pd
from msdparser import MSDParserError

from util import filter_name, remove_ext
import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine


def get_note_level_encoding(beat):
	note_levels = [1, 2, 3, 4, 6, 8]  # These multiplied by 4 are the note levels
	for i in range(len(note_levels)):
		if (beat%1*note_levels[i])%1 < 1e-6:
			return [j+1 for j in range(i+1) if note_levels[i] % note_levels[j] == 0]
	return len(note_levels)+1


def generate_chart_ts(note_data, engine, permutation=None):
	notes = []
	for note in note_data:
		if engine.hittable(note.beat) and \
				note.note_type in {NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD}:
			note_column = note.column
			if permutation is not None:
				note_column = permutation[note_column]
			notes += [(note.beat, engine.time_at(note.beat), note_column, note.note_type)]

	data = []
	last_beat = -100000
	last_time = notes[0][1] - 0.25
	note_count = 0
	number_of_beats = len(np.unique([beat for beat, _, _, _ in notes]))
	song_duration = notes[-1][1] - notes[0][1]
	ongoing_holds = np.zeros(4)

	for beat, time, column, ntype in notes:
		if beat != last_beat:
			note_count += 1
			last_beat = beat
			if len(data) > 0:
				data[-1][11 + 4:] += ongoing_holds
			data += [np.array([0] * 19, dtype=np.float32)]

			data[-1][0] = engine.bpm_at(beat) / 240

			data[-1][8] = note_count / number_of_beats

			data[-1][9] = time / song_duration

			data[-1][10] = min((time - last_time) * 4, 8)
			last_time = time

			data[-1][get_note_level_encoding(beat)] = 1

		match ntype:
			case NoteType.TAP:
				data[-1][11 + column] = 1
			case NoteType.TAIL:
				data[-1][11 + 4 + column] = 1
				ongoing_holds[column] = 0
			case NoteType.HOLD_HEAD | NoteType.ROLL_HEAD:
				data[-1][11 + column] = 1
				ongoing_holds[column] = 1
	data[-1][11 + 4:] += ongoing_holds

	return np.vstack(data).T


def generate_and_store_ts(link, store_dir=None, perms=({i: i for i in range(4)},), regenerate=False, file_ext='.pth'):
	song_meta = []
	if store_dir is None:
		generate = False
	else:
		generate = True

	simfileInstance = simfile.open(link)
	song_name = simfileInstance.title if simfileInstance.titletranslit == '' else simfileInstance.titletranslit
	song_artist = simfileInstance.artist if simfileInstance.artisttranslit == '' else simfileInstance.artisttranslit
	song_author = simfileInstance.credit
	song_identifier = filter_name(song_artist) + '-' + filter_name(song_name)
	engine = TimingEngine(TimingData(simfileInstance))
	for chart in simfileInstance.charts:
		# correct chart type
		if chart.stepstype != 'dance-single':
			continue
		# Note Data readable?
		try:
			note_data = NoteData(chart)
		except IndexError:
			continue
		chart_coarse_diff = chart.difficulty
		chart_fine_diff = chart.meter
		chart_author = filter_name(chart.description if song_author == '' else song_author)

		# todo: make perms more efficient -> swap resulting arrays?
		for i, perm in enumerate(perms):
			chart_identifier = song_identifier + '-' + chart_author + '-' + str(chart_fine_diff) + '_' + str(i)
			if generate:
				tensor_path = os.path.join(store_dir, chart_identifier+file_ext)
				if regenerate or not os.path.isfile(tensor_path):
					time_series = generate_chart_ts(note_data, engine, permutation=perm)
					torch.save(torch.from_numpy(time_series).to(dtype=torch.float), tensor_path)
			song_meta.append([song_identifier, chart_author, chart_fine_diff, i, chart_identifier+file_ext])

	return song_meta


if __name__ == "__main__":
	import argparse
	import os
	import torch

	parser = argparse.ArgumentParser()
	parser.add_argument('-input_dir', type=str, help='Input JSON directory', required=False)
	parser.add_argument('-output_dir', type=str, help='Output directory', required=False)
	# parser.add_argument('-extract_patt',  action='store_true', help='Extract pattern attributes', required=False)
	# parser.add_argument('-extract_ts', type=str, help='Extract a sequence representation. Samples regularly if >0, else once per non-zero step. Add "b" to process timing information in beats and not seconds.', required=False)
	parser.add_argument('-force_regen', action='store_true', help='Force re-generation of already computed charts')
	parser.set_defaults(
		input_dir='data/raw/',
		output_dir='data',
		# extract_patt=False,
		# extract_ts='0',
		force_regen=False,
	)
	args = parser.parse_args()
	assert os.path.isdir(args.input_dir)
	input_dir = os.path.abspath(args.input_dir)
	force_regen = args.force_regen
	# ts_code = args.extract_ts

	base_output_dir = args.output_dir
	output_dir_time_series = os.path.join(base_output_dir, 'time_series')
	repo_dir_ts = os.path.join(output_dir_time_series, 'repository')
	if not os.path.isdir(repo_dir_ts):
		os.makedirs(repo_dir_ts)
	# os.path.relpath()
	# os.path.split()

	# samples_per_beat = int(ts_code[:-1] if ts_code[-1] == 'b' else ts_code)
	# beats_not_seconds = ts_code[-1] == 'b'

	permutations = [{0: 0, 1: 1, 2: 2, 3: 3}]
	permutations.extend([{0: 0, 1: 2, 2: 1, 3: 3}, {0: 3, 1: 1, 2: 2, 3: 0}, {0: 3, 1: 2, 2: 1, 3: 0}])
	meta_data = []
	for (root, _, files) in os.walk(input_dir):
		root = os.path.abspath(root)
		filtered_files = []
		files_without_ext = []
		sm_waiting_list = []
		for f in files:
			# Assumes .dwi and .ssc don't occur together
			if f.endswith((".dwi", ".ssc")):
				filtered_files.append(f)
				base_name = remove_ext(f, '.')
				files_without_ext.append(base_name)
				if (base_name+'.sm') in sm_waiting_list:
					sm_waiting_list.remove(base_name+'.sm')
			elif f.endswith('.sm'):
				if not remove_ext(f, '.') in files_without_ext:
					sm_waiting_list.append(f)

		filtered_files.extend(sm_waiting_list)
		# likely single entry
		for f in filtered_files:
			split_path_to_file = os.path.normpath(os.path.relpath(root, input_dir)).split(os.sep)
			pack_idx = len(split_path_to_file) - 2
			if pack_idx >= 0:
				data_set_name = split_path_to_file[0]
			else:
				data_set_name = os.path.split(root)[1]
			data_set_name = filter_name(data_set_name)

			try:
				file_path = os.path.join(root, f)
				song_metadata = generate_and_store_ts(file_path, store_dir=repo_dir_ts, perms=permutations, regenerate=force_regen)
				for chart_meta in song_metadata:
					meta_data.append((data_set_name, chart_meta+[file_path]))
			except Exception as e:
				print('File {} at {} contains format errors. Cannot be extracted.'.format(f, root))
				print('Error message:', e)
				continue
	columns = ['Name', 'Author', 'Difficulty', "Permutation", "ts_file_name", "sm_fp"]
	meta_data.sort(key=lambda x: x[0])
	meta_data.append(('\\', -1))

	data_set_name = ''
	file_datatype = '.txt'
	per_dataset_meta = []
	for name, chart_meta in meta_data:
		# print(name)
		if name != data_set_name:
			# print(name, data_set_name)
			if len(per_dataset_meta) > 0:
				out_frame = pd.DataFrame(data=per_dataset_meta, columns=columns)
				data_set_out_path = os.path.join(output_dir_time_series, data_set_name + file_datatype)
				out_frame.to_csv(path_or_buf=data_set_out_path)
				print('Created Dataset {} of size {}'.format(data_set_name, len(out_frame)))
			data_set_name = name
			per_dataset_meta = []
		per_dataset_meta.append(chart_meta)


