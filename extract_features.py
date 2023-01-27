import numpy as np
import pandas as pd
from math import floor
from util import filter_name


def get_note_type(beat_number):
	note_type = round((beat_number % 1) * 48)
	note_type = note_type//2 + (.5 if note_type%2 == 1 else 0)
	note_type_idx = -1
	if note_type == 0:  # 4th
		note_type_idx = 0
	elif note_type == 12:  # 8th
		note_type_idx = 1
	elif note_type % 8 == 0:  # 12th
		note_type_idx = 2
	elif note_type % 6 == 0:  # 16th
		note_type_idx = 3
	elif note_type % 4 == 0:  # 24th
		note_type_idx = 4
	elif note_type % 3 == 0:  # 32nd
		note_type_idx = 5
	return note_type_idx


def count_steps_in_code(code):
	count = 0
	for char in code:
		if char != '0' and char != '3':
			count += 1
	return count


def extract_pattern_attributes_dataset(frame, name, meta, start_idx):
	chart_specific_attr = ['Difficulty', 'Stream', 'Voltage', 'Air', 'Freeze','Chaos', 'n_jump', 'v_jump', 'n_stair', 'v_stair', 'n_candle', 'v_candle', 'n_cross', 'v_cross',
	                       'n_drill', 'v_drill', 'n_jack', 'v_jack', 'n_step_j', 'v_step_j']

	for i, chart in enumerate(meta['charts']):
		difficulty, groove_radar, pattern_attr = calc_pattern_stats(chart)
		pattern_attr_flat = []
		for (n,v) in pattern_attr:
			pattern_attr_flat.extend((n,v/60))
		frame.loc[start_idx + i, chart_specific_attr] = [difficulty, *groove_radar[:5], *pattern_attr_flat]
	k = len(meta['charts'])
	frame.loc[start_idx:start_idx + k, ['Name']] = name
	frame.loc[start_idx:start_idx + k, ['sm_fp']] = meta['sm_fp']


def get_token(code):
	tokens = ['L', 'D', 'U', 'R']
	if count_steps_in_code(code) > 1:
		return 'J'
	else:
		pos = -1
		for char in code:
			pos += 1
			if char != '0' and char != '3':
				return tokens[pos]


def string_seq_automaton(state, definition, token, time_variable, time_delta):
	if token in definition[0]:
		return 1, 0.0, 0
	elif token in definition[state]:
		if state == len(definition)-1:
			return 0, time_variable+time_delta, 1
		else:
			return state+1, time_variable+time_delta, 0
	else:
		return 0, 0.0, 0


def calc_pattern_stats(chart):
	difficulty = chart['difficulty_fine']
	notes = chart['notes']
	groove_radar = chart['groove_radar']

	# Patterns to be found:
	# Jumps(Consecutive), Staircase, Candlelight, Crossovers, Drills, Jacks, Step Jumps
	pattern_features = [(0, 0.0)]*7

	# Initialization
	con_jumps, stair_left, stair_right, candle_up, candle_down, cross_right, cross_left, drill, jack = (0,)*9
	con_jumps_t, stair_left_t, stair_right_t, candle_up_t, candle_down_t, cross_right_t, cross_left_t, drill_t, jack_t = (0.0,) * 9
	drill_token1, drill_token2 = ['B']*2
	jack_token = 'B'
	step_jump_beat = -1
	last_t = notes[0][2]

	# Sequences
	seq_stair_l = ['R', 'U', 'D', 'L']
	seq_stair_r = ['L', 'D', 'U', 'R']
	seq_candle_d = [['U'], ['L', 'R'], ['D']]
	seq_candle_u = [['D'], ['L', 'R'], ['U']]
	seq_cross_l = [['R'], ['U', 'D'], ['L']]
	seq_cross_r = [['L'], ['U', 'D'], ['R']]

	for note in notes:
		token = get_token(note[3])
		time_delta = note[2] - last_t

		# Consecutive Jumps (infinite number to one occurence?)
		if token == 'J':
			con_jumps += 1
			if con_jumps > 1:
				con_jumps_t += time_delta
		else:
			if con_jumps > 1:
				pattern_features[0] = (pattern_features[0][0]+con_jumps-1, max(pattern_features[0][1], 60*(con_jumps-1)/con_jumps_t))
			con_jumps = 0
			con_jumps_t = 0.0

		# Staircase: LDUR or RUDL
		if token == seq_stair_r[0]:
			stair_right = 1
			stair_right_t = 0.0
		elif token == seq_stair_r[stair_right]:
			stair_right += 1
			stair_right_t += time_delta
			if stair_right == 4:
				pattern_features[1] = (pattern_features[1][0] + 1, max(pattern_features[1][1], 60 * 3 / stair_right_t))
				stair_right = 0
				stair_right_t = 0.0
		else:
			stair_right = 0
			stair_right_t = 0.0

		if token == seq_stair_l[0]:
			stair_left = 1
			stair_left_t = 0.0
		elif token == seq_stair_l[stair_left]:
			stair_left += 1
			stair_left_t += time_delta
			if stair_left == 4:
				pattern_features[1] = (pattern_features[1][0] + 1, max(pattern_features[1][1], 60 * 3 / stair_left_t))
				stair_left = 0
				stair_left_t = 0.0
		else:
			stair_left = 0
			stair_left_t = 0.0

		# Candlelight: Foot on first step to opposite panel on third step ( uld, urd, dlu, dru, and per def also crossover)
		candle_up, candle_up_t, res = string_seq_automaton(candle_up, seq_candle_u, token, candle_up_t, time_delta)
		if res:
			pattern_features[2] = (pattern_features[2][0] + 1, max(pattern_features[2][1], 60 * 2 / candle_up_t))
			candle_up_t = 0.
		candle_down, candle_down_t, res = string_seq_automaton(candle_down, seq_candle_d, token, candle_down_t, time_delta)
		if res:
			pattern_features[2] = (pattern_features[2][0] + 1, max(pattern_features[2][1], 60 * 2 / candle_down_t))
			candle_down_t = 0.0

		# Crossover: left foot on right or vice versa e.g. l u r or r d l
		cross_left, cross_left_t, res = string_seq_automaton(cross_left, seq_cross_l, token, cross_left_t, time_delta)
		if res:
			pattern_features[3] = (pattern_features[3][0] + 1, max(pattern_features[3][1], 60 * 2 / cross_left_t))
			cross_left_t = 0.

		cross_right, cross_right_t, res = string_seq_automaton(cross_right, seq_cross_r, token, cross_right_t, time_delta)
		if res:
			pattern_features[3] = (pattern_features[3][0] + 1, max(pattern_features[3][1], 60 * 2 / cross_right_t))
			cross_right_t = 0.0

		# Drills: at least five consecutive notes alternating two arrows
		if token == 'J':
			if drill > 4:
				pattern_features[4] = (pattern_features[4][0] + 1, max(pattern_features[4][1], 60 * (drill - 1) / drill_t))
			drill_token1, drill_token2 = ['B']*2
			drill = 0
			drill_t = 0.0
		elif drill_token1 == 'B':
			drill_token1 = token
		elif drill_token2 == 'B' and not token == drill_token1:
			drill_token2 = drill_token1
			drill_token1 = token
			drill = 2
			drill_t += time_delta
		elif drill_token2 != 'B':
			if token == drill_token2:
				drill += 1
				drill_t += time_delta
				drill_token1, drill_token2 = drill_token2, drill_token1
			else:
				if drill > 4:
					pattern_features[4] = (pattern_features[4][0] + 1, max(pattern_features[4][1], 60 * (drill-1)/ drill_t))
				if token == drill_token1:
					drill_token2 = 'B'
					drill = 1
					drill_t = 0
				else:
					drill_token2 = drill_token1
					drill_token1 = token
					drill = 2
					drill_t = time_delta

		# Jacks: Two or more on same arrow
		if jack_token == token and token != 'J':
			jack += 1
			jack_t += time_delta
		else:
			if jack > 1:
				pattern_features[5] = (pattern_features[5][0] + 1, max(pattern_features[5][1], 60 * (jack-1) / jack_t))
			jack_token = token
			jack = 1

		# Step Jump: classified as 8th singular steps followed by a jump
		if token == 'J' and note[1] - step_jump_beat < .6 and step_jump_beat > -1:
			pattern_features[6] = (pattern_features[6][0] + 1, max(pattern_features[6][1], 60/time_delta))
		elif token != 'J':
			step_jump_beat = note[1]

		last_t = note[2]

	# weighting  (grove radar ~0-1, frequency natural number)
	weighted_pattern_features = []
	for (frequency, speed) in pattern_features:
		weighted_pattern_features.append((frequency*0.05, speed*0.1))

	return difficulty, groove_radar, weighted_pattern_features


# Note Format: [no. of measure, beats in this measure, number of this beat in measure],
#              beat number of this beat (in quarter notes) like 20.5 (eight note after 20th quarter), time in seconds, code (str) of steps (e.g. "0001")]
def get_regular_time_series(notes, bpm_changes, samples_per_second=24, adaptive=False, beats_not_seconds=False):
	# setup: [bpm, unary notetype 1-7, step l-d-u-r, hold l-d-u-r]
	vector_size = 19
	time_series_l = []
	ref_tempo = 240
	bpm = bpm_changes[0][1]/ref_tempo
	bpm_number = 1
	step_vec_offset = vector_size-8

	eps = 1e-6
	assert samples_per_second > 0 or adaptive
	slot_length_per_sample = 1 / samples_per_second if not adaptive else 1

	first_sample_time_point = notes[0][1 if beats_not_seconds else 2]
	# is considered non-inclusive, i.e. previous slot has [last_time_point-slot_length, last_time_point) as interval
	next_time_point = min(0, floor(first_sample_time_point))
	last_time_point = notes[0][2]-0.25
	# notes[0][2] if not beats_not_seconds else notes[0][1]   # last_time_point time point or beat number
	song_duration = notes[-1][2] - (notes[0][2] if adaptive else 0)
	number_of_beats = len(notes)

	empty_sample = np.zeros(vector_size, dtype=float)
	empty_sample[0] = bpm
	empty_sample[step_vec_offset-2] = song_duration

	note_types = [4, 8, 12, 16, 24, 32, 1.3]

	ongoing_holds = np.zeros(4)
	n_steps_in_current_time_slot = 1
	averaged_features = [0, 8, 9]

	note_count = 0
	for beat_measure, beat_number, time, code in notes:
		note_count += 1
		if time < 0:
			print('Negative Time! ', time)
		this_point = time if not beats_not_seconds else beat_number
		# 		if not adaptive and last_time_point+slot_length_per_sample < this_point:
		# 			num_empty_samples = floor((this_point-last_time_point -eps)/slot_length_per_sample)  # epsilon for numerical stability

		# fills empty samples for all slots between two slots with notes
		if not adaptive and next_time_point+slot_length_per_sample <= this_point + eps:
			num_empty_samples = floor((this_point-next_time_point)/slot_length_per_sample + eps)  # epsilon for numerical stability

			if n_steps_in_current_time_slot > 1 and num_empty_samples > 0:
				# average bpm and holds
				time_series_l[-1][averaged_features] /= n_steps_in_current_time_slot
				n_steps_in_current_time_slot = 1
			for i in range(num_empty_samples):
				if i > 0:
					empty_sample = empty_sample.copy()
				empty_sample[step_vec_offset-2] = next_time_point/song_duration if not beats_not_seconds else empty_sample[0]*4*slot_length_per_sample/song_duration
				time_series_l.append(empty_sample)
				next_time_point += slot_length_per_sample

		vec = np.zeros(vector_size, dtype=float)

		bpm_change = False
		while bpm_number < len(bpm_changes) and bpm_changes[bpm_number][0] >= time:
			bpm = bpm_changes[bpm_number][1] / ref_tempo
			bpm_number += 1
			bpm_change = True
		vec[0] = bpm

		# relative beat
		vec[step_vec_offset-3] = note_count/number_of_beats
		# relative time
		vec[step_vec_offset-2] = time/song_duration

		# time delta
		vec[step_vec_offset - 1] = min((time - last_time_point) * 4, 8)  # (-log2(min((this_point - next_time_point)*(2 if beats_not_seconds else 4), 8) + eps)+3)/10  # which is last time point in this context
		last_time_point = time

		if not adaptive:
			# dealing with multiple notes falling in one slot
			# Todo: Check correct solution for changing tempi?
			if next_time_point > this_point + eps:
				# print(n_steps_in_current_time_slot+1, "in slot", this_point, next_time_point, slot_length_per_sample)
				n_steps_in_current_time_slot += 1
			else:
				next_time_point += slot_length_per_sample
				if n_steps_in_current_time_slot > 1:
					# time_series_l[-1] /= n_steps_in_current_time_slot  # sum vs. average
					# average bpm and holds
					time_series_l[-1][averaged_features] /= n_steps_in_current_time_slot
					n_steps_in_current_time_slot = 1

		note_type = get_note_type(beat_number)+1
		note_type = note_type if note_type > 0 else 7
		# hierarchical setting i.e. 16th is half an 8th, but not directly related of 12th
		vec[[i+1 for i in range(note_type) if note_types[note_type-1] % note_types[i] == 0]] = 1

		hold_change = False
		if len(code) == 4:  # only single pad songs
			for i, char in enumerate(code):
				if char == '0':
					continue
				elif char == '1':
					vec[step_vec_offset+i] = 1
				elif char == '3':
					vec[step_vec_offset+4+i] = 1
					ongoing_holds[i] = 0
					hold_change = True
				elif char == '2' or char == '4':
					vec[step_vec_offset+i] = 1
					ongoing_holds[i] = 1
					hold_change = True
					if n_steps_in_current_time_slot > 1:
						time_series_l[-1][step_vec_offset+4+i] += 1
		else:
			print("Code of incorrect length: ", code)

		if not adaptive:
			empty_sample = empty_sample.copy()
		if hold_change:
			empty_sample[step_vec_offset+4:] = ongoing_holds
		if bpm_change:
			empty_sample[0] = bpm
		empty_sample[step_vec_offset-3] = note_count/number_of_beats
		vec[step_vec_offset + 4:] += ongoing_holds

		if n_steps_in_current_time_slot > 1:
			delta = min(time_series_l[-1][step_vec_offset - 1], vec[step_vec_offset - 1])
			time_series_l[-1][:step_vec_offset+4] += vec[:step_vec_offset+4]  # don't add holds
			time_series_l[-1][step_vec_offset - 1] = delta
		else:
			time_series_l.append(vec)
	return np.vstack(time_series_l)


if __name__ == "__main__":
	import argparse
	import os
	import json
	import torch

	parser = argparse.ArgumentParser()
	parser.add_argument('-json_dir', type=str, help='Input JSON directory', required=False)
	parser.add_argument('-output_dir', type=str, help='Output directory', required=False)
	parser.add_argument('-extract_patt',  action='store_true', help='Extract pattern attributes', required=False)
	parser.add_argument('-extract_ts', type=str, help='Extract a sequence representation. Samples regularly if >0, else once per non-zero step. Add "b" to process timing information in beats and not seconds.', required=False)
	parser.set_defaults(
		json_dir='data/json/',
		output_dir='data',
		extract_patt=False,
		extract_ts=None,
	)
	args = parser.parse_args()
	assert os.path.isdir(args.json_dir)
	json_dir = os.path.abspath(args.json_dir)
	ts_code = args.extract_ts

	b_extract_pattern_attributes = args.extract_patt
	b_extract_time_series = ts_code is not None
	
	output_dir_pattern_attr = os.path.abspath(os.path.join(args.output_dir, 'pattern_attr'))
	if b_extract_pattern_attributes and not os.path.isdir(output_dir_pattern_attr):
		os.mkdir(output_dir_pattern_attr)
	output_dir_time_series = os.path.abspath(os.path.join(args.output_dir, 'time_series'))
	if b_extract_time_series and not os.path.isdir(output_dir_time_series):
		os.mkdir(output_dir_time_series)

	for data_set in os.scandir(json_dir):
		# if data_set.name != 'Speirmix':
		# 	continue
		if data_set.is_dir():
			pack_names = [obj.name for obj in os.scandir(data_set.path) if obj.is_dir()]  # Names of subfolders

			if b_extract_pattern_attributes:
				# sm_fp is excluded when extracting
				columns = ['Name', 'Difficulty', 'Stream', 'Voltage', 'Air', 'Freeze','Chaos', 'n_jump', 'v_jump', 'n_stair', 'v_stair', 'n_candle', 'v_candle', 'n_cross', 'v_cross',
	                       'n_drill', 'v_drill', 'n_jack', 'v_jack', 'n_step_j', 'v_step_j']
				default_vector = [['', 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.,
				                   0, 0., 0, 0., 0, 0., 0,
				                   0., 0, 0., 0, 0.]]
				out_frames = []
				for pack_name in pack_names:
					songs = []
					song_idx_start_map = {}
					idx = 0
					for obj in os.scandir(os.path.join(data_set.path, pack_name)):
						if obj.is_file():
							with open(obj.path, 'r') as f:
								song_meta = json.loads(f.read())
							song_id = obj.name
							name = "{}-{}-{}".format(pack_name, filter_name(song_meta['artist']),
							                         filter_name(song_meta['title']))
							songs.append([name, song_id, song_meta])

							k = len(song_meta['charts'])
							song_idx_start_map[song_id] = idx
							idx += k

					out_frame = pd.DataFrame(data=default_vector * idx,
					                         columns=columns)
					for song_name, song_id, song_meta in songs:
						extract_pattern_attributes_dataset(out_frame, song_name, song_meta, song_idx_start_map[song_id])

					out_frames.append(out_frame)
				out_frame = pd.concat(out_frames)
				file_datatype = '.txt'
				data_set_out_path = os.path.join(output_dir_pattern_attr, data_set.name + file_datatype)
				out_frame.to_csv(path_or_buf=data_set_out_path)

			if b_extract_time_series:
				samples_per_beat = int(ts_code[:-1] if ts_code[-1]=='b' else ts_code)
				beats_not_seconds = ts_code[-1] == 'b'

				adaptive = samples_per_beat == 0
				data_set_name = data_set.name+"_{}".format(samples_per_beat) + ("b" if beats_not_seconds else "")
				output_dir_data_set = os.path.abspath(os.path.join(output_dir_time_series, data_set_name))
				if not os.path.isdir(output_dir_data_set):
					os.mkdir(output_dir_data_set)
				columns = ['Name', 'Difficulty', "Permutation", "sm_fp"]
				out_list = []

				for pack_name in pack_names:
					for obj in os.scandir(os.path.join(data_set.path, pack_name)):
						if obj.is_file():
							with open(obj.path, 'r') as f:
								song_meta = json.loads(f.read())
							name = "{}-{}-{}".format(pack_name, filter_name(song_meta['artist']), filter_name(song_meta['title']))
							num_permutations = {}
							for chart in song_meta['charts']:
								difficulty = chart['difficulty_fine']
								if difficulty <= 20:
									if difficulty in num_permutations:
										num_permutations[difficulty] += 1
									else:
										num_permutations[difficulty] = 1
									freq = (song_meta['bpms'][0][1]/60)*samples_per_beat if not beats_not_seconds else samples_per_beat
									time_series_file_name = name+'-{}_{}.pt'.format(difficulty, num_permutations[difficulty])
									time_series = get_regular_time_series(chart['notes'], song_meta['bpms'], samples_per_second=freq, adaptive=adaptive, beats_not_seconds=beats_not_seconds)
									tensor_path = os.path.join(output_dir_data_set, time_series_file_name)
									torch.save(torch.from_numpy(time_series.T).to(dtype=torch.float), tensor_path)
									out_list.append([name, difficulty, num_permutations[difficulty], song_meta['sm_fp']])
				out_frame = pd.DataFrame(data=out_list, columns=columns)
				file_datatype = '.txt'
				data_set_out_path = os.path.join(output_dir_time_series, data_set_name + file_datatype)
				out_frame.to_csv(path_or_buf=data_set_out_path)
