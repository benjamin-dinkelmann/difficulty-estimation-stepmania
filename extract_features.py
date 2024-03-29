import numpy as np
import pandas as pd
from msdparser import MSDParserError

from util import filter_name, remove_ext
import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine
from simfile.timing import BeatValues
from simfile.notes.group import group_notes, SameBeatNotes, OrphanedNotes, NoteWithTail
from traceback import print_exception

# =================DataExtractionForPatternModel=======================
"""
Following Definition from
Caronongan, A. P., & Marcos, N. A. (2021). 
Predicting Chart Difficulty in Rhythm Games Through Classification Using Chart Pattern Derived Attributes. 
In Computational Science and Technology: 7th ICCST 2020, Pattaya, Thailand, 
29–30 August, 2020 (pp. 193-205). Springer Singapore.
"""


def string_seq_automaton(state, definition, token, time_variable, time_delta):
    if token in definition[0]:
        return 1, 0.0, 0
    elif token in definition[state]:
        if state == len(definition) - 1:
            return 0, time_variable + time_delta, 1
        else:
            return state + 1, time_variable + time_delta, 0
    else:
        return 0, 0.0, 0


def calc_pattern_stats(note_data, engine):
    notes = []
    last_time = float('-inf')
    last_tuple = None
    for note in note_data:
        if engine.hittable(note.beat) and \
                note.note_type in {NoteType.TAP, NoteType.HOLD_HEAD, NoteType.ROLL_HEAD}:
            token = ['L', 'D', 'U', 'R'][note.column]
            t = engine.time_at(note.beat)
            if t - last_time < 1e-10:
                token = 'J'
            else:
                notes.append(last_tuple)
            last_tuple = (note.beat, t, token, note.note_type)
            last_time = t
    notes.append(last_tuple)
    notes.pop(0)
    assert len(notes) > 0

    # Patterns to be found:
    # Jumps(Consecutive), Staircase, Candlelight, Crossovers, Drills, Jacks, Step Jumps
    pattern_features = [(0, 0.0)] * 7

    # Initialization
    con_jumps, stair_left, stair_right, candle_up, candle_down, cross_right, cross_left, drill, jack = (0,) * 9
    con_jumps_t, stair_left_t, stair_right_t, candle_up_t, candle_down_t, cross_right_t, cross_left_t, drill_t, jack_t = (
                                                                                                                             0.0,) * 9
    drill_token1, drill_token2 = ['B'] * 2
    jack_token = 'B'
    step_jump_beat = -1
    last_t = notes[0][1]

    # Sequences
    seq_stair_l = ['R', 'U', 'D', 'L']
    seq_stair_r = ['L', 'D', 'U', 'R']
    seq_candle_d = [['U'], ['L', 'R'], ['D']]
    seq_candle_u = [['D'], ['L', 'R'], ['U']]
    seq_cross_l = [['R'], ['U', 'D'], ['L']]
    seq_cross_r = [['L'], ['U', 'D'], ['R']]

    for beat, time, token, note_type in notes:
        time_delta = time - last_t

        # Consecutive Jumps (infinite number to one occurence?)
        if token == 'J':
            con_jumps += 1
            if con_jumps > 1:
                con_jumps_t += time_delta
        else:
            if con_jumps > 1:
                pattern_features[0] = (
                    pattern_features[0][0] + con_jumps - 1,
                    max(pattern_features[0][1], 60 * (con_jumps - 1) / con_jumps_t))
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
        candle_down, candle_down_t, res = string_seq_automaton(candle_down, seq_candle_d, token, candle_down_t,
                                                               time_delta)
        if res:
            pattern_features[2] = (pattern_features[2][0] + 1, max(pattern_features[2][1], 60 * 2 / candle_down_t))
            candle_down_t = 0.0

        # Crossover: left foot on right or vice versa e.g. l u r or r d l
        cross_left, cross_left_t, res = string_seq_automaton(cross_left, seq_cross_l, token, cross_left_t, time_delta)
        if res:
            pattern_features[3] = (pattern_features[3][0] + 1, max(pattern_features[3][1], 60 * 2 / cross_left_t))
            cross_left_t = 0.

        cross_right, cross_right_t, res = string_seq_automaton(cross_right, seq_cross_r, token, cross_right_t,
                                                               time_delta)
        if res:
            pattern_features[3] = (pattern_features[3][0] + 1, max(pattern_features[3][1], 60 * 2 / cross_right_t))
            cross_right_t = 0.0

        # Drills: at least five consecutive notes alternating two arrows
        if token == 'J':
            if drill > 4:
                pattern_features[4] = (
                    pattern_features[4][0] + 1, max(pattern_features[4][1], 60 * (drill - 1) / drill_t))
            drill_token1, drill_token2 = ['B'] * 2
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
                    pattern_features[4] = (
                        pattern_features[4][0] + 1, max(pattern_features[4][1], 60 * (drill - 1) / drill_t))
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
                pattern_features[5] = (
                    pattern_features[5][0] + 1, max(pattern_features[5][1], 60 * (jack - 1) / jack_t))
            jack_token = token
            jack = 1

        # Step Jump: classified as 8th singular steps followed by a jump
        if token == 'J' and beat - step_jump_beat < .6 and step_jump_beat > -1:
            pattern_features[6] = (pattern_features[6][0] + 1, max(pattern_features[6][1], 60 / time_delta))
        elif token != 'J':
            step_jump_beat = beat

        last_t = time

    # weighting+flatten  (grove radar ~0-1, frequency natural number, speed in bpm -> bars per sec)
    weighted_pattern_features = []
    for (frequency, speed) in pattern_features:
        weighted_pattern_features.extend([frequency * 0.05, speed / 240])

    return weighted_pattern_features


def extract_pattern_attributes_song(link):
    song_meta = []
    # chart_specific_attr = ['Difficulty', 'Stream', 'Voltage', 'Air', 'Freeze', 'Chaos', 'n_jump', 'v_jump', 'n_stair', 'v_stair', 'n_candle', 'v_candle', 'n_cross', 'v_cross', 'n_drill', 'v_drill', 'n_jack', 'v_jack', 'n_step_j', 'v_step_j']

    simfileInstance = simfile.open(link)
    song_name = simfileInstance.title if simfileInstance.titletranslit == '' else simfileInstance.titletranslit
    song_artist = simfileInstance.artist if simfileInstance.artisttranslit == '' else simfileInstance.artisttranslit
    song_author = simfileInstance.credit
    song_identifier = filter_name(song_artist) + '-' + filter_name(song_name)
    for chart in simfileInstance.charts:
        # correct chart type?
        if chart.stepstype != 'dance-single':
            continue
        # Note Data readable?
        try:
            note_data = NoteData(chart)
        except IndexError:
            continue

        engine = TimingEngine(TimingData(simfileInstance, chart))
        chart_coarse_diff = chart.difficulty
        chart_fine_diff = chart.meter
        chart_author = filter_name(chart.description if song_author == '' else song_author)
        groove_radar = [float(i) for i in chart.radarvalues.split(',')]  # property yields string not seperated values
        l_radar = len(groove_radar)
        if l_radar > 5:
            groove_radar = groove_radar[:5]
        elif l_radar < 5:
            groove_radar.extend([0.] * (5 - l_radar))

        pattern_attr = calc_pattern_stats(note_data, engine)
        song_meta.append([song_identifier, chart_author, chart_fine_diff, *groove_radar, *pattern_attr])
    return song_meta


# ===================DataExtractionForTimeSeriesModel=======================


def get_note_level_encoding(beat):
    note_levels = [1, 2, 3, 4, 6, 8]  # These multiplied by 4 are the note levels
    for i in range(len(note_levels)):
        if (beat % 1 * note_levels[i]) % 1 < 1e-6:
            return [j + 1 for j in range(i + 1) if note_levels[i] % note_levels[j] == 0]
    return len(note_levels) + 1


class BeatValuesIterator:  # kinda
    """
    Iterates a list of beat values by returning all novel changes between a given beat and the last (iterate)
    """

    def __init__(self, iterable_list: BeatValues):
        self.list = iterable_list
        self.counter = 0
        self.size = len(iterable_list)  # assumes list not changed

    def __len__(self):
        return self.size

    def has_next(self):
        return self.counter < self.size

    def _fetch_up_to(self, beat):
        result = []
        next_element = self.list[self.counter]
        while self.counter < self.size and next_element.beat < beat:
            # Assumes
            self.counter += 1
            result.append(next_element.value)
        return result

    def _fetch_at(self, beat):
        if self.has_next():
            next_element = self.list[self.counter]
            if next_element.beat == beat:
                self.counter += 1
                return [next_element.value]
        return []

    def fetch_change(self, beat, include_simultaneous_change=True):
        if not self.has_next():
            return [], []
        results = self._fetch_up_to(beat)
        if include_simultaneous_change:
            simultaneous = self._fetch_at(beat)
        else:
            simultaneous = []
        return results, simultaneous


def generate_chart_ts(note_data, engine, permutation=None):
    # Constants
    arrow_type_offset = 15
    number_channels = 4
    vector_size = arrow_type_offset + 4 * number_channels

    notes = []
    join_heads_to_tails = True
    for grouped_note in group_notes(note_data, same_beat_notes=SameBeatNotes.JOIN_ALL,
                                    join_heads_to_tails=join_heads_to_tails, orphaned_head=OrphanedNotes.KEEP_ORPHAN,
                                    orphaned_tail=OrphanedNotes.DROP_ORPHAN):
        beat = grouped_note[0].beat
        filtered_grouped_note = [note for note in grouped_note if
                                 note.note_type in {NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD,
                                                    NoteType.MINE}]

        if engine.hittable(beat):
            notes.append((beat, engine.time_at(beat), filtered_grouped_note))

    # Timing data gives Named Tuple name -> value;
    # Value: Warps: Beats skipped, Stop & Delay: seconds
    bpm_change_iter = BeatValuesIterator(engine.timing_data.bpms)
    warps_iter = BeatValuesIterator(engine.timing_data.warps)
    stops_iter = BeatValuesIterator(engine.timing_data.stops)  # notes at same time point need to be hit Before stop
    delays_iter = BeatValuesIterator(engine.timing_data.delays)  # notes at same time point need to be hit AFTER Delay

    # Init
    data = []
    # last_beat = -100000
    last_time = notes[0][1] - 0.25
    beats_with_notes_count = 0
    number_of_beats = len(np.unique([beat for beat, _, _ in notes]))
    song_duration = notes[-1][1] - notes[0][1]
    ongoing_holds = np.zeros(4)
    ongoing_rolls = np.zeros(4)

    for beat, time, notes_at_beat in notes:
        beats_with_notes_count += 1
        # last_beat = beat
        time_delta = time - last_time
        """if len(data) > 0:
			data[-1][arrow_type_offset + number_channels:arrow_type_offset + 2 * number_channels] = ongoing_holds
			data[-1][arrow_type_offset + 2 * number_channels:arrow_type_offset + 3 * number_channels] = ongoing_rolls"""
        ongoing_holds = np.maximum(ongoing_holds - time_delta, 0)
        ongoing_rolls = np.maximum(ongoing_rolls - time_delta, 0)
        data.append(np.zeros([vector_size], dtype=np.float32))

        # Current Tempo (bpm)
        data[-1][0] = engine.bpm_at(beat) / 240

        # Note type encoding (quarter, eight, twelfth, ...)
        data[-1][get_note_level_encoding(beat)] = 1

        # Chart progress in notes
        data[-1][8] = beats_with_notes_count / number_of_beats

        # Chart progress in time
        data[-1][9] = time / song_duration

        # Time passed since last note
        data[-1][10] = min(time_delta * 4, 8)
        last_time = time

        # Bpm change factor
        bpm_changes, change_at = bpm_change_iter.fetch_change(beat)
        if len(bpm_changes) + len(change_at) > 0 and len(data) > 1:  # why bother fetching changes at all?
            data[-1][11] = data[-1][0] / data[-2][0]
        else:
            data[-1][11] = 1.

        # Warps in beats simply as a number -> no input into progress and the like.
        warps, change_at = warps_iter.fetch_change(beat, include_simultaneous_change=False)
        if len(warps) + len(change_at) > 0:
            data[-1][12] = sum(warps)

        # Delays in seconds simply as a number -> no input into progress and the like.
        delays, change_at = delays_iter.fetch_change(beat)
        if len(delays) + len(change_at) > 0:
            data[-1][13] = sum(delays) + sum(change_at)

        # Stops in seconds simply as a number -> no input into progress and the like.
        stops, change_at = stops_iter.fetch_change(beat, include_simultaneous_change=False)
        if len(stops) + len(change_at) > 0:
            data[-1][14] = sum(stops)

        for note in notes_at_beat:

            note_column = note.column
            if permutation is not None:
                note_column = permutation[note_column]
            # Arrow type handling, distinguished by column
            match note.note_type:
                # Binary marking per channel (Is there a tap at this beat on this channel?)
                case NoteType.TAP:
                    data[-1][arrow_type_offset + note_column] = 1
                # Holds are encoded by the remaining time they need to be held (in each channel)
                # which extends the encoding past the single beat where they start
                case NoteType.HOLD_HEAD:
                    data[-1][arrow_type_offset + note_column] = 1
                    if isinstance(note, NoteWithTail):
                        ongoing_holds[note_column] = engine.time_at(note.tail_beat) - time
                    else:
                        ongoing_holds[note_column] = 1
                # analogous to holds
                case NoteType.ROLL_HEAD:
                    data[-1][arrow_type_offset + note_column] = 1
                    if isinstance(note, NoteWithTail):
                        ongoing_rolls[note_column] = engine.time_at(note.tail_beat) - time
                    else:
                        ongoing_rolls[note_column] = 1
                # Binary marking for mines per channel
                case NoteType.MINE:
                    data[-1][arrow_type_offset + 3 * number_channels + note_column] = 1

        data[-1][arrow_type_offset + number_channels:arrow_type_offset + 2 * number_channels] = ongoing_holds
        data[-1][arrow_type_offset + 2 * number_channels:arrow_type_offset + 3 * number_channels] = ongoing_rolls

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
    for chart in simfileInstance.charts:
        # correct chart type?
        if chart.stepstype != 'dance-single':
            continue
        # Note Data readable?
        try:
            note_data = NoteData(chart)
        except IndexError:
            continue

        engine = TimingEngine(TimingData(simfileInstance, chart))
        chart_coarse_diff = chart.difficulty
        chart_fine_diff = chart.meter
        chart_author = filter_name(chart.description if song_author == '' else song_author)

        # tests if all variants of this chart (permutations) are already given or a re-gen is forced
        to_generate = {}
        for i, perm in enumerate(perms):
            # Possible Problem: Perms are enumerated => would not force re-gen if the number remains the same (i.e. may need to refine permutation encoding in identifier)
            chart_identifier = song_identifier + '-' + chart_author + '-' + str(chart_fine_diff) + '_' + str(i)
            if generate:
                tensor_path = os.path.join(store_dir, chart_identifier + file_ext)
                if regenerate or not os.path.isfile(tensor_path):
                    to_generate[tensor_path] = perm
            song_meta.append([song_identifier, chart_author, chart_fine_diff, i, chart_identifier + file_ext])

        # all permutations of channels are generated by once extracting the time series and then permuting the channels of the matrix
        if len(to_generate) > 0:
            # print("Generating for", song_identifier + '-' + chart_author + '-' + str(chart_fine_diff))
            proto_time_series = generate_chart_ts(note_data, engine)
            # print("done")
            # NRows is the number of features
            nrows = proto_time_series.shape[0]
            for path, perm in to_generate.items():
                row_index = np.arange(nrows)
                # permutation is defined as the 4 channels which are mapped to any other channel (bijective)
                for k, v in perm.items():
                    nChannels = 4
                    # 5-1=4 is the number of times these channels come up, i.e. Taps, Holds, Roll, Mine
                    for j in range(1, 5):
                        row_index[-j * nChannels + k] = nrows - j * nChannels + v
                torch.save(torch.from_numpy(proto_time_series[row_index]).to(dtype=torch.float), path)

    return song_meta


def store_meta_dataframes(meta, cols, output_dir, dataset_type='TS'):
    """
    Stores the meta-data of all extracted charts
    grouped by the dataset name, that was determined earlier
    """
    meta.sort(key=lambda x: x[0])
    meta.append(('\\', -1))

    dataset_name = ''
    file_ext = '.txt'
    per_dataset_meta = []
    for name, chart in meta:
        # print(name)
        if name != dataset_name:
            # print(name, dataset_name)
            if len(per_dataset_meta) > 0:
                out_frame = pd.DataFrame(data=per_dataset_meta, columns=cols)
                save_path = os.path.join(output_dir, dataset_name + file_ext)
                out_frame.to_csv(path_or_buf=save_path)
                print('Created {} Dataset {} of size {}'.format(dataset_type, dataset_name, len(out_frame)))
            dataset_name = name
            per_dataset_meta = []
        per_dataset_meta.append(chart)


if __name__ == "__main__":
    import argparse
    import os
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('-root', type=str, help='Root directory. Simplifies definition of other directories',
                        required=False)
    parser.add_argument('-input_dir', type=str, help='Input JSON directory', required=False)
    parser.add_argument('-output_dir', type=str, help='Output directory', required=False)
    parser.add_argument('-extract_patt', action='store_true',
                        help='Extract pattern attributes. Deactivates time series extraction.', required=False)
    # parser.add_argument('-extract_ts', type=str, help='Extract a sequence representation. Samples regularly if >0, else once per non-zero step. Add "b" to process timing information in beats and not seconds.', required=False)
    parser.add_argument('-force_regen', action='store_true', help='Force re-generation of already computed charts')
    parser.set_defaults(
        root='',
        input_dir='data/raw/',
        output_dir='data',
        extract_patt=False,
        # extract_ts='0',
        force_regen=False,
    )
    args = parser.parse_args()
    root = args.root
    input_dir = os.path.join(root, args.input_dir)
    assert os.path.isdir(input_dir)
    input_dir = os.path.abspath(input_dir)
    force_regen = args.force_regen
    b_extract_pattern = args.extract_patt
    # ts_code = cmd_args.extract_ts

    base_output_dir = os.path.join(root, args.output_dir)
    output_dir_time_series = os.path.join(base_output_dir, 'time_series')
    output_dir_pattern = os.path.join(base_output_dir, 'pattern_attr')
    repo_dir_ts = os.path.join(output_dir_time_series, 'repository')
    if not b_extract_pattern and not os.path.isdir(repo_dir_ts):
        os.makedirs(repo_dir_ts)
    if b_extract_pattern and not os.path.isdir(output_dir_pattern):
        os.makedirs(output_dir_pattern)

    permutations = [{0: 0, 1: 1, 2: 2, 3: 3}]
    permutations.extend([{0: 0, 1: 2, 2: 1, 3: 3}, {0: 3, 1: 1, 2: 2, 3: 0}, {0: 3, 1: 2, 2: 1, 3: 0}])
    meta_data = []
    pattern_meta = []
    # Collects all StepMania files inside the input directory or its sub-folders
    # but doesn't ignore packaging structure
    # The pack for a found chart file is the directory in the input directory that contains the chart (if it exists)
    # or the directory, in which the chart was found (otherwise)
    # Note: If there are exactly two directories between the chart and the input directory,
    # then this follows standard StepMania Packs, where the songs in the pack have their own directory with music, image, etc. files.
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
                if (base_name + '.sm') in sm_waiting_list:
                    sm_waiting_list.remove(base_name + '.sm')
            elif f.endswith('.sm'):
                if not remove_ext(f, '.') in files_without_ext:
                    sm_waiting_list.append(f)

        filtered_files.extend(sm_waiting_list)
        # print("Start generating for folder", root)

        # filtered_files is likely a singleton set
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
                # The time series is stored separately, whereas pattern attributes are stored together with
                # the meta-data for organizing and grouping the data
                if not b_extract_pattern:
                    song_metadata = generate_and_store_ts(file_path, store_dir=repo_dir_ts, perms=permutations,
                                                          regenerate=force_regen)
                    for chart_meta in song_metadata:
                        meta_data.append((data_set_name, chart_meta + [file_path]))
                else:
                    song_metadata = extract_pattern_attributes_song(file_path)
                    for chart_meta in song_metadata:
                        pattern_meta.append((data_set_name, chart_meta))
            except Exception as e:  # Single chart error should not collapse whole extraction process
                print('File {} at {} contains format errors. Cannot be extracted.'.format(f, root))
                # print(f'Error message of {type(e)}:', e.__traceback__)
                print_exception(e)
                continue

    if not b_extract_pattern:
        # ts file name contains the location of the externally saved time series
        # sm_fp points to the original StepMania chart filed from which the time series was extracted (which may also be a .ssc or .dwi)
        ts_columns = ['Name', 'Author', 'Difficulty', "Permutation", "ts_file_name", "sm_fp"]
        store_meta_dataframes(meta_data, ts_columns, output_dir_time_series)
    else:
        patt_columns = ['Name', 'Author', 'Difficulty', 'Stream', 'Voltage', 'Air', 'Freeze', 'Chaos', 'n_jump',
                        'v_jump', 'n_stair',
                        'v_stair', 'n_candle', 'v_candle', 'n_cross', 'v_cross',
                        'n_drill', 'v_drill', 'n_jack', 'v_jack', 'n_step_j', 'v_step_j']
        store_meta_dataframes(pattern_meta, patt_columns, output_dir_pattern, dataset_type='Pattern')
