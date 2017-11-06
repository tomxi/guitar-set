""" A terminal-callable script to wrap analysis for a stem
"""

import argparse
import librosa
import vamp
import jams
import numpy as np
import sox


def get_features(y, fs, note, args):
    open_string = args.open_string_midi
    midi_note = librosa.hz_to_midi(note['values'][0])[0]
    fret_pos = round(midi_note - open_string)

    start_time = float(note['timestamp'])

    dur = float(note['duration'])
    end_time = start_time + dur

    note_y = y[int(round(start_time * fs)) : int(round(end_time * fs))]
    feature_list = [open_string, fret_pos, dur, np.max(note_y ** 2),
                    np.mean(note_y ** 2)]
    spread = librosa.feature.spectral_bandwidth(
            y=note_y, sr=fs, S=None, n_fft=2048, hop_length=512,
            freq=None, centroid=None, norm=True, p=2)
    avg_spread = np.mean(spread)
    max_spread = np.max(spread)
    feature_list.append(avg_spread)
    feature_list.append(max_spread)
    return feature_list


def output_to_jams(y, fs, notes, args):
    jam = jams.JAMS()
    jam.file_metadata.duration = sox.file_info.duration(args.stem_path)
    jam.file_metadata.title = args.stem_path
    ann = jams.Annotation(
        namespace='pitch_midi', time=0,
        duration=jam.file_metadata.duration
    )
    ann.annotation_metadata.data_source = str(args.open_string_midi)
    notes_features = []
    for note in notes:
        start_time = float(note['timestamp'])
        midi_note = librosa.hz_to_midi(note['values'][0])[0]
        dur = float(note['duration'])

        # notes_features.append(get_features(y, fs, note, args))
        # ann.append(time=start_time,
        #            value=midi_note,
        #            duration=dur,
        #            confidence=None)

        if midi_note >= args.open_string_midi-0.5:
            notes_features.append(get_features(y, fs, note, args))
            ann.append(time=start_time,
                       value=midi_note,
                       duration=dur,
                       confidence=None)
        else:
            print(
                'pyin: {} lower than open string {}, discarding'.format(
                    midi_note, args.open_string_midi)
            )
    notes_features = np.array(notes_features)
    ann.sandbox.features = notes_features

    jam.annotations.append(ann)

    return jam


def mono_anal(y, fs, param=None):
    """Run pyin on an audio signal y.

    Parameters
    ----------
    y : np.array
        audio signal
    fs : float
        audio sample rate
    param : dict default=None
        threshdistr : int, default=2
            Yin threshold distribution identifier.
                - 0 : uniform
                - 1 : Beta (mean 0.10)
                - 2 : Beta (mean 0.15)
                - 3 : Beta (mean 0.20)
                - 4 : Beta (mean 0.30)
                - 5 : Single Value (0.10)
                - 6 : Single Value (0.15)
                - 7 : Single Value (0.20)
        outputunvoiced : int, default=0
            Output estimates classified as unvoiced?
                - 0 : No
                - 1 : Yes
                - 2 : Yes, as negative frequencies
        precisetime : int, default=0
            If 1, use non-standard precise YIN timing (slow)
        lowampsuppression : float, default=0.005
            Threshold between 0 and 1 to supress low amplitude pitch estimates.


    """
    if param is None:
        param = {
            'threshdistr': 2,
            'outputunvoiced': 0,
            'precisetime': 0,
            'lowampsuppression': 0.005,
            'onsetsensitivity': 0.7
        }

    output_notes = vamp.collect(y, fs, 'pyin:pyin', output='notes',
                                parameters=param)

    return output_notes['list']


def main(args):
    """build a jams file next to the input file or to a specific directory"""
    print('loading {}'.format(args.stem_path))
    y, fs = librosa.load(args.stem_path, sr=44100)
    notes = mono_anal(y, fs)
    jam = output_to_jams(y, fs, notes, args)
    jam_path = args.stem_path.split('.')[0]+'.jams'
    jam.save(jam_path)
    print('jams file generated')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='analysis for a stem using pYin')
    parser.add_argument(
        'stem_path', type=str, help='path to the stem of interest')
    parser.add_argument(
        'open_string_midi', type=int,
        help='midi number of the open string note'
    )

    main(parser.parse_args())
