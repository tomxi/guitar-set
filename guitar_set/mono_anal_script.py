""" A terminal-callable script to wrap analysis for a stem
"""

import argparse
import librosa
import vamp
import jams
# import numpy as np


def output_to_jams(notes, dur, open_string_midi):
    jam = jams.JAMS()
    jam.file_metadata.duration = dur
    ann = jams.Annotation(
        namespace='pitch_midi', time=0,
        duration=jam.file_metadata.duration
    )
    ann.annotation_metadata.data_source = str(open_string_midi)
    for note in notes:
        start_time = float(note['timestamp'])
        midi_note = librosa.hz_to_midi(note['values'][0])[0]
        dur = float(note['duration'])
        if midi_note >= open_string_midi-0.5:
            print([midi_note, open_string_midi-0.5])
            ann.append(time=start_time,
                       value=midi_note,
                       duration=dur,
                       confidence=None)
        else:
            print('pyin: lower than open string, discarding')
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
            'lowampsuppression': 0.01,
            'onsetsensitivity': 0.3
        }

    output_notes = vamp.collect(y, fs, 'pyin:pyin', output='notes',
                                parameters=param)

    return output_notes['list'], len(y) / float(fs)


def main(args):
    """build a jams file next to the input file or to a specific directory"""
    print('loading audio')
    y, fs = librosa.load(args.stem_path, sr=44100)
    print('about to call vamp.collect')
    notes, dur = mono_anal(y, fs)
    print('finished calling vamp.collect')
    jam = output_to_jams(notes, dur, args.open_string_midi)
    jam_path = args.stem_path.split('.')[0]+'.jams'
    jam.save(jam_path)
    print('jams file generated')
    pass


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
