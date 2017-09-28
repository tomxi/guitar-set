"""Hex.annotator
"""
import vamp
import numpy as np

import jams
import pretty_midi
import librosa
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join, basename


def mono_pyin(y, fs, param=None):
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
            'lowampsuppression': 0.01
        }

    output = vamp.collect(y, fs, 'pyin:pyin', output='notes', parameters=param)
    return output['list'], len(y) / float(fs)


def wav_f_condition(f, dirpath):
    return isfile(join(dirpath, f)) & (f.split('.')[1] == 'wav')

def transcribe( y=None, fs=None, dirpath=None):
    """Run mono_pyin on a multi-ch signal `y` with sample rate `fs`. Or run
    mono_pyin on all wavs in a folder `dirpath`.


    """
    # first try loading from dirpath
    if dirpath is not None:
        files = [join(dirpath, f) for f in listdir(dirpath) if
                 wav_f_condition(f, dirpath)]
        y = []
        for f in files:
            print(f)
            y_mono, fs = librosa.load(f, sr=44100)
            print(fs)
            y.append(y_mono)
        y = np.array(y)
    elif y is None:
        raise ValueError("either y or dirpath has to not be None!")
    else:
        pass # use y as input.

    num_ch = y.shape[0] if y.ndim != 1 else 1
    # make dummy output
    output = []
    if num_ch != 1:
        for ch in range(num_ch):
            y_mono = y[ch]
            out_mono, dur = mono_pyin(y_mono, fs)
            output.append(out_mono)
    else: # mono sig
        output, dur = mono_pyin(y, fs)
    return output, dur


def output_to_jams(output, dur):
    """Interprets stored output into jams format

        Returns
        -------
        jam : JAMS object
            a jams file containing the annotation
        """
    output = output[0]

    jam = jams.JAMS()
    jam.file_metadata.duration = dur
    for s in range(len(output)):
        ann = jams.Annotation(
            namespace='pitch_midi', time=0,
            duration=jam.file_metadata.duration)
        ann.annotation_metadata.data_source = str(s)
        for i in range(len(output[s])):
            current_note = output[s][i+1]
            start_time = current_note['timestamp']
            midi_note = librosa.hz_to_midi(current_note['values'][0])
            dur = current_note['duration']
            ann.append(time=start_time,
                       value=midi_note,
                       duration=dur,
                       confidence=None)

        jam.annotations.append(ann)

    return jam


def output_to_midi(output):
    """Interprets stored output into MIDI format

    Parameters
    ----------
    output : list

    Returns
    -------
    midi : PrettyMidi object
        a pretty-midi object containing the annotation
    """

    # output = output[0]
    midi = pretty_midi.PrettyMIDI()
    for string_tran in output:
        ch = pretty_midi.Instrument(program=25)
        for note in string_tran:
            pitch = int(round(librosa.hz_to_midi(note['values'][0])[0]))
            end_time = float(note['timestamp'] + note['duration'])
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch, start=float(note['timestamp']),
                end=end_time
            )
            ch.notes.append(n)
        if len(ch.notes) != 0:
            midi.instruments.append(ch)
    return midi


def output_to_plot(output):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : 'p'}

    s = 0
    for string_tran in output:
        for note in string_tran:
            start_time = note['timestamp']
            midi_note = librosa.hz_to_midi(note['values'][0])
            dur = note['duration']
            plt.plot([start_time, start_time + dur],
                     [midi_note, midi_note],
                     style_dict[s])
        s += 1


def sonify(midi, fpath='resources/sonify_out/test.wav'):
    """midi to files sonification. Write sonified wave files to fpath
    """
    signal_out = midi.fluidsynth()
    librosa.output.write_wav(fpath, signal_out, 44100)


