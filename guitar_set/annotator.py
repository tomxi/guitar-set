"""Hex.annotator
"""

import numpy as np

import jams
import pretty_midi
import librosa
import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import isfile, join, basename


# def mono_pyin(y, fs, param=None):
#     """Run pyin on an audio signal y.
#
#     Parameters
#     ----------
#     y : np.array
#         audio signal
#     fs : float
#         audio sample rate
#     param : dict default=None
#         threshdistr : int, default=2
#             Yin threshold distribution identifier.
#                 - 0 : uniform
#                 - 1 : Beta (mean 0.10)
#                 - 2 : Beta (mean 0.15)
#                 - 3 : Beta (mean 0.20)
#                 - 4 : Beta (mean 0.30)
#                 - 5 : Single Value (0.10)
#                 - 6 : Single Value (0.15)
#                 - 7 : Single Value (0.20)
#         outputunvoiced : int, default=0
#             Output estimates classified as unvoiced?
#                 - 0 : No
#                 - 1 : Yes
#                 - 2 : Yes, as negative frequencies
#         precisetime : int, default=0
#             If 1, use non-standard precise YIN timing (slow)
#         lowampsuppression : float, default=0.005
#             Threshold between 0 and 1 to supress low amplitude pitch estimates.
#
#
#     """
#     if param is None:
#         param = {
#             'threshdistr': 2,
#             'outputunvoiced': 0,
#             'precisetime': 0,
#             'lowampsuppression': 0.005,
#             'onsetsensitivity': 0.7
#         }
#
#     output = vamp.collect(y, fs, 'pyin:pyin', output='notes', parameters=param)
#     return output['list'], len(y) / float(fs)


def mono_anal(stem_path, open_string_midi):
    done = False
    cmd = 'python mono_anal_script.py {} {}'.format(stem_path,
                                                    open_string_midi)
    while not done:
        err = os.system(cmd)
        if err:
            print('vamp.collect errored, trying again...')
        else: # successful, no seg fault
            done = True

    return 0


def ext_f_condition(f, dirpath, ext):
    return isfile(join(dirpath, f)) & (f.split('.')[1] == ext)


def transcribe(dirpath=None):
    """Run mono_pyin on a multi-ch signal `y` with sample rate `fs`. Or run
    mono_pyin on all wavs in a folder `dirpath`.
    """
    # first try loading from dirpath
    files = [join(dirpath, f) for f in listdir(dirpath) if
             ext_f_condition(f, dirpath, 'wav')]

    # make dummy output
    output = []

    str_midi_dict = {
        '0': 40,
        '1': 45,
        '2': 50,
        '3': 55,
        '4': 59,
        '5': 64
    }
    for f in files:
        string_id = basename(f).split('.')[0][1]  # the second char of f
        str_midi = str_midi_dict[string_id]
        mono_anal(f, str_midi)

    jams_files = [join(dirpath, f) for f in listdir(dirpath) if
             ext_f_condition(f, dirpath, 'jams')]
    return jams_files


def jams_to_midi(jams_files):
    midi = pretty_midi.PrettyMIDI()
    for j in jams_files:
        jam = jams.load(j)
        ch = pretty_midi.Instrument(program=25)
        ann = jam.search(namespace='pitch_midi')[0]
        for note in ann:
            pitch = int(round(note.value))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch, start=st,
                end=st+dur
            )
            ch.notes.append(n)
        if len(ch.notes) != 0:
            midi.instruments.append(ch)
    return midi


# def output_to_jams(output, dur):
#     """Interprets stored output into jams format
#
#         Returns
#         -------
#         jam : JAMS object
#             a jams file containing the annotation
#         """
#     output = output[0]
#
#     jam = jams.JAMS()
#     jam.file_metadata.duration = dur
#     for s in range(len(output)):
#         ann = jams.Annotation(
#             namespace='pitch_midi', time=0,
#             duration=jam.file_metadata.duration)
#         ann.annotation_metadata.data_source = str(s)
#         for i in range(len(output[s])):
#             current_note = output[s][i+1]
#             start_time = current_note['timestamp']
#             midi_note = librosa.hz_to_midi(current_note['values'][0])
#             dur = current_note['duration']
#             ann.append(time=start_time,
#                        value=midi_note,
#                        duration=dur,
#                        confidence=None)
#
#         jam.annotations.append(ann)
#
#     return jam
#
#
# def output_to_midi(output):
#     """Interprets stored output into MIDI format
#
#     Parameters
#     ----------
#     output : list
#
#     Returns
#     -------
#     midi : PrettyMidi object
#         a pretty-midi object containing the annotation
#     """
#
#     # output = output[0]
#     midi = pretty_midi.PrettyMIDI()
#     for string_tran in output:
#         ch = pretty_midi.Instrument(program=25)
#         for note in string_tran:
#             pitch = int(round(librosa.hz_to_midi(note['values'][0])[0]))
#             end_time = float(note['timestamp'] + note['duration'])
#             n = pretty_midi.Note(
#                 velocity=100 + np.random.choice(range(-5, 5)),
#                 pitch=pitch, start=float(note['timestamp']),
#                 end=end_time
#             )
#             ch.notes.append(n)
#         if len(ch.notes) != 0:
#             midi.instruments.append(ch)
#     return midi
#
#
# def output_to_plot(output):
#     style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
#
#     s = 0
#     for string_tran in output:
#         for note in string_tran:
#             start_time = note['timestamp']
#             midi_note = librosa.hz_to_midi(note['values'][0])
#             dur = note['duration']
#             plt.plot([start_time, start_time + dur],
#                      [midi_note, midi_note],
#                      style_dict[s])
#         s += 1
#
#     # TODO:Make legend and make the plot pretty. Make time axis
#
#     plt.show()
#
def sonify(midi, fpath='resources/sonify_out/test.wav'):
    """midi to files sonification. Write sonified wave files to fpath
    """
    signal_out = midi.fluidsynth()
    # TODO:write small wave files
    librosa.output.write_wav(fpath, signal_out, 44100)
#
#
