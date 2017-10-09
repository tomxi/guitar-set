"""annotator
"""

import numpy as np

import jams
import pretty_midi
import librosa
import csv
import sox
import matplotlib.pyplot as plt
import tempfile
import shutil
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
    cmd = 'python util/mono_anal_script.py {} {}'.format(stem_path,
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


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

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


def transcribe_hex(hex_path):
    temp_path = tempfile.mkdtemp()

    output_mapping = {'00': {1: [1]},
                      '01': {1: [2]},
                      '02': {1: [3]},
                      '03': {1: [4]},
                      '04': {1: [5]},
                      '05': {1: [6]}
                      }

    for mix_type, remix_dict in output_mapping.items():
        tfm = sox.Transformer()
        tfm.remix(remix_dictionary=remix_dict)
        output_path = os.path.join(temp_path, '{}.wav'.format(mix_type))
        tfm.build(hex_path, output_path)

    transcribe(temp_path)

    jam = jamses_to_jams(temp_path)
    shutil.rmtree(temp_path)
    return jam


def jamses_to_midi(jams_files_dir, q=1):
    jams_files = [join(jams_files_dir, f) for f in listdir(jams_files_dir) if
                  ext_f_condition(f, jams_files_dir, 'jams')]
    midi = pretty_midi.PrettyMIDI()
    for j in jams_files:
        jam = jams.load(j)
        ch = pretty_midi.Instrument(program=25)
        ann = jam.search(namespace='pitch_midi')[0]
        for note in ann:
            pitch = int(round(note.value))
            bend_amount = int(round((note.value - pitch) * 4096))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch, start=st,
                end=st+dur
            )
            pb = pretty_midi.PitchBend(pitch=bend_amount*q, time=st)
            ch.notes.append(n)
            ch.pitch_bends.append(pb)
        if len(ch.notes) != 0:
            midi.instruments.append(ch)
    return midi


def jams_to_midi(jam, q=1):
    midi = pretty_midi.PrettyMIDI()
    annos = jam.search(namespace='pitch_midi')
    for anno in annos:
        midi_ch = pretty_midi.Instrument(program=25)
        for note in anno:
            pitch = int(round(note.value))
            bend_amount = int(round((note.value - pitch) * 4096))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch, start=st,
                end=st + dur
            )
            pb = pretty_midi.PitchBend(pitch=bend_amount*q, time=st)
            midi_ch.notes.append(n)
            midi_ch.pitch_bends.append(pb)
        if len(midi_ch.notes) != 0:
            midi.instruments.append(midi_ch)
    return midi


def sonify(midi, fpath='resources/sonify_out/test.wav'):
    """midi to files sonification. Write sonified wave files to fpath
    """
    signal_out = midi.fluidsynth()
    # TODO:write small wave files
    librosa.output.write_wav(fpath, signal_out, 44100)
    return 0


def jamses_to_jams(jams_files_dir):
    jams_files = [join(jams_files_dir, f) for f in listdir(jams_files_dir) if
                  ext_f_condition(f, jams_files_dir, 'jams')]
    jam = jams.JAMS()
    for j in jams_files:
        mono_jams = jams.load(j)
        jam.annotations.append(mono_jams.search(namespace='pitch_midi')[0])
        jam.file_metadata.duration = mono_jams.file_metadata.duration
    return jam


def csvs_to_jams(csv_files_dir):
    jam = jams.JAMS()
    wav_files = [join(csv_files_dir, f) for f in listdir(csv_files_dir) if
                 ext_f_condition(f, csv_files_dir, 'wav')]
    cvs_files = [join(csv_files_dir, f) for f in listdir(csv_files_dir) if
                 ext_f_condition(f, csv_files_dir, 'csv')]
    jam.file_metadata.duration = sox.file_info.duration(wav_files[0])
    for c in cvs_files:
        with open(c, 'r') as stream:
            ann_cvs = csv.reader(stream)
            ann_jam = jams.Annotation(namespace='pitch_midi', time=0,
                                      duration=jam.file_metadata.duration)
            for row in ann_cvs:
                try:
                    st = float(row[0])
                    midi_note = librosa.hz_to_midi(float(row[1]))[0]
                    dur = float(row[2])
                    ann_jam.append(time=st, value=midi_note, duration=dur,
                                   confidence=None)
                except:
                    print('empty row')
                    pass
        jam.annotations.append(ann_jam)

    return jam


def visualize_jams(jam):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}

    s = 0
    for string_tran in jam.search(namespace='pitch_midi'):
        for note in string_tran:
            start_time = note[0]
            midi_note = note[2]
            dur = note[1]
            plt.plot([start_time, start_time + dur],
                     [midi_note, midi_note],
                     style_dict[s])
        s += 1

    plt.show()



