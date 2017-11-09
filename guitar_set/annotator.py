"""annotator
"""
import csv
import os
import shutil
import tempfile
from os import listdir
from os.path import join, basename

import jams
import librosa
import numpy as np
import pretty_midi
import sox

from guitar_set.util import ext_f_condition


def mono_anal(stem_path, open_string_midi):
    done = False
    cmd = 'python scripts/mono_anal_script.py {} {}'.format(stem_path,
                                                    open_string_midi)
    while not done:
        err = os.system(cmd)
        if err:
            print('vamp.collect errored, trying again...')
        else: # successful, no seg fault
            done = True

    return 0


def transcribe(dirpath):
    """run
    mono_pyin on all wavs in a folder `dirpath`.
    """
    # first load from dirpath
    files = [join(dirpath, f) for f in listdir(dirpath) if
             ext_f_condition(f, dirpath, 'wav')]
    files.sort()
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
    jam.file_metadata.title = os.path.basename(hex_path)
    shutil.rmtree(temp_path)
    print(jam.file_metadata.title)
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


def jamses_to_jams(jams_files_dir):
    jams_files = [join(jams_files_dir, f) for f in listdir(jams_files_dir) if
                  ext_f_condition(f, jams_files_dir, 'jams')]
    jams_files.sort()
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



