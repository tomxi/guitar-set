"""interpreter
"""
import numpy as np
import pretty_midi
from matplotlib import lines as mlines, pyplot as plt

from guitar_set import util as util


def jams_to_midi(jam, q=1):
    # q = 1: with pitch bend. q = 0: without pitch bend.
    midi = pretty_midi.PrettyMIDI()
    annos = jam.search(namespace='note_midi')
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


def sonify_jams(jam, fpath='resources/sonify_out/test.wav', q=1):
    midi = jams_to_midi(jam, q) # q=1 : with pitchbend
    signal_out = midi.fluidsynth()
    util.save_small_wav(fpath, signal_out, 44100)
    return fpath


def visualize_jams(jam, save_path=None):
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    string_dict = {0: 'E', 1: 'A', 2: 'D', 3: 'G', 4: 'B', 5: 'e' }
    s = 0
    handle_list = []
    plt.figure()
    for string_tran in jam.search(namespace='note_midi'):
        handle_list.append(mlines.Line2D([], [], color=style_dict[s],
                                         label=string_dict[s]))
        for note in string_tran:
            start_time = note[0]
            midi_note = note[2]
            dur = note[1]
            plt.plot([start_time, start_time + dur],
                     [midi_note, midi_note],
                     style_dict[s], label=string_dict[s])
        s += 1
    plt.xlabel('Time (sec)')
    plt.ylabel('Pitch (midi note number)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handle_list)
    plt.title(jam.file_metadata.title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def tablaturize_jams(jam, save_path=None):
    str_midi_dict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    style_dict = {0 : 'r', 1 : 'y', 2 : 'b', 3 : '#FF7F50', 4 : 'g', 5 : '#800080'}
    s = 0
    plt.figure()
    for string_tran in jam.search(namespace='note_midi'):
        for note in string_tran:
            start_time = note[0]
            midi_note = note[2]
            fret = int(round(midi_note - str_midi_dict[s]))
            plt.scatter(start_time, s, marker="${}$".format(fret), color = style_dict[s])
        s += 1
    plt.xlabel('Time (sec)')
    plt.ylabel('String Number')
    plt.title(jam.file_metadata.title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()
