"""Hex.annotator
"""
import vamp
import numpy as np

import jams
import pretty_midi
import librosa
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join


class Annotator(object):
    """annotator using probabalistic yin pitch tracker.

    Parameters
    ----------
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
    lowampsuppression : float, default=0.1
        Threshold between 0 and 1 to supress low amplitude pitch estimates.

    """
    def __init__(
            self, threshdistr=2, outputunvoiced=0, precisetime=0,
            lowampsuppression=0.1):
        """init method
        """

        self.pyin_param = {
            'threshdistr': threshdistr,
            'outputunvoiced': outputunvoiced,
            'precisetime': precisetime,
            'lowampsuppression': lowampsuppression
        }
        self.ann_param = {}
        self.trans_output = None

    def _mono_pyin(self, y, fs):
        """Run pyin on an audio signal y.

        Parameters
        ----------
        y : np.array
            audio signal
        fs : float
            audio sample rate
        """
        output = vamp.collect(
            y, fs, 'pyin:pyin', output='notes',
            parameters=self.pyin_param
        )

        return output['list']

    def transcribe(self, y=None, fs=None, dirpath=None):
        """Run mono_pyin on a multi-ch signal `y` with sample rate `fs`. Or run
        mono_pyin on all wavs in a folder `dirpath`.


        """
        # first try loading from dirpath
        if dirpath is not None:
            files = [join(dirpath, f)
                     for f in listdir(dirpath) if isfile(join(dirpath, f))]
            y = []
            for f in files:
                y_mono, fs = librosa.load(f, sr=None)
                y.append(y_mono)
            y = np.array(y)


        num_ch = y.shape[0] if y.ndim != 1 else 1
        # make dummy output
        output = []
        if num_ch != 1:
            for ch in range(num_ch):
                y_mono = y[ch]
                out_mono = self._mono_pyin(y_mono, fs)
                output.append(out_mono)
        else: # mono sig
            output = self._mono_pyin(y_mono, fs)
        self.trans_output = output

    def output_to_jams(self):
        """Interprets stored output into jams format

        Returns
        -------
        jam : JAMS object
            a jams file containing the annotation
        """
        # changes self.output
        pass

    def output_to_midi(self):
        """Interprets stored output into MIDI format

        Returns
        -------
        midi : PrettyMidi object
            a pretty-midi object containing the annotation
        """
        pass

    def output_to_plot(self):
        """Plot Notes, one color per string
        """
        pass

    def sonify(self):
        pass

    def tablature(self):
        pass

    def rh_pattern(self):
        pass

    def notes_to_jams(self, output, dur, plot):
        jam = jams.JAMS()
        jam.file_metadata.duration = dur
        ann = jams.Annotation(
            namespace='pitch_midi', time=0,
            duration=jam.file_metadata.duration)

        for i in range(len(output['list'])):
            current_note = output['list'][i]
            start_time = current_note['timestamp']
            midi_note = librosa.hz_to_midi(current_note['values'][0])
            dur = current_note['duration']
            ann.append(time=start_time,
                       value=midi_note,
                       duration=dur,
                       confidence=None)
            if plot:
                plt.plot([start_time, start_time + dur],
                         [midi_note, midi_note],
                         '#FFA500')

        jam.annotations.append(ann)

        if plot:
            plt.show()

    def notes_to_midi(self, output):
        string = pretty_midi.PrettyMIDI()
        for i in range(len(output['list'])):
            current_note = output['list'][i]
            start_time = current_note['timestamp']
            midi_note = librosa.hz_to_midi(current_note['values'][0])
