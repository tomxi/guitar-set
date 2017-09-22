"""Pyin pitch tracker
"""
# import glob
import jams
import librosa
import numpy as np
# import os
import vamp
from vamp import vampyhost
import matplotlib.pyplot as plt


class Pyin(object):
    """probabalistic yin pitch tracker.

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

        self.parameters = {
            'threshdistr': threshdistr,
            'outputunvoiced': outputunvoiced,
            'precisetime': precisetime,
            'lowampsuppression': lowampsuppression
        }

    def run(self, y, fs, plot=0):
        """Run pyin on an audio signal y.

        Parameters
        ----------
        y : np.array
            audio signal
        fs : float
            audio sample rate

        Returns
        -------
        jam : JAMS
            JAMS object with pyin output
        """
        output = vamp.collect(
            y, fs, 'pyin:pyin', output='notes',
            parameters=self.parameters
        )

        jam = jams.JAMS()
        jam.file_metadata.duration = len(y) / float(fs)
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
                plt.plot([start_time, start_time+dur], [midi_note, midi_note])

        jam.annotations.append(ann)

        if plot:
            plt.show()

        return jam

