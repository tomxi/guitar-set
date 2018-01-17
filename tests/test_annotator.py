import unittest

import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

from guitar_set import pyin_annotator as ghex


# Constant function for this file.
def rELPATH(path):
    out_path = os.path.join(os.path.dirname(__file__), path)
    return out_path

class TestAnnotator(unittest.TestCase):
    def test_mono_pyin(self):
        path = rELPATH('resources/vincent/04_G.wav')
        y, fs = librosa.core.load(path, sr=None)
        output = ghex.mono_pyin(y, fs)
        self.assertEqual(len(output), 38)

    def test_transcribe(self):
        dirpath = rELPATH('resources/vincent')
        output, dur = ghex.transcribe(dirpath=dirpath)
        self.assertIsNotNone(output)

    def test_output_to_jams(self):
        dirpath = rELPATH('resources/vincent')
        output, dur = ghex.transcribe(dirpath=dirpath)
        jam = ghex.output_to_jams(output, dur)
        self.assertIsNotNone(jam)

    def test_output_to_midi(self):
        dirpath = rELPATH('resources/vincent')
        output = ghex.transcribe(dirpath=dirpath)
        midi = ghex.output_to_midi(output)
        self.assertIsNotNone(midi)

    def test_output_to_plot(self):
        dirpath = rELPATH('resources/vincent')
        output = ghex.transcribe(dirpath=dirpath)
        ghex.output_to_plot(output)


    def test_sonify(self):
        raise NotImplementedError
        #TODO



