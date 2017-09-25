import unittest

import os
import librosa
import matplotlib.pyplot as plt

from guitar_set import annotator


# Constant function for this file.
def rELPATH(path):
    out_path = os.path.join(os.path.dirname(__file__), path)
    return out_path


class TegitstAnnotator(unittest.TestCase):
    def setUp(self):
        self.app = annotator.Annotator(lowampsuppression=0.001)

    def test_mono_pyin(self):
        path = rELPATH('./resources/hex_guitar/04_G.wav')
        y, fs = librosa.core.load(path, sr=None)
        output = self.app._mono_pyin(y, fs)
        print(output[0:3])
        self.assertEqual(len(output), 39)

    def test_transcribe(self):
        dirpath = rELPATH('./resources/hex_guitar')
        self.app.transcribe(dirpath=dirpath)
        self.assertIsNotNone(self.app.trans_output)

    # def test_output_to_jams(self):
    #     raise NotImplementedError
    #
    # def test_output_to_midi(self):
    #     raise NotImplementedError
    #
    # def test_output_to_plot(self):
    #     raise NotImplementedError
    #
    # def test_sonify(self):
    #     raise NotImplementedError
    #
    #




