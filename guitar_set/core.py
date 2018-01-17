import os
import jams
import librosa
import librosa.display
import jams.display
import numpy as np
import shutil
import tempfile
import sox
import matplotlib.pyplot as plt
from guitar_set import acf_annotator as acf
from guitar_set import pyin_annotator as pyin


class HexRecording(object):
    def __init__(self, hexpath):
        self.hexpath = hexpath
        self.ys = []
        self.sr = None


        print('Loading {}'.format(hexpath))
        temp_path = tempfile.mkdtemp()
        self._sox_it_out(temp_path)
        self._load_dirpath(temp_path)
        shutil.rmtree(temp_path)
        print('finished loading')

        self.jam_acf = None
        self.jam_pyin = None

    def _load_dirpath(self, mono_dir_path):
        ys = []
        for string_number in range(6):
            mono_path = os.path.join(mono_dir_path,
                                     '{}.wav'.format(string_number))
            y, self.sr = librosa.load(mono_path)
            print('length of y: {}, sr: {}'.format(len(y), self.sr))
            ys.append(y)
        self.ys = ys

    def _init_jam(self):
        jam = jams.JAMS()
        jam.file_metadata.duration = \
            np.asarray(self.ys).shape[1] / float(self.sr)
        jam.file_metadata.title = os.path.basename(self.hexpath)
        return jam

    def _sox_it_out(self, temp_path):
        output_mapping = {'0': {1: [1]},
                          '1': {1: [2]},
                          '2': {1: [3]},
                          '3': {1: [4]},
                          '4': {1: [5]},
                          '5': {1: [6]}
                          }
        for mix_type, remix_dict in output_mapping.items():
            tfm = sox.Transformer()
            tfm.remix(remix_dictionary=remix_dict)
            output_path = os.path.join(temp_path, '{}.wav'.format(mix_type))
            tfm.build(self.hexpath, output_path)

    def detect_jams_acf(self, voicing_quantile=70, voicing_q=None):
        acf0s, fret_energys = acf.acf_from_hexrec(self)
        Ps = acf.energy_to_prob(fret_energys, acf0s, voicing_quantile,
                                voicing_q)
        weight_mat = acf.create_weight_mat(Ps.shape[1])
        states = acf.joint_viterbi(Ps, weight_mat)

        self.jam_acf = self._init_jam()
        anno_list = acf.state_detects_to_anno_list(
            states, self.jam_acf.file_metadata.duration)
        for anno in anno_list:
            self.jam_acf.annotations.append(anno)
        # return states

    def detect_jams_pyin(self):
        jam = pyin.transcribe_hex(self.hexpath)
        self.jam_pyin = pyin.ref_jam_update(jam)

    def visualize_f0s(self):
        for str_num in range(6):
            y = self.ys[str_num]
            anno = self.jam_acf.annotations[str_num]
            plt.figure()
            S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=256)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                     y_axis='mel', sr=self.sr, x_axis='time')

            jams.display.display(anno, color='w', linewidth=3)
            plt.show()



