"""util
"""
import os
import tempfile
from genericpath import isfile
from os.path import join

import librosa
import sox


def ext_f_condition(f, dirpath, ext):
    return isfile(join(dirpath, f)) & (f.split('.')[1] == ext)


def get_all_ext(a_dir, ext='jams'):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if name.endswith(ext)]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def save_small_wav(out_path, y, fs):
    fhandle, tmp_file = tempfile.mkstemp(suffix='.wav')

    librosa.output.write_wav(tmp_file, y, fs)

    tfm = sox.Transformer()
    tfm.convert(bitdepth=16)
    tfm.build(tmp_file, out_path)
    os.close(fhandle)
    os.remove(tmp_file)
