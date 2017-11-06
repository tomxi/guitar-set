"""util
"""
import os
from genericpath import isfile
from os.path import join


def ext_f_condition(f, dirpath, ext):

    return isfile(join(dirpath, f)) & (f.split('.')[1] == ext)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
