import pyin
import numpy as np
import matplotlib.pyplot as plt
import librosa

p_yin = pyin.Pyin(lowampsuppression=0.001)
path4 = 'hex_guitar/04_G.wav'
string4, fs = librosa.core.load(path4, sr=None)

out_jams = p_yin.run(string4, fs, plot=1)


