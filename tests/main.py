import librosa
import mir_eval.sonify as sonify

from guitar_set import annotator

p_yin = annotator.Annotator(lowampsuppression=0.001)
path4 = './resources/hex_guitar/04_G.wav'
string4, fs = librosa.core.load(path4, sr=None)

out_jams = p_yin.run(string4, fs, plot=1)

sonify.pitch_contour()




