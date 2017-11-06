
import numpy as np
from scipy.optimize import minimize
import librosa


def get_feature_audio(filename):
    sr = 8192
    y, fs = librosa.load(filename, mono=True, sr=sr)
    feature = y ** 2.0
    return feature


def linear_model(x, A, y):
    return np.linalg.norm(np.dot(A, x) - y, ord=2)


def analyze_mix_audio(mtrack):
    mixfile = mtrack.mix_path
    mix_audio = get_feature_audio(mixfile)

    stems = mtrack.stems
    stem_indices = list(stems.keys())
    n_stems = len(stem_indices)
    stem_files = [stems[k].audio_path for k in stem_indices]
    stem_audio = np.array(
        [get_feature_audio(_) for _ in stem_files]
    )

    # force weights to be between 0.01 and 100
    bounds = tuple([(0.01, 100.0) for _ in range(n_stems)])
    res = minimize(
        linear_model, x0=np.ones((n_stems,)), args=(stem_audio.T, mix_audio.T),
        bounds=bounds
    )
    coefs = res['x']

    mixing_coeffs = {
        int(i): float(c) for i, c in zip(stem_indices, coefs)
    }
    return mixing_coeffs
