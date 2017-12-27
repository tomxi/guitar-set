import librosa
import mir_eval.display
import librosa.display
import numpy as np
import os
import numba
import scipy.signal as signal
import matplotlib.pyplot as plt

STR_MIDI_DICT = {
    0: 40,
    1: 45,
    2: 50,
    3: 55,
    4: 59,
    5: 64
}

STATE_INDEX = np.arange(-2,24,0.25)

#TODO: Rename STR_MIDI_DICT to all caps



def eng_to_voicing_prob(A0, quantile=75, epsilon=5e-7, Q=None):
    if Q is None:
        Q = np.percentile(A0, quantile)
    else:
        quantile = None
    print(quantile, Q)
    V = np.minimum(A0, Q) / (Q + epsilon)
    return V


def constrained_acf(y, sr, open_str_midi):
    midi_min = open_str_midi - 2
    midi_max = open_str_midi + 24
    fmin = librosa.midi_to_hz(midi_min)
    fmax = librosa.midi_to_hz(midi_max)

    max_lag = librosa.time_to_samples(1. / fmin, sr=sr)[0]
    min_lag = librosa.time_to_samples(1. / fmax, sr=sr)[0]

    print('max_lag:{} min_lag:{}'.format(max_lag, min_lag))
    yf = librosa.util.frame(y, frame_length=max_lag)
    acf = librosa.autocorrelate(yf, axis=0)
    print('acf.shape:{}'.format(acf.shape))

    return acf, min_lag


def lag_to_fret(cliped_acf_frame, min_lag, sr, open_str_midi):
    real_lag = range(len(cliped_acf_frame))[:] + min_lag
    freq = 1. / librosa.samples_to_time(real_lag, sr=sr)
    pitch = librosa.hz_to_midi(freq)
    fret = pitch - open_str_midi
    new_x_cord = np.arange(-2, 24, 0.25)
    frame_fret_energy = np.interp(new_x_cord, fret[::-1],
                                  cliped_acf_frame[::-1])
    return fret, frame_fret_energy


def compute_fret_energy(cliped_acf, min_lag, sr, open_str_midi):
    fret_energy = []
    for frame in cliped_acf.T:
        fret, frame_fret_energy = lag_to_fret(frame, min_lag, sr,
                                              open_str_midi)
        fret_energy.append(frame_fret_energy)

    return fret, np.array(fret_energy).T


@numba.jit(nopython=True)
def _viterbi(logp, logA, V, ptr, S):
    T, m = logp.shape

    V[0] = logp[0]

    for t in range(1, T):
        # Want V[t, j] <- p[t, j] * max_k V[t-1, k] * A[k, j]
        #    assume at time t-1 we were in state k
        #    transition k -> j

        # Broadcast over rows:
        #    Tout[k, j] = V[t-1, k] * A[k, j]
        #    then take the max over columns
        #
        # We'll do this in log-space for stability

        Tout = V[t - 1] + logA.T

        # Unroll the max/argmax loop to enable numba support
        for j in range(m):
            ptr[t, j] = np.argmax(Tout[j])
            V[t, j] = logp[t, j] + np.max(Tout[j])

    # Now roll backward

    # Get the last state
    S[-1] = np.argmax(V[-1])

    for t in range(T - 2, -1, -1):
        S[t] = ptr[t + 1, S[t + 1]]

    return S, V, ptr


def viterbi(p, A):
    '''Viterbi decoding for discriminative HMMs

    Parameters
    ----------
    p : np.ndarray [shape=(T, m)], non-negative
        p[t] is the distribution over states at time t.
        Each row must sum to 1.

    A : np.ndarray [shape=(m, m)], non-negative
        A[i,j] is the probability of a transition from i->j.
        Each row must sum to 1.

    Returns
    -------
    s : np.ndarray [shape=(T,)]
        The most likely state sequence
    '''

    T, m = p.shape

    assert A.shape == (m, m)

    assert np.all(A >= 0)
    assert np.all(p >= 0)
    assert np.allclose(A.sum(axis=1), 1)
    assert np.allclose(p.sum(axis=1), 1)

    V = np.zeros((T, m), dtype=float)

    ptr = np.zeros((T, m), dtype=int)

    logA = np.log(A + 1e-10)
    logp = np.log(p + 1e-10)
    S = np.zeros(T, dtype=int)

    return _viterbi(logp, logA, V, ptr, S)


def make_self_transition(N, alpha=0.05):
    T = np.empty((N, N), dtype=float)
    T[:] = alpha / (N - 1)

    np.fill_diagonal(T, 1 - alpha)

    return T


def make_bump_transition(N, window='hann', width=5, alpha=0.05):
    T = np.eye(N, dtype=float)
    w = librosa.filters.get_window(window, width, fftbins=False)[np.newaxis]
    T = signal.convolve(T, w, mode='same', )

    T = (1 - alpha) * T + alpha
    return T / T.sum(axis=1, keepdims=True)


def mod_bump_transition(N, window='hann', width=5, alpha=0.05):
    T = np.eye(N, dtype=float)
    w = librosa.filters.get_window(window, width, fftbins=False)[np.newaxis]
    T = signal.convolve(T, w, mode='same', )

    T = (1 - alpha) * T + alpha
    # treat last row differently
    T = T / T.sum(axis=1, keepdims=True)

    T[-1, :-1] = (1 - T[-1, -1]) / (N - 1)
    T[:-1, -1] = (1 - T[-1, -1]) / (N - 1)
    return T / T.sum(axis=1, keepdims=True)


def mod_string_bump_transition(N, window='hann', width=5, alpha=0.005):
    T_i = np.eye(N, dtype=float)
    w = librosa.filters.get_window(window, width, fftbins=False)[np.newaxis]
    T_i = signal.convolve(T_i, w, mode='same')
    # open string
    T_o = np.zeros((N, N), dtype=float)
    T_o[:, 8] = 1
    T_o = signal.convolve(T_o, w, mode='same')
    T_o = T_o + T_o.T
    # combine
    T = T_o + T_i
    # unifrom bed
    T = (1 - alpha) * T + alpha
    # treat last row differently
    T = T / T.sum(axis=1, keepdims=True)

    T[-1, :-1] = (1 - T[-1, -1]) / (N - 1)
    T[:-1, -1] = (1 - T[-1, -1]) / (N - 1)
    return T / T.sum(axis=1, keepdims=True)


def energy_to_prob(fret_energys, acf0s):
    voiced_probs = eng_to_voicing_prob(acf0s, Q=0.005)
    unvoiced_probs = (1 - voiced_probs) * np.max(fret_energys)
    Ps = []
    for fe, up in zip(fret_energys, unvoiced_probs):
        total_energy = np.vstack((fe, up))
#         total_energy = fe
        sigsq = np.mean(np.abs(total_energy))
        P = np.exp(0.5 * total_energy / sigsq)
        P = librosa.util.normalize(P.astype(np.float64), axis=0, norm=1)
        Ps.append(P)
    return np.asarray(Ps)


def create_weight_mat(n_pitches):
    weight = (np.arange(n_pitches) + 25) / float(n_pitches + 25)
    weight = weight[::-1]
    weight[-1] = 1
    weight_mat = np.diag(weight)
    return weight_mat


def state_to_freq(states, open_str_midi):
    f_detect = []
    for s in states:
        if s > 0 :  # voiced
            fret = STATE_INDEX[s]
            midi = open_str_midi + fret
            freq = librosa.midi_to_hz(midi)[0]
            f_detect.append(freq)
        else:  # unvoiced
            f_detect.append(-1)

    return f_detect


def joint_viterbi(Ps, weight_mat):
    state_detects = []
    T = mod_bump_transition(Ps.shape[1], width=10, alpha=5e-2)
    for string_num in range(6):
        P = np.dot(weight_mat, Ps[string_num])
        P = librosa.util.normalize(P.astype(np.float64), axis=0, norm=1)
        seq, values, pointers = viterbi(P.T, T)
        state_detect = seq
        state_detect[seq == Ps.shape[1] - 1] *= -1 # Ps.shape[1] is n_pitches
        state_detects.append(state_detect)
    return state_detects


def acf_from_dirpath(mono_dir_path):
    acf0 = []
    fret_energys = []
    acf0_size = None
    for string_number in range(6):
        mono_path = os.path.join(mono_dir_path, '{}.wav'.format(string_number))

        y, sr = librosa.load(mono_path)
        print(len(y))
        acf, min_lag = constrained_acf(y, sr, STR_MIDI_DICT[string_number])
        if acf0_size is None:
            acf0_size = acf.shape[1]
        cliped_acf = np.clip(acf[min_lag:], 0, None) / acf[0]
        fret, fret_energy = compute_fret_energy(cliped_acf, min_lag, sr,
                                                STR_MIDI_DICT[string_number])
        acf0.append(acf[0, :acf0_size])
        fret_energys.append(fret_energy[:, :acf0_size])

    acf0 = np.asarray(acf0)
    fret_energys = np.asarray(fret_energys)
    return acf0, fret_energys


def visualize_joint_states(state_detects, mono_dir_path):
    for string_number in range(6):
        mono_path = os.path.join(mono_dir_path, '{}.wav'.format(string_number))
        y, sr = librosa.load(mono_path)

        states = state_detects[string_number]
        open_str_midi = STR_MIDI_DICT[string_number]
        f_detect = state_to_freq(states, open_str_midi)


        plt.figure()

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, x_axis='time')

        times = librosa.frames_to_time(np.arange(len(f_detect)))
        mir_eval.display.pitch(times, f_detect, color='w', linewidth=3)
        plt.show()


def compute_novelty_spec_flux(y, fs, hop_size=256):
    """ Compute novelty function using spectural flux.
    Parameters
    ----------
    y : ndarray
    (T,) mono time domian signal
    fs : int
    sample rate of y
    win_size : int
    hop_size : int

    Returns
    -------
    novelty : ndarray
    times : ndarray
    fs_novelty : float
    """
    S = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=256,
                                       hop_length=hop_size)
    S_dB = librosa.power_to_db(S, ref=np.max)
    dS_dt = np.diff(S, axis=1)
    # HW Rectify dS_dt
    for x in np.nditer(dS_dt, op_flags=['readwrite']):
        x[...] = max(x, 0)
    #

    weight_vec = np.append(np.zeros(128), np.ones(128))
    weight_vec += 1

    weighted_sf = np.dot(np.diag(weight_vec), dS_dt)

    # output gathering
    novelty = np.mean(weighted_sf, axis=0)
    fs_novelty = float(fs) / hop_size
    times = np.array([float(i) / fs_novelty for i in range(len(novelty))])

    return novelty, times, fs_novelty


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def onsets_from_novelty(novelty, times, fs, w_c=3, medfilt_len=11,
                        offset=0.001):
    # smoothing using butterworth and normalize
    nyq = fs / 2.0
    B, A = signal.butter(1, w_c / nyq, 'low')
    novelty_smoothed = signal.filtfilt(B, A, novelty)
    novelty_smoothed = novelty_smoothed / np.max(np.abs(novelty_smoothed))

    # adaptive thresholding
    thresh = medfilt(novelty_smoothed, medfilt_len) + offset

    # onset detection
    peak_idx = librosa.util.peak_pick(novelty_smoothed, 3, 3, 3, 5, 0, 10)

    pruned_peak_idx = []
    for p_idx in peak_idx:
        if novelty_smoothed[p_idx] > thresh[p_idx]:  # it made it
            pruned_peak_idx.append(p_idx)
    return pruned_peak_idx, novelty_smoothed, thresh


def run(mono_dir_path):
    acf0s, fret_energys = acf_from_dirpath(mono_dir_path)
    Ps = energy_to_prob(fret_energys=fret_energys, acf0s=acf0s)
    weight_mat = create_weight_mat(Ps.shape[1])
    state_detects = joint_viterbi(Ps, weight_mat)
    visualize_joint_states(state_detects=state_detects,
                           mono_dir_path=mono_dir_path)
