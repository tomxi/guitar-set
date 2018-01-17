import numpy as np
from scipy import signal
import numba
import librosa
import librosa.display
import jams


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


def acf_to_voicing_prob(A, quantile=75, epsilon=5e-3):
    Q = np.percentile(A[0], quantile)
    V = np.minimum(A[0], Q) / (Q + epsilon)
    return V


def constrained_pitch_prob(y, sr, str_midi):
    midi_min = str_midi - 15
    midi_max = str_midi + 25
    fmin = librosa.midi_to_hz(midi_min)
    fmax = librosa.midi_to_hz(midi_max)

    max_lag = librosa.time_to_samples(1. / fmin, sr=sr)[0]
    min_lag = librosa.time_to_samples(1. / fmax, sr=sr)[0]

    yf = librosa.util.frame(y, frame_length=max_lag, hop_length=256)
    A = librosa.autocorrelate(yf, axis=0)

    V = acf_to_voicing_prob(A)

    Corig = np.clip(A[min_lag:], 0, None) / A[0]
    C = np.vstack((Corig, 1 - V))

    sigsq = np.mean(np.abs(C))
    P = np.exp(0.5 * C / sigsq)

    P = librosa.util.normalize(P.astype(np.float64), axis=0, norm=1)

    return P, min_lag


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


def f0seg_to_note(f_seg, start_frame, sr, big_ann):
    """Given a f0 segmented by onset, output the note associated"""

    s_frames = []
    e_frames = []
    for idx in range(len(f_seg)):
        f_frame = f_seg[idx]
        if f_seg[idx] <= 0:  # unvoiced
            if idx == 0:
                continue
            elif f_seg[idx - 1] <= 0:
                continue
            else:  # just switched from voiced to unvoiced
                note_end_frame = start_frame + idx
                e_frames.append(note_end_frame)
        else:  # voiced
            if idx == 0:
                note_start_frame = start_frame + idx
                s_frames.append(note_start_frame)
            elif f_seg[idx - 1] <= 0:  # just switched from unvoiced to voiced
                note_start_frame = start_frame + idx
                s_frames.append(note_start_frame)
            else:
                continue

    if len(e_frames) != len(s_frames):
        e_frames.append(start_frame + len(f_seg) - 1)

    s_times = librosa.frames_to_time(frames=s_frames, sr=sr, hop_length=256)
    e_times = librosa.frames_to_time(frames=e_frames, sr=sr, hop_length=256)

    for idx in range(len(s_times)):
        if len(f_seg) == 0:
            print('zero length fseg')
            continue
        pitch_hz = np.mean(f_seg)
        print(pitch_hz)
        pitch = librosa.hz_to_midi(pitch_hz)[0]
        print((s_times[idx], e_times[idx] - s_times[idx], pitch, len(f_seg)))
        big_ann.append(time=s_times[idx],
                       duration=e_times[idx] - s_times[idx],
                       value=pitch,
                       confidence=None)

    return big_ann


def detect_f0(y, sr, open_string_midi, onset_frames=[]):
    pitch_prob, min_lag = constrained_pitch_prob(y, sr, open_string_midi)

    n_pitches = pitch_prob.shape[0]
    T = make_bump_transition(n_pitches, width=15, alpha=1e-3)

    onset_frames.insert(0, 0)

    f_detect = np.zeros((pitch_prob.shape[1],))
    big_ann = jams.Annotation(namespace='note_midi')
    big_ann.duration = len(y) / float(sr)
    for frame in range(len(onset_frames)):
        print(frame, len(onset_frames))
        start_frame = onset_frames[frame]
        try:
            end_frame = onset_frames[frame + 1]
        except IndexError:
            end_frame = None
        print('start:{} end:{}'.format(start_frame, end_frame))
        segment_pitch_prob = pitch_prob[:, start_frame:end_frame]
        print(segment_pitch_prob.shape)
        if segment_pitch_prob.shape[1] <= 2:
            print('zero lenth pitch prob matrix')
            continue
        seq, values, pointers = viterbi(segment_pitch_prob.T, T)
        f_detect_seg = float(sr) / (seq + min_lag)
        f_detect_seg[seq == n_pitches - 1] *= -1
        f_detect[start_frame:end_frame] = f_detect_seg

        # create note for each segment
        big_ann = f0seg_to_note(f_detect_seg, start_frame, sr, big_ann)
    return f_detect, big_ann