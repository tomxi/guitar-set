"""annotator
"""
import csv
import os
import shutil
import tempfile
from os import listdir
from os.path import join, basename

import numba
import jams
import librosa
from scipy import signal
import numpy as np
import pretty_midi
import sox

from guitar_set.util import ext_f_condition

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


import scipy.signal


def make_bump_transition(N, window='hann', width=5, alpha=0.05):
    T = np.eye(N, dtype=float)
    w = librosa.filters.get_window(window, width, fftbins=False)[np.newaxis]
    T = scipy.signal.convolve(T, w, mode='same', )

    T = (1 - alpha) * T + alpha
    return T / T.sum(axis=1, keepdims=True)


def acf_to_voicing_prob(A, quantile=75, epsilon=5e-3):
    Q = np.percentile(A[0], quantile)
    V = np.minimum(A[0], Q) / (Q + epsilon)
    return V


def constrained_pitch_prob(y, sr, open_string_midi):
    midi_min = open_string_midi - 10
    midi_max = open_string_midi + 30
    fmin = librosa.midi_to_hz(midi_min)
    fmax = librosa.midi_to_hz(midi_max)

    max_lag = librosa.time_to_samples(1. / fmin, sr=sr)[0]
    min_lag = librosa.time_to_samples(1. / fmax, sr=sr)[0]

    yf = librosa.util.frame(y, frame_length=max_lag, hop_length=256)
    A = librosa.autocorrelate(yf, axis=0)

    V = acf_to_voicing_prob(A)

    Corig = np.clip(A[min_lag:max_lag], 0, None) / A[0]
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
                        offset=0.0001):
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
            continue
        note_region = \
            f_seg[s_frames[idx] - start_frame: e_frames[idx] - start_frame]
        pitch = librosa.hz_to_midi(np.mean(note_region))[0]
        print((s_times[idx], len(note_region) / float(sr), pitch, None))
        big_ann.append(time=s_times[idx],
                       duration= len(note_region) / float(sr) * 256,
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
        if segment_pitch_prob.shape[1] == 0:
            continue
        seq, values, pointers = viterbi(segment_pitch_prob.T, T)
        f_detect_seg = float(sr) / (seq + min_lag)
        f_detect_seg[seq == n_pitches - 1] *= -1
        f_detect[start_frame:end_frame] = f_detect_seg

        # create note for each segment
        big_ann = f0seg_to_note(f_detect_seg, start_frame, sr, big_ann)
    return f_detect, big_ann


def mono_anal_homebrew(stem_path, open_string_midi):
    """save jams with same stem_path"""
    y, sr = librosa.load(stem_path)

    sf, times_on, fs_novelty_on = compute_novelty_spec_flux(y, sr)
    onset_frames, novelty_smoothed, thresh = onsets_from_novelty(sf, times_on,
                                                                 fs_novelty_on)
    f_detect, big_ann = detect_f0(y, sr, open_string_midi,
                                  onset_frames=onset_frames)

    jam = jams.JAMS()
    jam.file_metadata.duration = big_ann.duration
    jam.annotations.append(big_ann)
    jam_path = stem_path.split('.')[0] + '.jams'
    jam.save(jam_path)
    print('jams file generated')

    return 0


def mono_anal(stem_path, open_string_midi):
    """save jams with same stem_path"""
    done = False
    cmd = 'python guitar_set/scripts/mono_anal_script.py {} {}'.format(
        stem_path,
                                                    open_string_midi)
    while not done:
        err = os.system(cmd)
        if err:
            print('vamp.collect errored, trying again...')
        else: # successful, no seg fault
            done = True

    return 0

def transcribe(dirpath):
    """run
    mono_pyin on all wavs in a folder `dirpath`.
    """
    # first load from dirpath
    files = [join(dirpath, f) for f in listdir(dirpath) if
             ext_f_condition(f, dirpath, 'wav')]
    files.sort()
    # make dummy output
    output = []

    str_midi_dict = {
        '0': 40,
        '1': 45,
        '2': 50,
        '3': 55,
        '4': 59,
        '5': 64
    }
    for f in files:
        string_id = basename(f).split('.')[0][1]  # the second char of f
        str_midi = str_midi_dict[string_id]
        mono_anal(f, str_midi)


def transcribe_hex(hex_path):
    temp_path = tempfile.mkdtemp()

    output_mapping = {'00': {1: [1]},
                      '01': {1: [2]},
                      '02': {1: [3]},
                      '03': {1: [4]},
                      '04': {1: [5]},
                      '05': {1: [6]}
                      }

    for mix_type, remix_dict in output_mapping.items():
        tfm = sox.Transformer()
        tfm.remix(remix_dictionary=remix_dict)
        output_path = os.path.join(temp_path, '{}.wav'.format(mix_type))
        tfm.build(hex_path, output_path)

    transcribe(temp_path)

    jam = jamses_to_jams(temp_path)
    jam.file_metadata.title = os.path.basename(hex_path)
    shutil.rmtree(temp_path)
    print(jam.file_metadata.title)
    return jam


def jamses_to_midi(jams_files_dir, q=1):
    jams_files = [join(jams_files_dir, f) for f in listdir(jams_files_dir) if
                  ext_f_condition(f, jams_files_dir, 'jams')]
    midi = pretty_midi.PrettyMIDI()
    for j in jams_files:
        jam = jams.load(j)
        ch = pretty_midi.Instrument(program=25)
        ann = jam.search(namespace='note_midi')[0]
        for note in ann:
            pitch = int(round(note.value))
            bend_amount = int(round((note.value - pitch) * 4096))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch, start=st,
                end=st+dur
            )
            pb = pretty_midi.PitchBend(pitch=bend_amount*q, time=st)
            ch.notes.append(n)
            ch.pitch_bends.append(pb)
        if len(ch.notes) != 0:
            midi.instruments.append(ch)
    return midi


def jamses_to_jams(jams_files_dir):
    jams_files = [join(jams_files_dir, f) for f in listdir(jams_files_dir) if
                  ext_f_condition(f, jams_files_dir, 'jams')]
    jams_files.sort()
    jam = jams.JAMS()
    for j in jams_files:
        mono_jams = jams.load(j)
        jam.annotations.append(mono_jams.search(namespace='note_midi')[0])
        jam.file_metadata.duration = mono_jams.file_metadata.duration
    return jam


def csvs_to_jams(csv_files_dir):
    jam = jams.JAMS()
    wav_files = [join(csv_files_dir, f) for f in listdir(csv_files_dir) if
                 ext_f_condition(f, csv_files_dir, 'wav')]
    cvs_files = [join(csv_files_dir, f) for f in listdir(csv_files_dir) if
                 ext_f_condition(f, csv_files_dir, 'csv')]
    jam.file_metadata.duration = sox.file_info.duration(wav_files[0])
    for c in cvs_files:
        with open(c, 'r') as stream:
            ann_cvs = csv.reader(stream)
            ann_jam = jams.Annotation(namespace='note_midi', time=0,
                                      duration=jam.file_metadata.duration)
            for row in ann_cvs:
                try:
                    st = float(row[0])
                    midi_note = librosa.hz_to_midi(float(row[1]))[0]
                    dur = float(row[2])
                    ann_jam.append(time=st, value=midi_note, duration=dur,
                                   confidence=None)
                except:
                    print('empty row')
                    pass
        jam.annotations.append(ann_jam)

    return jam



