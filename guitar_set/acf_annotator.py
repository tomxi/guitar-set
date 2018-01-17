import librosa
import numpy as np
import jams
import guitar_set.viterbi as vtb

STR_MIDI_DICT = {
    0: 40,
    1: 45,
    2: 50,
    3: 55,
    4: 59,
    5: 64
}

STATE_INDEX = np.arange(-2,24,0.25)


def eng_to_voicing_prob(A0, quantile=65, epsilon=5e-7, Q=None):
    if Q is None:
        Q = np.percentile(A0, quantile)
    else:
        quantile = None
    print("voicing threshold: at {}%, or amplitude: {}".format(quantile, Q))
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


def energy_to_prob(fret_energys, acf0s, voicing_quantile, voicing_q):
    voiced_probs = eng_to_voicing_prob(A0=acf0s, quantile=voicing_quantile,
                                       Q=voicing_q)
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
    weight = (np.arange(n_pitches) + 5) / float(n_pitches + 5)
    weight = weight[::-1]
    weight[-1] = 1
    weight_mat = np.diag(weight)
    return weight_mat

def joint_viterbi(Ps, weight_mat):
    state_detects = []
    T = vtb.mod_bump_transition(Ps.shape[1], width=15, alpha=5e-2)
    for string_num in range(6):
        P = np.dot(weight_mat, Ps[string_num])
        P = librosa.util.normalize(P.astype(np.float64), axis=0, norm=1)
        seq, values, pointers = vtb.viterbi(P.T, T)
        state_detect = seq
        state_detect[seq == Ps.shape[1] - 1] *= -1 # Ps.shape[1] is n_pitches
        state_detects.append(state_detect)
    return state_detects

def acf_from_hexrec(hexrec):
    acf0 = []
    fret_energys = []
    acf0_size = None
    for string_number in range(6):
        y = hexrec.ys[string_number]
        sr = hexrec.sr

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


def state_detects_to_anno_list(state_detects, dur):
    annotations = []
    for string_number in range(6):
        anno = jams.Annotation(namespace='pitch_contour', duration=dur)
        anno.annotation_metadata.curator = str(string_number)
        states = state_detects[string_number]
        midi = list(map(lambda s: STATE_INDEX[s] + STR_MIDI_DICT[
            string_number],
                    states))
        freqs = librosa.midi_to_hz(midi)
        times = librosa.frames_to_time(
            np.arange(np.asarray(state_detects).shape[1]))
        voiced = np.asarray(states) > 0

        # print(type(freqs))
        # print(freqs)
        for (f, t, v) in zip(freqs, times, voiced):
            anno.append(time=t, duration=0.0,
                        value={'voiced': v, 'index': 0,
                               'frequency': f if v else 0}
                        )
            # print(f, t, v)
        annotations.append(anno)
    return annotations

