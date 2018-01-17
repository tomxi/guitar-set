import librosa
import librosa.display
import numpy as np
import numba
import scipy.signal as signal

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