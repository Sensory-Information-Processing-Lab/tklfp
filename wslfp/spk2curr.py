import numpy as np
import numba
from scipy import sparse
from scipy.optimize import fsolve


@numba.njit
def biexp_kernel(t, tau1, tau2, normalize=False):
    """Biexponential kernel"""
    if normalize:
        tau_rise = tau1 * tau2 / (tau1 - tau2)
        B = 1 / (
            (tau2 / tau1) ** (tau_rise / tau1) - (tau2 / tau1) ** (tau_rise / tau2)
        )
    else:
        B = 1
    # print(np.sum(np.isnan((np.exp(-t / tau1) - np.exp(-t / tau2)) * (t >= 0))))
    # print(np.shape(np.exp(-t / tau1) - np.exp(-t / tau2)))
    # print(np.shape(t >= 0))
    # print(np.sum(np.isnan(t >= 0)))
    return B * (np.exp(-t / tau1) - np.exp(-t / tau2)) * (t >= 0)


@numba.njit(parallel=True)
def _postsyn_spikes_to_biexp_currents(
    t_eval_ms, t_spk_ms, j_spk, J_spk, n_curr_sources, tau1, tau2, normalize
):
    n_spk = len(t_spk_ms)
    T = len(t_eval_ms)

    # # uses a crazy amount of memory since there are so many postsynaptic spikes
    # t_spk_conv = t_eval_ms[..., np.newaxis] - t_spk_ms
    # assert t_spk_conv.shape == (T, n_spk)

    # I_syn_per_spike = kernel(t_spk_conv, **kernel_kwargs) * J_spk
    # assert I_syn_per_spike.shape == (T, n_spk)

    # # need to sum across spikes that go to same current source
    # I_syn_per_src = np.zeros((T, n_curr_sources))
    # for j_src in range(n_curr_sources):
    #     I_syn_per_src[:, j_src] = np.sum(I_syn_per_spike[:, j_spk == j_src], axis=1)
    I_syn_per_src = np.zeros((T, n_curr_sources))
    for i_t, t in enumerate(t_eval_ms):
        t_spk_conv = t - t_spk_ms
        assert t_spk_conv.shape == (n_spk,)
        I_syn_per_spike = biexp_kernel(t_spk_conv, tau1, tau2, normalize) * J_spk
        # print(np.isnan(I_syn_per_spike).sum())
        for j_src, I_syn in zip(j_spk, I_syn_per_spike):
            I_syn_per_src[i_t, j_src] += I_syn

    return I_syn_per_src


# def spikes_to_currents(
#     t_eval_ms, t_spk_ms, i_spk, J, kernel, syn_delay_ms, **kernel_kwargs
# ):
# TODO
# @numba.njit(parallel=True)
def spikes_to_biexp_currents(
    t_eval_ms,
    t_spk_ms,
    i_spk,
    J,
    tau1,
    tau2,
    syn_delay_ms=1,
    normalize=False,
    threshold=0.001,
):
    if not isinstance(t_spk_ms, np.ndarray):
        t_spk_ms = np.array(t_spk_ms)
    if not isinstance(i_spk, np.ndarray):
        i_spk = np.array(i_spk)
    n_targets = J.shape[1]
    T = len(t_eval_ms)
    n_spk = len(t_spk_ms)
    assert i_spk.shape == (n_spk,)

    t_spk_conv = t_eval_ms[..., np.newaxis] - (t_spk_ms + syn_delay_ms)
    assert t_spk_conv.shape == (T, n_spk)
    assert np.all(np.diff(t_spk_ms) >= 0), "assuming t_spk_ms is sorted"

    assert tau1 > tau2, "tau1 must be greater than tau2"

    # Define a function for the difference between the biexp_kernel and the threshold
    def biexp(t):
        return biexp_kernel(t, tau1, tau2, normalize=True) - threshold

    # Use fsolve to find the time when the biexp_kernel drops to the threshold
    t_end = fsolve(biexp, 6 * tau1)[0]
    # print(f"t_end = {t_end}")
    assert t_end > tau1

    I_syn = np.zeros((T, n_targets))

    window_sizes = np.zeros(T, dtype=int)
    for t in numba.prange(T):
        # for t in range(T):
        # each row goes backward in time as later and later spike times are subtracted
        # flip the array to sort in ascending order
        spk_left, spk_right = n_spk - np.searchsorted(t_spk_conv[t, ::-1], [t_end, 0])
        assert spk_left <= spk_right, (spk_left, spk_right)
        if spk_left - spk_right == 0:
            continue
        window_sizes[t] = spk_right - spk_left

        I_syn_t = biexp_kernel(t_spk_conv[t, spk_left:spk_right], tau1, tau2, normalize)

        J_t = J[i_spk[spk_left:spk_right], :]
        # numpy doesn't handle multiplication with sparse matrices
        if sparse.issparse(J_t):
            J_t = J_t.toarray()
        I_syn[t] = (I_syn_t[:, np.newaxis] * J_t).sum(axis=0)

    return I_syn
    # n_each_spk = (J[i_spk, :] != 0).sum(axis=1)
    # t_postsyn_spk_ms = np.repeat(t_spk_ms + syn_delay_ms, n_each_spk)
    # i_postsyn_spk = np.repeat(i_spk, n_each_spk)

    # if isinstance(J, np.ndarray):
    #     i_postsyn_spk = np.repeat(i_spk, n_each_spk)
    #     assert len(t_postsyn_spk_ms) == np.sum(n_each_spk)

    #     spike_index, j_postsyn_spk = np.nonzero(J[i_spk, :])
    #     J_postsyn_spk = J[i_postsyn_spk, j_postsyn_spk]

    # elif sparse.issparse(J):
    #     spike_index, j_postsyn_spk, J_postsyn_spk = sparse.find(J[i_spk, :])

    # assert np.all(np.diff(spike_index) >= 0), "assuming nonzero goes by row"
    # assert i_postsyn_spk.shape == j_postsyn_spk.shape == J_postsyn_spk.shape

    # return _postsyn_spikes_to_biexp_currents(
    #     t_eval_ms,
    #     t_postsyn_spk_ms,
    #     j_postsyn_spk,
    #     J_postsyn_spk,
    #     J.shape[1],
    #     tau1,
    #     tau2,
    #     normalize,
    # )


# def spikes_to_biexp_currents(
#     t_eval_ms, t_spk_ms, i_spk, J, tau1, tau2, syn_delay_ms=1, normalize=False
# ):
#     """J[i, j] = weight from source i to target j
#     tau1: fall time constant. 2 ms for AMPA, 5 ms for GABA
#     tau2: rise time constant. 0.4 ms for AMPA, 0.25 ms for GABA"""
#     return spikes_to_currents(
#         t_eval_ms,
#         t_spk_ms,
#         i_spk,
#         J,
#         biexp_kernel,
#         syn_delay_ms,
#         tau1=tau1,
#         tau2=tau2,
#         normalize=normalize,
#     )
