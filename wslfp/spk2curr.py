import numpy as np
import numba
from scipy import sparse


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
    print(np.sum(np.isnan((np.exp(-t / tau1) - np.exp(-t / tau2)) * (t >= 0))))
    # print(np.shape(np.exp(-t / tau1) - np.exp(-t / tau2)))
    # print(np.shape(t >= 0))
    # print(np.sum(np.isnan(t >= 0)))
    return B * (np.exp(-t / tau1) - np.exp(-t / tau2)) * (t >= 0)


@numba.njit
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
def spikes_to_biexp_currents(
    t_eval_ms, t_spk_ms, i_spk, J, tau1, tau2, syn_delay_ms=1, normalize=False
):
    n_each_spk = (J[i_spk, :] != 0).sum(axis=1)
    t_postsyn_spk_ms = np.repeat(t_spk_ms + syn_delay_ms, n_each_spk)
    i_postsyn_spk = np.repeat(i_spk, n_each_spk)

    if isinstance(J, np.ndarray):
        i_postsyn_spk = np.repeat(i_spk, n_each_spk)
        assert len(t_postsyn_spk_ms) == np.sum(n_each_spk)

        spike_index, j_postsyn_spk = np.nonzero(J[i_spk, :])
        J_postsyn_spk = J[i_postsyn_spk, j_postsyn_spk]

    elif sparse.issparse(J):
        spike_index, j_postsyn_spk, J_postsyn_spk = sparse.find(J[i_spk, :])

    assert np.all(np.diff(spike_index) >= 0), "assuming nonzero goes by row"
    assert i_postsyn_spk.shape == j_postsyn_spk.shape == J_postsyn_spk.shape

    return _postsyn_spikes_to_biexp_currents(
        t_eval_ms,
        t_postsyn_spk_ms,
        j_postsyn_spk,
        J_postsyn_spk,
        J.shape[1],
        tau1,
        tau2,
        normalize,
    )


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
