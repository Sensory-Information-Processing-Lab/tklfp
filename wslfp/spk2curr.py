import numpy as np
from scipy import sparse
from scipy.optimize import fsolve


def biexp_kernel(t, tau1, tau2, normalize=False):
    """Biexponential kernel"""
    if normalize:
        tau_rise = tau1 * tau2 / (tau1 - tau2)
        B = 1 / (
            (tau2 / tau1) ** (tau_rise / tau1) - (tau2 / tau1) ** (tau_rise / tau2)
        )
    else:
        B = 1
    return B * (np.exp(-t / tau1) - np.exp(-t / tau2)) * (t >= 0)


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
    assert t_end > tau1

    I_syn = np.zeros((T, n_targets))

    window_sizes = np.zeros(T, dtype=int)
    for t in range(T):
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
