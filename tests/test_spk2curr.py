import numpy as np
import pytest
from scipy import sparse

from wslfp import spikes_to_biexp_currents


@pytest.mark.parametrize("seed", [18300406, 18051223])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("sparse_J", [False, True])
@pytest.mark.parametrize("tau1, tau2", [(10, 5), (5, 2)])
@pytest.mark.parametrize("delay", [0, 2, 5])
def test_spikes_to_biexp_currents(delay, tau1, tau2, sparse_J, normalize, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(100)

    # shuffle the rows of J
    row1, row2, row3 = [
        [0, 1, 2, 3],
        [0, 1, -1, 0],
        [0, 0, 0, 0],
    ]
    i1, i2, i3 = rng.choice(3, 3, replace=False)
    J = np.zeros((3, 4))
    J[i1, :] = row1
    J[i2, :] = row2
    J[i3, :] = row3

    if sparse_J:
        J = sparse.csr_array(J)

    def s2c(t_spk, i_spk, threshold=0.01):
        return spikes_to_biexp_currents(
            t,
            t_spk,
            i_spk,
            J,
            tau1,
            tau2,
            syn_delay_ms=delay,
            normalize=normalize,
            threshold=threshold,
        )

    # more spikes produce more current
    # same source
    result = s2c([0.9, 49.8, 49.9, 79.7, 79.8, 79.9], np.full(6, i1))
    assert np.all(result[1 + delay, 1:] < result[50 + delay, 1:])
    assert np.all(result[50 + delay, 1:] < result[80 + delay, 1:])
    # different sources
    result = s2c([0.9, 49.8, 49.9], [i1, i1, i2])
    assert np.all(result[1 + delay, 1:] < result[50 + delay, 1:])

    # higher weights lead to more current
    result = s2c([0], [i1])
    assert np.all(np.diff(result[tau2 + delay]) > 0)

    # don't miss spikes
    result = s2c([-(delay + 0.1)], [i1])
    assert np.all(result[0, 1:] > 0)

    # 0 when no incoming spikes with weights
    result = s2c([0, 4, 6, 30], [i3] * 4)
    assert np.all(result == 0)

    # lower threshold gets more spikes (higher LFP for only positive weights)
    t_spk = tau1 * np.linspace(-10, 10, 100)
    res_low_thresh = s2c(t_spk, np.full(100, i1), threshold=1e-6)
    res_high_thresh = s2c(t_spk, np.full(100, i1), threshold=1e-1)
    assert np.all(res_low_thresh[:, 1:] > res_high_thresh[:, 1:])

    # back to 0 after long time
    result = s2c([-(delay + 10 * tau1)], [i1])
    assert np.all(result == 0)

    # delay works
    result = s2c([0], [i1])
    assert np.all(result[delay] == 0)
    assert np.all(result[delay + tau2, 1:] > 0)

    # negative weights work
    result = s2c([0], [i2])
    assert result[tau2 + delay, 1] > 0
    assert result[tau2 + delay, 2] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
