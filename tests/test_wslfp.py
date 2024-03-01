import numpy as np
import pytest

from wslfp import WSLFP


@pytest.mark.parametrize(
    "t_ampa, t_gaba, t_eval, success, lfp_increasing",
    [
        # Valid times, should pass
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], False, False),
        # t_evals out of range, should fail
        ([5, 6, 7, 8, 9], [5, 6, 7, 8, 9], [10, 11, 12], False, False),
        # length of time and current array must equal
        ([1, 2, 4], [1, 2, 10], [9], True, True),
        # missing I_gaba at t=9
        ([1, 2, 10], [1, 2, 4], [9], False, False),
        ([3, 2, 1], [3, 2, 1], [2], False, False),  # Non-increasing order, should fail
        ([], [], [], False, False),  # Empty arrays, should fail
        ([1, 5, 9, 12, 17], [1, 3, 5, 9, 12], [8, 11], True, True),
        ([1, 5, 9, 12, 17], [1, 3, 5, 9, 12], [], False, False),
        ([2, 4, 6, 23, 25], [7, 10, 15, 21, 25], [12, 20, 24], True, True),
        ([2, 3, 4, 5, 6], [50, 60, 70, 80, 90], [70, 100], False, False),
    ],
)
@pytest.mark.parametrize("n_elec", [1, 20])
def test_time_ranges(t_ampa, t_gaba, t_eval, success, lfp_increasing, n_elec):
    # Dummy data for currents and coordinates
    n_src = 3
    n_elec = 2
    I_ampa = np.arange(len(t_ampa))
    I_ampa = I_ampa[:, np.newaxis] + np.arange(n_src)
    I_gaba = -np.arange(len(t_gaba)) - 2
    I_gaba = I_gaba[:, np.newaxis] - np.arange(n_src)

    xs = ys = zs = np.array([1, 2, 3])
    source_coords = np.column_stack((xs, ys, zs))
    elec_coords = np.random.uniform(-500, 500, (n_elec, 3))

    wslfp = WSLFP.from_xyz_coords(elec_coords, source_coords, strict_boundaries=True)
    if success:
        lfp_values = wslfp.compute(t_ampa, I_ampa, t_gaba, I_gaba, t_eval)
        assert lfp_values.shape == (len(t_eval), n_elec)
        assert lfp_increasing == np.all(
            np.diff(lfp_values, axis=0) >= 0
        ), "LFP increasing order does not match expectation"
    else:
        with pytest.raises(ValueError):
            wslfp.compute(t_ampa, I_ampa, t_gaba, I_gaba, t_eval)


if __name__ == "__main__":
    pytest.main([__file__, "--lf", "-x", "-s"])
