import numpy as np
import pytest

import wslfp


@pytest.mark.parametrize(
    "seed",
    [17320222, 17991214, 18220427, 18850723, 18090212, 18650415],
)
@pytest.mark.parametrize("n_elec", [1, 20])
@pytest.mark.parametrize(
    "t_ampa, t_gaba, t_eval, success",
    [
        # Valid times, should pass
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], False),
        # t_evals out of range, should fail
        ([5, 6, 7, 8, 9], [5, 6, 7, 8, 9], [10, 11, 12], False),
        # length of time and current array must equal
        ([1, 2, 4], [1, 2, 10], [9], True),
        # missing I_gaba at t=9
        ([1, 2, 10], [1, 2, 4], [9], False),
        ([3, 2, 1], [3, 2, 1], [2], False),  # Non-increasing order, should fail
        ([], [], [], False),  # Empty arrays, should fail
        ([1, 5, 9, 12, 17], [1, 3, 5, 9, 12], [8, 11], True),
        ([1, 5, 9, 12, 17], [1, 3, 5, 9, 12], [], False),
        ([2, 4, 6, 23, 25], [7, 10, 15, 21, 25], [12, 20, 24], True),
        ([2, 3, 4, 5, 6], [50, 60, 70, 80, 90], [70, 100], False),
    ],
)
def test_calculate(t_ampa, t_gaba, t_eval, success, n_elec, seed):
    rng = np.random.default_rng(seed)
    # Dummy data for currents and coordinates
    n_src = 3
    n_elec = 2
    I_ampa = np.arange(len(t_ampa))
    I_ampa = I_ampa[:, np.newaxis] + np.arange(n_src)
    I_gaba = -np.arange(len(t_gaba)) - 2
    I_gaba = I_gaba[:, np.newaxis] - np.arange(n_src)

    # xs = ys = zs = np.array([1, 2, 3])
    # source_coords = np.column_stack((xs, ys, zs))
    source_coords = rng.uniform(-10, 10, (n_src, 3))
    elec_coords = rng.uniform(-500, 500, (n_elec, 3))

    calc = wslfp.from_xyz_coords(elec_coords, source_coords, strict_boundaries=True)
    if success:
        lfp_values = calc.calculate(t_eval, t_ampa, I_ampa, t_gaba, I_gaba)
        assert lfp_values.shape == (len(t_eval), n_elec)
        # increasing could be negative or positive
        # can't use np.abs since normalization can put values on either side of 0
        assert np.all(
            np.logical_or(
                np.all(np.diff(lfp_values, axis=0) > 0, axis=0),
                np.all(np.diff(lfp_values, axis=0) < 0, axis=0),
            )
        ), "LFP increasing order does not match expectation"
    else:
        with pytest.raises(ValueError):
            calc.calculate(t_eval, t_ampa, I_ampa, t_gaba, I_gaba)


if __name__ == "__main__":
    pytest.main([__file__, "--lf", "-x", "-s"])
