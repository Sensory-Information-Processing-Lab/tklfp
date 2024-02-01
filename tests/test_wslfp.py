import numpy as np
import pytest
from wslfp import WSLFP
import matplotlib.pyplot as plt

@pytest.mark.parametrize(
    "ampa_times, gaba_times, t_evals, success, lfp_increasing",
    [
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], False, False),  # Valid times, should pass
        ([5, 6, 7, 8, 9], [5, 6, 7, 8, 9], [10, 11, 12], False, False),  # t_evals out of range, should fail
        ([1, 2, 4], [1, 2, 10], [9], False, False),  # length of time and current array must equal
        ([3, 2, 1], [3, 2, 1], [2], False, False),  # Non-increasing order, should fail
        ([], [], [], False, False),  # Empty arrays, should fail
        ([1, 5, 9, 12, 17], [1, 3, 5, 9, 12], [8, 11], True, True),
        ([1, 5, 9, 12, 17], [1, 3, 5, 9, 12], [], False, False),
        ([2, 4, 6, 23, 25], [7, 10, 15, 21, 25], [12, 20, 24], True, True),
        ([2, 3, 4, 5, 6], [50, 60, 70, 80, 90], [70, 100], False, False)
    ]
)
def test_wslfp_time_ranges(ampa_times, gaba_times, t_evals, success, lfp_increasing):
    # Dummy data for currents and coordinates
    ampa_currents = np.array([1, 2, 3, 4, 5])
    gaba_currents = np.array([-1, -2, -3, -4, -5])
    xs = ys = zs = np.array([1, 2, 3])
    elec_coords = np.array([[1, 1, 1], [2, 2, 2]])

    wslfp = WSLFP(xs, ys, zs, elec_coords, ampa_times, ampa_currents, gaba_times, gaba_currents)
    if success:
        try:
            lfp_values = wslfp.calculate_lfp(t_evals)
            print("Calculated LFP values:", lfp_values)
            assert lfp_increasing == np.all(np.diff(lfp_values) >= 0), "LFP increasing order does not match expectation"
        except ValueError:
            assert False, "Test expected to succeed but failed"
    else:
        with pytest.raises(ValueError):
            wslfp.calculate_lfp(t_evals)


