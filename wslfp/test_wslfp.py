import numpy as np
import pytest
import sys
import sys
sys.path.append('/Users/alissa/Documents/WSLFP/wslfp')
from wslfp import WSLFP

# @pytest.mark.parametrize (
#     "t_ampa, t_gaba, t_eval, success",
#     ([10], [4], [10], True),
#     ([5], [5], [10], False),
#     ([10], [4.1], [10], False),
#     ([9.9], [4], [10], False),
#     ([9.9, 10.1], [4], [10], True),
#     ([11], [5], [11], True),
#     ([2], [8], [6], False),
#     ([7, 8, 9], [3, 4, 5], [10, 20, 30], True),  # Increasing
#     ([10, 10, 10], [4, 4, 4], [10, 20, 30], True),  # Increasing
#     ([10, 10, 10], [4, 4, 4], [30, 20, 10], False),  # Non-increasing
#     ([10, 10, 10], [4, 4, 4], [10, 10, 10], True),  # Increasing
#     ([10, 10, 10], [4, 4, 4], [5, 10, 15], False),  # Non-increasing
#)

# Create a fixture for a sample WSLFP object
# @pytest.fixture
# def sample_wslfp():
#     xs = np.array([1, 2, 3])
#     ys = np.array([4, 5, 6])
#     zs = np.array([7, 8, 9])
#     elec_coords = np.array([[1, 1, 1], [2, 2, 2]])
#     wslfp_obj = WSLFP(xs, ys, zs, elec_coords)
#     return wslfp_obj



# Test the compute_gaba_curr method
# def test_compute_gaba_curr(sample_wslfp):
#     t_gaba_ms = np.array([1.0, 2.0, 3.0])
#     tau_gaba = 0.5
#     t_eval_ms = 2.0
#     gaba = np.array([-1,-1,-1])
#     gaba_curr = sample_wslfp.compute_gaba_curr(gaba, t_gaba_ms, tau_gaba, t_eval_ms)
#     print(gaba_curr.type())
#     assert isinstance(gaba_curr, float)  # Check if the result is a float

# Test the compute_ampa_curr method
# def test_compute_ampa_curr(sample_wslfp):
#     t_ampa_ms = np.array([1.0, 2.0, 3.0])
#     tau_ampa = 0.5
#     t_eval_ms = 2.0
#     ampa = [1,1,1]
#     ampa_curr = sample_wslfp.compute_ampa_curr(ampa, t_ampa_ms, tau_ampa, t_eval_ms)
#     assert isinstance(ampa_curr, float)  # Check if the result is a float

# Test the lfp_ws_proxy method
# def test_lfp_ws_proxy(sample_wslfp):
#     t_ampa_ms = np.array([1.0, 2.0, 3.0])
#     t_gaba_ms = np.array([0.5, 1.5, 2.5])
#     tau_ampa = 0.5
#     tau_gaba = 0.2
#     t_eval = 2.0
#     lfp_ws = sample_wslfp.lfp_ws_proxy(t_ampa_ms, t_gaba_ms, tau_ampa, tau_gaba, t_eval)
#     assert isinstance(lfp_ws, float)  # Check if the result is a float

# test_data = [
#     (np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 2.5]), 0.5, 0.2, 2.0, True),
#     (np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 2.5]), 0.5, 0.2, 2.0, False),  # Add more test cases as needed
# ]

def test_compute_lfp():
    xs = np.array([1, 2, 3])
    ys = np.array([4, 5, 6])
    zs = np.array([7, 8, 9])
    elec_coords = np.array([[1, 1, 1], [2, 2, 2]])
    sample = WSLFP(xs, ys, zs, elec_coords)
    ampa = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    t_ampa_ms = np.array([1,2])
    gaba = np.array([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5]])
    t_gaba_ms = np.array([1, 2])
    t_eval_ms = np.array([2.5])
    compute_lfp = sample._lfp_ws_proxy(t_ampa_ms, t_gaba_ms, sample.tau_ampa_ms, sample.tau_gaba_ms, t_eval_ms[0])
    ampa_shifted = ampa[1]  # Assuming t_eval_ms - tau_ampa is close to the second time point
    gaba_shifted = gaba[1]
    expected_lfp = np.maximum(np.sum(ampa_shifted) - np.sum(gaba_shifted), 0.0)
    assert np.isclose(compute_lfp, expected_lfp), f"Expected {expected_lfp}, but got {compute_lfp}"

# Use the @pytest.mark.parametrize decorator to run the test with each set of data
# @pytest.mark.parametrize("t_ampa_ms, t_gaba_ms, tau_ampa, tau_gaba, t_eval, expected_result", test_data)
# def test_lfp_ws_proxy_negative_sum(sample_wslfp, t_ampa_ms, t_gaba_ms, tau_ampa, tau_gaba, t_eval, expected_result):
#     # Check if the sum of ampa and gaba is negative
#     ampa_sum = np.sum(sample_wslfp.compute_ampa_curr(t_ampa_ms, tau_ampa, t_eval))
#     gaba_sum = np.sum(sample_wslfp.compute_gaba_curr(t_gaba_ms, tau_gaba, t_eval))
#     total_sum = ampa_sum - gaba_sum
    
#     # Check if the result matches the expected outcome
#     assert (total_sum < 0) == expected_result, f"Expected result: {expected_result}, Actual result: {total_sum < 0}"

# Add more test cases as needed

# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()



# @pytest.mark.parametrize (
#     "t_ampa, t_gaba, t_eval, success",
#     ([10], [4], [10], True),
#     ([5], [5], [10], False),
#     ([10], [4.1], [10], False),
#     ([9.9], [4], [10], False),
#     ([9.9, 10.1], [4], [10], True),
#     ([11], [5], [11], True),
#     ([2], [8], [6], False),
#     ([7, 8, 9], [3, 4, 5], [10, 20, 30], True),  # Increasing
#     ([10, 10, 10], [4, 4, 4], [10, 20, 30], True),  # Increasing
#     ([10, 10, 10], [4, 4, 4], [30, 20, 10], False),  # Non-increasing
#     ([10, 10, 10], [4, 4, 4], [10, 10, 10], True),  # Increasing
#     ([10, 10, 10], [4, 4, 4], [5, 10, 15], False),  # Non-increasing
#)
