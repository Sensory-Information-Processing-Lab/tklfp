from cgi import test
import numpy as np
import lfp_amplitude_function

def test_compute_length():
    # check that length is 0 when points are the same
    cords = np.random.rand(5, 3) * 10
    length = lfp_amplitude_function.compute_length(cords, cords)
    for arr in range(len(length)):
        assert length[arr] == 0
    # check that flipping the neuron results in the same length
    length1 = lfp_amplitude_function.compute_length([[0,2,0]], [[0,0,0]])
    length2 = lfp_amplitude_function.compute_length([[0,0,0]], [[0,2,0]])
    for arr in range(len(length1)):
        assert length1[arr] == length2[arr]

def test_compute_midpoint():
    # check that midpoint is the same if the neuron is flipped
    midpoint1 = lfp_amplitude_function.compute_midpoint([[0,2,0]], [[0,0,0]])
    midpoint2 = lfp_amplitude_function.compute_midpoint([[0,0,0]], [[0,2,0]])
    for arr in range(len(midpoint1)):
        assert midpoint1[arr] == midpoint2[arr]
    # check that fucntion can handle negative values
    midpoint = lfp_amplitude_function.compute_midpoint([[0,-1,-3]], [[4,8,-10]])
    for arr in range(len(midpoint)):
        assert midpoint[arr] == [2, 3.5, -6.5]

def test_compute_distance():
    # check that distance is negative when electrode is below neuron
    distance = lfp_amplitude_function.compute_distance([[0,2,0]], [[0,0,0]])
    for arr in range(len(distance)):
        assert distance[arr] <= 0
    # check that distance is positive when electrode is above neuron
    distance1 = lfp_amplitude_function.compute_distance([[0,2,0]], [[0,4,0]])
    for arr in range(len(distance1)):
        assert distance1[arr] >= 0
    # check that distance is smaller when electrode is closer to neuron and vise versa
    distance2 = lfp_amplitude_function.compute_distance([[0,0,0]], [[0,4,0]])
    for arr in range(len(distance2)):
        assert distance1[arr] < distance2[arr]
    # check that distance is inverse when electrode is rotated around neuron
    distance3 = lfp_amplitude_function.compute_distance([[0,0,0]], [[0,-4,0]])
    for arr in range(len(distance3)):
        assert -distance2[arr] == distance3[arr]

def test_compute_angle():
    # check that angle is the same if the electrode is above or below the neuron
    angle1 = lfp_amplitude_function.compute_angle([[0,2,0]], [[0,1,0]])
    angle2 = lfp_amplitude_function.compute_angle([[0,2,0]], [[0,4,0]])
    assert angle1 == angle2

def test_compute_amp():
    # check that LFP Amplitude is positive above the midpoint
    amp1 = lfp_amplitude_function.compute_amp([[1,1,1]], [[2,2,2]], [[5,5,5]])
    for arr in range(len(amp1)):
        assert amp1[arr] >= 0
    # check that LFP Amplitude is negarive below the midpoint
    amp2 = lfp_amplitude_function.compute_amp([[1,1,1]], [[2,2,2]], [[-5,-5,-5]])
    for arr in range(len(amp2)):
        assert amp2[arr] <= 0
    # check that LFP Amplitude is stronger closer to the neuron
    amp3 = lfp_amplitude_function.compute_amp([[1,1,1]],[[2,2,2]],[[3,3,3]])
    amp4 = lfp_amplitude_function.compute_amp([[1,1,1]],[[2,2,2]],[[6,6,6]])
    for arr in range(len(amp3)):
        assert amp3[arr] > amp4[arr]
    # check that flipping the neuron results in the same LFP
    amp5 = lfp_amplitude_function.compute_amp([[1,1,1]],[[2,2,2]],[[3,3,3]])
    amp6 = lfp_amplitude_function.compute_amp([[2,2,2]],[[1,1,1]],[[3,3,3]])
    for arr in range(len(amp5)):
        assert amp5[arr] == amp6[arr]
    # check input of multiple neurons
    soma = np.random.rand(5, 3) * 10
    dendrite = np.random.rand(5, 3) * 10
    electrode = np.random.rand(5, 3) * 10

def test_random_orientation():
    rot_mat = rotation_matrix()
    soma1 = [1,1,1]
    dendrite1 = [2,2,2]
    electrode1 = [3,3,3]
    soma2 = np.dot(rot_mat, soma1)
    dendrite2 = np.dot(rot_mat, dendrite1)
    electrode2 = np.dot(rot_mat, electrode1)
    amp1 = lfp_amplitude_function.compute_amp([soma1], [dendrite1], [electrode1])
    amp2 = lfp_amplitude_function.compute_amp([soma2], [dendrite2], [electrode2])
    for arr in range(len(amp1)):
        assert amp1[arr] == (amp2[arr])
    
def rotation_matrix(rng: np.random.Generator = np.random.default_rng()):
    τ = 2 * np.pi
    θxy, θyz = τ * rng.random(2)
    # multiply rotation matrices for xy and yz planes together
    rot_mat = np.array(
        [[np.cos(θxy), -np.sin(θxy), 0], [np.sin(θxy), np.cos(θxy), 0], [0, 0, 1]]
    ) @ np.array(
        [[1, 0, 0], [0, np.cos(θyz), -np.sin(θyz)], [0, np.sin(θyz), np.cos(θyz)]]
    )
    return rot_mat

def test_multiple_inputs():
    soma = np.random.rand(5, 3) * 10
    dendrite = np.random.rand(5, 3) * 10
    electrode = np.random.rand(5, 3) * 10
    midpoint = lfp_amplitude_function.compute_midpoint(soma, dendrite)
    print(lfp_amplitude_function.compute_amp(soma, dendrite, electrode))

def test_fewer_electrodes():
    soma = np.random.rand(5, 3) * 10
    dendrite = np.random.rand(5, 3) * 10
    electrode = np.random.rand(2, 3) * 10
    print(lfp_amplitude_function.compute_amp(soma, dendrite, electrode))

test_compute_amp()
test_compute_angle()
test_compute_distance()
test_compute_length()
test_compute_midpoint()
test_multiple_inputs()
test_random_orientation()
test_fewer_electrodes()