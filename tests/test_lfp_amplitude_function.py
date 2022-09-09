import numpy as np
import lfp_amplitude_function 
def test_ComputeLength():
    # check that length is 0 when points are the same
    length = lfp_amplitude_function.computeLength([10,10,10], [10,10,10])
    assert length == 0
    # check that flipping the neuron results in the same length
    length1 = lfp_amplitude_function.computeLength([0,2,0], [0,0,0])
    length2 = lfp_amplitude_function.computeLength([0,0,0], [0,2,0])
    assert length1 == length2

def test_ComputeMidpoint():
    # check that midpoint is the same if the neuron is flipped
    midpoint1 = lfp_amplitude_function.computeMidPoint([0,2,0], [0,0,0])
    midpoint2 = lfp_amplitude_function.computeMidPoint([0,0,0], [0,2,0])
    assert midpoint1 == midpoint2
    # check that fucntion can handle negative values
    midpoint = lfp_amplitude_function.computeMidPoint([0,-1,-3], [4,8,-10])
    assert midpoint == [2, 3.5, -6.5]

def test_ComputeDistance():
    # check that distance is negative when electrode is below neuron
    distance = lfp_amplitude_function.computeDistance([0,2,0], [0,0,0])
    assert distance <= 0
    # check that distance is positive when electrode is above neuron
    distance1 = lfp_amplitude_function.computeDistance([0,2,0], [0,4,0])
    assert distance1 >= 0
    # check that distance is smaller when electrode is closer to neuron and vise versa
    distance2 = lfp_amplitude_function.computeDistance([0,0,0], [0,4,0])
    assert distance1 < distance2
    # check that distance is inverse when electrode is rotated around neuron
    distance3 = lfp_amplitude_function.computeDistance([0,0,0], [0,-4,0])
    assert -distance2 == distance3

def test_ComputeAngle():
    # check that angle is the same if the electrode is above or below the neuron
    angle1 = lfp_amplitude_function.computeAngle([0,2,0], [0,1,0])
    angle2 = lfp_amplitude_function.computeAngle([0,2,0], [0,4,0])
    assert angle1 == angle2

def test_ComputeAmp():
    # check that LFP Amplitude is positive above the midpoint
    amp1 = lfp_amplitude_function.computeAmp([[1,1,1]], [[2,2,2]], [[5,5,5]])
    assert amp1 >= 0
    # check that LFP Amplitude is negarive below the midpoint
    amp2 = lfp_amplitude_function.computeAmp([[1,1,1]], [[2,2,2]], [[-5,-5,-5]])
    assert amp2 <= 0
    # check that LFP Amplitude is stronger closer to the neuron
    amp3 = lfp_amplitude_function.computeAmp([[1,1,1]],[[2,2,2]],[[3,3,3]])
    amp4 = lfp_amplitude_function.computeAmp([[1,1,1]],[[2,2,2]],[[6,6,6]])
    assert amp3 > amp4
    # check that flipping the neuron results in the same LFP
    amp5 = lfp_amplitude_function.computeAmp([[1,1,1]],[[2,2,2]],[[3,3,3]])
    amp6 = lfp_amplitude_function.computeAmp([[2,2,2]],[[1,1,1]],[[3,3,3]])
    assert amp5 == amp6

def test_random_orientation():
    rot_mat = rotation_matrix()
    soma1 = [1,1,1]
    dendrite1 = [2,2,2]
    electrode1 = [3,3,3]
    soma2 = np.dot(rot_mat, soma1)
    dendrite2 = np.dot(rot_mat, dendrite1)
    electrode2 = np.dot(rot_mat, electrode1)
    amp1 = lfp_amplitude_function.computeAmp([soma1], [dendrite1], [electrode1])
    amp2 = lfp_amplitude_function.computeAmp([soma2], [dendrite2], [electrode2])
    assert amp1 == amp2
    
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
