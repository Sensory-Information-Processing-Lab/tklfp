import numpy as np
import pytest

import wslfp


@pytest.mark.parametrize("seed", [18300406, 18360406])
@pytest.mark.parametrize("n_elec", [1, 3])
def test_orientation_sensitivity(seed, n_elec):
    n_nrns = 4
    rng = np.random.default_rng(seed)
    elec_coords = rng.random((n_elec, 3))
    # neurons in same spot
    nrn_coords = np.zeros((n_nrns, 3))
    # but random orientations
    orientation = rng.uniform(-1, 1, (n_nrns, 3))
    lfp_calc = wslfp.from_xyz_coords(
        elec_coords, nrn_coords, source_orientation=orientation
    )
    amp = lfp_calc.amp_uV
    assert not (amp[:, :-1] == amp[:, 1:]).all(axis=1).any()
    # each neuron should produce sth different since
    # they are all oriented differently


def _rand_rot_mat(rng: np.random.Generator = np.random.default_rng()):
    τ = 2 * np.pi
    θxy, θyz = τ * rng.random(2)
    # multiply rotation matrices for xy and yz planes together
    rot_mat = np.array(
        [[np.cos(θxy), -np.sin(θxy), 0], [np.sin(θxy), np.cos(θxy), 0], [0, 0, 1]]
    ) @ np.array(
        [[1, 0, 0], [0, np.cos(θyz), -np.sin(θyz)], [0, np.sin(θyz), np.cos(θyz)]]
    )
    return rot_mat.T  # transpose for right instead of left multiplication


@pytest.mark.parametrize("seed", [421, 385])
@pytest.mark.parametrize("n_nrns", [1, 4])
@pytest.mark.parametrize("n_elec", [1, 3])
def test_rotation_invariance(seed, n_nrns, n_elec):
    """If orientations work correctly, we should be able to rotate the whole system"""
    rng = np.random.default_rng(seed)
    nrn_coords = rng.uniform(-1, 1, (n_nrns, 3))
    orientation = rng.uniform(-1, 1, (n_nrns, 3))
    elec_coords = rng.uniform(-1, 1, (n_elec, 3))

    amp = wslfp.from_xyz_coords(
        elec_coords, nrn_coords, source_orientation=orientation
    ).amp_uV
    # now if we rotate both the neurons, orientations, and the electrode coordinates
    # they should have the same relative positions yield the same results every time
    for i_rot in range(6):  # try 6 rotations
        rot_mat = _rand_rot_mat(rng)
        # rotate with right-multiplication for convenience (since mats are nx3)
        amp_rot = wslfp.from_xyz_coords(
            elec_coords @ rot_mat,
            nrn_coords @ rot_mat,
            source_orientation=orientation @ rot_mat,
        ).amp_uV
        assert np.allclose(amp, amp_rot)
