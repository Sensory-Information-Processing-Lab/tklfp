import warnings

import numpy as np
from attrs import define, field
from scipy.interpolate import PchipInterpolator

from wslfp.amplitude import (
    aussel18,
    aussel18_mod,
    mazzoni15_nrn,
    mazzoni15_pop,
    plot_amp,
)
from wslfp.spk2curr import spikes_to_biexp_currents


def xyz_to_rd_coords(
    source_coords,
    elec_coords,
    source_orientation,
):
    n_sources = np.shape(source_coords)[0]
    n_elec = np.shape(elec_coords)[0]
    # n_elec X n_nrns X 3
    xyz_dist = elec_coords[:, np.newaxis, :] - source_coords[np.newaxis, :, :]
    assert xyz_dist.shape == (n_elec, n_sources, 3)

    # theta = arccos(o*d/(||o||*||d||))
    dist = np.linalg.norm(xyz_dist, axis=2)
    assert dist.shape == (n_elec, n_sources)
    # since 0 dist leads to division by 0 and numerator of 0 is "invalid"
    old_settings = np.seterr(divide="ignore", invalid="ignore")
    theta = np.nan_to_num(
        np.arccos(
            np.sum(
                source_orientation * xyz_dist, axis=2
            )  # multiply elementwise then sum across x,y,z to get dot product
            / (1 * dist)  # norm of all orientation vectors should be 1
        )
    )
    assert theta.shape == (n_elec, n_sources)
    np.seterr(**old_settings)

    d_um = dist * np.cos(theta)
    r_um = dist * np.sin(theta)
    assert r_um.shape == d_um.shape == (n_elec, n_sources)

    return r_um, d_um


@define
class WSLFPCalculator:
    amp_uV: np.ndarray = field()
    """(n_elec, n_sources) array of amplitudes in μV"""
    alpha: float = 1.65
    tau_ampa_ms: float = 6
    tau_gaba_ms: float = 0
    strict_boundaries: bool = False

    @property
    def n_elec(self):
        return self.amp_uV.shape[0]

    @property
    def n_sources(self):
        return self.amp_uV.shape[1]

    def _interp_currents(self, t_ms, I, delay_ms, t_eval_ms):
        if not np.all(np.diff(t_ms) > 0):
            raise ValueError("t_ms must be monotonically increasing")

        t_eval_delayed = np.subtract(t_eval_ms, delay_ms)
        t_needed = (np.min(t_eval_delayed), np.max(t_eval_delayed))
        t_provided = (np.min(t_ms), np.max(t_ms))

        if t_needed[0] < t_provided[0] or t_needed[1] > t_provided[1]:
            if self.strict_boundaries:
                raise ValueError(
                    "Insufficient current data to interpolate for the requested times. "
                    f"Needed [{t_needed[0]}, {t_needed[1]}] ms, "
                    f"provided [{t_provided[0]}, {t_provided[1]}] ms."
                )
            else:
                warnings.warn(
                    "Insufficient current data to interpolate for the requested times. "
                    "Assuming 0 current for out-of-range times. "
                    f"Needed [{t_needed[0]}, {t_needed[1]}] ms, "
                    f"provided [{t_provided[0]}, {t_provided[1]}] ms."
                )
        # I_interp = np.interp(t_eval_delayed, t_ms, I, left=0, right=0)
        I_interp = PchipInterpolator(t_ms, I, extrapolate=False)(t_eval_delayed)
        assert I_interp.shape == (len(t_eval_ms), self.n_sources)
        I_interp[np.isnan(I_interp)] = 0
        return I_interp

    def calculate(
        self, t_eval_ms, t_ampa_ms, I_ampa, t_gaba_ms, I_gaba, normalize=True
    ):
        I_ampa = np.reshape(I_ampa, (-1, self.n_sources))
        assert I_ampa.shape == (
            len(t_ampa_ms),
            self.n_sources,
        ), f"{I_ampa.shape} != ({len(t_ampa_ms)}, {self.n_sources})"
        I_gaba = np.reshape(I_gaba, (-1, self.n_sources))
        assert I_gaba.shape == (
            len(t_gaba_ms),
            self.n_sources,
        ), f"{I_gaba.shape} != ({len(t_gaba_ms)}, {self.n_sources})"

        I_ampa_eval = self._interp_currents(
            t_ampa_ms, I_ampa, self.tau_ampa_ms, t_eval_ms
        )
        I_gaba_eval = self._interp_currents(
            t_gaba_ms, I_gaba, self.tau_gaba_ms, t_eval_ms
        )

        wsum = self.amp_uV * (I_ampa_eval - self.alpha * I_gaba_eval)[:, np.newaxis, :]
        assert wsum.shape == (len(t_eval_ms), self.n_elec, self.n_sources)
        wsum = np.sum(wsum, axis=2)
        assert wsum.shape == (len(t_eval_ms), self.n_elec)
        if normalize:
            wsum = wsum - np.mean(wsum, axis=0)
            if len(t_eval_ms) > 1:
                wsum /= np.std(wsum, axis=0)
            assert wsum.shape == (len(t_eval_ms), self.n_elec)

        assert np.allclose(wsum.mean(axis=0), 0)
        if len(t_eval_ms) > 1:
            assert np.allclose(wsum.std(axis=0), 1)

        wsum *= np.abs(self.amp_uV.mean(axis=1))
        return wsum


def from_rec_radius_depth(
    r_um,
    d_um,
    source_dendrite_length_um=250,
    amp_func=aussel18,
    amp_kwargs={},
    **kwargs,
):
    amplitude_per_source = amp_func(
        r_um, d_um, L_um=source_dendrite_length_um, **amp_kwargs
    )
    return WSLFPCalculator(amp_uV=amplitude_per_source, **kwargs)


def from_xyz_coords(
    elec_coords_um,
    source_coords_um,
    source_coords_are_somata=True,
    source_dendrite_length_um=250,
    source_orientation=(0, 0, 1),
    amp_func=aussel18,
    amp_kwargs={},
    **kwargs,
):
    elec_coords_um = np.reshape(elec_coords_um, (-1, 3))
    source_coords_um = np.reshape(source_coords_um, (-1, 3))
    ornt_shape = np.shape(source_orientation)
    assert len(ornt_shape) in [1, 2] and ornt_shape[-1] == 3
    # normalize orientation vectors
    source_orientation = source_orientation / np.linalg.norm(
        source_orientation, axis=-1, keepdims=True
    )

    if source_coords_are_somata:
        source_coords_um = source_coords_um + np.multiply(
            source_orientation, source_dendrite_length_um / 2
        )

    r_um, d_um = xyz_to_rd_coords(source_coords_um, elec_coords_um, source_orientation)
    return from_rec_radius_depth(
        r_um,
        d_um,
        source_dendrite_length_um=source_dendrite_length_um,
        amp_func=amp_func,
        amp_kwargs=amp_kwargs,
        **kwargs,
    )
