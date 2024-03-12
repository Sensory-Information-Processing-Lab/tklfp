import warnings
from typing import Union

import numpy as np
from attrs import define, field
from scipy.interpolate import PchipInterpolator

from wslfp.amplitude import (
    aussel18,
    aussel18_mod,
    f_amp,
    mazzoni15_nrn,
    mazzoni15_pop,
    plot_amp,
)
from wslfp.spk2curr import spikes_to_biexp_currents


def xyz_to_rd_coords(
    source_coords: np.ndarray,
    elec_coords: np.ndarray,
    source_orientation: np.ndarray,
):
    n_sources = np.shape(source_coords)[0]
    n_elec = np.shape(elec_coords)[0]
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
    """Calculator for WSLFP given current sources and electrode coordinates.

    Use `wslfp.from_rec_radius_depth` or `wslfp.from_xyz_coords` to initialize."""

    amp_uV: np.ndarray = field()
    """(n_elec, n_sources) array of amplitudes in μV"""
    alpha: float = 1.65
    """Weight on GABAergic currents"""
    tau_ampa_ms: float = 6
    """Delay of AMPAergic currents' contribution to LFP in ms"""
    tau_gaba_ms: float = 0
    """Delay of GABAergic currents' contribution to LFP in ms"""
    strict_boundaries: bool = False
    """Whether or not to raise an error when evaluation times
    require currents outside the provided time range"""

    @property
    def n_elec(self) -> int:
        """Number of electrodes in the WSLFP calculator"""
        return self.amp_uV.shape[0]

    @property
    def n_sources(self) -> int:
        """Number of current sources (neurons or populations) in the calculator"""
        return self.amp_uV.shape[1]

    def _interp_currents(self, t_ms, I, delay_ms, t_eval_ms):
        if not np.all(np.diff(t_ms) > 0):
            raise ValueError("t_ms must be monotonically increasing")
        if len(t_ms) == 0:
            return np.zeros((len(t_eval_ms), self.n_sources))

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
        if len(t_ms) > 1:
            interpolator = PchipInterpolator(t_ms, I, extrapolate=False)
        elif len(t_ms) == 1:
            interpolator = lambda t_eval: (t_eval == t_ms[0]) * I[0:1]

        I_interp = interpolator(t_eval_delayed)
        assert I_interp.shape == (len(t_eval_ms), self.n_sources)
        I_interp[np.isnan(I_interp)] = 0
        return I_interp

    def calculate(
        self,
        t_eval_ms: np.ndarray,
        t_ampa_ms: np.ndarray,
        I_ampa: np.ndarray,
        t_gaba_ms: np.ndarray,
        I_gaba: np.ndarray,
        normalize: bool = True,
        wsum_mean_std_for_norm: tuple[np.ndarray, np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate WSLFP at requested times for initialized coordinates given currents

        Args:
            t_eval_ms (np.ndarray): Times at which to evaluate the WSLFP, in milliseconds
            t_ampa_ms (np.ndarray): Time points at which AMPAergic currents are given,
                in milliseconds
            I_ampa (np.ndarray): AMPAergic currents, shape (len(t_ampa_ms), n_sources)
            t_gaba_ms (np.ndarray): Time points at which AMPAergic currents are given,
                in milliseconds
            I_gaba (np.ndarray): GABAergic currents, shape (len(t_ampa_ms), n_sources)
            normalize (bool, optional): Whether to normalize to mean of 0 and variance of 1.
                The main reason not to normalize is if you are computing one time step at a time
                (see `notebooks/stepwise.ipynb`).  Defaults to True.
            wsum_mean_std_for_norm (tuple[np.ndarray, np.ndarray], optional): If provided,
                the mean and standard deviation of the weighted sum term are used to normalize
                before a realistic amplitude is applied. (see `notebooks/stepwise.ipynb`).
                Defaults to None.

        Returns:
            np.ndarray: (len(t_eval_ms), n_elec) array of WSLFP at requested times for
                each electrode
        """
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
            if wsum_mean_std_for_norm:
                wsum_mean, wsum_std = wsum_mean_std_for_norm
            else:
                wsum_mean = np.mean(wsum, axis=0)
                if len(t_eval_ms) > 1:
                    wsum_std = np.std(wsum, axis=0)
                else:
                    wsum_std = 1

            lfp = (wsum - wsum_mean) / wsum_std
            assert lfp.shape == (len(t_eval_ms), self.n_elec)

            if not wsum_mean_std_for_norm:
                assert np.allclose(lfp.mean(axis=0), 0)
                if len(t_eval_ms) > 1:
                    assert np.allclose(lfp.std(axis=0), 1)

            lfp *= np.abs(self.amp_uV.mean(axis=1))
        else:
            lfp = wsum

        if normalize and wsum_mean_std_for_norm:
            return lfp, wsum
        else:
            return lfp


def from_rec_radius_depth(
    r_um: np.ndarray,
    d_um: np.ndarray,
    source_dendrite_length_um=250,
    amp_func: callable = mazzoni15_nrn,
    amp_kwargs={},
    **kwargs,
) -> WSLFPCalculator:
    """Initalize a `WSLFPCalculator` from recording radius and depth coordinates

    Args:
        r_um (np.ndarray): (n_elec, n_sources) array of lateral distance from source to electrode
        d_um (np.ndarray): (n_elec, n_sources) array of vertical distance from source
            to electrode. Convention follows Mazzoni 2015, measuring distance from dipole
            center.
        source_dendrite_length_um (int or np.ndarray, optional): Length of apical dendrites.
            Defaults to 250.
        amp_func (callable, optional): Amplitude function that follows signature of
            wslfp.f_amp. Defaults to `mazzoni15_nrn`.
        amp_kwargs (dict, optional): Passed to `amp_func`. Defaults to {}.

    Returns:
        WSLFPCalculator: Calculator object with per-source, per-electrode amplitude
            properly initialized.
    """
    amplitude_per_source = amp_func(
        r_um, d_um, L_um=source_dendrite_length_um, **amp_kwargs
    )
    return WSLFPCalculator(amp_uV=amplitude_per_source, **kwargs)


def from_xyz_coords(
    elec_coords_um: np.ndarray,
    source_coords_um: np.ndarray,
    source_coords_are_somata: bool = True,
    source_dendrite_length_um: Union[int, np.ndarray] = 250,
    source_orientation: np.ndarray = (0, 0, 1),
    amp_func: callable = mazzoni15_nrn,
    amp_kwargs={},
    **kwargs,
) -> WSLFPCalculator:
    """Initializes calculator from electrode and source coordinates

    Args:
        elec_coords_um (np.ndarray): (n_elec, 3) array of electrode coordinates in μm
        source_coords_um (np.ndarray): (n_sources, 3) array of source (neuron or population)
            coordinates in μm
        source_coords_are_somata (bool, optional): Whether source_coords represent somata
            (as opposed to dipole midpoint). Defaults to True.
        source_dendrite_length_um (Union[int, np.ndarray], optional): Length of apical dendrite,
            in μm, used in some amplitude functions. Defaults to 250.
        source_orientation (np.ndarray, optional): The vector(s) pointing "up," from soma
            to apical dendrite. Can have shape (3,) or (n_src, 3). Defaults to (0, 0, 1).
        amp_func (callable, optional): Amplitude function that follows signature of
            wslfp.f_amp. Defaults to `mazzoni15_nrn`.
        amp_kwargs (dict, optional): Passed to `amp_func`. Defaults to {}.

    Returns:
        WSLFPCalculator: Calculator object with per-source, per-electrode amplitude
            properly initialized.
    """
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
