from importlib import resources
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator


def aussel18(r_um, d_um, L_um=250, sigma=0.3):
    """Follows the closed-form formula for per-neuron contributions from Aussel et al., 2018.
    It is then scaled so that summed over a population, the scale resembles that of
    Mazzoni, Lindén et al., 2015 (about -0.1 to 0.1 μV).

    THe formula is :math:`A \\frac{L \\cos(\\theta)}{4 \\pi \\sigma r^2}`."""
    A = 0.021
    dist = np.sqrt(r_um**2 + d_um**2)
    costheta = -d_um / dist
    return A * (L_um * costheta) / (4 * np.pi * sigma * dist**2)


def aussel18_mod(r_um, d_um, L_um=250, sigma=0.3):
    dist = np.sqrt(r_um**2 + d_um**2)
    costheta = -d_um / dist
    return (L_um * costheta) / (4 * np.pi * sigma * dist)


def mazzoni15_pop(r_um, d_um, L_um=250, sigma=0.3):
    """The profile of the LFP amplitude extracted from Mazzoni, Lindén et al., 2015,
    Figure 2B. See `notebooks/mazzoni_data_extrapolation.ipynb`"""
    rdf_samples = np.load(resources.files("wslfp") / "mazzoni15-rdf.npy")
    pos_interp = LinearNDInterpolator(rdf_samples[:, :2], rdf_samples[:, 2])
    d_um_sign = np.sign(d_um)
    f_interp = pos_interp(r_um, np.abs(d_um)) * d_um_sign
    return np.nan_to_num(f_interp)


def mazzoni15_nrn(r_um, d_um, L_um=250, sigma=0.3):
    """`mazzoni15_pop`, but rescaled. See `notebooks/amplitude_comparison.ipynb.`"""
    rscale = 1.31724
    dscale = 1.22438
    fscale = 1.82849
    return fscale * mazzoni15_pop(r_um * rscale, d_um * dscale, L_um, sigma)


def f_amp(
    r_um: np.ndarray,
    d_um: np.ndarray,
    L_um: Union[float, np.ndarray] = 250,
    sigma: float = 0.3,
    method: str = "mazzoni15_nrn",
) -> np.ndarray:
    """Compute the amplitude of the LFP at a given position.

    Args:
        r_um (np.ndarray): Lateral distance from dipole, in μm
        d_um (np.ndarray): Depth from dipole midpoint, in μm, where pyramidal apices are positive.
        L_um (int, optional): Apical dendrite length, in μm. Defaults to 250.
        sigma (float, optional): Conductivity, in S/m. Defaults to 0.3.
        method (str, optional): Name of amplitude function to use. Defaults to "mazzoni15_nrn".

    Returns:
        np.ndarray: LFP amplitudes with shape matching r_um and d_um
    """
    assert r_um.shape == d_um.shape
    return globals()[method](r_um, d_um, L_um, sigma)


def plot_amp(
    f,
    extent=None,
    title=None,
    vlim=None,
    fig=None,
    ax=None,
    cbar=True,
    labels=True,
    **kwargs,
):
    if not fig and not ax:
        fig, ax = plt.subplots(figsize=(3, 3))
    im_kwargs = dict(cmap="RdBu", extent=extent, origin="lower")
    if vlim:
        kwargs["vmin"] = -vlim
        kwargs["vmax"] = vlim
    im_kwargs.update(kwargs)
    im = ax.imshow(f, **im_kwargs)
    if cbar:
        fig.colorbar(im)
    if labels:
        ax.set(
            title="LFP amplitude (μV)",
            ylabel="Electrode depth (μm)",
            xlabel="Electrode lateral distance (μm)",
        )
    if title:
        ax.set_title(title)
    return fig, ax
