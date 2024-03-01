from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.interpolate import LinearNDInterpolator


def aussel18(r_um, d_um, L_um=250, sigma=0.3):
    dist = np.sqrt(r_um**2 + d_um**2)
    costheta = -d_um / dist
    return (L_um * costheta) / (4 * np.pi * sigma * dist**2)


def aussel18_mod(r_um, d_um, L_um=250, sigma=0.3):
    dist = np.sqrt(r_um**2 + d_um**2)
    costheta = -d_um / dist
    return (L_um * costheta) / (4 * np.pi * sigma * dist)


def mazzoni15(r_um, d_um, L_um=250, sigma=0.3):
    # params_file = resources.open_binary("wslfp", "mazzoni15-beta-params.npz")
    params = np.load(resources.files("wslfp") / "mazzoni15-beta-params.npz")
    a, b, scale, dmax, rslices = (
        params["a"],
        params["b"],
        params["scale"],
        params["dmax"],
        params["rslices"],
    )
    d_um_sign = np.sign(d_um)
    d_um_abs = np.abs(d_um)
    amp_at_rslices = np.zeros((len(a) + 1, *d_um_abs.shape))  # add dummy slice
    for i in range(len(a)):
        amp_at_rslices[i, ...] = beta.pdf(
            d_um_abs / dmax[i], a[i], b[i], loc=0, scale=scale[i]
        )
    # dummy slice to facilitate interpolation
    amp_at_rslices[-1, ...] = 0
    rslices = np.append(rslices, rslices[-1] + 1)

    #################################################################
    # homebrewed 1D interpolation, since we only interpolate in the lateral direction
    #################################################################
    homebrewed = True
    if homebrewed:
        f_interp = np.zeros_like(r_um)

        i_left_slice = np.searchsorted(rslices, r_um, side="right") - 1
        i_right_slice = i_left_slice + 1
        in_range = i_left_slice < len(rslices) - 1

        amp_at_rslices_filt = amp_at_rslices[:, in_range]

        i_left_slice_filt = i_left_slice[in_range]
        i_right_slice_filt = i_right_slice[in_range]
        r_um_filt = r_um[in_range]

        rleft = rslices[i_left_slice_filt]
        rright = rslices[i_right_slice_filt]
        assert np.all((rleft <= r_um_filt) & (r_um_filt < rright))
        amp_left, amp_right = np.zeros_like(rleft), np.zeros_like(rright)
        for i, (i_left, i_right) in enumerate(
            zip(i_left_slice_filt, i_right_slice_filt)
        ):
            amp_left[i] = amp_at_rslices_filt[i_left, i]
            amp_right[i] = amp_at_rslices_filt[i_right, i]

        f_interp[in_range] = (
            amp_left * (rright - r_um_filt) + amp_right * (r_um_filt - rleft)
        ) / (rright - rleft)

        f_interp *= -d_um_sign
        f_interp = np.nan_to_num(f_interp)

    #################################################################
    # interpn: again, fails on large arrays
    # f_interp = interpn(
    #     (rslices,),
    #     amp_at_rslices.reshape((len(rslices), -1)),
    #     r_um.flatten(),
    #     fill_value=0,
    #     bounds_error=False,
    # )
    # f_interp = f_interp.diagonal().copy()
    # print(f_interp.shape)

    # numpy version: doesn't work on multi-dim arrays
    # f_interp = np.interp(r_um.flatten(), rslices, amp_at_rslices, left=0, right=0)

    #################################################################
    # scipy LinearNDINterpolator: works, but slow, fails on large arrays
    else:
        amp_at_rslices = np.moveaxis(amp_at_rslices, 0, -1)
        print(amp_at_rslices.shape)
        r_points, d_points = np.broadcast_arrays(rslices, d_um_abs[..., None])
        print(r_points.shape, d_points.shape)
        points = np.array([r_points.flatten(), d_points.flatten()]).T
        interp = LinearNDInterpolator(points, amp_at_rslices.flatten(), fill_value=0)
        f_interp = interp(np.array([r_um.flatten(), d_um_abs.flatten()]).T)
        # f_interp = np.moveaxis(f_interp, 0, -1)
        print(f_interp.shape)
        f_interp *= -d_um_sign.flatten()
        f_interp = np.nan_to_num(f_interp).reshape(r_um.shape)

    #################################################################
    # scipy interpn: not working
    # points_interp = np.array([r_um.flatten(), d_um_abs.flatten()]).T
    # print(points_interp.shape)
    # f_interp = interpn(
    #     (rslices, d_um_abs[:, 0]),
    #     amp_at_rslices.flatten(),
    #     points_interp,
    #     fill_value=0,
    # )

    return f_interp


def f_amp(r_um, d_um, L_um=250, sigma=0.3, method="mazzoni15"):
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
