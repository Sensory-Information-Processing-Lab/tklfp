# wslfp

[![DOI](https://zenodo.org/badge/440986279.svg)](https://zenodo.org/badge/latestdoi/440986279)

This is a lightweight package for computing the LFP approximation from 
[Mazzoni et al., 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004584). 

## How to install:
Simply install from pypi:
```bash
pip install wslfp
```

## How to use:

### Initialization
First you must initialize a `WSLFP` object which computes and caches the per-spike contribution of each neuron to the total LFP. You will need X, Y, and Z coordinates of the neurons and the coordinates of the electrode(s):
```python
from wslfp import WSLFP
wslfp = WSLFP(xs_mm, ys_mm, zs_mm, elec_coords_mm)
```

The first three arguments must all have the same length N_n, the total number of neurons. `elec_coords_mm` must an N_e by 3 array-like object, where N_e is the number of recording sites.

### Computing LFP
LFP can then be computed with the ampa, gaba, the time point we want to get the synaptic current, and the time point we want to evaluate the LFP.
```python
def compute(ampa: n_neuronsXn_timepoints, gaba: n_neuronsXn_timepoints, t_ms, t_eval_ms)
```

A complete example, reworking the demo from the original paper, can be found [here](https://github.com/kjohnsen/tklfp/blob/master/notebooks/demo_lfp_kernel.ipynb). Basic usage information is also accessible in docstrings.

### Cortical orientation
The `TKLFP` constructor can also take an `orientation` argument which represents which direction is "up," that is, towards the surface of the cortex or parallel to the apical dendrites of pyramidal cells.
The default is `[0, 0, 1]`, indicating that the positive z axis is "up."
In the case your population isn't a sheet of neurons with uniform orientation (for a curved cortical area, for example), you can pass an N_n by 3 array containing the individual orientation vectors for all the neurons.

## Future development
The package uses [parameters from the original 2020 paper](https://github.com/kjohnsen/tklfp/blob/master/notebooks/param_prep.ipynb) by default. This can be changed by passing in an alternate parameter dictionary on initialization:
```python
tklfp = TKLFP(..., params=new_params)
```

The new params must have the same content as the default [`tklfp.params2020`](https://github.com/kjohnsen/tklfp/blob/master/tklfp/__init__.py#:~:text=_sig_i%20%3D%202.1-,params2020%20%3D,-%7B). The `A0_by_depth` params are scipy interpolation objects, but could theoretically be any callable that will return A0 (in μV) for an arbitrary depth (in mm).

