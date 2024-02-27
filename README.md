# wslfp

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

## When to use
According to Mazzoni et al., 2015, the WSLFP method is a good proxy for LFP when:
- There's enough network activity for the LFP to be sizable
- Morphologies are sufficiently "pyramidal," i.e., the centers of GABA and AMPA dendritic bushes are sufficiently separated (>= 150 μm)

## Future development
These features might be useful to add in the future:
- amplitude and $alpha$ that vary by axon length as well as by recording position