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
