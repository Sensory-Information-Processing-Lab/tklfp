from logging import raiseExceptions
from scipy.interpolate import interp1d
from numpy.linalg import norm
import pickle
import importlib.resources as pkg_resources
from numpy import array
import numpy as np
from src import lfp_amplitude_function

class WSLFP:
    def __init__(self, xs, ys, zs, elec_coords, ampa_times, ampa_currents, gaba_times, gaba_currents, alpha=1.65, tau_ampa_ms=6, tau_gaba_ms=0):
        """xs, ys, zs are n_neurons-length 1D arrays
        elec_coords is a (n_elec, 3) array
        [[1, 1, 1],
         [2, 2, 2]] """
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.elec_coords = elec_coords
        self.ampa_times = ampa_times
        self.ampa_currents = ampa_currents
        self.gaba_times = gaba_times
        self.gaba_currents = gaba_currents
        self.alpha = alpha
        self.tau_ampa_ms = tau_ampa_ms
        self.tau_gaba_ms = tau_gaba_ms
    
    def get_current(self, times, currents, t_eval, kind='linear'):
        if len(times) != len(currents):
            raise ValueError("Length of times and currents arrays must be equal.")
        if len(times) == 1:
            if np.all(times[0] == t_eval):
                return np.full_like(t_eval, currents[0])
            else:
                raise ValueError("Evaluation time does not match the single available time point.")
        else:
            if np.any(t_eval < times[0]) or np.any(t_eval > times[-1]):
                raise ValueError("Evaluation time is out of the current data range.")
            interpolate = interp1d(times, currents, kind=kind, bounds_error=True)
            return interpolate(t_eval)
    
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def calculate_lfp(self, t_evals):
        t_evals = np.array(t_evals)
        if len(t_evals) == 0:
            raise ValueError("Evaluation time is not given.")
        if len(self.ampa_times) == 0 or len(self.gaba_times) == 0:
            raise ValueError("AMPA times and GABA times cannot be empty")
        if not np.all(np.diff(self.ampa_times) >= 0):
            raise ValueError("AMPA times are not in increasing order")
        if not np.all(np.diff(self.gaba_times) >= 0):
            raise ValueError("GABA times are not in increasing order")
        if np.any(t_evals - self.tau_ampa_ms < self.ampa_times[0]) or np.any(t_evals - self.tau_ampa_ms > self.ampa_times[-1]):
            raise ValueError("Some evaluation times are out of the AMPA time range.")
        if np.any(t_evals - self.tau_gaba_ms < self.gaba_times[0]) or np.any(t_evals - self.tau_gaba_ms > self.gaba_times[-1]):
            raise ValueError("Some evaluation times are out of the GABA time range.")
        ampa_currents = np.array([self.get_current(self.ampa_times, self.ampa_currents, t_eval - self.tau_ampa_ms) for t_eval in t_evals])
        gaba_currents = np.array([self.get_current(self.gaba_times, self.gaba_currents, t_eval - self.tau_gaba_ms) for t_eval in t_evals])
        ws = ampa_currents - self.alpha * gaba_currents
        amplitudes = lfp_amplitude_function.compute_amp([[1,1,1]], [[2,2,2]], [[5,5,5]])
        lfp_normalized = self.normalize(amplitudes * ws)
        return lfp_normalized
        

    
