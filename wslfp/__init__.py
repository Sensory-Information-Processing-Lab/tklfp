from logging import raiseExceptions
from scipy.interpolate import interp1d
from numpy.linalg import norm
import pickle
import importlib.resources as pkg_resources
from numpy import array
import numpy as np
from wslfp import lfp_amplitude_function

class WSLFP:
    def __init__(self, xs, ys, zs, elec_coords, alpha=1.65, tau_ampa_ms=6, tau_gaba_ms=0):
        """xs, ys, zs are n_neurons-length 1D arrays
        elec_coords is a (n_elec, 3) array
        [[1, 1, 1],
         [2, 2, 2]] """
        self.a = self._amplitude(xs, ys, zs, elec_coords)
        self.alpha = alpha
        self.tau_ampa_ms = tau_ampa_ms
        self.tau_gaba_ms = tau_gaba_ms

    def _amplitude(self, xs, ys, zs, elec_coords):
        ...

    #def compute(ampa: np.ndarray, t_ampa_ms, gaba: np.ndarray, t_gaba_ms, t_eval_ms: np.ndarray):
        """_summary_
​
        Parameters
        ----------
        ampa : np.ndarray
            (n_timepoints, n_neurons)  e.g., 5 timepoints * 1000 neurons => 5000
        t_ampa_ms : np.ndarray
            (n_timepoints), e.g., [1, 2, 3, 4, 5]
        gaba : np.ndarray
            (n_timepoints, n_neurons)
        t_gaba_ms : _type_
            _description_
        t_eval_ms : _type_
            _description_
​
        Example
        -------
        # just 1 timepoint
        lfp = wslfp.compute(..., t_gaba_ms=[now_ms], t_eval_ms=[now_ms])
        # multiple timepoint
        lfp = wslfp.compute(..., t_gaba_ms=[multiple, gaba, measurements], t_eval_ms=[a, whole, bunch, of, timepoints])
        """

        #try:
           # _check_timepoints(t_ampa_ms, t_gaba_ms, t_eval_ms)
    
    # 1. combine check timepoints to compute curr
    # 2.
    
    #substract tau from t_eval NEW CODE
    def compute_gaba_curr(self, t_gaba_ms, tau_gaba, t_eval_ms): # evaluate at 1 timepoint, so t_eval_ms is a float
        gaba = np.array(gaba)
        #row = np.where(t_gaba_ms == t_eval_ms) 
        # #use interpolation
        gaba_interp = self.check_gaba_timepoints(gaba, t_gaba_ms, t_eval_ms, tau_gaba)
        
        return gaba_interp[t_eval_ms] #return gaba currents of all neurons at t_eval

    #def compute_ampa_curr(self, ampa, t)
    

    def compute_ampa_curr(self,t_ampa_ms, tau_ampa, t_eval_ms): 
        ampa = np.array(ampa)
        #np.substract(ampa_time_arr, tau_ampa)
        ampa_interp = self.check_ampa_timepoints(ampa, t_ampa_ms, t_eval_ms, tau_ampa)
        return ampa_interp[t_eval_ms]


    def check_ampa_timepoints(ampa, t_ampa_ms, t_eval_ms, tau_ampa): 
        # need exact timepoints if just one measurement is given. Otherwise, let interpolation throw an error
        # when out of range
        # check t_ampa_ms: ranging from at least tau_ampa ms before the first eval point
        # up to 6 ms before the last eval point

        # check if gaba is at least 6 ms before eval
        # ampa has to be equal to or smaller than eval
        # for a range of ampa, if eval is within range, use linear/quadratic interpolation
        # if not, raise exception
        for t in [np.min(t_eval_ms), np.max(t_eval_ms)]:
            if len(t_ampa_ms) == 1:
                if t > t_ampa_ms[0]:
                    raise Exception("ampa not valid")
            elif t - tau_ampa < np.min(t_ampa_ms) or t - tau_ampa > np.max(t_ampa_ms):
                raise Exception("ampa not valid")
            else:
                interp_ampa = interp1d(t_ampa_ms, ampa, kind= 'quadratic')
        return interp_ampa

            #else:
                #t_ampa_chosen = interp1d(t_ampa_ms, t_eval_ms, kind = int)
                #if t > t_ampa_chosen:
                    #raise Exception("ampa not valid")
    def check_gaba_timepoints(gaba, t_gaba_ms, t_eval_ms, tau_gaba) :
        for t in [np.min(t_eval_ms), np.max(t_eval_ms)]:
            if t - tau_gaba < np.min(t_gaba_ms) or t - tau_gaba > np.max(t_gaba_ms):
                raise Exception("gaba not valid")
            else:
                interp_gaba = interp1d(t_gaba_ms, gaba, kind= 'quadratic')
        return interp_gaba
        
 
         


        # if not ampa_valid:
        #     raise Exception("ampa not valid")
        # check t_gaba_ms: ranging from tau_gaba before the first eval point
        # up to the last eval point
        # if not gaba_valid:
        #     raise Exception("gaba not valid")

