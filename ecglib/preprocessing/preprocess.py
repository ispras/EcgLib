import copy
from typing import Union

import numpy as np
import pandas as pd
import pywt
from scipy import signal
from scipy.stats import zscore

from . import functional as F


__all__ = [
    "FrequencyResample",
    "Padding",
    "EdgeCut",
    "Normalization",
    "ButterworthFilter",
    "IIRNotchFilter",
    "BaselineWanderRemoval",
    "WaveletTransform",
    "LeadCrop",
    "RandomLeadCrop",
    "TimeCrop",
    "RandomTimeCrop",
    "SumAug",
    "RandomSumAug",
    "ReflectAug",
    "ConvexAug",
    "RandomConvexAug",
]


class FrequencyResample:
    """
    Apply frequency resample
    :param ecg_frequency: sampling frequency of a signal
    :param requested_frequency: sampling frequency of a preprocessed signal

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        ecg_frequency: int, 
        requested_frequency: int = 500,
    ):
        if isinstance(ecg_frequency, (int, float)):
            self.ecg_frequency = ecg_frequency
        else:
            raise ValueError('ecg_frequency must be scalar')
        if isinstance(requested_frequency, (int, float)):
            self.requested_frequency = requested_frequency
        else:
            raise ValueError('requested_frequency must be scalar')
        self.func = F.ecg_to_one_frequency

    def __call__(self, x):

        x = self.func(x, int(self.ecg_frequency), int(self.requested_frequency))
        
        return x
    

class Padding:
    """
    Apply padding. If ECG is longer than the observed_ecg_length the record is cut.
    :param observed_ecg_length: length of padded signal in seconds
    :param frequency: sampling frequency of a signal
    :param pad_mode: padding mode

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        observed_ecg_length: float = 10, 
        frequency: int = 500, 
        pad_mode: str = "constant",
    ):
        self.observed_ecg_length = observed_ecg_length
        self.frequency = frequency
        self.pad_mode = pad_mode

    def __call__(self, x):

        if self.observed_ecg_length*self.frequency - x.shape[1] > 0:
            x = np.pad(x, ((0, 0), (0, self.observed_ecg_length*self.frequency - x.shape[1])), mode=self.pad_mode)
        else:
            x = x[:, :self.observed_ecg_length*self.frequency]
            
        return x


class EdgeCut:
    """
    Cut signal edges 
    :param cut_range: cutting parameters
    :param frequency: sampling frequency of a signal

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        cut_range: list = [0, 0], 
        frequency: int = 500,
    ):
        self.cut_range = cut_range
        self.frequency = frequency
        self.func = F.cut_ecg
        
    def __call__(self, x):

        x = self.func(x, self.cut_range, self.frequency)
        
        return x
        
    
class Normalization:
    """
    Apply normalization
    :param norm_type: type of normalization ('z_norm' or 'cycle') 
    :param leads: leads to be normalized

    :return: preprocessed data
    """

    def __init__(
        self, 
        norm_type: str = "z_norm", 
        leads: list = None,
    ):
        if leads is None:
            self.leads = list(range(12))
        elif isinstance(leads, list):
            self.leads = leads
        else:
            raise ValueError('leads must be list type')
        self.norm_type = norm_type
        if norm_type == "cycle":
            self.func = F.cycle_normalization
        elif norm_type == "min_max":
            self.func = F.minmax_normalization
        elif norm_type == "z_norm":
            self.func = F.z_normalization
        else:
            raise ValueError('norm_type must be one of [cycle, min_max, z_norm]')

    def __call__(self, x):
        
        if self.norm_type is not None:
            return self.func(x[self.leads, :])
        else:
            return x[self.leads, :]

    
class ButterworthFilter:
    """
    Apply Butterworth filter augmentation
    :param filter_type: type of Butterworth filter ('bandpass', 'lowpass' or 'highpass') 
    :param leads: leads to be filtered
    :param n: filter order
    :param Wn: cutoff frequency(ies)
    :param fs: filtered signal frequency

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        filter_type: str = "bandpass", 
        leads: list = None, 
        n: int = 10, 
        Wn: Union[float,list] = [3, 30], 
        fs: int = 500,
    ):
        if leads is None:
            self.leads = list(range(12))
        elif isinstance(leads, list):
            self.leads = leads
        else:
            raise ValueError('leads must be list type')
        self.filter_type = filter_type
        if filter_type == "bandpass":
            self.func = F.butterworth_bandpass_filter
        elif filter_type == "lowpass":
            self.func = F.butterworth_lowpass_filter
        elif filter_type == "highpass":
            self.func = F.butterworth_highpass_filter
        else:
            raise ValueError("Filter type must be one of [bandpass, lowpass, highpass]")
        self.n = n
        if filter_type == "bandpass" and not isinstance(Wn, list):
            raise ValueError('Wn must be list type in case of bandpass filter')
        elif (filter_type == "highpass" or filter_type == "lowpass") and not isinstance(Wn, (int, float)):
            raise ValueError(f'Wn must be a scalar in case of {filter_type} filter')
        self.Wn = Wn
        self.fs = fs

    def __call__(self, x):
        
        return np.apply_along_axis(self.func, axis=1, arr=x[self.leads, :], n=self.n, Wn=self.Wn, fs=self.fs)

    
class IIRNotchFilter:
    """
    Apply IIR notch filter augmentation
    :param leads: leads to be filtered
    :param w0: frequency to remove from a signal
    :param Q: quality factor
    :param fs: sampling frequency of a signal

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        leads: list = None, 
        w0: float = 50, 
        Q: float = 30, 
        fs: int = 500,
    ):
        if leads is None:
            self.leads = list(range(12))
        elif isinstance(leads, list):
            self.leads = leads
        else:
            raise ValueError('leads must be list type')
        self.w0 = w0
        self.Q = Q
        self.fs = fs
        self.func = F.IIR_notch_filter

    def __call__(self, x):
        
        return np.apply_along_axis(self.func, axis=1, arr=x[self.leads, :], w0=self.w0, Q=self.Q, fs=self.fs)


class BaselineWanderRemoval:
    """
    Remove baseline wander using wavelets (see article https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.6789&rep=rep1&type=pdf)
    :param leads: leads to be preprocessed
    :param wavelet: wavelet name

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        leads: list = None, 
        wavelet: str = 'db4',
    ):
        if leads is None:
            self.leads = list(range(12))
        elif isinstance(leads, list):
            self.leads = leads
        else:
            raise ValueError('leads must be list type')
        self.wavelet = wavelet
        self.func = F.DWT_BW

    def __call__(self, x):
        
        return np.apply_along_axis(self.func, axis=1, arr=x[self.leads, :], wavelet=self.wavelet)

    
class WaveletTransform:
    """
    Apply wavelet transform augmentation
    :param wt_type: type of wavelet transform ('DWT' with soft thresholding or 'SWT') 
    :param leads: leads to be transformed
    :param wavelet: wavelet name
    :param level: decomposition level
    :param threshold: thresholding value for all coefficients except the first one (only for DWT)
    :param low: thresholding value for the first coefficient (only for DWT)

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        wt_type: str = "DWT", 
        leads: list = None, 
        wavelet: str = 'db4', 
        level: int = 3, 
        threshold: float = 2, 
        low: float = 1e6,
    ):
        if leads is None:
            self.leads = list(range(12))
        elif isinstance(leads, list):
            self.leads = leads
        else:
            raise ValueError('leads must be list type')
        self.wt_type = wt_type
        self.wavelet = wavelet
        self.level = level
        if wt_type == "DWT":
            self.threshold = threshold
            self.low = low
            self.func = F.DWT_filter
        elif wt_type == "SWT":
            self.threshold = None
            self.low = None
            self.func = F.SWT_filter
        else:
            raise ValueError('wt_type must be one of [DWT, SWT]')
            
    def __call__(self, x):
        
        if self.wt_type == "DWT":
            return np.apply_along_axis(self.func, axis=1, arr=x[self.leads, :], wavelet=self.wavelet, 
                                       level=self.level, threshold=self.threshold, low=self.low)
        elif self.wt_type == "SWT":
            return np.apply_along_axis(self.func, axis=1, arr=x[self.leads, :], wavelet=self.wavelet, 
                                       level=self.level)


class LeadCrop:
    """
    Apply lead crop augmentation
    :param leads: leads to be cropped

    :return: preprocessed data
    """
    
    def __init__(
        self,
        leads: list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        self.leads = leads
        self.func = F.lead_crop
        
    def __call__(self, x):
        
        return self.func(x, leads=self.leads)
    
    
class RandomLeadCrop(LeadCrop):
    """
    Apply random lead crop augmentation
    :param n: number of leads to be cropped (chosen randomly)

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        n: int = 11,
    ):
        ls = np.arange(12, dtype='int')
        leads_to_remove = np.random.choice(ls, size=n, replace=False)
        super().__init__(leads_to_remove)
        self.n = n

    def __call__(self, x):
        
        return self.func(x, leads=self.leads)

    
class TimeCrop:
    """
    Apply time crop augmentation
    :param time: length of time segment to be cropped (the same units as signal)
    :param leads: leads to be cropped

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        time: int = 100, 
        leads: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        self.time = time
        self.leads = leads
        self.func = F.time_crop
        
    def __call__(self, x):
        
        return self.func(x, time=self.time, leads=self.leads)

    
class RandomTimeCrop(TimeCrop):
    """
    Apply random time crop augmentation
    :param time: length of time segment to be cropped (the same units as signal)
    :param n: number of leads to be cropped (chosen randomly)

    :return: preprocessed data
    """
    
    def __init__(
        self, 
        time: int = 100, 
        n: int = 12,
    ):
        ls = np.arange(12, dtype='int')
        leads_to_modify = np.random.choice(ls, size=n, replace=False)
        super().__init__(time, leads_to_modify)
        self.time = time
        self.n = n
        
    def __call__(self, x):

        return self.func(x, time=self.time, leads=self.leads)
    
    
class SumAug:
    """
    Apply sum augmentation to selected leads
    :param leads: leads to be replaced by sum of all leads

    :return: preprocessed data
    """
    
    def __init__(
        self,
        leads: list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        self.leads = leads
        self.func = F.sum_augmentation
        
    def __call__(self, x):
        
        return self.func(x, leads=self.leads)
    
    
class RandomSumAug(SumAug):
    """
    Apply random sum augmentation
    :param n: number of leads to be replaced by sum of all leads (chosen randomly)

    :return: preprocessed data
    """
    
    def __init__(
        self,
        n: int = 11,
    ):
        ls = np.arange(12, dtype='int')
        leads_to_remove = np.random.choice(ls, size=n, replace=False)
        super().__init__(leads_to_remove)
        self.n = n
        
    def __call__(self, x):
        
        return self.func(x, leads=self.leads)
    
    
class ReflectAug:
    """
    Apply reflection augmentation

    :return: preprocessed data
    """
    
    def __init__(
        self,
    ):
        self.func = F.reflect_augmentation
        
    def __call__(self, x):
        
        return self.func(x)
    
    
class ConvexAug:
    """
    Apply convex augmentation
    :param leads: leads to be replaced by convex combination of some leads (chosen randomly)

    :return: preprocessed data
    """
    
    def __init__(
        self,
        leads: list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        self.leads = leads
        self.func = F.convex_augmentation
        
    def __call__(self, x):
        
        return self.func(x, leads=self.leads)
    
    
class RandomConvexAug(ConvexAug):
    """
    Apply random convex augmentation
    :param n: number of leads (chosen randomly) to be replaced by convex combination of some leads (chosen randomly)

    :return: preprocessed data
    """
    
    def __init__(
        self,
        n: int = 11,
    ):
        ls = np.arange(12, dtype='int')
        leads_to_remove = np.random.choice(ls, size=n, replace=False)
        super().__init__(leads_to_remove)
        self.n = n
        
    def __call__(self, x):
        
        return self.func(x, leads=self.leads)