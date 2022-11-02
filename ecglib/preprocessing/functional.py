import copy

import numpy as np
import pywt
from scipy import signal
from scipy.stats import zscore


__all__ = [
    "butterworth_bandpass_filter",
    "butterworth_highpass_filter",
    "butterworth_lowpass_filter",
    "IIR_notch_filter",
    "elliptic_bandpass_filter",
    "minmax_normalization",
    "z_normalization",
    "cycle_normalization",
    "DWT_filter",
    "SWT_filter",
    "lead_crop",
    "time_crop",
    "sum_augmentation",
    "convex_augmentation",
    "reflect_augmentation",
    "ecg_to_one_frequency",
    "DWT_BW",
    "cut_ecg",
]


def butterworth_bandpass_filter(
    s: np.ndarray, 
    n: int = 10, 
    Wn: list = [3, 30], 
    fs: float = 500,
) -> np.ndarray:
    """
    Butterworth bandpass filter augmentation
    :param s: one lead signal
    :param n: filter order
    :param Wn: cutoff frequencies
    :param fs: filtered signal frequency

    :return: preprocessed data
    """
    
    sos = signal.butter(N=n, btype='bandpass', Wn=Wn, fs=fs, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, s)
    return filtered_signal


def butterworth_highpass_filter(
    s: np.ndarray, 
    n: int = 7, 
    Wn: float = 0.5, 
    fs: int = 500,
) -> np.ndarray:
    """
    Butterworth highpass filter augmentation
    :param s: one lead signal
    :param n: filter order
    :param Wn: cutoff frequency
    :param fs: filtered signal frequency

    :return: preprocessed data
    """
    
    sos = signal.butter(N=n, btype='highpass', Wn=Wn, fs=fs, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, s)
    return filtered_signal


def butterworth_lowpass_filter(
    s: np.ndarray, 
    n: int = 6, 
    Wn: float = 20, 
    fs: int = 500,
) -> np.ndarray:
    """
    Butterworth lowpass filter augmentation
    :param s: one lead signal
    :param n: filter order
    :param Wn: cutoff frequency
    :param fs: filtered signal frequency

    :return: preprocessed data
    """
    
    sos = signal.butter(N=n, btype='lowpass', Wn=Wn, fs=fs, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, s)
    return filtered_signal


def IIR_notch_filter(
    s: np.ndarray, 
    w0: float = 50, 
    Q: float = 30, 
    fs: int = 500,
) -> np.ndarray:
    """
    IIR notch filter augmentation
    :param s: one lead signal
    :param w0: frequency to remove from a signal
    :param Q: quality factor
    :param fs: sampling frequency of a signal

    :return: preprocessed data
    """
    
    b, a = signal.iirnotch(w0=w0, Q=Q, fs=fs)
    filtered_signal = signal.filtfilt(b, a, s)
    return filtered_signal


def elliptic_bandpass_filter(
    s: np.ndarray, 
    n: int = 10, 
    rp: float = 4, 
    rs: float = 5, 
    Wn: list = [0.5, 50], 
    fs: int = 500,
) -> np.ndarray:
    """
    Elliptic bandpass filter
    :param s: one lead signal
    :param n: filter order
    :param rp: maximum ripple allowed below unity gain in the passband
    :param rs: minimum attenuation required in the stop band
    :param Wn: cutoff frequencies
    :param fs: filtered signal frequency
    
    :return: preprocessed data
    """
    
    sos = signal.ellip(N=n, btype='bandpass', rp=rp, rs=rs, Wn=Wn, fs=fs, output='sos')
    filtered_signal = signal.sosfiltfilt(sos, s)
    return filtered_signal


def minmax_normalization(
    s: np.ndarray,
) -> np.ndarray:
    """
    minmax_normalization
    :param s: signal

    :return: preprocessed data
    """
    
    return (s-np.min(s))/(np.max(s)-np.min(s))


def z_normalization(
    s: np.ndarray,
) -> np.ndarray:
    """
    Z-normalization
    :param s: signal

    :return: preprocessed data
    """
    
    return zscore(s, axis=1, nan_policy='raise')


def cycle_normalization(
    s: np.ndarray,
) -> np.ndarray:
    """
    Cycle normalization
    :param s: one lead signal

    :return: preprocessed data
    """

    smin = np.min(s)
    smax = np.max(s)
    n = len(s) - 1
    i = np.arange(len(s))
    return ((n - i) / n) * ((s - smin) / (smax - smin)) + (i / n) * ((s - smin) / (smax - smin))


def DWT_filter(
    s: np.ndarray, 
    wavelet: str = 'db4', 
    level: int = 3, 
    threshold: float = 2, 
    low: float = 1e6,
) -> np.ndarray:
    """
    Discrete wavelet transform augmentation
    :param s: one lead signal
    :param wavelet: wavelet name
    :param level: decomposition level
    :param threshold: thresholding value for all coefficients except the first one
    :param low: thresholding value for the first coefficient

    :return: preprocessed data
    """
    
    w = pywt.Wavelet(wavelet)
    maxlev = pywt.dwt_max_level(len(s), w.dec_len)

    assert maxlev >= level

    coeffs = pywt.wavedec(s, w, level=level)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * np.sqrt(np.log2(len(coeffs[i]))), mode='soft')
    coeffs[0] = pywt.threshold(coeffs[0], low, mode='less')
    return pywt.waverec(coeffs, wavelet, mode='periodic')


def SWT_filter(
    s: np.ndarray, 
    wavelet: str = 'db4', 
    level: int = 6,
) -> np.ndarray:
    """
    Stationary wavelet transform augmentation
    :param s: one lead signal
    :param wavelet: wavelet name
    :param level: decomposition level

    :return: preprocessed data
    """
    
    if len(s) % 2 == 0:
        width = (2**int(np.ceil(np.log2(len(s)))) - len(s)) // 2
        s_padded = np.pad(s, pad_width=width, mode='symmetric')
    else:
        width1 = (2**int(np.ceil(np.log2(len(s)))) - len(s)) // 2
        width2 = (2**int(np.ceil(np.log2(len(s)))) - len(s)) // 2 + 1
        s_padded = np.pad(s, pad_width=(width1, width2), mode='symmetric')
    
    w = pywt.Wavelet(wavelet)
    maxlev = pywt.swt_max_level(len(s_padded))

    assert maxlev >= level

    coeffs = pywt.swt(s_padded, w, level=level, trim_approx=True, norm=True)
    return pywt.iswt(coeffs, wavelet, norm=True)


def lead_crop(
    record: np.ndarray, 
    leads: list,
) -> np.ndarray:
    """
    Lead crop augmentation
    :param record: signal
    :param leads: leads to be cropped

    :return: preprocessed data
    """
    
    record[leads, :] = np.zeros(record.shape[1])
    return record


def time_crop(
    record: np.ndarray, 
    time: int, 
    leads: list,
) -> np.ndarray:
    """
    Time crop augmentation
    :param record: signal
    :param time: length of time segment to be cropped (the same units as signal)
    :param leads: leads to be cropped

    :return: preprocessed data
    """
    
    assert time <= record.shape[1]
    for lead in leads:
        ls = np.arange(0, len(record[lead])-time, dtype="int")
        start = np.random.choice(ls)
        record[lead, start:start+time] = 0
    return record


def sum_augmentation(
    record: np.ndarray,
    leads: list,
) -> np.ndarray:
    """
    Signal summation augmentation
    
    :param record: signal
    :param leads: leads to be replaced by sum of all leads

    :return: preprocessed data
    """
    
    record[leads, :] = np.sum(record, axis=0)
    return record


def reflect_augmentation(
    record: np.ndarray,
) -> np.ndarray:
    """
    Reflection augmentation
    
    :param record: signal

    :return: preprocessed data
    :rtype: numpy 2d array
    """
    
    return -record


def convex_augmentation(
    record: np.ndarray,
    leads: list,
) -> np.ndarray:
    """
    Convex augmentation
    
    :param record: signal
    :param leads: leads to be replaced by convex combination of some leads (chosen randomly)

    :return: preprocessed data
    """
    
    result = copy.deepcopy(record)
    ls = np.arange(12, dtype='int')
    for ltr in leads:
        ln = np.random.randint(12, size=1)[0]
        leads_to_sum = np.random.choice(ls, size=ln, replace=False)
        convex_coeffs = np.random.dirichlet(np.ones(ln), size=1)[0]
        if ln != 0:
            result[ltr, :] = np.dot(convex_coeffs, record[leads_to_sum, :])
    return result


def ecg_to_one_frequency(
    ecg_record: np.ndarray, 
    ecg_frequency: int, 
    requested_frequency: int,
) -> np.ndarray:
    """
    Frequency resample
    :param record: signal
    :param ecg_frequency: sampling frequency of a signal
    :param requested_frequency: sampling frequency of a preprocessed signal

    :return: preprocessed data
    """
    
    if ecg_frequency == requested_frequency:
        return ecg_record
    ecg_record = signal.resample(ecg_record, int(ecg_record.shape[1] * requested_frequency / ecg_frequency), axis=1)
    return ecg_record


def DWT_BW(
    s: np.ndarray, 
    wavelet: str = 'db4',
)-> np.ndarray:
    """
    Remove baseline wander using wavelets (see article https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.6789&rep=rep1&type=pdf)
    :param s: one lead signal
    :param wavelet: wavelet name

    :return: preprocessed data
    """
    
    w = pywt.Wavelet(wavelet)
    maxlev = pywt.dwt_max_level(len(s), w.dec_len)
    
    diffs = []
    for i in range(1, maxlev + 1):
        coeffs = pywt.wavedec(s, w, level=i, mode='periodic')
        diff = np.sum(np.square(coeffs[0])) - np.sum([np.sum(np.square(coeffs[j])) for j in range(1, i)])
        diffs.append(diff)
    diffs = np.array(diffs)
    if np.max(diffs) > 6500:
        ixs = np.where(diffs > 6500)[0]
        ix = ixs[np.argmin(diffs[ixs])]
        coeffs = pywt.wavedec(s, w, level=ix, mode='periodic')
        coeffs[0] = np.array([0]* len(coeffs[0]))
    else:
        ix = np.argmin(diffs[-3:])
        ix += len(diffs) - 3
        coeffs = pywt.wavedec(s, w, level=ix, mode='periodic')
        coeffs[0] = np.array([0]* len(coeffs[0]))
    return pywt.waverec(coeffs, wavelet, mode='periodic')


def cut_ecg(
    data: np.ndarray, 
    cut_range: list, 
    frequency: int,
) -> np.ndarray:
    """
    Cut signal edges 
    :param data: signal
    :param cut_range: cutting parameters
    :param frequency: sampling frequency of a signal

    :return: preprocessed data
    """
    
    cut_data = []
    start = int(cut_range[0]*frequency)
    for rec in data:
        end = -int(cut_range[1]*frequency) if cut_range[1] != 0 else len(rec)
        cut_data.append(rec[start: end])

    return np.array(cut_data)