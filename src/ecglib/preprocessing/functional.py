import copy
from typing import Union

import numpy as np
import pywt
from scipy import signal
from scipy.stats import zscore


__all__ = [
    "frequency_resample",
    "cut_ecg",
    "butterworth_filter",
    "IIR_notch_filter",
    "elliptic_filter",
    "minmax_normalization",
    "z_normalization",
    "DWT_filter",
    "SWT_filter",
    "lead_null",
    "time_null",
    "time_crop",
    "sum_augmentation",
    "convex_augmentation",
    "DWT_BW",
]


def frequency_resample(
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
    ecg_record = signal.resample(
        ecg_record,
        int(ecg_record.shape[1] * requested_frequency / ecg_frequency),
        axis=1,
    )
    return ecg_record


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
    start = int(cut_range[0] * frequency)
    for rec in data:
        end = -int(cut_range[1] * frequency) if cut_range[1] != 0 else len(rec)
        cut_data.append(rec[start:end])

    return np.array(cut_data)


def butterworth_filter(
    s: np.ndarray,
    leads: list,
    btype: str = "bandpass",
    n: int = 10,
    Wn: Union[float, int, list] = [3, 30],
    fs: float = 500,
) -> np.ndarray:
    """
    Butterworth bandpass filter augmentation
    :param s: ECG signal
    :param leads: leads to be filtered
    :param btype: type of Butterworth filter ('bandpass', 'lowpass' or 'highpass')
    :param n: filter order
    :param Wn: cutoff frequency(ies)
    :param fs: filtered signal frequency

    :return: preprocessed data
    """
    if btype == "bandpass" and not isinstance(Wn, list):
        raise ValueError("Wn must be list type in case of bandpass filter")
    elif (btype == "highpass" or btype == "lowpass") and not isinstance(
        Wn, (int, float)
    ):
        raise ValueError(f"Wn must be a scalar in case of {btype} filter")
    sos = signal.butter(N=n, btype=btype, Wn=Wn, fs=fs, output="sos")
    s[leads, :] = signal.sosfiltfilt(sos, s[leads, :])
    return s


def IIR_notch_filter(
    s: np.ndarray,
    leads: list,
    w0: float = 50,
    Q: float = 30,
    fs: int = 500,
) -> np.ndarray:
    """
    IIR notch filter augmentation
    :param s: ECG signal
    :param leads: leads to be filtered
    :param w0: frequency to remove from a signal
    :param Q: quality factor
    :param fs: sampling frequency of a signal

    :return: preprocessed data
    """
    b, a = signal.iirnotch(w0=w0, Q=Q, fs=fs)
    s[leads, :] = signal.filtfilt(b, a, s[leads, :])
    return s


def elliptic_filter(
    s: np.ndarray,
    leads: list,
    btype: str = "bandpass",
    n: int = 10,
    rp: float = 4,
    rs: float = 5,
    Wn: Union[float, int, list] = [0.5, 50],
    fs: int = 500,
) -> np.ndarray:
    """
    Elliptic filter
    :param s: ECG signal
    :param leads: leads to be filtered
    :param btype: type of elliptic filter ('bandpass', 'lowpass' or 'highpass')
    :param n: filter order
    :param rp: maximum ripple allowed below unity gain in the passband
    :param rs: minimum attenuation required in the stop band
    :param Wn: cutoff frequency(ies)
    :param fs: filtered signal frequency

    :return: preprocessed data
    """
    if btype == "bandpass" and not isinstance(Wn, list):
        raise ValueError("Wn must be list type in case of bandpass filter")
    elif (btype == "highpass" or btype == "lowpass") and not isinstance(
        Wn, (int, float)
    ):
        raise ValueError(f"Wn must be a scalar in case of {btype} filter")
    sos = signal.ellip(N=n, btype=btype, rp=rp, rs=rs, Wn=Wn, fs=fs, output="sos")
    s[leads, :] = signal.sosfiltfilt(sos, s[leads, :])
    return s


def minmax_normalization(
    s: np.ndarray,
) -> np.ndarray:
    """
    minmax_normalization
    :param s: signal

    :return: preprocessed data
    """
    smin = np.min(s)
    smax = np.max(s)
    s = (s - smin) / (smax - smin)
    return s


def z_normalization(
    s: np.ndarray,
    handle_constant_axis: bool=False,
) -> np.ndarray:
    """
    Z-normalization
    :param s: signal
    :param handle_constant_axis: Flag indicating whether to handle constant values in the signal.

    :return: preprocessed data
    """
    s_norm = zscore(s, axis=1, nan_policy="raise")
    if handle_constant_axis:
        same_values = np.all(s == s[:, 0][:, np.newaxis], axis=1)
        s_norm[same_values] = 0
    return s_norm

def identical_nomralization(
    s: np.ndarray, 
) -> np.ndarray:
    """function to identical normalization

    Args:
        s (np.ndarray): signal

    Returns:
        np.ndarray: preprocessed (identical) signal
    """
    return s


def DWT_filter(
    s: np.ndarray,
    wavelet: str = "db4",
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
        coeffs[i] = pywt.threshold(
            coeffs[i], threshold * np.sqrt(np.log2(len(coeffs[i]))), mode="soft"
        )
    coeffs[0] = pywt.threshold(coeffs[0], low, mode="less")
    return pywt.waverec(coeffs, wavelet, mode="periodic")


def SWT_filter(
    s: np.ndarray,
    wavelet: str = "db4",
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
        width = (2 ** int(np.ceil(np.log2(len(s)))) - len(s)) // 2
        s_padded = np.pad(s, pad_width=width, mode="symmetric")
    else:
        width1 = (2 ** int(np.ceil(np.log2(len(s)))) - len(s)) // 2
        width2 = (2 ** int(np.ceil(np.log2(len(s)))) - len(s)) // 2 + 1
        s_padded = np.pad(s, pad_width=(width1, width2), mode="symmetric")

    w = pywt.Wavelet(wavelet)
    maxlev = pywt.swt_max_level(len(s_padded))

    assert maxlev >= level

    coeffs = pywt.swt(s_padded, w, level=level, trim_approx=True, norm=True)
    return pywt.iswt(coeffs, wavelet, norm=True)


def lead_null(
    record: np.ndarray,
    leads: list,
) -> np.ndarray:
    """
    Lead nulling augmentation
    :param record: signal
    :param leads: leads to be nulled

    :return: preprocessed data
    """

    record[leads, :] = np.zeros(record.shape[1])
    return record


def time_null(
    record: np.ndarray,
    time: int,
    leads: list,
) -> np.ndarray:
    """
    Time nulling augmentation
    :param record: signal
    :param time: length of time segment to be nulled (the same units as signal)
    :param leads: leads to be nulled

    :return: preprocessed data
    """

    assert time <= record.shape[1]
    for lead in leads:
        ls = np.arange(0, len(record[lead]) - time, dtype="int")
        start = np.random.choice(ls)
        record[lead, start : start + time] = 0
    return record


def time_crop(
    record: np.ndarray,
    time: int,
) -> np.ndarray:
    """
    Time crop augmentation
    :param record: signal
    :param time: length of time segment to be cropped (the same units as signal)
    :param leads: leads to be cropped

    :return: preprocessed data
    """
    assert time <= record.shape[1]
    ls = np.arange(0, len(record[0]) - time, dtype="int")
    start = np.random.choice(ls)
    return record[:, start:start+time]


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
    ls = np.arange(12, dtype="int")
    for ltr in leads:
        ln = np.random.randint(1, 13, size=1)[0]
        leads_to_sum = np.random.choice(ls, size=ln, replace=False)
        convex_coeffs = np.random.dirichlet(np.ones(ln), size=1)[0]
        if ln != 0:
            result[ltr, :] = np.dot(convex_coeffs, record[leads_to_sum, :])
    return result


def DWT_BW(
    s: np.ndarray,
    wavelet: str = "db4",
) -> np.ndarray:
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
        coeffs = pywt.wavedec(s, w, level=i, mode="periodic")
        diff = np.sum(np.square(coeffs[0])) - np.sum(
            [np.sum(np.square(coeffs[j])) for j in range(1, i)]
        )
        diffs.append(diff)
    diffs = np.array(diffs)
    if np.max(diffs) > 6500:
        ixs = np.where(diffs > 6500)[0]
        ix = ixs[np.argmin(diffs[ixs])]
        coeffs = pywt.wavedec(s, w, level=ix, mode="periodic")
        coeffs[0] = np.array([0] * len(coeffs[0]))
    else:
        ix = np.argmin(diffs[-3:])
        ix += len(diffs) - 3
        coeffs = pywt.wavedec(s, w, level=ix, mode="periodic")
        coeffs[0] = np.array([0] * len(coeffs[0]))
    return pywt.waverec(coeffs, wavelet, mode="periodic")
