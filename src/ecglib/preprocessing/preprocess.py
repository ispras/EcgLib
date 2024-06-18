from typing import Union

import numpy as np

from . import functional as F
import ecglib


__all__ = [
    "FrequencyResample",
    "Padding",
    "EdgeCut",
    "Normalization",
    "ButterworthFilter",
    "IIRNotchFilter",
    "EllipticFilter",
    "BaselineWanderRemoval",
    "WaveletTransform",
    "LeadNull",
    "RandomLeadNull",
    "TimeNull",
    "RandomTimeNull",
    "TimeCrop",
    "SumAug",
    "RandomSumAug",
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
            raise ValueError("ecg_frequency must be scalar")
        if isinstance(requested_frequency, (int, float)):
            self.requested_frequency = requested_frequency
        else:
            raise ValueError("requested_frequency must be scalar")
        self.func = F.frequency_resample

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.func(
                x.signal, int(x.frequency), int(self.requested_frequency)
            )
            x.preprocessing_info.append(
                f"applied FrequencyResample to change the frequency from {x.frequency} to {self.requested_frequency}"
            )
            x.frequency = self.requested_frequency
        else:
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

    def apply_pad(self, x, frequency):
        if self.observed_ecg_length * frequency - x.shape[1] > 0:
            x = np.pad(
                x,
                ((0, 0), (0, int(self.observed_ecg_length * frequency - x.shape[1]))),
                mode=self.pad_mode,
            )
        else:
            x = x[:, : int(self.observed_ecg_length * frequency)]
        return x

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_pad(x.signal, x.frequency)
            x.duration = self.observed_ecg_length
            x.preprocessing_info.append(
                f"applied Padding with a length of {self.observed_ecg_length}"
            )
        else:
            x = self.apply_pad(x, self.frequency)

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

    def apply_edge_cut(self, x, frequency):
        if x.shape[1] / frequency <= sum(self.cut_range):
            raise ValueError(
                f"cut_range must be < length of the input signal ({x.shape[1]/frequency})"
            )

        x = self.func(x, self.cut_range, frequency)

        return x

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_edge_cut(x.signal, x.frequency)
            x.duration = x.signal.shape[1] / x.frequency
            x.preprocessing_info.append(
                f"applied EdgeCut with a cut_range of {self.cut_range}"
            )
        else:
            x = self.apply_edge_cut(x, self.frequency)

        return x


class Normalization:
    """
    Apply normalization
    :param norm_type: type of normalization ('z_norm', 'z_norm_constant_handle', 'min_max' and 'identical')

    :return: preprocessed data
    """

    def __init__(
        self,
        norm_type: str = "z_norm",
    ):
        self.norm_type = norm_type
        if norm_type == "min_max":
            self.func = F.minmax_normalization
        elif norm_type == "z_norm" or norm_type == "z_norm_constant_handle":
            self.func = F.z_normalization
        elif norm_type == "identical":
            self.func = F.identical_nomralization
        else:
            raise ValueError(
                "norm_type must be one of [min_max, z_norm, z_norm_constant_handle, identical]"
            )

    def apply_normalization(self, x):
        if self.norm_type is not None:
            if self.norm_type == "z_norm_constant_handle":
                return self.func(x, handle_constant_axis=True)
            return self.func(x)
        else:
            return x

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_normalization(x.signal)
            x.preprocessing_info.append(
                f"applied Normalization with norm_type {self.norm_type}"
            )
        else:
            x = self.apply_normalization(x)
        return x


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
        Wn: Union[float, int, list] = [3, 30],
        fs: int = 500,
    ):
        self.leads = leads
        self.filter_type = filter_type
        self.func = F.butterworth_filter
        if filter_type not in ["bandpass", "lowpass", "highpass"]:
            raise ValueError("Filter type must be one of [bandpass, lowpass, highpass]")
        self.n = n
        if filter_type == "bandpass" and not isinstance(Wn, list):
            raise ValueError("Wn must be list type in case of bandpass filter")
        elif (filter_type == "highpass" or filter_type == "lowpass") and not isinstance(
            Wn, (int, float)
        ):
            raise ValueError(f"Wn must be a scalar in case of {filter_type} filter")
        self.Wn = Wn
        self.fs = fs

    def apply_butterworth(self, x, frequency):
        if self.leads is None:
            self.leads = np.arange(x.shape[0])

        return self.func(
            x,
            leads=self.leads,
            btype=self.filter_type,
            n=self.n,
            Wn=self.Wn,
            fs=frequency,
        )

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_butterworth(x.signal, x.frequency)
            x.preprocessing_info.append(
                f"applied Butterworth filter with filter_type {self.filter_type} "
                f"on leads {self.leads} with parameters  Wn={self.Wn} and n={self.n}"
            )
        else:
            x = self.apply_butterworth(x, self.fs)

        return x


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
        self.leads = leads
        self.w0 = w0
        self.Q = Q
        self.fs = fs
        self.func = F.IIR_notch_filter

    def apply_iirnotch(self, x, frequency):
        if self.leads is None:
            self.leads = np.arange(x.shape[0])

        return self.func(x, leads=self.leads, w0=self.w0, Q=self.Q, fs=frequency)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_iirnotch(x.signal, x.frequency)
            x.preprocessing_info.append(
                f"applied IIRNotch filter "
                f"on leads {self.leads} with parameters w0={self.w0} and Q={self.Q}"
            )
        else:
            x = self.apply_iirnotch(x, self.fs)

        return x


class EllipticFilter:
    """
    Apply elliptic filter augmentation
    :param filter_type: type of elliptic filter ('bandpass', 'lowpass' or 'highpass')
    :param leads: leads to be filtered
    :param n: filter order
    :param rp: maximum ripple allowed below unity gain in the passband
    :param rs: minimum attenuation required in the stop band
    :param Wn: cutoff frequency(ies)
    :param fs: filtered signal frequency

    :return: preprocessed data
    """

    def __init__(
        self,
        filter_type: str = "bandpass",
        leads: list = None,
        n: int = 10,
        rp: float = 4,
        rs: float = 5,
        Wn: Union[float, int, list] = [0.5, 50],
        fs: int = 500,
    ):
        self.leads = leads
        self.filter_type = filter_type
        self.func = F.elliptic_filter
        if filter_type not in ["bandpass", "lowpass", "highpass"]:
            raise ValueError("Filter type must be one of [bandpass, lowpass, highpass]")
        self.n = n
        if filter_type == "bandpass" and not isinstance(Wn, list):
            raise ValueError("Wn must be list type in case of bandpass filter")
        elif (filter_type == "highpass" or filter_type == "lowpass") and not isinstance(
            Wn, (int, float)
        ):
            raise ValueError(f"Wn must be a scalar in case of {filter_type} filter")
        self.rp = rp
        self.rs = rs
        self.Wn = Wn
        self.fs = fs

    def apply_elliptic(self, x, frequency):
        if self.leads is None:
            self.leads = np.arange(x.shape[0])

        return self.func(
            x,
            leads=self.leads,
            btype=self.filter_type,
            n=self.n,
            rp=self.rp,
            rs=self.rs,
            Wn=self.Wn,
            fs=frequency,
        )

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_elliptic(x.signal, x.frequency)
            x.preprocessing_info.append(
                f"applied Elliptic filter with filter_type {self.filter_type}"
                f"on leads {self.leads} with parameters n={self.n}, rp={self.rp}, rs={self.rs}, Wn={self.Wn}"
            )
        else:
            x = self.apply_elliptic(x, self.fs)

        return x


class BaselineWanderRemoval:
    """
    Remove baseline wander using wavelets
    (see article https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.6789&rep=rep1&type=pdf)
    :param leads: leads to be processed
    :param wavelet: wavelet name

    :return: preprocessed data
    """

    def __init__(
        self,
        leads: list = None,
        wavelet: str = "db4",
    ):
        self.leads = leads
        self.wavelet = wavelet
        self.func = F.DWT_BW

    def apply_bas_wander(self, x):
        if self.leads is None:
            self.leads = np.arange(x.shape[0])
        for lead in self.leads:
            func_result = self.func(x[lead, :], wavelet=self.wavelet)
            if len(func_result) >= x.shape[1]:
                x[lead, :] = func_result[: x.shape[1]]
            else:
                x[lead, :] = np.pad(func_result, (0, x.shape[1] - len(func_result)))
        return x

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_bas_wander(x.signal)
            x.preprocessing_info.append(
                f"applied BaselineWanderRemoval filter with wavelet {self.wavelet} on leads {self.leads}"
            )
        else:
            x = self.apply_bas_wander(x)

        return x


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
        wavelet: str = "db4",
        level: int = 3,
        threshold: float = 2,
        low: float = 1e6,
    ):
        self.leads = leads
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
            raise ValueError("wt_type must be one of [DWT, SWT]")

    def apply_wavelet_transform(self, x):
        if self.leads is None:
            self.leads = np.arange(x.shape[0])
        for lead in self.leads:
            if self.wt_type == "DWT":
                func_result = self.func(
                    x[lead, :],
                    wavelet=self.wavelet,
                    level=self.level,
                    threshold=self.threshold,
                    low=self.low,
                )
            elif self.wt_type == "SWT":
                func_result = self.func(
                    x[lead, :], wavelet=self.wavelet, level=self.level
                )
            if len(func_result) >= x.shape[1]:
                x[lead, :] = func_result[: x.shape[1]]
            else:
                x[lead, :] = np.pad(func_result, (0, x.shape[1] - len(func_result)))
        return x

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_wavelet_transform(x.signal)
            x.preprocessing_info.append(
                f"applied WaveletTransform filter with wavelet {self.wavelet}, level {self.level},"
                f"and wt_type {self.wt_type} on leads {self.leads}"
            )
        else:
            x = self.apply_wavelet_transform(x)

        return x


class LeadNull:
    """
    Apply lead null augmentation
    :param leads: leads to be nulled

    :return: preprocessed data
    """

    def __init__(
        self,
        leads: list = None,
    ):
        self.leads = leads
        self.func = F.lead_null

    def apply_lead_null(self, x):
        if self.leads is None:
            self.leads = [0]
        return self.func(x, leads=self.leads)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_lead_null(x.signal)
            x.preprocessing_info.append(
                f"applied LeadNull filter on leads {self.leads}"
            )
        else:
            x = self.apply_lead_null(x)

        return x


class RandomLeadNull(LeadNull):
    """
    Apply random lead null augmentation
    :param leads: leads to be potentially nulled
    :param n: number of leads to be nulled (chosen randomly)

    :return: preprocessed data
    """

    def __init__(
        self,
        leads: list = None,
        n: int = None,
    ):
        super().__init__(leads=leads)
        self.n = n

    def apply_random_lead_null(self, x):
        if isinstance(self.leads, list):
            if self.n is None:
                self.n = 1
            if isinstance(self.n, int) and self.n > len(self.leads):
                raise ValueError(f"n must be <= {len(self.leads)}")
            leads_to_null = list(
                np.random.choice(self.leads, size=self.n, replace=False)
            )
            self.leads = leads_to_null
        else:
            self.leads = np.arange(x.shape[0])
            if self.n is None:
                self.n = 1
            leads_to_null = list(
                np.random.choice(self.leads, size=self.n, replace=False)
            )
            self.leads = leads_to_null

        return self.func(x, leads=self.leads)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_random_lead_null(x.signal)
            x.preprocessing_info.append(
                f"applied RandomLeadNull filter for {self.n} leads from the leads {self.leads}"
            )
        else:
            x = self.apply_random_lead_null(x)

        return x


class TimeNull:
    """
    Apply time null augmentation
    :param time: length of time segment to be nulled (the same units as signal)
    :param leads: leads to be nulled

    :return: preprocessed data
    """

    def __init__(
        self,
        time: int = 100,
        leads: list = None,
    ):
        self.time = time
        self.leads = leads
        self.func = F.time_null

    def apply_time_null(self, x):
        if self.leads is None:
            self.leads = np.arange(x.shape[0])
        return self.func(x, time=self.time, leads=self.leads)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_time_null(x.signal)
            x.preprocessing_info.append(
                f"applied TimeNull filter with a size of {self.time} from the leads {self.leads}"
            )
        else:
            x = self.apply_time_null(x)

        return x


class RandomTimeNull(TimeNull):
    """
    Apply random time null augmentation
    :param time: length of time segment to be nulled (the same units as signal)
    :param leads: leads to be potentially nulled
    :param n: number of leads to be nulled (chosen randomly)

    :return: preprocessed data
    """

    def __init__(
        self,
        time: int = 100,
        leads: list = None,
        n: int = None,
    ):
        super().__init__(time=time, leads=leads)
        self.n = n

    def apply_random_time_null(self, x):
        if isinstance(self.leads, list):
            if self.n is None:
                self.n = len(self.leads)
            if isinstance(self.n, int) and self.n > len(self.leads):
                raise ValueError(f"n must be <= {len(self.leads)}")
            leads_to_null = list(
                np.random.choice(self.leads, size=self.n, replace=False)
            )
            self.leads = leads_to_null
        else:
            self.leads = np.arange(x.shape[0])
            if self.n is None:
                self.n = len(self.leads)
            else:
                leads_to_null = list(
                    np.random.choice(self.leads, size=self.n, replace=False)
                )
                self.leads = leads_to_null

        return self.func(x, time=self.time, leads=self.leads)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_random_time_null(x.signal)
            x.preprocessing_info.append(
                f"applied RandomTimeNull filter with a size of {self.time} to {self.n} leads "
                f"from the leads {self.leads}"
            )
        else:
            x = self.apply_random_time_null(x)

        return x
    

class TimeCrop:
    """
    Apply time crop augmentation
    :param time: length of time segment to be cropped (the same units as signal)

    :return: preprocessed data
    """
    def __init__(
        self,
        time: int = 100,
    ):
        self.time = time
        self.func = F.time_crop

    def apply_time_crop(self, x):
        return self.func(x, time=self.time)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_time_crop(x.signal)
            x.preprocessing_info.append(
                f"applied TimeCrop filter with a size of {self.time}"
            )
        else:
            x = self.apply_time_crop(x)

        return x


class SumAug:
    """
    Apply sum augmentation to selected leads
    :param leads: leads to be replaced by sum of all leads

    :return: preprocessed data
    """

    def __init__(
        self,
        leads: list = None,
    ):
        self.leads = leads
        self.func = F.sum_augmentation

    def apply_sum_aug(self, x):
        if self.leads is None:
            self.leads = np.arange(x.shape[0])
        return self.func(x, leads=self.leads)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_sum_aug(x.signal)
            x.preprocessing_info.append(
                f"applied SumAug filter on the leads {self.leads}"
            )
        else:
            x = self.apply_sum_aug(x)

        return x


class RandomSumAug(SumAug):
    """
    Apply random sum augmentation
    :param leads: leads to be potentially modified
    :param n: number of leads to be replaced by sum of all leads (chosen randomly)

    :return: preprocessed data
    """

    def __init__(
        self,
        leads: list = None,
        n: int = None,
    ):
        super().__init__(leads=leads)
        self.n = n

    def apply_random_sum_aug(self, x):
        if isinstance(self.leads, list):
            if self.n is None:
                self.n = len(self.leads)
            if isinstance(self.n, int) and self.n > len(self.leads):
                raise ValueError(f"n must be <= {len(self.leads)}")
            leads_to_sum = list(
                np.random.choice(self.leads, size=self.n, replace=False)
            )
            self.leads = leads_to_sum
        else:
            self.leads = np.arange(x.shape[0])
            if self.n is None:
                self.n = len(self.leads)
            else:
                leads_to_sum = list(
                    np.random.choice(self.leads, size=self.n, replace=False)
                )
                self.leads = leads_to_sum

        return self.func(x, leads=self.leads)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_random_sum_aug(x.signal)
            x.preprocessing_info.append(
                f"applied RandomSumAug filter to {self.n} leads from the leads {self.leads}"
            )
        else:
            x = self.apply_random_sum_aug(x)

        return x


class ConvexAug:
    """
    Apply convex augmentation
    :param leads: leads to be replaced by convex combination of some leads (chosen randomly)

    :return: preprocessed data
    """

    def __init__(
        self,
        leads: list = None,
    ):
        self.leads = leads
        self.func = F.convex_augmentation

    def apply_convex_aug(self, x):
        if self.leads is None:
            self.leads = np.arange(x.shape[0])
        return self.func(x, leads=self.leads)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_convex_aug(x.signal)
            x.preprocessing_info.append(
                f"applied ConvexAug filter on the leads {self.leads}"
            )
        else:
            x = self.apply_convex_aug(x)

        return x


class RandomConvexAug(ConvexAug):
    """
    Apply random convex augmentation
    :param leads: leads to be returned
    :param n: number of leads (chosen randomly) to be replaced by convex combination of some leads (chosen randomly)

    :return: preprocessed data
    """

    def __init__(
        self,
        leads: list = None,
        n: int = None,
    ):
        super().__init__(leads=leads)
        self.n = n

    def apply_random_convex_aug(self, x):
        if isinstance(self.leads, list):
            if self.n is None:
                self.n = len(self.leads)
            if isinstance(self.n, int) and self.n > len(self.leads):
                raise ValueError(f"n must be <= {len(self.leads)}")
            leads_to_convex = list(
                np.random.choice(self.leads, size=self.n, replace=False)
            )
            self.leads = leads_to_convex
        else:
            self.leads = np.arange(x.shape[0])
            if self.n is None:
                self.n = len(self.leads)
            else:
                leads_to_convex = list(
                    np.random.choice(self.leads, size=self.n, replace=False)
                )
                self.leads = leads_to_convex

        return self.func(x, leads=self.leads)

    def __call__(self, x):
        if isinstance(x, ecglib.data.ecg_record.EcgRecord):
            x.signal = self.apply_random_convex_aug(x.signal)
            x.preprocessing_info.append(
                f"applied RandomConvexAug filter to {self.n} leads from the leads {self.leads}"
            )
        else:
            x = self.apply_random_convex_aug(x)

        return x
