from dataclasses import dataclass, field
from typing import Optional, Dict

import ecg_plot
import numpy as np
import torch

from ..preprocessing import preprocess


@dataclass(repr=True, eq=True)
class EcgRecord:
    """
    Class that describe ECG record
    :param signal: ECG signal
    :param frequency: ECG record frequency
    :param name: ECG name
    :param lead_order: order of ECG leads in signal
    :param duration: ECG record length
    :param leads_num: number of leads
    :param patient_id: patient_id
    :param ecg_segment_info: location of ECG record peaks, intervals and segments
    :param ecg_metadata: ECG signal metadata
    :param patient_metadata: patient's metadata
    :param annotation_info: annotation of the record
    :param preprocessing_info: a list of preprocessing techniques applied to the signal
    """

    signal: np.ndarray
    frequency: int
    name: str = "ecg_record"
    lead_order: list = field(
        default_factory=lambda: [
            "I",
            "II",
            "III",
            "AVR",
            "AVL",
            "AVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
    )
    duration: float = 0
    leads_num: Optional[int] = 12
    patient_id: Optional[str] = ""
    ecg_segment_info: dict = field(default_factory=lambda: {})
    ecg_metadata: Optional[Dict[str, float]] = field(default_factory=lambda: {})
    patient_metadata: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "age": None,
            "sex": None,
            "weight": None,
            "height": None,
        }
    )
    annotation_info: Optional[Dict[str, list]] = field(default_factory=lambda: {})
    preprocessing_info: list = field(default_factory=lambda: [])

    def __post_init__(self):
        self.duration = len(self.signal[0]) / self.frequency
        self.leads_num = len(self.signal)

    def ecg_plot(self, path="./", save_img=False):
        ecg = self.signal / np.max(
            self.signal
        )
        ecg_plot.plot(ecg, sample_rate=self.frequency)
        if save_img:
            ecg_plot.save_as_png(path + self.name)

    def to_tensor(self):
        ecg_tensor = torch.tensor(self.signal, dtype=torch.float)
        patient_tensor_values = []
        for meta in self.patient_metadata.values():
            if meta is not None:
                patient_tensor_values += [float(param) for param in meta]
            else:
                patient_tensor_values += [None]
        ecg_metadata_tensor_values = []
        for meta in self.ecg_metadata.values():
            if meta is not None:
                ecg_metadata_tensor_values += [float(param) for param in meta]
            else:
                ecg_metadata_tensor_values += [None]
        patient_tensor_values = np.array(patient_tensor_values, dtype=float)
        np.nan_to_num(patient_tensor_values, copy=False)
        patient_tensor = torch.tensor(patient_tensor_values)
        ecg_metadata_tensor_values = np.array(ecg_metadata_tensor_values, dtype=float)
        np.nan_to_num(ecg_metadata_tensor_values, copy=False)
        ecg_metadata_tensor = torch.tensor(ecg_metadata_tensor_values)
        return list([ecg_tensor, patient_tensor, ecg_metadata_tensor])

    def frequency_resample(self, requested_frequency=500):
        self.signal = preprocess.FrequencyResample(
            ecg_frequency=self.frequency, requested_frequency=requested_frequency
        )(self.signal)
        self.frequency = requested_frequency
        self.preprocessing_info.append(
            f"changed frequency from {self.frequency} to {requested_frequency}"
        )

    def cut_ecg(self, cut_range=[0, 0]):
        self.signal = preprocess.EdgeCut(cut_range=cut_range, frequency=self.frequency)(
            self.signal
        )
        self.duration = self.duration - cut_range[0] - cut_range[1]
        self.preprocessing_info.append(
            f"cut ecg record {cut_range[0]} seconds from the beginning"
            "and {cut_range[1]} seconds from the end leaving {self.duration} seconds"
        )

    def get_fixed_length(self, requested_length=10):
        self.signal = preprocess.Padding(
            observed_ecg_length=requested_length, frequency=self.frequency
        )(self.signal)
        self.duration = requested_length
        self.preprocessing_info.append(
            f"changed length of ecg record from {self.duration} seconds"
            "to {requested_length} seconds"
        )

    def normalize(self, norm_type="z_norm"):
        self.signal = preprocess.Normalization(norm_type=norm_type)(self.signal)
        self.preprocessing_info.append(f"applied {norm_type} normalization")

    def remove_baselinewander(self, wavelet="db4"):
        self.signal = preprocess.BaselineWanderRemoval(wavelet=wavelet)(self.signal)
        self.preprocessing_info.append(
            f"removed baseline wander with wavelet {wavelet}"
        )
