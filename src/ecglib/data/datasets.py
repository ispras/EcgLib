from pathlib import Path
from typing import Callable, Optional, Union

import ecg_plot
import numpy as np
import pandas as pd
import torch
import wfdb
from ecglib import preprocessing as P
from .ecg_record import EcgRecord
from torch.utils.data import Dataset


__all__ = [
    "EcgDataset",
]


class EcgDataset(Dataset):
    """
    EcgDataset
    :param ecg_data: dataframe with ecg info
    :param target: a list of targets
    :param frequency: frequency for signal resampling
    :param leads: a list of leads
    :param ecg_length: length of ECG signal after padding / cropping
    :param cut_range: cutting parameters
    :param pad_mode: padding mode
    :param classes: number of classes
    :param use_meta: whether to use metadata or not
    :param augmentation: a bunch of augmentations and other preprocessing techniques
    """

    def __init__(
        self,
        ecg_data: pd.DataFrame,
        target: list,
        frequency: int = 500,
        leads: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        data_type: str = "wfdb",
        ecg_length: Union[int, float] = 10,
        cut_range: list = [0, 0],
        pad_mode: str = "constant",
        norm_type: str = "z_norm",
        classes: int = 2,
        augmentation: Callable = None,
    ):
        super().__init__()
        if "fpath" not in ecg_data.columns:
            raise ValueError("column 'fpath' not in ecg_data")
        self.ecg_data = ecg_data
        self.target = target
        self.frequency = frequency
        self.leads = leads
        self.data_type = data_type
        self.ecg_length = ecg_length
        self.cut_range = cut_range
        self.pad_mode = pad_mode
        self.norm_type = norm_type
        self.classes = classes
        self.augmentation = augmentation

    def __len__(self):
        return self.ecg_data.shape[0]

    def get_fpath(self, index: int) -> str:
        """
        Returns path to file with ECG leads
        :param index: Index of ECG in dataset

        :return: Path to ECG file
        """

        return self.ecg_data.iloc[index]["fpath"]

    def get_name(self, index: int) -> str:
        """
        Returns name of ECG file
        :param index: Index of ECG in dataset

        :return: ECG file name
        """

        return str(Path(self.get_fpath(index)).stem)

    def read_ecg_record(
        self, file_path, data_type, leads=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ):
        if data_type == "npz":
            ecg_record = np.load(file_path)["arr_0"].astype("float64")
            frequency = None
        elif data_type == "wfdb":
            ecg_record, ann = wfdb.rdsamp(file_path, channels=leads)
            ecg_record = ecg_record.T
            ecg_record = ecg_record.astype("float64")
            frequency = ann['fs']
        else:
            raise ValueError(
                'data_type can have only values from the list ["npz", "wfdb"]'
            )
        return ecg_record, frequency
    
    def take_metadata(self, index: int):
        """
        Take metadata and convert them into dictionary

        Args:
            index (int): index of a row in self.ecg_data

        Returns:
            Tuple: (patient_meta, ecg_record_meta) -- tuple with metadata
        """
        patient_meta = (
            self.ecg_data.iloc[index]["patient_metadata"]
            if "patient_metadata" in self.ecg_data.iloc[index]
            else dict()
        )
        ecg_record_meta = (
            self.ecg_data.iloc[index]["ecg_metadata"]
            if "ecg_metadata" in self.ecg_data.iloc[index]
            else dict()
        )
        patient_meta = {
            key: patient_meta[key]
            if isinstance(patient_meta[key], list)
            else [patient_meta[key]]
            for key in patient_meta
        }

        ecg_record_meta = {
            key: ecg_record_meta[key]
            if isinstance(ecg_record_meta[key], list)
            else [ecg_record_meta[key]]
            for key in ecg_record_meta
        }

        return (patient_meta, ecg_record_meta)

    def __getitem__(self, index):
        ecg_frequency = float(self.ecg_data.iloc[index]["frequency"])
        patient_meta, ecg_record_meta = self.take_metadata(index)
        file_path = self.ecg_data.iloc[index]["fpath"]

        # data standartization (scaling, resampling, cuts off, normalization and padding/truncation)
        ecg_record, _ = self.read_ecg_record(file_path, self.data_type, self.leads)
        full_ecg_record_info = EcgRecord(
            signal=ecg_record[self.leads, :],
            frequency=ecg_frequency,
            name=file_path,
            lead_order=self.leads,
            ecg_metadata=ecg_record_meta,
            patient_metadata=patient_meta,
        )

        # data standartization:
        # resampling
        full_ecg_record_info.frequency_resample(requested_frequency=self.frequency)
        # cuts off
        full_ecg_record_info.cut_ecg(self.cut_range)
        # normalization
        full_ecg_record_info.normalize(self.norm_type)
        if self.ecg_length:
            # padding/truncation
            full_ecg_record_info.get_fixed_length(self.ecg_length)

        assert not full_ecg_record_info.check_nans(), f"ecg_record = {full_ecg_record_info.signal}, index = {index}"

        # data preprocessing if specified (augmentation, filtering)
        if self.augmentation is not None:
            full_ecg_record_info = self.augmentation(full_ecg_record_info)

        target = self.target[index]

        result = [
            full_ecg_record_info.to_tensor(),
            torch.tensor(target, dtype=torch.float),
        ]

        return (index, result)

    def save_as_png(
        self, index: int, dest_path: str, postfix: Optional[str] = None
    ) -> None:
        """
        Saves the image of ecg record

        :param index: Index of ECG
        :param dest_path: Directory to save the image
        :param postfix: Subdirectory where the image will be saved, defaults to None
        """

        ecg = (np.load(self.get_fpath(index))["arr_0"].astype("float64"),)
        ecg = np.squeeze(ecg)

        if "frequency" in self.ecg_data.columns:
            frequency = self.ecg_data.iloc[index]["frequency"]
        else:
            frequency = self.frequency
        ecg = ecg / np.max(
            ecg
        )  # added to fit the record to the visible part of the plot
        ecg_plot.plot(ecg, sample_rate=frequency)
        ecg_fname = self.get_name(index)

        if postfix:
            dest_path = str(Path(dest_path).joinpath(postfix))

        dest_path = (
            "{}/".format(dest_path) if not dest_path.endswith("/") else dest_path
        )

        if not Path(dest_path).exists():
            Path(dest_path).mkdir(parents=True, exist_ok=True)

        ecg_plot.save_as_png(file_name=ecg_fname, path=dest_path)

    @staticmethod
    def for_train_from_config(
        data: pd.DataFrame,
        target: list,
        augmentation: Callable,
        config: dict,
        classes_num: int,
    ):
        """
        A wrapper with just four parameters to create `TisDataset` for training and validation
        :param data: dataframe with ecg info
        :param target: a list of targets
        :param augmentation: a bunch of augmentations and other preprocessing techniques
        :param config: config dictionary
        :param classes_num: number of classes

        :return: EcgDataset
        """

        return EcgDataset(
            data,
            target,
            frequency=config.ecg_record_params.resampled_frequency,
            leads=config.ecg_record_params.leads,
            data_type=config.ecg_record_params.data_type,
            ecg_length=config.ecg_record_params.observed_ecg_length,
            norm_type=config.ecg_record_params.normalization,
            classes=classes_num,
            cut_range=config.ecg_record_params.ecg_cut_range,
            augmentation=augmentation,
        )
