import os
from typing import List, Union

import pandas as pd
import numpy as np
import torch

from ecglib.models.model_builder import (
    create_model,
    Combination,
)
from ecglib import preprocessing as P
from ecglib.models.config.model_configs import BaseConfig
from ecglib.data.datasets import EcgDataset


def tabular_metadata_handler(
    patient_metadata: torch.Tensor, ecg_metadata: torch.Tensor, axis: int = 1
) -> torch.Tensor:
    """
    Concatenates patient and ECG metadata along a specified axis.

    :param patient_metadata: torch.Tensor, patient metadata
    :param ecg_metadata: torch.Tensor, ECG metadata
    :param axis: int, axis along which the tensors will be concatenated (default is 1)

    :return: torch.Tensor, concatenated patient and ECG metadata
    """
    assert isinstance(patient_metadata, torch.Tensor)
    assert isinstance(ecg_metadata, torch.Tensor)
    return torch.concat((patient_metadata, ecg_metadata), axis)


def get_full_record(
    frequency: int,
    record: str,
    patient_meta: dict,
    ecg_meta: dict,
    normalization: str = "z_norm",
    use_metadata: bool = False,
    preprocess: list = None,
) -> list:
    """
    Returns a full record from raw record, patient metadata, ECG metadata, and configuration.

    :param frequency: int, frequency of the ECG record
    :param record: str, path to ECG record
    :param patient_meta: dict, patient metadata
    :param ecg_meta: dict, ECG metadata
    :param normalization: str, normalization type
    :param use_metadata: bool, whether to use metadata or not
    :param preprocess: list of preprocessing methods applied to ECG record

    :return: list, processed ECG record along with its metadata
    """

    record = record[:,]
    frequency = int(frequency)
    patient_meta = patient_meta
    trained_freq = frequency
    if preprocess:
        record_processed = P.Compose(transforms=preprocess, p=1.0)(record)
    else:
        record_processed = P.Compose(
            transforms=[
                P.FrequencyResample(
                    ecg_frequency=frequency, requested_frequency=trained_freq
                ),
                P.Normalization(norm_type=normalization),
            ],
            p=1.0,
        )(record)

    assert not np.isnan(record_processed).any(), "NaN values in record"

    ecg_tensor = torch.tensor(record_processed, dtype=torch.float)
    if use_metadata:
        patient_meta_list = [float(param) for param in patient_meta.values()]
        ecg_meta_list = [float(param) for param in ecg_meta.values()]
    else:
        patient_meta_list = []
        ecg_meta_list = []
    return [ecg_tensor, patient_meta_list, ecg_meta_list]


class Predict:
    def __init__(
        self,
        weights_path: str,
        model_name: str,
        pathologies: list,
        frequency: int,
        device: str,
        threshold: float,
        leads: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        model: torch.nn.Module = None,
        model_config: Union[BaseConfig, List[BaseConfig]] = None,
        combine: Combination = Combination.SINGLE,
        use_metadata: bool = False,
        use_sigmoid: bool = True,
        normalization: str = "z_norm",
    ):
        """
        Class for making predictions using a trained model.

        :param weights_path: str, path to the model weights
        :param model_name: str, name of the model
        :param pathologies: list, list of pathologies
        :param frequency: int, frequency of the ECG record
        :param device: str, device to be used for computations
        :param threshold: float, threshold for the model
        :param leads: list, list of leads
        :param model: torch.nn.Module, model to be used for predictions
        :param model_config: config with parameters of trained model
        :param combine: Combination to select which type of model to use
        :param use_metadata: bool, whether to use metadata or not
        :param use_sigmoid: bool, whether to apply sigmoid after model output
        :param normalization: str, normalization type
        """

        if not isinstance(pathologies, list):
            pathologies = [pathologies]

        self.leads_num = len(leads)
        self.leads = leads
        self.device = device
        self.frequency = frequency
        self.use_sigmoid = use_sigmoid

        if model is None:
            self.model = create_model(
                model_name=model_name,
                pretrained=True,
                pretrained_path=weights_path,
                pathology=pathologies,
                leads_count=self.leads_num,
                config=model_config,
                combine=combine,
            )
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

        self.handler = None

        self.use_metadata = use_metadata
        self.normalization = normalization

        if use_metadata:
            self.handler = tabular_metadata_handler
        self.threshold = threshold

    def predict(
        self,
        record,
        ecg_meta=None,
        patient_meta=None,
        channels_first=True,
    ):
        """
        Function that evaluates the model on a single ECG record.

        :param record: np.array or torch.tensor, ECG record
        :param ecg_meta: dict, ECG metadata (default is None)
        :param patient_meta: dict, patient metadata (default is None)
        :param channels_first: bool, whether the channels are the first dimension in the input data (default is False)

        :return: dict, predicted probability, raw output, and labels
        """

        if patient_meta is None and ecg_meta is None and self.use_metadata:
            raise ValueError("Patient or ECG metadata is required")
        self.patient_meta = patient_meta
        self.ecg_meta = ecg_meta
        self.record = record
        self.channels_first = channels_first

        if not self.channels_first:
            if isinstance(self.record, torch.Tensor):
                self.record = self.record.permute(1, 0)
            elif isinstance(self.record, np.ndarray):
                self.record = self.record.transpose(1, 0)
            else:
                raise ValueError(
                    "Record type must be either torch.tensor or np.array. Given type: {}".format(
                        type(self.record)
                    )
                )

        input_ = get_full_record(
            self.frequency,
            self.record,
            self.patient_meta,
            self.ecg_meta,
            self.normalization,
            self.use_metadata,
        )

        ecg_signal = input_[0]
        patient_meta = input_[1]
        ecg_meta = input_[2]
        inp = (
            ecg_signal
            if not self.handler
            else [ecg_signal, self.handler(patient_meta, ecg_meta)]
        )

        inp = (
            [item.to(self.device) for item in inp]
            if isinstance(inp, list)
            else inp.to(self.device)
        )
        inp = inp.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(inp)
            if self.use_sigmoid:
                probability = torch.nn.Sigmoid()(outputs)
            else:
                probability = outputs
            label = (torch.nn.Sigmoid()(outputs) > self.threshold).float()

        return {"raw_out": outputs, "prob_out": probability, "label_out": label}

    def predict_directory(
        self,
        directory,
        file_type,
        write_to_file=None,
        ecg_meta=None,
        patient_meta=None,
    ):
        """
        Evaluates the model on all ECG records in a directory.

        :param directory: str, path to the directory with ECG records
        :param file_type: str, file type of the ECG records
        :param write_to_file: str, path to the file where the predictions will be written (default is None), or None if the predictions should not be written to a file
        :param ecg_meta: list of dicts, each dict contains "filename" and "data" keys. ECG metadata (default is None)
        :param patient_meta: list of dicts, each dict contains "filename" and "data" keys. Patient metadata (default is None)

        :return: pd.DataFrame, dataframe with the predictions
        """
        if ecg_meta:
            ecg_meta = sorted(ecg_meta, key=lambda k: k["filename"])
        if patient_meta:
            patient_meta = sorted(patient_meta, key=lambda k: k["filename"])

        all_files = os.listdir(directory)

        # filter by file_type
        if file_type == "wfdb":
            record_files = [file[:-4] for file in all_files if file.endswith(".dat")]
        else:
            record_files = [file for file in all_files if file.endswith(file_type)]
        record_files = sorted(record_files)

        answer_df = pd.DataFrame(
            columns=["filename", "raw_out", "prob_out", "label_out"]
        )

        ecg_meta_counter = 0
        patient_meta_counter = 0
        for record in record_files:
            ecg_meta_ = None
            patient_meta_ = None

            if ecg_meta:
                if ecg_meta[ecg_meta_counter]["filename"] == record:
                    ecg_meta_counter += 1
                    ecg_meta_ = ecg_meta[ecg_meta_counter]["data"]

            if patient_meta:
                if patient_meta[patient_meta_counter]["filename"] == record:
                    patient_meta_counter += 1
                    patient_meta_ = patient_meta[patient_meta_counter]["data"]

            record_ = EcgDataset.read_ecg_record(
                None, os.path.join(directory, record), file_type
            )
            record_answer = self.predict(record_, ecg_meta_, patient_meta_)

            answer_df_current = pd.DataFrame(
                {
                    "filename": record,
                    "raw_out": record_answer["raw_out"].cpu().numpy().item(),
                    "prob_out": record_answer["prob_out"].cpu().numpy().item(),
                    "label_out": record_answer["label_out"].cpu().numpy().item(),
                },
                index=[0],
            )
            answer_df = pd.concat([answer_df_current, answer_df], ignore_index=True)

        if write_to_file:
            answer_df.to_csv(write_to_file)

        return answer_df
