# EcgLib

## Table of contents

- [Introduction](#introduction)
- [Credits](#credits)
- [Installation](#installation)
- [Data](#data)
- [Models](#models)
- [Preprocessing](#preprocessing)
- [Predict](#predict)
- [Generation](#generation)

### Introduction

**Ecg** **lib**rary (`ecglib`) is a tool for ECG signal analysis. The library helps with preprocessing ECG signals, downloading the datasets, creating Dataset classes to train models to detect ECG pathologies and EcgRecord classes to store their records. The library allows researchers to use model architectures pretrained on more than 500,000 ECG records to fine-tune them on their own datasets.

### Credits

If you find this tool useful in your research, please consider citing the paper:

1) **Deep Neural Networks Generalization and Fine-Tuning for 12-lead ECG Classification** - We demonstrate that training deep neural networks on a large dataset and fine-tuning it on a small dataset from another domain outperforms the networks trained only on one of the datasets.

        @article{avetisyan2023deep,
            title={Deep Neural Networks Generalization and Fine-Tuning for 12-lead ECG Classification},
            author={Avetisyan, Aram and Tigranyan, Shahane and Asatryan, Ariana and Mashkova, Olga and Skorik, Sergey and Ananev, Vladislav and Markin, Yury},
            journal={arXiv preprint arXiv:2305.18592},
            year={2023}
        }

### Installation

To install the latest version from PyPI:

```
pip install ecglib
```

### Data
This module allows user to load and store ECG datasets and records in different formats and to extract meta-information about each single ECG signal (i.e. frequency, full path to file, scp-codes, patient age etc.). 

Via `load_datasets.py` one can download [PTB-XL ECG database](https://physionet.org/content/ptb-xl/1.0.2/) in its original *wfdb* format and to store information concerning each record in a *csv* file.

```python
# download PTB-XL ECG dataset

from ecglib.data import load_ptb_xl

ptb_xl_info = load_ptb_xl(download=True)
```

Via `ecg_record.py` one can create class *EcgRecord* to store important information about an ECG record.

```python
# creating EcgRecord class for example file

from ecglib.data import EcgRecord
import wfdb

ecg_signal = wfdb.rdsamp("wfdb_file")[0] # for example 00001_hr from PTB-XL dataset
ecg_record = EcgRecord(signal=ecg_signal.T, frequency=500, patient_id=1)

```

Via `datasets.py` one can create class *EcgDataset* to store ECG datasets. It stores main features of your ECG dataset such as number of leads, number of classes to predict, augmentation etc. It is also possible to plot each record using [ecg-plot](https://pypi.org/project/ecg-plot/).

```python
# create EcgDataset class from 
# fit targets for 'AFIB' binary classification

from ecglib.data import EcgDataset 

targets = [[1.0] if 'AFIB' in eval(ptb_xl_info.iloc[i]['scp_codes']).keys() else [0.0] 
           for i in range(ptb_xl_info.shape[0])]
ecg_data = EcgDataset(ecg_data=ptb_xl_info, target=targets)
```

### Models
This module comprises components of model architectures and open weights for models derived from binary classification experiments in several pathologies.

`create_model` function allows user to create a model from scratch (supported architectures include *resnet1d18*, *resnet1d50*, *resnet1d101*, *densenet1d121*, *densenet1d201*) as well as load a pretrained model checkpoint from `weights` folder (supported architectures include *resnet1d18*, *resnet1d50*, *resnet1d101*, *densenet1d121*). `create_model` also allows to use both ECG record and metadata during training by concating FCN to the network that takes ECG record as an input.

```python
# create 'densenet1d121' model from scratch for binary classification 12-lead experiment

from ecglib.models.model_builder import create_model

model = create_model(model_name='densenet1d121', pathology='1AVB', pretrained=False)

# create 'cnntabular' model with 'densenet1d121' architecture for ECG record and FCN for metadata. Number of input features is set to 5 by default and can be changed by adding config

from ecglib.models.model_builder import Combination
from ecglib.models.config.model_configs import DenseNetConfig, TabularNetConfig

densenet_config = DenseNetConfig()
tabular_config = TabularNetConfig(inp_features=50)
model = create_model(model_name=['densenet1d121', 'tabular'],
                     config=[densenet_config, tabular_config],
                     combine=Combination.CNNTAB,
                     pathology='1AVB',
                     pretrained=False)
```

```python
# load pretrained 'densenet1d121' model from 'weights' folder for binary classification 12-lead experiment

from ecglib.models import create_model

model = create_model(model_name='densenet1d121', pathology='AFIB', pretrained=True)
```

`architectures` folder includes model architectures.

In `weights` folder one can find file with paths to the models derived from the following binary classification 12-lead experiments. Available pathologies (scp-codes): *AFIB*, *1AVB*, *STACH*, *SBRAD*, *IRBBB*, *CRBBB*, *PVC*.

### Preprocessing
This module includes framework inspired by [Albumentations](https://albumentations.ai/) Python library for preprocessing and augmenting ECG data.

`composition.py` script contains *SomeOf*, *OneOf* and *Compose* structures for building your own preprocessing and augmentation pipeline.

`preprocess.py` and `functional.py` both comprise classes and functions respectively describing different preprocessing and augmentation techniques. You can preprocess either numpy data and EcgRecord data. For more information see code commentary.

```python
# augmentation example
import torch
from ecglib.preprocessing.preprocess import *
from ecglib.preprocessing.composition import *

# provide an ecg record in a `numpy.ndarray` form
ecg_signal = wfdb.rdsamp("wfdb_file")[0] # for example 00001_hr from PTB-XL dataset
ecg_record = EcgRecord(signal=ecg_signal.T, frequency=500, patient_id=1)

augmented_record = Compose(transforms=[
    SumAug(leads=[0, 6, 11]), 
    RandomConvexAug(n=4), 
    OneOf(transforms=[ButterworthFilter(), IIRNotchFilter()], transform_prob=[0.8, 0.2])
], p=0.5)(ecg_record) # ecg_signal can be used instead of ecg_record
```

### Predict
This module allows users to test trained model with the architecture from `ecglib`. You can get the prediction for the specific ECG record or the prediction for all the records in the directory. For NPZ-typed records, the `ecg_frequency` parameter should be specified either as an integer for all records in the directory or as a dictionary with record names and their corresponding frequency.

```python
# Predict example
from ecglib.predict import Predict

ecg_signal, ann = wfdb.rdsamp("wfdb_file") # for example 00001_hr from PTB-XL dataset
ecg_frequency = ann["fs"]

predict = Predict(
    weights_path="/path/to/model_weights",
    model_name="densenet1d121",
    pathologies=["AFIB"],
    model_frequency=500,
    device="cuda:0",
    threshold=0.5
)

ecg_prediction = predict.predict(ecg_signal, ecg_frequency, channels_first=False)

result_df_wfdb = predict.predict_directory(directory="path/to/data_to_predict",
                                           file_type="wfdb")

result_df_npz = predict.predict_directory(directory="path/to/data_to_predict",
                                          file_type="npz",
                                          ecg_frequency=1000)
```

### Generation
`ecglib` contains the architecture of the diffusion model `SSSD_ECG_nle`, with which you can obtain synthetic signals. The training and generation pipeline is presented in `notebooks/sssd_ecg_nle.ipynb`. 