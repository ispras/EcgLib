# EcgLib

## Table of contents

- [Introduction](#datasets)
- [Datasets](#datasets)
- [Models](#models)
- [Preprocessing](#preprocessing)
- [ToDo](#todo)
- [Credits](#credits)

### Introduction

**Ecg** **lib**rary (`ecglib`) is a tool for ECG signal analysis. The library helps with preprocessing ECG signals, downloading the datasets, creating Dataset classes to train models to detect ECG pathologies. The library allows researchers to use model architectures pretrained on more than 500,000 ECG records to fine-tune them on their own datasets.

### Datasets
This module allows user to load and store ECG datasets in different formats and to extract meta-information about each single ECG signal (i.e. frequency, full path to file, scp-codes, patient age etc.). 

Via `load_datasets.py` one can download [PTB-XL ECG database](https://physionet.org/content/ptb-xl/1.0.2/) in its original *wfdb* format and to store information concerning each record in a *csv* file.

```python
# download PTB-XL ECG dataset

from ecglib.datasets import load_ptb_xl

ptb_xl_info = load_ptb_xl(download=True)
```

`datasets.py` script contains classes for storing ECG datasets.
- *ECGDataset* is a general class for storing main features of your ECG dataset such as number of leads, number of classes to predict, augmentation etc..
- *PTBXLDataset* is a child class with respect to *ECGDataset*; one can load each record from *wfdb* or *npz* format and preprocess it before further utilization. It is also possible to create a *png* picture of each record using [ecg-plot](https://pypi.org/project/ecg-plot/). 

```python
# create PTBXLDataset class from PTB-XL map file
# fit targets for 'AFIB' binary classification

from ecglib.datasets import PTBXLDataset 

targets = [[0.0] if 'AFIB' in eval(ptb_xl_info.iloc[i]['scp_codes']).keys() else [1.0] 
           for i in range(ptb_xl_info.shape[0])]
ecg_data = PTBXLDataset(ecg_data=ptb_xl_info, target=targets)
```

### Models
This module comprises components of model architectures and open weights for models derived from binary classification experiments in several pathologies.

`create_model` function allows user to create a model from scratch (currently supported architectures include *densenet1d121*, *densenet1d201*) as well as load a pretrained model checkpoint from `weights` folder (currently supported architectures include *densenet1d121*).

```python
# create 'densenet1d121' model from scratch for binary classification 12-lead experiment

from ecglib.models import create_model

model = create_model(model_name='densenet1d121', pathology='1AVB', pretrained=False, leads_num=12)
```

```python
# load pretrained 'densenet1d121' model from 'weights' folder for binary classification 12-lead experiment

from ecglib.models import create_model

model = create_model(model_name='densenet1d121', pathology='AFIB', pretrained=True, leads_num=12)
```

`architectures` folder includes model architectures.

`config` folder contains default parameter dataclasses for building a model. 

In `weights` folder one can find file with paths to the models derived from the following binary classification 12-lead experiments. Currently avaliable pathologies (scp-codes): *AFIB*, *1AVB*, *STACH*, *SBRAD*, *RBBB*, *LBBB*, *PVC*, *LVH*.

### Preprocessing
This module includes framework inspired by [Albumentations](https://albumentations.ai/) Python library for preprocessing and augmenting ECG data.

`composition.py` script contains *SomeOf*, *OneOf* and *Compose* structures for building your own preprocessing and augmentation pipeline.

`preprocess.py` and `functional.py` both comprise classes and functions respectively describing different preprocessing and augmentation techniques. For more information see code commentary.

```python
# augmentation example
import torch
from ecglib.preprocessing.preprocess import *
from ecglib.preprocessing.composition import *

# provide an ecg-record in `numpy.ndarray` form
ecg_record = read_any_ECG_ndarray_type

augmented_record = Compose(transforms=[
    SumAug(leads=[0, 6, 11]), 
    RandomConvexAug(n=4), 
    OneOf(transforms=[ButterworthFilter(), IIRNotchFilter()], transform_prob=[0.8, 0.2])
], p=0.5)(ecg_record)
```

### ToDo
**Next release in December 2022**
- **Datasets**: add support for more data formats and datasets. Change TisDataset/PTBXLDataset to remove duplicates
- **Models**: add more model architectures and weights of these models for different pathologies
- **Preprocessing**: add class ECGrecord and update preprocessing methods
- Add possibility to use metadata for analysis
- Add complex segmentation methods

### Credits
This project is made possible by:

- [Aram Avetisyan](https://github.com/avetisyanaram) (a.a.avetisyan@gmail.com)
- [Olga Mashkova](https://github.com/omashkova)
- [Vladislav Ananev](https://github.com/Survial53)
- [Shahane Tigranyan](https://github.com/decoder-99)
- [Ariana Asatryan](https://github.com/arianasatryan)
- [Sergey Skorik](https://github.com/Skorik99)
- [Yury Markin](https://github.com/grandkarabas)
