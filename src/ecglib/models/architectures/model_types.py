from enum import IntEnum

__all__ = ["MType"]


class MType(IntEnum):
    RESNET = 0
    DENSENET = 1
    TABULAR = 2
    CNN = 3
    SSSD = 4
    OTHER = 5  # use to sign custom models

    @staticmethod
    def from_string(label: str) -> IntEnum:
        label = label.lower()
        if "resnet" in label:
            return MType.RESNET
        elif "densenet" in label:
            return MType.DENSENET
        elif "tabular" in label:
            return MType.TABULAR
        elif "cnn1d" in label:
            return MType.CNN
        elif "sssd" in label:
            return MType.SSSD
        elif "other" in label:
            return MType.OTHER
        else:
            raise ValueError
