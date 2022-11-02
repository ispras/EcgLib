import random

import numpy as np


__all__ = [
    "Compose",
    "SomeOf",
    "OneOf",
]


class Compose:
    """
    Apply successively a list of preprocessing techniques (normalization, augmentation etc.)
    :param transforms: a list of preprocessing techniques
    :param p: probability of preprocessing pipeline

    :return: preprocessed data
    """

    def __init__(
        self,
        transforms: list,
        p: float,
    ):
        self.transforms = transforms
        if p > 1.0 or p < 0.0:
            raise ValueError("Probability should be between 0.0 and 1.0")
        else:
            self.p = p

    def __call__(self, x):
        
        idx = np.random.RandomState(random.randint(0, (1 << 32) - 1)).choice(2, p=[self.p, 1.0-self.p])
        if idx == 0:
            for t in self.transforms:
                x = t(x)

        return x


class SomeOf:
    """
    Apply some of preprocessing techniques from a given list (normalization, augmentation etc.)
    :param transforms: a list of preprocessing techniques
    :param n: a number of preprocessing techniques to apply
    :param transform_prob: a list of probabilities

    :return: preprocessed data
    """

    def __init__(
        self,
        transforms: list,
        n: int,
        transform_prob: list = None,
    ):
        self.transforms = transforms
        self.n = n
        if transform_prob is None:
            self.transform_prob = [1 / len(self.transforms)]*len(self.transforms)
        else:
            if sum(transform_prob) > 1.0:
                raise ValueError("Sum of probabilities should be equal to 1.0")
            else:
                self.transform_prob = transform_prob

    def __call__(self, x):
        
        idx = np.random.RandomState(random.randint(0, (1 << 32) - 1)).choice(len(self.transforms), size=self.n, p=self.transform_prob, replace=False)
        for i in idx:
            t = self.transforms[i]
            data = t(x)
            
        return data
    

class OneOf:
    """
    Apply one of preprocessing techniques from a given list (normalization, augmentation etc.)
    :param transforms: a list of preprocessing techniques
    :param transform_prob: a list of probabilities

    :return: preprocessed data
    """
    
    def __init__(
        self,
        transforms: list,
        transform_prob: list = None,
    ):
        self.transforms = transforms
        if transform_prob is None:
            self.transform_prob = [1 / len(self.transforms)]*len(self.transforms)
        else:
            if sum(transform_prob) > 1.0:
                raise ValueError("Sum of probabilities should be equal to 1.0")
            else:
                self.transform_prob = transform_prob

    def __call__(self, x):
        
        idx = np.random.RandomState(random.randint(0, (1 << 32) - 1)).choice(len(self.transforms), p=self.transform_prob)
        t = self.transforms[idx]
        x = t(x)
        
        return x