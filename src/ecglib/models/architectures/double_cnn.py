from dataclasses import dataclass, field
from typing import Optional, Type, Union

import torch
import torch.nn as nn
from fastai.layers import AdaptiveConcatPool1d, LinBnDrop


class DoubleCNN(torch.nn.Module):
    def __init__(
        self,
        cnn1_backbone,
        cnn1_out_features,
        cnn2_model,
        cnn2_out_features,
        classes,
        head_ftrs=[512],
        head_drop_prob=0.2,
    ):
        super(DoubleCNN, self).__init__()

        self.cnn1_backbone = cnn1_backbone
        self.cnn2_model = cnn2_model
        self.head_layers = head_ftrs

        self.head_inp_features = cnn1_out_features * 2 + cnn2_out_features * 2

        self.cnn_pooling = torch.nn.Sequential(
            AdaptiveConcatPool1d(), torch.nn.Flatten(),
        )

        self.head = torch.nn.Sequential()
        for i, f in enumerate(self.head_layers):
            if not self.head:
                self.head.add_module(
                    "input",
                    LinBnDrop(
                        n_in=self.head_inp_features,
                        n_out=f,
                        bn=True,
                        p=head_drop_prob,
                        act=torch.nn.ReLU(),
                        lin_first=False,
                    ),
                )

            self.head.add_module(
                "hidden{}".format(i),
                LinBnDrop(
                    n_in=f,
                    n_out=f,
                    bn=True,
                    p=head_drop_prob,
                    act=torch.nn.ReLU(),
                    lin_first=False,
                ),
            )

        self.head.add_module(
            "output",
            LinBnDrop(
                n_in=self.head_layers[-1],
                n_out=classes,
                bn=True,
                p=head_drop_prob,
                act=None,
                lin_first=False,
            ),
        )

    def forward(self, input):
        y_cnn = self.cnn1_backbone(input[0])
        y_cnn = self.cnn_pooling(y_cnn)
        y_cnn_model = self.cnn2_model(input[1])
        y_cnn_model = self.cnn_pooling(y_cnn_model)
        y = torch.cat((y_cnn, y_cnn_model), dim=-1)
        y = self.head(y)
        return y
