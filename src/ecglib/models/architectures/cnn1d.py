import torch
import torch.nn as nn


class CNN1d(torch.nn.Module):
    def __init__(
            self,
            inp_channels,
            inp_features,
            cnn_ftrs=[16, 16],
    ):
        super(CNN1d, self).__init__()

        self.out_size = cnn_ftrs[-1]

        self.cnn = torch.nn.Sequential()
        self.cnn.add_module(
            "input", nn.Conv1d(in_channels=inp_channels, out_channels=cnn_ftrs[0], kernel_size=3, stride=1, padding=1),
        )
        for i, (n_inp, n_out) in enumerate(zip(cnn_ftrs[:-1], cnn_ftrs[1:])):
            self.cnn.add_module(
                f"bn{i}", nn.BatchNorm1d(n_inp),
            )
            self.cnn.add_module(
                f"act{i}", nn.ReLU(),
            )
            self.cnn.add_module(
                f"hidden{i}", nn.Conv1d(in_channels=n_inp, out_channels=n_out, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):
        return self.cnn(x)


def cnn1d(**kwargs):
    return CNN1d(**kwargs)
