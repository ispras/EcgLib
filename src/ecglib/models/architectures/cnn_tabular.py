import torch
from fastai.layers import AdaptiveConcatPool1d, LinBnDrop


class CnnTabular(torch.nn.Module):
    def __init__(
        self,
        cnn_backbone,
        cnn_out_features,
        tabular_model,
        tabular_out_features,
        classes,
        head_ftrs=[512],
        head_drop_prob=0.2,
    ):
        super(CnnTabular, self).__init__()

        self.cnn_backbone = cnn_backbone
        self.tabular = tabular_model
        self.head_layers = head_ftrs

        self.head_inp_features = cnn_out_features * 2 + tabular_out_features

        self.cnn_pooling = torch.nn.Sequential(
            AdaptiveConcatPool1d(),
            torch.nn.Flatten(),
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
        y_cnn = self.cnn_backbone(input[0])
        y_cnn = self.cnn_pooling(y_cnn)
        y_tabular = self.tabular(input[1])
        y = torch.cat((y_cnn, y_tabular), dim=-1)
        y = self.head(y)
        return y
