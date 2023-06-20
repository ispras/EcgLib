import torch
from fastai.layers import LinBnDrop
from fastcore.basics import listify


class TabularNet(torch.nn.Module):
    def __init__(
        self,
        inp_features,
        lin_ftrs=[10, 10, 10, 10, 10, 8],
        drop=[0.5],
        act_fn="relu",
        bn_last=True,
        act_last=True,
        drop_last=True,
    ):
        super(TabularNet, self).__init__()

        criterion = (
            torch.nn.ReLU(inplace=True)
            if act_fn == "relu"
            else torch.nn.ELU(inplace=True)
        )
        actns = [criterion] * (len(lin_ftrs) - 2)
        bns = [True] * (len(lin_ftrs) - 2)

        actns.append(criterion) if act_last else actns.append(None)
        bns.append(True) if bn_last else bns.append(False)

        if isinstance(drop, int):
            drop = listify(drop)

        if len(drop) == 1:
            drop_ps = [drop[0] / 2] * (len(lin_ftrs) - 2)
            drop_ps.append(drop[0]) if drop_last else drop_ps.append(0.0)
        elif len(drop) != len(lin_ftrs):
            raise

        self.mlp = torch.nn.Sequential()
        for i, (n_inp, n_out, drop_p, bn, act_fn) in enumerate(
            zip(lin_ftrs[:-1], lin_ftrs[1:], drop_ps, bns, actns)
        ):
            if not self.mlp:
                self.mlp.add_module(
                    "input",
                    LinBnDrop(
                        n_in=inp_features,
                        n_out=n_inp,
                        bn=bn,
                        p=drop_p,
                        act=act_fn,
                        lin_first=True,
                    ),
                )

            self.mlp.add_module(
                "hidden{}".format(i),
                LinBnDrop(
                    n_in=n_inp, n_out=n_out, bn=bn, p=drop_p, act=act_fn, lin_first=True
                ),
            )

        self.out_size = lin_ftrs[-1]

    def forward(self, x):
        return self.mlp(x)


def tabular(**kwargs):
    """Constructs an TabularNet model"""
    return TabularNet(**kwargs)
