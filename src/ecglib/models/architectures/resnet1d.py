import torch.nn as nn
from fastai.layers import AdaptiveConcatPool1d, LinBnDrop
from fastcore.basics import listify


class ResidualBlock1d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, out_ftrs, stride=1, kernel_size=3, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv1d(inplanes, out_ftrs, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ftrs)

        self.conv2 = nn.Conv1d(
            out_ftrs,
            out_ftrs,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_ftrs)

        self.conv3 = nn.Conv1d(out_ftrs, out_ftrs * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ftrs * 4)

        self.out_features = out_ftrs * 4

        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):
    """Paper: https://arxiv.org/pdf/1512.03385.pdf"""

    def __init__(
        self,
        block,
        layers,
        kernel_size=3,
        num_classes=2,
        input_channels=3,
        inplanes=64,
        fix_feature_dim=False,
        kernel_size_stem=None,
        stride_stem=2,
        pooling_stem=True,
        stride=2,
        lin_ftrs_head=None,
        ps_head=0.5,
        bn_final_head=False,
        bn_head=True,
        act_head="relu",
        concat_pooling=True,
    ):
        super(ResNet1d, self).__init__()

        self.stem = None
        self.backbone = None
        self.pooling_adapter = None
        self.head = None
        self.inplanes = inplanes

        self.kernel_size_stem = (
            kernel_size if kernel_size_stem is None else kernel_size_stem
        )

        # stem
        self.stem = self._make_stem(
            in_channels=input_channels,
            inplanes=inplanes,
            kernel_size=self.kernel_size_stem,
            stride=stride_stem,
            pooling=pooling_stem,
        )
        # backbone
        self.backbone = self._make_backbone(
            inplanes=inplanes,
            bb_layers=layers,
            bb_block=block,
            feature_dim=fix_feature_dim,
            bb_kernel_size=kernel_size,
            bb_stride=stride,
        )
        # head
        head_ftrs = (
            inplanes if fix_feature_dim else (2 ** len(layers) * inplanes)
        ) * block.expansion
        self.head = self._make_head(
            n_features=head_ftrs,
            n_classes=num_classes,
            lin_ftrs=lin_ftrs_head,
            ps=ps_head,
            bn_final=bn_final_head,
            bn=bn_head,
            act=act_head,
            concat_pooling=concat_pooling,
        )

    def _make_stem(self, in_channels, inplanes, kernel_size, stride, pooling):
        stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                inplanes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if pooling else None,
        )
        return stem

    def _make_backbone(
        self, bb_layers, bb_block, inplanes, feature_dim, bb_kernel_size, bb_stride
    ):
        backbone = nn.Sequential()

        for i, blocks in enumerate(bb_layers):
            if not backbone:
                out_ftrs = inplanes
            else:
                out_ftrs = inplanes if feature_dim else (2**i) * inplanes

            backbone.add_module(
                "hidden_block{}".format(i),
                self._make_block(
                    bb_block=bb_block,
                    out_ftrs=out_ftrs,
                    blocks=bb_layers[i],
                    stride=bb_stride,
                    kernel_size=bb_kernel_size,
                ),
            )

        return backbone

    def _make_block(self, bb_block, out_ftrs, blocks, stride=1, kernel_size=3):
        downsample = None
        if stride != 1 or self.inplanes != out_ftrs * bb_block.expansion:
            downsample = self._perform_downsample(bb_block, out_ftrs, stride)

        block_layers = nn.Sequential()
        block_layers.add_module(
            f"{bb_block.__name__}_layer0",
            bb_block(self.inplanes, out_ftrs, stride, kernel_size, downsample),
        )

        self.inplanes = out_ftrs * bb_block.expansion

        for i in range(1, blocks):
            block_layers.add_module(
                f"{bb_block.__name__}_layer{i}", bb_block(self.inplanes, out_ftrs)
            )

        return block_layers

    def _perform_downsample(self, block, out_ftrs, stride):
        downsample = nn.Sequential()

        downsample.add_module(
            f"{block.__name__}_downsample",
            nn.Conv1d(
                self.inplanes,
                out_ftrs * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
        )
        downsample.add_module(
            f"{block.__name__}_normalize",
            nn.BatchNorm1d(out_ftrs * block.expansion),
        )
        return downsample

    def _make_head(
        self,
        n_features,
        n_classes,
        lin_ftrs=None,
        ps=0.5,
        bn_final=False,
        bn=True,
        act="relu",
        concat_pooling=True,
    ):
        lin_ftrs = (
            [n_features if concat_pooling else n_features, n_classes]
            if lin_ftrs is None
            else [2 * n_features if concat_pooling else n_features]
            + lin_ftrs
            + [n_classes]
        )

        probs = listify(ps)
        if len(probs) == 1:
            probs = [probs[0] / 2] * (len(lin_ftrs) - 2) + probs

        actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (
            len(lin_ftrs) - 2
        ) + [None]

        pooling_adapter = nn.Sequential(
            AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2),
            nn.Flatten(),
        )
        layers = nn.Sequential()
        layers.add_module("pooling_adapter", pooling_adapter)

        for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], probs, actns):
            layers.add_module("lin_bn_drop", LinBnDrop(ni, no, bn, p, actn))

        if bn_final:
            layers.add_module("bn_final", nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))

        return layers

    def get_cnn(self):
        return (
            nn.Sequential(self.stem, self.backbone),
            self.backbone[-1][-1].out_features,
        )

    def forward(self, x):
        y = self.stem(x)
        y = self.backbone(y)
        y = self.head(y)
        return y


def resnet1d18(**kwargs):
    kwargs["block"] = ResidualBlock1d
    kwargs["layers"] = [1, 2, 2, 1]
    return ResNet1d(**kwargs)


def resnet1d50(**kwargs):
    kwargs["block"] = ResidualBlock1d
    kwargs["layers"] = [3, 4, 6, 3]
    return ResNet1d(**kwargs)


def resnet1d101(**kwargs):
    kwargs["block"] = ResidualBlock1d
    kwargs["layers"] = [3, 4, 23, 3]
    return ResNet1d(**kwargs)
