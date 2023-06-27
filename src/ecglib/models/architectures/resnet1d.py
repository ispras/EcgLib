import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.layers import LinBnDrop, AdaptiveConcatPool1d

"Copyright [resnet1d] [hsd1503] Licensed under the Apache License, Version 2.0 (the «License»);"


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    :param in_channels: number of input channels
    :param out_channels: number of hidden chanels
    :param kernel_size: size of kernel filters
    :param stride: filter stride
    :param bias: flag to use bias in convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(MyConv1dPadSame, self).__init__()
        # print("In channels", in_channels)
        # print("out_channels", out_channels)
        # print("kernel_sie", kernel_size)
        # print("stride", stride)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=bias,
        )

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = torch.div(
            in_dim + self.stride - 1, self.stride, rounding_mode="floor"
        )
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = torch.div(p, 2, rounding_mode="floor")
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    :param kernel_size: size of pooler filters
    :param stride: pooler stride
    """

    def __init__(self, kernel_size, stride=1):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=self.kernel_size, stride=self.stride
        )

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = torch.div(
            in_dim + self.stride - 1, self.stride, rounding_mode="floor"
        )
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = torch.div(p, 2, rounding_mode="floor")
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock1d(nn.Module):
    """
    ResNet1d Basic Block

    :param in_channels: number of input channels
    :param out_channels: number of chanels on output of the block
    :param kernel_size: size of kernel filters
    :param stride: filter stride for the first layer in block
    :param drop_prob: probability in dropout layers
    :param downsample: nn.Module layers to downsample the input if input_size != output_size
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        drop_prob=0.0,
        downsample=None,
    ):
        super(BasicBlock1d, self).__init__()

        self.downsample = downsample

        # the first conv
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=drop_prob)

        # the second conv
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        identity = x

        # the first conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.do1(out)

        # the second conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.do2(out)

        if self.downsample:
            identity = self.downsample(identity)

        # shortcut
        out = out + identity

        return out


class BottleneckBlock1d(nn.Module):
    """
    ResNet1d Bottleneck Block

    :param in_channels: number of input channels
    :param out_channels: number of chanels on output of the block
    :param kernel_size: size of kernel filters for the second layer in block
    :param stride: filter stride for the second layer in block
    :param drop_prob: probability in dropout layers
    :param downsample: nn.Module layers to downsample the input if input_size != output_size
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        drop_prob=0.0,
        downsample=None,
    ):
        super(BottleneckBlock1d, self).__init__()

        self.downsample = downsample

        # the first conv
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=drop_prob)

        # the second conv
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=drop_prob)

        # the third conv
        self.conv3 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(out_channels * 4)
        self.relu3 = nn.ReLU()
        self.do3 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        identity = x

        # the first conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.do1(out)

        # the second conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.do2(out)

        # the third conv
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.do3(out)

        if self.downsample:
            identity = self.downsample(identity)

        # shortcut
        out = out + identity

        return out


class ResNet1d(nn.Module):
    """
    Residual Network for 1d data

    :param block_type: type of block in backbone. must be one of: \"BasicBlock1d\", \"BottleneckBlock1d\"
    :param layers: list of numbers of repeating blocks in backbone
    :param input_channels: number of input channels
    :param base_filters: output channels for the first block in backbone
    :param kernel_size: size of kernel filters in backbone 
    :param stride: stride in conv1d for the first block in each group of blocks in backbone
    :param num_classes: number of prediction classes
    :param drop_prob: probability in dropout layers in backbone
    :param fix_feature_dim: flag to save `base_filters` value for groups of blocks. Otherwise base_filters * 2
    :param kernel_size_stem: size of kernel filters in stem 
    :param stride: stride in conv1d in stem
    :param pooling_stem: flag to use MaxPool1d in stem
    :param concat_pooling: flag to use AdaptiveConcatPool1d in head. Otherwise use AdaptiveMaxPool1d
    :param hidden_layers_head: list of num of hidden neurons in each layer of head
    :param dropout_prob_head: probability in dropout layers in head
    :param act_head: type of activation layer in head. act_head must be one of: \"relu\", \"elu\". 
    """

    def __init__(
        self,
        block_type,
        layers,
        input_channels,
        base_filters,
        kernel_size,
        stride,
        num_classes,
        dropout_prob,
        fix_feature_dim,
        kernel_size_stem,
        stride_stem,
        pooling_stem,
        concat_pooling,
        hidden_layers_head,
        dropout_prob_head,
        act_head,
        bn_head,
        bn_final_head,
    ):
        super(ResNet1d, self).__init__()

        self.stem = None
        self.backbone = None
        self.head = None
        self.base_filters = base_filters
        self.in_channels = input_channels

        # block type correctness
        if block_type.__name__ == BasicBlock1d.__name__:
            self.block_str = block_type.__name__
            self.block_expansion = 1
        elif block_type.__name__ == BottleneckBlock1d.__name__:
            self.block_str = block_type.__name__
            self.block_expansion = 4
        else:
            raise ValueError(
                'block type must be one of: "BasicBlock1d", "BottleneckBlock1d"'
            )

        # kernel size correctness
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(len(layers))]
        else:
            assert isinstance(kernel_size, list)
            assert len(kernel_size) == len(layers)

        # kernel size stem correctness
        kernel_size_stem = (
            kernel_size[0] if kernel_size_stem is None else kernel_size_stem
        )

        # stem
        self.stem = nn.Sequential(
            MyConv1dPadSame(
                in_channels=self.in_channels,
                out_channels=self.base_filters,
                kernel_size=kernel_size_stem,
                stride=stride_stem,
                bias=False,
            ),
            nn.BatchNorm1d(self.base_filters),
            nn.ReLU(),
            MyMaxPool1dPadSame(kernel_size=3, stride=2) if pooling_stem else None,
        )

        # backbone
        self.backbone = nn.Sequential()

        for i, num_blocks in enumerate(layers):

            if not self.backbone:
                self.in_channels = self.base_filters
            else:
                self.in_channels = self.base_filters * self.block_expansion
                self.base_filters = (
                    self.base_filters if fix_feature_dim else 2 * self.base_filters
                )

            tmp_block = self.create_block(
                block_type=block_type,
                num_blocks=num_blocks,
                stride=stride,
                kernel_size=kernel_size[i],
                dropout_prob=dropout_prob,
            )

            self.backbone.add_module(f"{self.block_str}_block{i}", tmp_block)

        # head
        self.head = nn.Sequential()

        pooling_adapter = nn.Sequential(
            AdaptiveConcatPool1d(size=1) if concat_pooling else nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.head.add_module("pooling_adapter_head", pooling_adapter)

        input_shape_head = self.in_channels * 2 if concat_pooling else self.in_channels
        if hidden_layers_head is None:
            hidden_layers_head = []
        if act_head == "relu":
            act = nn.ReLU()
        elif act_head == "elu":
            act = nn.ELU()
        else:
            raise ValueError('act_head must be one of: "relu", "elu"')

        for i, hidden_layer in enumerate(hidden_layers_head):
            self.head.add_module(
                f"lin_bn_drop_head_{i}",
                LinBnDrop(
                    input_shape_head,
                    hidden_layer,
                    bn=bn_head,
                    p=dropout_prob_head,
                    act=act,
                ),
            )
            input_shape_head = hidden_layer

        self.head.add_module(
            f"lin_bn_drop_head_final",
            LinBnDrop(input_shape_head, num_classes, bn=bn_final_head, p=dropout_prob_head),
        )

    def create_block(self, block_type, num_blocks, stride, kernel_size, dropout_prob):
        downsample = None

        # dotted skip connection conditions
        if stride != 1:
            downsample = nn.Sequential(
                MyConv1dPadSame(
                    self.in_channels,
                    self.base_filters * self.block_expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(self.base_filters * self.block_expansion),
            )

        layers = nn.Sequential()
        layers.add_module(
            f"{self.block_str}_layer0",
            block_type(
                self.in_channels,
                self.base_filters,
                kernel_size,
                stride,
                dropout_prob,
                downsample,
            ),
        )
        self.in_channels = self.base_filters * self.block_expansion
        for i in range(1, num_blocks):
            layers.add_module(
                f"{self.block_str}_layer{i}",
                block_type(
                    self.in_channels,
                    self.base_filters,
                    kernel_size=3,
                    stride=1,
                    drop_prob=dropout_prob,
                ),
            )

        return layers

    def get_cnn(self):
        """
        method for metadata adaptation

        return: tuple of
        nn.Seqiential of stem and backbone
        number of hidden channels after backbone
        """
        return (nn.Sequential(self.stem, self.backbone), self.in_channels)

    def forward(self, x):
        y = self.stem(x)
        y = self.backbone(y)
        y = self.head(y)
        return y


def resnet1d18(**kwargs):
    """Constructs a ResNet-18 model."""
    kwargs["block_type"] = BasicBlock1d
    kwargs["layers"] = [2, 2, 2, 2]
    return ResNet1d(**kwargs)


def resnet1d34(**kwargs):
    """Constructs a ResNet-34 model."""
    kwargs["block_type"] = BasicBlock1d
    kwargs["layers"] = [3, 4, 6, 3]
    return ResNet1d(**kwargs)


def resnet1d50(**kwargs):
    """Constructs a ResNet-50 model."""
    kwargs["block_type"] = BottleneckBlock1d
    kwargs["layers"] = [3, 4, 6, 3]
    return ResNet1d(**kwargs)


def resnet1d101(**kwargs):
    """Constructs a ResNet-101 model."""
    kwargs["block_type"] = BottleneckBlock1d
    kwargs["layers"] = [3, 4, 23, 3]
    return ResNet1d(**kwargs)


def resnet1d152(**kwargs):
    """Constructs a ResNet-152 model."""
    kwargs["block_type"] = BottleneckBlock1d
    kwargs["layers"] = [3, 8, 36, 3]
    return ResNet1d(**kwargs)


if __name__ == "__main__":
    input_channels = 12
    num_classes = 8
    inp = torch.randn(4, 12, 1000)
    model = ResNet1d(
        block_type=BottleneckBlock1d,
        layers=[3, 4, 6, 3],
        input_channels=input_channels,
        base_filters=64,
        kernel_size=3,
        stride=2,
        num_classes=num_classes,
        dropout_prob=0.0,
        fix_feature_dim=True,
        kernel_size_stem=3,
        stride_stem=2,
        pooling_stem=True,
        concat_pooling=False,
        hidden_layers_head=[512],
        dropout_prob_head=0.5,
        act_head="relu",
        bn_head=True,
        bn_final_head=False
    )
    # print(model)
    print(model(inp))
    backbone, out_features = model.get_cnn()
    print(out_features)