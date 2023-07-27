import torch
from torch import nn
from torch.nn.functional import elu
from .modules import Expression, Ensure4d, Ensure3d
from .functions import squeeze_final_output
import numpy as np
from torch import nn
from torch.nn import init
from .util import np_to_th
from .modules import Expression, Ensure4d
from .functions import (
    safe_log, square, transpose_time_to_spat, squeeze_final_output
)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNetv4Binary(nn.Sequential):
    """EEGNet v4 model from Lawhern et al 2018.
    See details in [EEGNet4]_.
    Parameters
    ----------
    in_chans : int
        XXX
    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.
    References
    ----------
    .. [EEGNet4] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2018).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """

    def __init__(
            self,
            in_chans,
            n_classes=1,
            input_window_samples=None,
            final_conv_length="auto",
            pool_mode="mean",
            F1=8,
            D=2,
            F2=16,  # usually set to F1*D (?)
            kernel_length=64,
            third_kernel_size=(8, 4),
            drop_prob=0.25,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.add_module("ensuredims", Ensure4d())
        # b c 0 1
        # now to b 1 0 c
        self.add_module("dimshuffle", Expression(_transpose_to_b_1_c_0))

        self.add_module(
            "conv_temporal",
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_length),
                stride=1,
                bias=False,
                padding=(0, self.kernel_length // 2),
            ),
        )
        self.add_module(
            "bnorm_temporal",
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module(
            "conv_spatial",
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.in_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=self.F1,
                padding=(0, 0),
                ),
        )

        self.add_module(
            "bnorm_1",
            nn.BatchNorm2d(
                self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3
            ),
        )
        self.add_module("elu_1", Expression(elu))

        self.add_module("pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4)))
        self.add_module("drop_1", nn.Dropout(p=self.drop_prob))

        # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7
        self.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, 16),
                stride=1,
                bias=False,
                groups=self.F1 * self.D,
                padding=(0, 16 // 2),
                ),
        )
        self.add_module(
            "conv_separable_point",
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                (1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
                ),
        )

        self.add_module(
            "bnorm_2",
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
        )
        self.add_module("elu_2", Expression(elu))
        self.add_module("pool_2", pool_class(kernel_size=(1, 8), stride=(1, 8)))
        self.add_module("drop_2", nn.Dropout(p=self.drop_prob))

        out = self(
            torch.ones(
                (1, self.in_chans, self.input_window_samples, 1),
                dtype=torch.float32
            )
        )
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.F2,
                self.n_classes,
                (n_out_virtual_chans, self.final_conv_length),
                bias=True,
            ),
        )
        self.add_module("sigmoid", nn.Sigmoid()) #FIXME
        # Transpose back to the the logic of braindecode,
        # so time in third dimension (axis=2)
        self.add_module("permute_back", Expression(_transpose_1_0))
        self.add_module("squeeze", Expression(squeeze_final_output))

        _glorot_weight_zero_bias(self)


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_to_b_1_c(x):
    return x.permute(0, 2, 1)


def _transpose_1(x):
    return x.permute(0, 2, 1)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)
def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.
    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class ShallowFBCSPNetBinary(nn.Sequential):
    """Shallow ConvNet model from Schirrmeister et al 2017.
    Model described in [Schirrmeister2017]_.
    Parameters
    ----------
    in_chans : int
        Number of EEG input channels.
    n_classes: int
        Number of classes to predict (number of output filters of last layer).
    input_window_samples: int | None
        Only used to determine the length of the last convolutional kernel if
        final_conv_length is "auto".
    n_filters_time: int
        Number of temporal filters.
    filter_time_length: int
        Length of the temporal filter.
    n_filters_spat: int
        Number of spatial filters.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    final_conv_length: int | str
        Length of the final convolution layer.
        If set to "auto", input_window_samples must not be None.
    conv_nonlin: callable
        Non-linear function to be used after convolution layers.
    pool_mode: str
        Method to use on pooling layers. "max" or "mean".
    pool_nonlin: callable
        Non-linear function to be used after pooling layers.
    split_first_layer: bool
        Split first layer into temporal and spatial layers (True) or just use temporal (False).
        There would be no non-linearity between the split layers.
    batch_norm: bool
        Whether to use batch normalisation.
    batch_norm_alpha: float
        Momentum for BatchNorm2d.
    drop_prob: float
        Dropout probability.
    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
            self,
            in_chans,
            n_classes=1,
            input_window_samples=None,
            n_filters_time=40,
            filter_time_length=25,
            n_filters_spat=40,
            pool_time_length=75,
            pool_time_stride=15,
            final_conv_length=30,
            conv_nonlin=square,
            pool_mode="mean",
            pool_nonlin=safe_log,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            drop_prob=0.5,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.conv_nonlin = conv_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob

        self.add_module("ensuredims", Ensure4d())
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        if self.split_first_layer:
            self.add_module("dimshuffle", Expression(transpose_time_to_spat))
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                ),
            )
            self.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module(
                "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, momentum=self.batch_norm_alpha, affine=True
                ),
            )
        self.add_module("conv_nonlin_exp", Expression(self.conv_nonlin))
        self.add_module(
            "pool",
            pool_class(
                kernel_size=(self.pool_time_length, 1),
                stride=(self.pool_time_stride, 1),
            ),
        )
        self.add_module("pool_nonlin_exp", Expression(self.pool_nonlin))
        self.add_module("drop", nn.Dropout(p=self.drop_prob))
        self.eval()
        if self.final_conv_length == "auto":
            out = self(
                np_to_th(
                    np.ones(
                        (1, self.in_chans, self.input_window_samples, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        self.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_conv,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )
        self.add_module("sigmoid", nn.Sigmoid())
        self.add_module("squeeze", Expression(squeeze_final_output))

        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)