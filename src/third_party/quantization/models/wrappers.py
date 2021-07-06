# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Wrappers around on some nn functions, mainly to support empty tensors.

Ideally, add support directly in PyTorch to empty tensors in those functions.

These can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math
from typing import List
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple

# from detectron2.utils.env import TORCH_VERSION

import logging
import_quantization = True
try:
    import numpy as np
    import torch.nn.functional as F
    from .quant import quantization as Quantization
    from .dorefa import RoundSTE
except:
    import_quantization = False


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

        # add for quantization support
        self.quantization = None
        self.quant_activation = None
        self.quant_weight = None
        self.pads = None
        self.force_fp = True

    def convert_to_quantization_version(self, quantization=None, index=-1):
        self.quantization = quantization
        logger = logging.getLogger(__name__ + '.Quantization')
        if self.quantization is not None and import_quantization:
            if index == 0:
                for i in ['proxquant', 'custom-update', 'real_skip']:
                    if i in quantization.keyword:
                        logger.info("warning keyword {} not support".format(i))
            self.pads = tuple(x for x in self.padding for _ in range(2))
            self.quant_activation = Quantization(self.quantization, 'fm', [1, self.in_channels, 1, 1], logger=logger)
            self.quant_weight = Quantization(self.quantization, 'wt', [self.out_channels, self.in_channels, *self.kernel_size], logger=logger)
            self.padding_after_quant = getattr(self.quantization, 'padding_after_quant', False)
            self.quant_activation.update_quantization(index=index)
            self.quant_weight.update_quantization(index=index)
            device = self.weight.device
            self.quant_activation.to(device)
            self.quant_weight.to(device)
            self.force_fp = False

    def update_quantization_parameter(self, **parameters):
        if not self.force_fp:
            feedback = dict()
            def merge_dict(feedback, fd):
                if fd is not None:
                    for k in fd:
                        if k in feedback:
                            if isinstance(fd[k], list) and isinstance(feedback[k], list):
                                feedback[k] = feedback[k] + fd[k]
                        else:
                            feedback[k] = fd[k]
            fd = self.quant_activation.update_quantization(**parameters)
            merge_dict(feedback, fd)
            fd = self.quant_weight.update_quantization(**parameters)
            merge_dict(feedback, fd)
            #fd = self.quant_output.update_quantization(**parameters)
            #merge_dict(feedback, fd)
            return feedback
        else:
            return None

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

            if x.numel() == 0 and TORCH_VERSION <= (1, 4):
                assert not isinstance(
                    self.norm, torch.nn.GroupNorm
                ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
                # When input is empty, we want to return a empty tensor with "correct" shape,
                # So that the following operations will not panic
                # if they check for the shape of the tensor.
                # This computes the height and width of the output tensor
                output_shape = [
                    (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                    for i, p, di, k, s in zip(
                        x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                    )
                ]
                output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
                empty = _NewEmptyTensorOp.apply(x, output_shape)
                if self.training:
                    # This is to make DDP happy.
                    # DDP expects all workers to have gradient w.r.t the same set of parameters.
                    _dummy = sum([x.view(-1)[0] for x in self.parameters()]) * 0.0
                    return empty + _dummy
                else:
                    return empty

        if self.quantization is not None:
            weight = self.quant_weight(self.weight)
            if self.padding_after_quant:
                x = self.quant_activation(x)

            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                x = F.pad(x, expanded_padding, mode='circular')
            else:
                x = F.pad(x, self.pads, 'constant', 0)
            padding = (0, 0)

            if not self.padding_after_quant:
                x = self.quant_activation(x)

            x = F.conv2d(x, weight, self.bias, self.stride, padding, self.dilation, self.groups)
        else:
            x = F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# if TORCH_VERSION > (1, 4):
#     ConvTranspose2d = torch.nn.ConvTranspose2d
# else:

    class ConvTranspose2d(torch.nn.ConvTranspose2d):
        """
        A wrapper around :class:`torch.nn.ConvTranspose2d` to support zero-size tensor.
        """

        def forward(self, x):
            if x.numel() > 0:
                return super(ConvTranspose2d, self).forward(x)
            # get output shape

            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
                for i, p, di, k, d, op in zip(
                    x.shape[-2:],
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                    self.stride,
                    self.output_padding,
                )
            ]
            output_shape = [x.shape[0], self.out_channels] + output_shape
            # This is to make DDP happy.
            # DDP expects all workers to have gradient w.r.t the same set of parameters.
            _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
            return _NewEmptyTensorOp.apply(x, output_shape) + _dummy

class BatchNorm2d(torch.nn.BatchNorm2d):
    """
    A wrapper around :class:`torch.nn.BatchNorm2d` to support zero-size tensor.
    """

    def __init__(self, args, kwargs):
        super().__init__(kwargs)

        # quantization related attributes
        self.enable = True
        self.args = args
        self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info
        self.index = -1
        self.input_scale = 1.0
        self.tag = 'norm'
        self.input_index = ""
        def identify(x):
            return x
        self.quant_functions = { "RoundSTE": RoundSTE.apply, "identify": identify }
        self.choice = 'RoundSTE'

    def convert_norm_to_quantization_version(self, args=None, index=-1):
        self.args = args
        self.update_norm_quantization_parameter(index=index)
        if args is not None:
            assert hasattr(args, 'global_buffer'), "no global_buffer found in quantization args"

    def update_norm_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.verbose('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(v, str):
                                    v = v.replace("'", "").replace('"', '')
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                elif isinstance(getattr(self, k), float):
                                    v = float(v)
                                elif isinstance(getattr(self, k), str):
                                    if k == 'input_index' and 'same' in v:
                                        v = v.replace('same', str(self.index))
                                if not isinstance(getattr(self, k), torch.Tensor):
                                    setattr(self, "{}".format(k), v)
                                    self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))
                                    if k == 'input_index':
                                        for tag in ['fm', 'wt']:
                                            exist = 'Yes' if self.input_index + "-{}".format(tag) in self.args.global_buffer else 'No'
                                            self.verbose("input_index of tag({}) exist in global buffer ? {}".format(tag, exist))
        if self.enable:
            assert self.args is not None, "args should not be None"
            assert hasattr(self.args, 'global_buffer'), "no global_buffer found in quantization args"
            assert self.choice in self.quant_functions, "unknown choice of quant_function {}".format(self.choice)

    def forward(self, x):
        if x.numel() > 0:
            if self.enable:
                scale = self.weight * (self.running_var + self.eps).rsqrt()
                bias = self.bias / scale - self.running_mean
                input_scale = self.input_scale
                if self.input_index + "-fm" in self.args.global_buffer:
                    input_scale = input_scale * self.args.global_buffer[self.input_index + "-fm"].abs().item()
                if self.input_index + "-wt" in self.args.global_buffer:
                    input_scale = input_scale * self.args.global_buffer[self.input_index + "-wt"].abs().item()
                self.args.global_buffer[self.input_index + "-norm"] = input_scale * scale.cpu().detach().numpy()
                bias = self.quant_functions[self.choice](bias / input_scale) * input_scale
                scale = scale.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
                return (x + bias) * scale
            else:
                return super(BatchNorm2d, self).forward(x)
        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)

    def __repr__(self):
        base = super(BatchNorm2d, self).__repr__()
        if self.enable:
            base = base + "-choice({})-index({})-input({})".format(self.choice, self.index, self.input_index)
        return base

class Linear(torch.nn.Linear):
    """
    A wrapper around :class:`torch.nn.Linear` to support empty inputs and more features.
    Because of https://github.com/pytorch/pytorch/issues/34202
    """

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

        # add for quantization support
        self.quantization = None
        self.quant_activation = None
        self.quant_weight = None
        self.force_fp = True

    def convert_to_quantization_version(self, quantization=None, index=-1):
        args = quantization
        logger = logging.getLogger(__name__ + '.Quantization')
        if import_quantization and args is not None and hasattr(args, 'keyword'):
            self.quantization = quantization
            self.force_fp = False
            self.quant_activation = Quantization(self.quantization, 'fm', [1, self.in_features, 1, 1], logger=logger)
            self.quant_weight = Quantization(self.quantization, 'wt', [1, 1, self.in_features, self.out_features], logger=logger)
            self.quant_activation.update_quantization(index=index)
            self.quant_weight.update_quantization(index=index)
            device = self.weight.device
            self.quant_activation.to(device)
            self.quant_weight.to(device)

    def update_quantization_parameter(self, **parameters):
        if not self.force_fp:
            feedback = dict()
            def merge_dict(feedback, fd):
                if fd is not None:
                    for k in fd:
                        if k in feedback:
                            if isinstance(fd[k], list) and isinstance(feedback[k], list):
                                feedback[k] = feedback[k] + fd[k]
                        else:
                            feedback[k] = fd[k]
            fd = self.quant_activation.update_quantization(**parameters)
            merge_dict(feedback, fd)
            fd = self.quant_weight.update_quantization(**parameters)
            merge_dict(feedback, fd)
            return feedback
        else:
            return None

    def forward(self, x):
        if x.numel() == 0:
            output_shape = [x.shape[0], self.weight.shape[0]]

            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        if self.quantization is not None and not self.force_fp:
            shape = self.weight.shape 
            weight = self.quant_weight(self.weight)
            weight = weight.reshape(shape)
            inputs = self.quant_activation(x)

            x = F.linear(inputs, weight, self.bias)
        else:
            x = super().forward(x)

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# if TORCH_VERSION > (1, 4):
#     interpolate = torch.nn.functional.interpolate
# else:

    def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
        """
        A wrapper around :func:`torch.nn.functional.interpolate` to support zero-size tensor.
        """
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners=align_corners
            )

        def _check_size_scale_factor(dim):
            if size is None and scale_factor is None:
                raise ValueError("either size or scale_factor should be defined")
            if size is not None and scale_factor is not None:
                raise ValueError("only one of size or scale_factor should be defined")
            if (
                scale_factor is not None
                and isinstance(scale_factor, tuple)
                and len(scale_factor) != dim
            ):
                raise ValueError(
                    "scale_factor shape must match input shape. "
                    "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
                )

        def _output_size(dim):
            _check_size_scale_factor(dim)
            if size is not None:
                return size
            scale_factors = _ntuple(dim)(scale_factor)
            # math.floor might return float in py2.7
            return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

        output_shape = tuple(_output_size(2))
        output_shape = input.shape[:-2] + output_shape
        return _NewEmptyTensorOp.apply(input, output_shape)


def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if x.dim() == 0:
        return x.unsqueeze(0).nonzero().unbind(1)
    return x.nonzero().unbind(1)

class EltWiseModule(torch.nn.Module):
    def __init__(self, operator='sum', args=None):
        super(EltWiseModule, self).__init__()

        # quantization related attributes
        self.enable = True
        self.args = args
        self.index = -1
        self.tag = 'eltwise'
        self.x_index = []
        self.y_index = []
        self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info

    def convert_eltwise_to_quantization_version(self, args=None, index=-1):
        if args is not None and import_quantization:
            self.args = args
            self.update_eltwise_quantization_parameter(index=index)

    def update_eltwise_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.verbose('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(v, str):
                                    v = v.replace("'", "").replace('"', '')
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                elif isinstance(getattr(self, k), float):
                                    v = float(v)
                                elif isinstance(getattr(self, k), list):
                                    if k in ['x_index', 'y_index']:
                                        v = v.split(',') if ',' in v else v.split(' ')
                                setattr(self, "{}".format(k), v)
                                self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))

        if self.enable:
            assert self.args is not None, "args should not be None"
            assert hasattr(self.args, 'global_buffer'), "no global_buffer found in quantization args"

    def coordinate(self, mark_x=0, mark_y=0):
        if mark_x < len(self.x_index):
            alphaX = self.args.global_buffer[self.x_index[mark_x]]
        else:
            raise RuntimeError("Cannot find mark")
        if mark_y < len(self.y_index):
            alphaY = self.args.global_buffer[self.y_index[mark_y]]
        else:
            raise RuntimeError("Cannot find mark")

        alpha = np.ones_like(alphaX)
        scale = np.ones_like(alphaX)
        for i, j, m, n in zip(alphaX, alphaY, alpha, scale):
            m = j if i >= j else i
            n = i / j if i >= j else j / i
        self.args.global_buffer['alpha-{}-{}'.format(self.index, self.tag)] = alpha

        error = np.ones_like(alphaX)
        shift = np.ones_like(alphaX)
        multi = np.ones_like(alphaX)
        for i in range(16):
            for cur, his, shi in zip(scale, error, shift):
                cur = cur * pow(2.0, i)
                cur = abs(round(cur) - cur)
                if cur < his:
                    shi = i
                    his = cur

        for denominator, numerator, fraction in zip(shift, multi, scale):
            denominator = pow(2.0, denominator)
            numerator = round(fraction * denominator)

        scale = multi / shift/ scale
        scale_x = np.ones_like(alphaX)
        scale_y = np.ones_like(alphaX)
        for x, y, z, m, n in zip(scale_x, scale_y, scale, alphaX, alphaY):
            x = z if m >= n  else 1.0
            y = 1.0 if m >= n  else z

        return scale_x, scale_y

    def coordinate_addition(self, x, y, mark_x=0, mark_y=0):
        scale_x, scale_y = self.coordinate(mark_x, mark_y)
        scale_x = torch.from_numpy(scale_x).to(x.device)
        scale_y = torch.from_numpy(scale_y).to(x.device)
        scale_x = scale_x.reshape(1, -1, 1, 1)
        scale_y = scale_y.reshape(1, -1, 1, 1)
        x = x * scale_x
        y = y * scale_y
        return x, y

    def forward(self, x, y, mark_x=0, mark_y=0):
        if self.enable:
            x, y = self.coordinate_addition(x, y, mark_x, mark_y)
        output = x + y
        return output

    def __repr__(self):
        base = 'EltWiseModule()'
        if self.enable:
            base = base + "-index({})".format(self.index)
        return base


