
import torch
import torch.nn as nn

from .quant import conv3x3, conv1x1, conv0x0
from .wrappers import BatchNorm2d

def seq_c_b_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    out = relu(out)
    if skip_enable:
        out += skip
    return out

def seq_c_b_s_a(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    if skip_enable:
        out += skip
    out = relu(out)
    return out

def seq_c_a_b_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = relu(out)
    out = bn(out)
    if skip_enable:
        out += skip
    return out

def seq_b_c_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = bn(x)
    out = conv(out)
    out = relu(out)
    if skip_enable:
        out += skip
    return out

def seq_b_a_c_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = bn(x)
    out = relu(out)
    out = conv(out)
    if skip_enable:
        out += skip
    return out

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class StaticBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class ReverseBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__(num_features, eps=eps, affine=False)
        assert affine, "Affine should be True for ReverseBatchNorm2d"
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)

    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        x = x * scale + bias
        x = super(ReverseBatchNorm2d, self).forward(x)
        return x

def norm(channel, args=None, feature_stride=None, affine=False):
    keyword = None
    if args is not None:
        keyword = getattr(args, "keyword", None)

    if keyword is None:
        if args.bn_quant:
            return BatchNorm2d(args, channel)
        else:
            return nn.BatchNorm2d(channel)

    if "group-norm" in keyword:
        group = getattr(args, "fm_quant_group", 2)
        return nn.GroupNorm(group, channel)

    if "static-bn" in keyword:
        return StaticBatchNorm2d(channel)

    if "freeze-bn" in keyword:
        return FrozenBatchNorm2d(channel)

    if "reverse-bn" in keyword:
        return ReverseBatchNorm2d(channel)

    if "instance-norm" in keyword:
        return nn.InstanceNorm2d(channel, affine=affine)

    if args.bn_quant:
        return BatchNorm2d(args, channel)
    else:
        return nn.BatchNorm2d(channel)

class ShiftReLU(nn.ReLU):
    def __init__(self, args):
        super(ShiftReLU, self).__init__(inplace=True)
        self.shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x = x + self.shift
        x = super(ShiftReLU, self).forward(x)
        return x


def actv(args=None, negative_slope=0.01):
    keyword = None
    if args is not None:
        keyword = getattr(args, "keyword", None)

    if keyword is None:
        return nn.ReLU(inplace=True)

    if 'PReLU' in keyword:
        return nn.PReLU()

    if 'NReLU' in keyword:
        return nn.Sequential()

    if 'SReLU' in keyword:
        return ShiftReLU(args)

    if 'ReLU6' in keyword:
        return nn.ReLU6(inplace=True)

    if 'LReLU' in keyword:
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    return nn.ReLU(inplace=True)

# TResNet: High Performance GPU-Dedicated Architecture (https://arxiv.org/pdf/2003.13630v1.pdf)
class TResNetStem(nn.Module):
    def __init__(self, out_channel, in_channel=3, stride=4, kernel_size=1, args=None):
        super(TResNetStem, self).__init__()
        self.stride = stride
        force_fp = True
        if hasattr(args, 'keyword'):
            force_fp = 'real_skip' in args.keyword
        assert kernel_size in [1, 3], "Error reshape conv kernel"
        if kernel_size == 1:
            self.conv = conv1x1(in_channel*stride*stride, out_channel, args=args, force_fp=force_fp)
        elif kernel_size == 3:
            self.conv = conv3x3(in_channel*stride*stride, out_channel, args=args, force_fp=force_fp)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.stride, self.stride, W // self.stride, self.stride)
        x = x.transpose(4, 3).reshape(B, C, 1, H // self.stride, W // self.stride, self.stride * self.stride)
        x = x.transpose(2, 5).reshape(B, C * self.stride * self.stride, H // self.stride, W // self.stride)
        x = self.conv(x)
        return x

class DuplicateModule(nn.Module):
    def __init__(self, hyper_parameter, number):
        super(DuplicateModule, self).__init__()
        instance_type = hyper_parameter.get('type', 'identify')
        if instance_type == 'identify':
            self.duplicates = nn.ModuleList([nn.Sequential() for i in range(number)])
        elif instance_type == 'conv':
            module = hyper_parameter.get('module')
            in_channel = hyper_parameter.get('in_channel')
            out_channel = hyper_parameter.get('out_channel')
            stride = hyper_parameter.get('stride', 1)
            args = hyper_parameter.get('args', None)
            force_fp = hyper_parameter.get('force_fp', True)
            self.duplicates = nn.ModuleList([module(in_channel, out_channel, stride=stride, args=args, force_fp=force_fp) for i in range(number)])

    def forward(self, x):
        result = []
        for model in self.duplicates:
            result.append(model(x))

        if len(result) > 1:
            return torch.cat(result, dim=1)
        else:
            return result[0]

def duplicate(hyper_parameter, number):
    return DuplicateModule(hyper_parameter, number)

class ConcatModule(nn.Module):
    def __init__(self, model_list):
        super(ConcatModule, self).__init__()
        assert isinstance(model_list, nn.ModuleList), "concat nn.ModuleList only"
        self.duplicates = model_list

    def forward(self, x):
        result = []
        for model in self.duplicates:
            result.append(model(x))

        if len(result) > 1:
            return torch.cat(result, dim=1)
        else:
            return result[0]

def concat(model_list):
    return ConcatModule(model_list)

class FlattenModule(nn.Module):
    def __init__(self):
        super(FlattenModule, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

def flatten():
    return FlattenModule()


