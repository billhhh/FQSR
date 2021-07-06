import math

import torch
import torch.nn as nn
import torch.nn.functional as F

### Quantization option begin
quantization_enable = False
try:
    from third_party.quantization.models.quant import conv5x5, conv3x3, conv1x1, conv0x0, eltwise
    from third_party.quantization.models.layers import actv, norm
    quantization_enable = True
    # quantization_enable = False
    print('Import third_party code Success')
except (ImportError, RuntimeError, FileNotFoundError) as e:
    print('Import third_party code Failed in model/common.py', e)

def default_conv(in_channels, out_channels, kernel_size, bias=True, args=None):
    if quantization_enable:
        assert kernel_size in [0, 1, 3, 5], "Unexcept kernel_size %d" % kernel_size
        if kernel_size == 0:
            return conv0x0(in_channels, out_channels, args=args)
        if kernel_size == 1:
            return conv1x1(in_channels, out_channels, bias=bias, args=args)
        if kernel_size == 3:
            return conv3x3(in_channels, out_channels, bias=bias, args=args)
        if kernel_size == 5:
            return conv5x5(in_channels, out_channels, bias=bias, args=args)

    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        args = None
        if hasattr(conv, 'keywords') and 'args' in conv.keywords and quantization_enable:
            args = conv.keywords['args']

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            if args is not None:
                m.append(norm(out_channels, args))
            else:
                m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            if args is not None:
                m.append(actv(args))
            else:
                m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        args = None
        if hasattr(conv, 'keywords') and 'args' in conv.keywords and quantization_enable:
            args = conv.keywords['args']

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                if args is not None:
                    m.append(norm(n_feats, args))
                else:
                    m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                if args is not None:
                    m.append(actv(args))
                else:
                    m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.quantization_version = quantization_enable
        if self.quantization_version:
            self.eltwise = eltwise(channels=n_feats, args=args)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        if self.quantization_version:
            res = self.eltwise(res, x)
        else:
            res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        args = None
        if hasattr(conv, 'keywords') and 'args' in conv.keywords and quantization_enable:
            args = conv.keywords['args']

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    if args is not None:
                        m.append(norm(n_feats, args))
                    else:
                        m.append(nn.BatchNorm2d(n_feats))
                ## whether control ReLU/PReLU here ?
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                if args is not None:
                    m.append(norm(n_feats, args))
                else:
                    m.append(nn.BatchNorm2d(n_feats))
            ## whether control ReLU/PReLU here ?
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

