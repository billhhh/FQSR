import torch
import torch.nn as nn
import math
from functools import partial
import matplotlib.pyplot as plt
from utility import draw_hist

### Quantization option begin
quantization_enable = False
try:
    from third_party.quantization.models.quant import custom_conv, custom_linear
    from third_party.quantization.models.layers import norm, actv
    quantization_enable = True
except (ImportError, RuntimeError, FileNotFoundError) as e:
    print('Import third_party code Failed in %s' % __file__, e)


def make_model(args, parent=False):
    return _NetG(args)


class _Residual_Block(nn.Module):
    def __init__(self, args=None):
        super(_Residual_Block, self).__init__()

        if quantization_enable:
            conv_func = custom_conv
            conv_func = partial(conv_func, args=args)
            normalization = norm
            normalization = partial(normalization, args=args)
            non_linear = actv
            non_linear = partial(non_linear, args=args)
            non_linear = partial(non_linear, negative_slope=0.2)
        else:
            conv_func = nn.Conv2d
            normalization = nn.InstanceNorm2d
            non_linear = nn.LeakyReLU
            non_linear = partial(non_linear, negative_slope=0.2)

        self.conv1 = conv_func(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = normalization(64, affine=True)
        self.relu = non_linear()
        self.conv2 = conv_func(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = normalization(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class _NetG(nn.Module):
    def __init__(self, args):
        super(_NetG, self).__init__()
        if quantization_enable:
            conv_func = custom_conv
            conv_func = partial(conv_func, args=args)
            normalization = norm
            normalization = partial(normalization, args=args)
            non_linear = actv
            non_linear = partial(non_linear, args=args)
            non_linear = partial(non_linear, negative_slope=0.2)
        else:
            conv_func = nn.Conv2d
            normalization = nn.InstanceNorm2d
            non_linear = nn.LeakyReLU
            non_linear = partial(non_linear, negative_slope=0.2)

        self.conv_input = conv_func(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = non_linear()

        res_layers = args.srresnet_lite_layers
        self.residual = self.make_layer(_Residual_Block, res_layers, args=args)

        self.conv_mid = conv_func(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = normalization(64, affine=True)

        self.upscale4x = nn.Sequential(
            conv_func(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            non_linear(),
            conv_func(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            non_linear(),
        )

        self.upscale2x = nn.Sequential(
            conv_func(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            # normalization(256, affine=True),
            nn.PixelShuffle(2),
            # normalization(64, affine=True),
            non_linear(),
            # conv_func(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
            # non_linear(),
        )

        self.conv_output = conv_func(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

        self.scale = args.scale[0]
        self.three_layers = args.three_layers

    def make_layer(self, block, num_of_layer, args=None):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(args=args))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.relu(self.conv_input(x))
        out = self.conv_input(x)
        # draw_hist(out[0], 'layer0_before_relu')
        out = self.relu(out)

        if not self.three_layers:
            residual = out
            out = self.residual(out)
            out = self.bn_mid(self.conv_mid(out))
            out = torch.add(out, residual)

        if self.scale == 2:
            out = self.upscale2x(out)
        elif self.scale == 4:
            out = self.upscale4x(out)

        out = self.conv_output(out)
        return out


# from leftthomas/SRGAN
class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


# original discrim
# class _NetD(nn.Module):
#     def __init__(self):
#         super(_NetD, self).__init__()
#
#         self.features = nn.Sequential(
#
#             # input is (3) x 96 x 96
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (64) x 96 x 96
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (64) x 96 x 96
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (64) x 48 x 48
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (128) x 48 x 48
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (256) x 24 x 24
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (256) x 12 x 12
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (512) x 12 x 12
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
#         self.fc1 = nn.Linear(512 * 6 * 6, 1024)
#         self.fc2 = nn.Linear(1024, 1)
#         self.sigmoid = nn.Sigmoid()
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(0.0, 0.02)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.normal_(1.0, 0.02)
#                 m.bias.data.fill_(0)
#
#     def forward(self, input):
#
#         out = self.features(input)
#
#         # state size. (512) x 6 x 6
#         out = out.view(out.size(0), -1)
#
#         # state size. (512 x 6 x 6)
#         out = self.fc1(out)
#
#         # state size. (1024)
#         out = self.LeakyReLU(out)
#
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         return out.view(-1, 1).squeeze(1)


# Discriminator quant
# class _NetD(nn.Module):
#     def __init__(self, args):
#         super(_NetD, self).__init__()
#         if quantization_enable:
#             conv_func = custom_conv
#             conv_func = partial(conv_func, args=args)
#             normalization = norm
#             normalization = partial(normalization, args=args)
#             non_linear = actv
#             non_linear = partial(non_linear, args=args)
#             non_linear = partial(non_linear, negative_slope=0.2)
#         else:
#             conv_func = nn.Conv2d
#             normalization = nn.InstanceNorm2d
#             non_linear = nn.LeakyReLU
#             non_linear = partial(non_linear, negative_slope=0.2)
#
#         self.features = nn.Sequential(
#
#             # input is (3) x 96 x 96
#             conv_func(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (64) x 96 x 96
#             conv_func(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (64) x 96 x 96
#             conv_func(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (64) x 48 x 48
#             conv_func(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (128) x 48 x 48
#             conv_func(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (256) x 24 x 24
#             conv_func(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (256) x 12 x 12
#             conv_func(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (512) x 12 x 12
#             conv_func(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
#         self.fc1 = nn.Linear(512 * 6 * 6, 1024)
#         # self.fc1 = custom_linear(512 * 6 * 6, 1024, bias=True, args=args)
#         self.fc2 = nn.Linear(1024, 1)
#         # self.fc2 = custom_linear(1024, 1, bias=True, args=args)
#         self.sigmoid = nn.Sigmoid()
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(0.0, 0.02)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.normal_(1.0, 0.02)
#                 m.bias.data.fill_(0)
#
#     def forward(self, input):
#
#         out = self.features(input)
#
#         # state size. (512) x 6 x 6
#         out = out.view(out.size(0), -1)
#
#         # state size. (512 x 6 x 6)
#         out = self.fc1(out)
#
#         # state size. (1024)
#         out = self.LeakyReLU(out)
#
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         return out.view(-1, 1).squeeze(1)
