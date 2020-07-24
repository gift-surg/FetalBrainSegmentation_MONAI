# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm, Act
from monai.networks.layers.convutils import same_padding
from custom_convolution import AnisotropicConvolution


class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.dimensions = 2
        self.in_channels = 1
        self.out_channels = 1
        self.channels = (16, 32, 64, 128, 256)
        self.strides = (2, 2, 2, 2, 1)
        self.kernel_size = 3
        self.up_kernel_size = 3
        self.act = Act.PRELU
        self.norm = Norm.INSTANCE
        self.dropout = 0

        assert len(self.channels) == len(self.strides)

        # encoding layers
        self.conv_down0 = Convolution(self.dimensions, self.in_channels, self.channels[0], self.strides[0],
                                      kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)
        self.conv_down1 = Convolution(self.dimensions, self.channels[0], self.channels[1], self.strides[1],
                                      kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)
        self.conv_down2 = Convolution(self.dimensions, self.channels[1], self.channels[2], self.strides[2],
                                      kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)
        self.conv_down3 = Convolution(self.dimensions, self.channels[2], self.channels[3], self.strides[3],
                                      kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)
        self.conv_bottom = Convolution(self.dimensions, self.channels[3], self.channels[4], self.strides[4],
                                       kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)

        # decoding layers
        self.conv_up3 = Convolution(self.dimensions, self.channels[4] + self.channels[3], self.channels[3],
                                    self.strides[3],
                                    kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout,
                                    is_transposed=True)
        self.conv_up2 = Convolution(self.dimensions, self.channels[3] + self.channels[2], self.channels[2],
                                    self.strides[2],
                                    kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout,
                                    is_transposed=True)
        self.conv_up1 = Convolution(self.dimensions, self.channels[2] + self.channels[1], self.channels[1],
                                    self.strides[1],
                                    kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout,
                                    is_transposed=True)
        self.conv_up0 = Convolution(self.dimensions, self.channels[1] + self.channels[0], self.channels[0],
                                    self.strides[0],
                                    kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout,
                                    is_transposed=True)
        self.conv_out = Convolution(self.dimensions, self.channels[0], self.out_channels, 1,
                                    kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout,
                                    is_transposed=True, conv_only=True)

    def forward(self, x):

        # encoding path
        conv_down0 = self.conv_down0(x)
        conv_down1 = self.conv_down1(conv_down0)
        conv_down2 = self.conv_down2(conv_down1)
        conv_down3 = self.conv_down3(conv_down2)
        conv_bottom = self.conv_bottom(conv_down3)

        # decoding path
        up_3 = torch.cat([conv_bottom, conv_down3], dim=1)
        conv_up3 = self.conv_up3(up_3)

        up_2 = torch.cat([conv_up3, conv_down2], dim=1)
        conv_up2 = self.conv_up2(up_2)

        up_1 = torch.cat([conv_up2, conv_down1], dim=1)
        conv_up1 = self.conv_up1(up_1)

        up_0 = torch.cat([conv_up1, conv_down0], dim=1)
        conv_up0 = self.conv_up0(up_0)

        out = self.conv_out(conv_up0)

        return out


class CustomUNet25(nn.Module):
    def __init__(self):
        super().__init__()

        self.dimensions = 3
        self.in_channels = 1
        self.out_channels = 1
        self.channels = (16, 32, 64, 128, 256)
        self.strides = ([2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [1, 1, 1])
        self.kernel_size = (3, 3, 1)
        self.up_kernel_size = 3
        self.act = Act.PRELU
        self.norm = Norm.INSTANCE
        self.dropout = 0

        assert len(self.channels) == len(self.strides)

        # encoding layers
        self.conv_down0 = Convolution(self.dimensions, self.in_channels, self.channels[0], self.strides[0],
                                      kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)
        self.conv_down1 = Convolution(self.dimensions, self.channels[0], self.channels[1], self.strides[1],
                                      kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)
        self.conv_down2 = Convolution(self.dimensions, self.channels[1], self.channels[2], self.strides[2],
                                      kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)
        self.conv_down3 = Convolution(self.dimensions, self.channels[2], self.channels[3], self.strides[3],
                                      kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)
        self.conv_bottom = Convolution(self.dimensions, self.channels[3], self.channels[4], self.strides[4],
                                       kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout)

        # decoding layers
        self.conv_up3 = AnisotropicConvolution(self.dimensions, self.channels[4] + self.channels[3], self.channels[3],
                                               self.strides[3],
                                               kernel_size=self.kernel_size, act=self.act, norm=self.norm,
                                               dropout=self.dropout,
                                               is_transposed=True)
        self.conv_up2 = AnisotropicConvolution(self.dimensions, self.channels[3] + self.channels[2], self.channels[2],
                                               self.strides[2],
                                               kernel_size=self.kernel_size, act=self.act, norm=self.norm,
                                               dropout=self.dropout,
                                               is_transposed=True)
        self.conv_up1 = AnisotropicConvolution(self.dimensions, self.channels[2] + self.channels[1], self.channels[1],
                                               self.strides[1],
                                               kernel_size=self.kernel_size, act=self.act, norm=self.norm,
                                               dropout=self.dropout,
                                               is_transposed=True)
        self.conv_up0 = AnisotropicConvolution(self.dimensions, self.channels[1] + self.channels[0], self.channels[0],
                                               self.strides[0],
                                               kernel_size=self.kernel_size, act=self.act, norm=self.norm,
                                               dropout=self.dropout,
                                               is_transposed=True)
        self.conv_out = AnisotropicConvolution(self.dimensions, self.channels[0], self.out_channels, 1,
                                               kernel_size=self.kernel_size, act=self.act, norm=self.norm,
                                               dropout=self.dropout,
                                               is_transposed=True, conv_only=True)

        # self.conv_up3 = nn.ConvTranspose3d(self.channels[4] + self.channels[3], self.channels[3],
        #                                    self.kernel_size, self.strides[3],
        #                                    same_padding(self.kernel_size, dilation=1),
        #                                    [i - 1 for i in self.strides[3]], 1, bias=True, dilation=1)
        # self.conv_up2 = nn.ConvTranspose3d(self.channels[3] + self.channels[2], self.channels[2],
        #                                    self.kernel_size, self.strides[2],
        #                                    same_padding(self.kernel_size, dilation=1),
        #                                    [i - 1 for i in self.strides[2]], 1, bias=True, dilation=1)
        # self.conv_up1 = nn.ConvTranspose3d(self.channels[2] + self.channels[1], self.channels[1],
        #                                    self.kernel_size, self.strides[1],
        #                                    same_padding(self.kernel_size, dilation=1),
        #                                    [i - 1 for i in self.strides[1]], 1, bias=True, dilation=1)
        # self.conv_up0 = nn.ConvTranspose3d(self.channels[1] + self.channels[0], self.channels[0],
        #                                    self.kernel_size, self.strides[0],
        #                                    same_padding(self.kernel_size, dilation=1),
        #                                    [i - 1 for i in self.strides[0]], 1, bias=True, dilation=1)
        # self.conv_out = Convolution(self.dimensions, self.channels[0], self.out_channels, 1,
        #                             kernel_size=self.kernel_size, act=self.act, norm=self.norm, dropout=self.dropout,
        #                             is_transposed=True, conv_only=True)

    def forward(self, x):

        # encoding path
        conv_down0 = self.conv_down0(x)
        conv_down1 = self.conv_down1(conv_down0)
        conv_down2 = self.conv_down2(conv_down1)
        conv_down3 = self.conv_down3(conv_down2)
        conv_bottom = self.conv_bottom(conv_down3)

        # decoding path
        up_3 = torch.cat([conv_bottom, conv_down3], dim=1)
        conv_up3 = self.conv_up3(up_3)

        up_2 = torch.cat([conv_up3, conv_down2], dim=1)
        conv_up2 = self.conv_up2(up_2)

        up_1 = torch.cat([conv_up2, conv_down1], dim=1)
        conv_up1 = self.conv_up1(up_1)

        up_0 = torch.cat([conv_up1, conv_down0], dim=1)
        conv_up0 = self.conv_up0(up_0)

        out = self.conv_out(conv_up0)

        return out