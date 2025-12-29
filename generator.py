import torch
import torch.nn as nn
import math

import random
import numpy as np

ngf = 64


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


class lowpass(nn.Module):
    def __init__(self, channels=3, k=None, kernel_size=17, sigma=4):
        super(lowpass, self).__init__()
        if k is not None and k > 0:
            kernel_size = 4 * k + 1
            sigma = float(k)

        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.channels = channels
    
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                        kernel_size=kernel_size, groups=channels, 
                                        padding=self.pad, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class Generator(nn.Module):
    def __init__(self, inception=False, device='cuda', num_head=1, nz=16, loc=[1, 1, 1], context_dim=512, k=None):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 to 3x299x299.
        :param use_lfaa: if True, use LFAA methodology (perturb low-frequency components)
        '''
        super(Generator, self).__init__()
        self.inception = inception
        self.device = device
        self.snlinear = snlinear(in_features=512, out_features=nz, bias=False)
        self.loc = loc

        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3 + nz * self.loc[0], ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf + nz * self.loc[1], ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2 + nz * self.loc[2], ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )
        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

        self.lowpass = lowpass(channels=3, k=k)

    def forward(self, input, cond, eps=16/255.):
        z_cond = self.snlinear(cond.float())

        input_low_freq = self.lowpass(input)

        z_img = z_cond[:, :, None, None].expand(z_cond.size(0), z_cond.size(1), input.size(2), input.size(3))
        x = self.block1(torch.cat((input, z_img), dim=1))

        z_img = z_cond[:, :, None, None].expand(z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        x = self.block2(torch.cat((x, z_img), dim=1)) if self.loc[1] else self.block2(x)

        z_img = z_cond[:, :, None, None].expand(z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        x = self.block3(torch.cat((x, z_img), dim=1)) if self.loc[2] else self.block3(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)

        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)

        if self.inception:
            x = self.crop(x)
        x = torch.tanh(x)

        adv = x * eps  
        adv = input_low_freq + adv
        adv = torch.max(torch.min(adv, input + eps), input - eps)
        return adv
        


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual
