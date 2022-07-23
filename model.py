# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:04:43 2022

@author: Paul
@file: model.py
@dependencies:
    env pt3.7
    python 3.7.13
    torch >= 1.7.1
    torchvision >= 0.8.2

Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
    Tuple is structured by (filters, kernel_size, stride) 
    Every conv is a same convolution. 
    List is structured by "B" indicating a residual block followed by the number of repeats
    "S" is for scale prediction block and computing the yolo loss
    "U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],     # ["B", 1],
    (128, 3, 2),
    ["B", 2],     # ["B", 2],
    (256, 3, 2),
    ["B", 8],     # ["B", 8],
    (512, 3, 2),
    ["B", 8],     # ["B", 8],
    (1024, 3, 2),
    ["B", 4],     # ["B", 4], To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        # if we do use bn activation function in the block, then we do not want to use bias, its unnecessary
        # **kwargs will be the kernal size, the stride and padding as well
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(negative_slope=0.1) # default negative_slope=0.01
        self.use_bn_act = bn_act # indicating if the block is going to use a batch norm NN activation function

    def forward(self, x):
        # using if-else statement in the forward pass might lose on some performance, negligible?
        # we use bn activation by default, except for scale prediction
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x))) # bn_act()
        # for scale prediction we don't want to use batch norm LeakyReLU on our output, just normal Conv
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats): # repeat for num_repeats
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1, padding=0), # down samples or reduces the number of filters
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1), # then brings it back again
                )
            ]
        # 1. why specify use_residual in a ResidualBlock? is because in some cases we are going to use skip 
        # connections, in some cases we just going through the config file and build the ordinary ResidualBlock
        # 2. why we need to store these? 
        self.use_residual = use_residual # indicating using residual
        self.num_repeats = num_repeats   # number of repeats set to 1 by default

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
            # if self.use_residual:
            #     # x = x + layer(x)
            #     x = layer(x) + x
            # else:
            #     x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        # for every single cell grid we have 3 anchor boxes, for every anchor box we have 1 node for each of the classes
        # for each bounding box we have [P(Object), x, y, w, h] and that's 5 values
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, 3 * (num_classes + 5), bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # we want to return the prediction of x, then we want to reshape it to the number of examples in our batch
        # split out_channel "3 * (num_classes + 5)" into two different dimensions "3, (num_classes + 5)", instead of 
        # having a long vector of bounging boxes, and change the order of the dimensions
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]) 
            .permute(0, 1, 3, 4, 2) 
        ) 
        # [x.shape[0], 3, x.shape[2], x.shape[3], self.num_classes + 5], e.g. [N, 3, 13, 13, 5+num_classes]
        # for scale one, we have N examples in our batch, each example has 3 anchors, each anchor has 13-by-13 grid
        # and every cell has (5+num_classes) output, output dimension = N x 3 x 13 x 13 x (5+num_classes)


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers() # we want to create the layers using the config files

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    # create the layers using the config files
    def _create_conv_layers(self):
        layers = nn.ModuleList()       # keep track of all the layers in a ModuleList, which supports tools like model.eval() 
        in_channels = self.in_channels # 

        # go through and parse the config file and construct the model line by line
        for module in config:
            # if it's a tuple, then it's just a CNNBlock
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module # we want to take out the (filters, kernel_size, stride)
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0, # if kernel_size == 1 then padding = 0
                    )
                )
                # the in_channels for the next block is going to be the out_channels of this block
                in_channels = out_channels 

            # if it's a List, then it's a ResidualBlock
            elif isinstance(module, list):
                num_repeats = module[1] # we want to take out the number of repeats
                # module[0] should be "B" and it's useless, just indicating this is a ResidualBlock
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            # if it's a String, then it's a 
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers


if __name__ == "__main__":
    num_classes = 1 # 20
    IMAGE_SIZE = 16 # 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)) # x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    stride = [8, 4, 2] 
    # stride = [32, 16, 8] 
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//stride[0], IMAGE_SIZE//stride[0], num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//stride[1], IMAGE_SIZE//stride[1], num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//stride[2], IMAGE_SIZE//stride[2], num_classes + 5)
    print("Success!")

