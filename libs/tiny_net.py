import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from libs import utils
import pdb

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x


class TinyYoloNet(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(TinyYoloNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.cnn = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d(3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),

            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),

            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, 2)),

            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),
            ('pool6', MaxPoolStride1()),

            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8
            ('conv8', nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(1024)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('output', nn.Conv2d(1024, num_anchors*(num_classes+5), 1, 1, 0)),
        ]))

    def forward(self, x):
        x = self.cnn(x)
        bs, _, h, w = x.size()
        out = x.view(bs, self.num_anchors, self.num_classes+5, h, w)
        x_pred = F.sigmoid(out[:, :, 0, :, :])
        y_pred = F.sigmoid(out[:, :, 1, :, :])
        w_pred = out[:, :, 2, :, :]
        h_pred = out[:, :, 3, :, :]
        iou_pred = F.sigmoid(out[:, :, 4, :, :])
        prob_pred = F.softmax(out[:, :, 5:, :, :], dim=2)
        return x_pred, y_pred, w_pred, h_pred, iou_pred, prob_pred

    def print_network(self):
        print(self)

    def load_weights(self, path):
        buf = np.fromfile(path, dtype = np.float32)
        start = 4
        
        start = utils.load_conv_bn(buf, start, self.cnn[0], self.cnn[1])
        start = utils.load_conv_bn(buf, start, self.cnn[4], self.cnn[5])
        start = utils.load_conv_bn(buf, start, self.cnn[8], self.cnn[9])
        start = utils.load_conv_bn(buf, start, self.cnn[12], self.cnn[13])
        start = utils.load_conv_bn(buf, start, self.cnn[16], self.cnn[17])
        start = utils.load_conv_bn(buf, start, self.cnn[20], self.cnn[21])
        
        start = utils.load_conv_bn(buf, start, self.cnn[24], self.cnn[25])
        start = utils.load_conv_bn(buf, start, self.cnn[27], self.cnn[28])
        start = utils.load_conv(buf, start, self.cnn[30])

