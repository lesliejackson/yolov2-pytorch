import torch
import torch.nn as nn
import torch.nn.functional as F


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        a = x[:, :, ::self.stride, ::self.stride]
        b = x[:, :, ::self.stride, 1::self.stride]
        c = x[:, :, 1::self.stride, ::self.stride]
        d = x[:, :, 1::self.stride, 1::self.stride]
        return torch.cat((a, b, c, d), 1)


class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 ksize, stride=1, activation=F.relu,
                 padding='SAME'):
        super(Conv2d_BN, self).__init__()
        padding_ = int((ksize-1)/2) if padding == 'SAME' else 0
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, padding_)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            return self.activation(x)
        return x


class Darknet_test(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(Darknet_test, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = Conv2d_BN(in_channels, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = Conv2d_BN(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = Conv2d_BN(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = Conv2d_BN(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = Conv2d_BN(256, 512, 3)
        self.pool5 = nn.MaxPool2d(2)
        self.conv6 = Conv2d_BN(512, 1024, 3)
        self.conv7 = Conv2d_BN(1024+512*4, self.num_anchors*(self.num_classes+5), 3)
        self.reorg = Reorg()

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        pool5 = self.pool5(conv5)
        conv6 = self.conv6(pool5)
        passthrough = self.reorg(conv5)
        out = torch.cat([conv6, passthrough], 1)
        out = self.conv7(out)
        bs, _, h, w = out.size()
        out = out.view(bs, h, w, self.num_anchors, self.num_classes+5)
        xy_pred = F.sigmoid(out[:, :, :, :, :2])
        wh_pred = out[:, :, :, :, 2:4]
        bbox_pred = torch.cat((xy_pred, wh_pred), -1)
        iou_pred = F.sigmoid(out[:, :, :, :, 4])
        score_pred = out[:, :, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

        return bbox_pred, iou_pred, prob_pred
