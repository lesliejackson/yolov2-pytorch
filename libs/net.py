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
                 ksize, stride=1, activation=None,
                 padding='SAME'):
        super(Conv2d_BN, self).__init__()
        padding_ = int((ksize-1)/2) if padding == 'SAME' else 0
        self.conv = nn.Conv2d(in_channels, out_channels, ksize,
                              stride, padding_, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1) if activation else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def block(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = block(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(Conv2d_BN(in_channels,
                                        out_channels,
                                        ksize))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels


class Darknet_19(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(Darknet_19, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        net_cfgs = [
            # conv1
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(1024, 3)]
        ]
        self.conv1, oc1 = block(3, net_cfgs[:5])
        self.conv2, oc2 = block(oc1, net_cfgs[5])
        self.conv3, oc3 = block(oc2, net_cfgs[6])
        self.reorg = Reorg(stride=2)
        self.conv4, oc4 = block(oc3 + oc1*2*2, net_cfgs[7])
        self.conv5 = Conv2d_BN(oc4,
                               self.num_anchors*(self.num_classes+5),
                               ksize=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        reorg = self.reorg(conv1)
        cat_conv1_3 = torch.cat([conv3, reorg], 1)
        conv4 = self.conv4(cat_conv1_3)
        out = self.conv5(conv4)
        bs, _, h, w = out.size()
        out = out.view(bs, h, w, self.num_anchors, self.num_classes+5)
        xy_pred = F.sigmoid(out[:, :, :, :, :2])
        wh_pred = out[:, :, :, :, 2:4]
        bbox_pred = torch.cat((xy_pred, wh_pred), -1)
        iou_pred = F.sigmoid(out[:, :, :, :, 4])
        score_pred = out[:, :, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

        return bbox_pred, iou_pred, prob_pred