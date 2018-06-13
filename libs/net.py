import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

# class Reorg(nn.Module):
#     def __init__(self, stride=2):
#         super(Reorg, self).__init__()
#         self.stride = stride

#     def forward(self, x):
#         a = x[:, :, ::self.stride, ::self.stride]
#         b = x[:, :, ::self.stride, 1::self.stride]
#         c = x[:, :, 1::self.stride, ::self.stride]
#         d = x[:, :, 1::self.stride, 1::self.stride]
#         return torch.cat((a, b, c, d), 1)
class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x


class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 ksize, stride=1, activation=False,
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


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 ksize, stride=1, activation=False,
                 padding='SAME'):
        super(Conv2d, self).__init__()
        padding_ = int((ksize-1)/2) if padding == 'SAME' else 0
        self.conv = nn.Conv2d(in_channels, out_channels, ksize,
                              stride, padding_)
        self.relu = nn.LeakyReLU(0.1) if activation else None

    def forward(self, x):
        x = self.conv(x)
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
                                        ksize,
                                        activation=True))
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
        self.conv1s, oc1 = block(3, net_cfgs[:5])
        self.conv2, oc2 = block(oc1, net_cfgs[5])
        self.conv3, oc3 = block(oc2, net_cfgs[6])
        self.reorg = Reorg(stride=2)
        self.conv4, oc4 = block(oc3 + oc1*2*2, net_cfgs[7])
        self.conv5 = Conv2d(oc4,
                            self.num_anchors*(self.num_classes+5),
                            ksize=1)

    def forward(self, x):
        conv1 = self.conv1s(x)
        # pdb.set_trace()
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        reorg = self.reorg(conv1)
        cat_conv1_3 = torch.cat([conv3, reorg], 1)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        bs, _, h, w = conv5.size()
        out = conv5.view(bs, self.num_anchors, self.num_classes+5, h, w)
        x_pred = F.sigmoid(out[:, :, 0, :, :])
        y_pred = F.sigmoid(out[:, :, 1, :, :])
        w_pred = out[:, :, 2, :, :]
        h_pred = out[:, :, 3, :, :]
        iou_pred = F.sigmoid(out[:, :, 4, :, :])
        prob_pred = F.softmax(out[:, :, 5:, :, :], dim=2)
        return x_pred, y_pred, w_pred, h_pred, iou_pred, prob_pred

    def load_from_npz(self, fname, num_conv=None):
        dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases',
                    'bn.weight': 'gamma', 'bn.bias': 'biases',
                    'bn.running_mean': 'moving_mean',
                    'bn.running_var': 'moving_variance'}
        params = np.load(fname)
        own_dict = self.state_dict()
        keys = list(own_dict.keys())

        for i, start in enumerate(range(0, len(keys), 5)):
            if num_conv is not None and i >= num_conv:
                break
            end = min(start+5, len(keys))
            for key in keys[start:end]:
                list_key = key.split('.')
                ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
                src_key = '{}-convolutional/{}:0'.format(i, ptype)
                print((src_key, own_dict[key].size(), params[src_key].shape))
                param = torch.from_numpy(params[src_key])
                if ptype == 'kernel':
                    param = param.permute(3, 2, 0, 1)
                own_dict[key].copy_(param)
