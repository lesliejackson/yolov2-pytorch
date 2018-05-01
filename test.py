import argparse
import logging
import sys

import numpy as np
from libs.data import VOCdataset
from libs.net import Darknet_19
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torch.optim as optim
import pdb
import os


parser = argparse.ArgumentParser(description='PyTorch YOLOv2')
parser.add_argument('--anchor_scales', type=str,
                    default=('1.3221,1.73145,'
                             '3.19275,4.00944,'
                             '5.05587,8.09892,'
                             '9.47112,4.84053,'
                             '11.2364,10.0071'),
                    help='anchor scales')
parser.add_argument('--resume', type=str, default=None,
                    help='path to latest checkpoint')
parser.add_argument('--num_classes', type=int, default=20,
                    help='number of classes')
parser.add_argument('--num_anchors', type=int, default=5,
                    help='number of anchors per cell')                    
parser.add_argument('--threshold', type=float, default=0.4,
                    help='iou threshold')
parser.add_argument('--test_dir', type=str, help='path to test dataset')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size must be 1')


def transform_center(xy):
    b, h, w, num_anchors, _ = xy.size()
    x = xy[..., 0:1]
    y = xy[..., 1:2]
    # pdb.set_trace()
    offset_x = torch.arange(w).view(1, 1, w, 1, 1)
    offset_y = torch.arange(h).view(1, h, 1, 1, 1)
    x = (x + offset_x)/w
    y = (y + offset_y)/h
    return torch.cat([x,y], dim=-1)


def transform_size(wh, anchor_scales):
    b, h, w, num_anchors, _ = wh.size()
    anchor_scales = torch.from_numpy(anchor_scales)
    return torch.cat([
                        torch.exp(wh[..., 0:1])*anchor_scales[:, 0:1]/w,
                        torch.exp(wh[..., 1:2])*anchor_scales[:, 1:2]/h
                     ], dim=-1)



def transform_center2corner(bbox):
    bbox[..., 0], bbox[..., 2] = bbox[..., 0] - bbox[..., 2]/2, bbox[..., 0] + bbox[..., 2]/2
    bbox[..., 1], bbox[..., 3] = bbox[..., 1] - bbox[..., 3]/2, bbox[..., 1] + bbox[..., 3]/2
    return bbox


def test(data_loader, model, anchor_scales):
    model.eval()
    classes_dict = data_loader.dataset.classes
    reverse_cls_dict = {v:k for k,v in classes_dict.items()}

    for imgs, filename in data_loader:
        imgs = imgs.cuda()
        with torch.no_grad():
            bbox_pred, iou_pred, prob_pred = model(imgs)
        bbox_pred = bbox_pred.cpu()
        xy_transform = transform_center(bbox_pred[..., :2])
        wh_transform = transform_size(bbox_pred[..., 2:4], anchor_scales)
        # pdb.set_trace()
        bbox_transform = torch.cat([xy_transform, wh_transform], dim=-1)
        bbox_transform = bbox_transform.cpu().numpy()
        bbox_corner = transform_center2corner(bbox_transform)
        iou_pred = iou_pred.cpu().numpy()
        prob_pred = prob_pred.cpu().numpy()

        idxs = np.where(iou_pred > args.threshold)
        ious = iou_pred[idxs]
        probs = prob_pred[idxs]
        classes = np.argmax(probs, axis=1)
        probs = np.amax(probs, axis=1)
        final_probs = probs * ious
        bboxs = bbox_corner[idxs]

        np.clip(bboxs, 0, 1, out=bboxs)
        img = Image.open(filename[0])
        h, w = img.size
        draw = ImageDraw.Draw(img)
        # pdb.set_trace()
        for i in range(ious.shape[0]):
            draw.rectangle((bboxs[i][0]*w, bboxs[i][1]*h, bboxs[i][2]*w, bboxs[i][3]*h))
            # pdb.set_trace()
            draw.text((bboxs[i][0]*w, bboxs[i][1]*h), reverse_cls_dict[classes[i]])
        img.save('d:\\YOLOV2\\test_results\\{}.jpg'.format(os.path.basename(filename[0])))


def main():
    global args
    args = parser.parse_args()
    assert args.batch_size == 1
    anchor_scales = map(float, args.anchor_scales.split(','))
    anchor_scales = np.array(list(anchor_scales), dtype=np.float32).reshape(-1, 2)

    data_transform = transforms.Compose(
            [
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    test_dataset = VOCdataset(usage='test', transform=data_transform, test_dir=args.test_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
						                       num_workers=4,
                                               pin_memory=True,
                                               drop_last=False)

    darknet = Darknet_19(3, args.num_anchors, args.num_classes)
    darknet.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("load checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            darknet.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint success: '{}'".format(args.resume))
        else:
            print("no checkpoint found at {}".format(args.resume))
    # print('train')
    test(test_loader, darknet, anchor_scales)


if __name__ == '__main__':
    main()
