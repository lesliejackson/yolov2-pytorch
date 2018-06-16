import argparse

import numpy as np
from libs.data import VOCdataset_single
from libs.net import Darknet_19
from libs.tiny_net import TinyYoloNet
from torchvision import transforms
from PIL import Image, ImageDraw

import torch
import pdb
import os


parser = argparse.ArgumentParser(description='PyTorch YOLOv2')
parser.add_argument('--anchor_scales', type=str,
                    default='1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52',
                    help='anchor scales')
parser.add_argument('--resume', type=str, default=None,
                    help='path to latest checkpoint')
parser.add_argument('--num_classes', type=int, default=20,
                    help='number of classes')
parser.add_argument('--num_anchors', type=int, default=5,
                    help='number of anchors per cell')                    
parser.add_argument('--threshold', type=float, default=0.25,
                    help='iou threshold')
parser.add_argument('--test_jpg', type=str, help='path to test jpg')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size must be 1')


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


def transform_center(x, y):
    nB, nA, nH, nW = x.size()
    offset_x = torch.arange(nW).view(1, 1, 1, nW).cuda()
    offset_y = torch.arange(nH).view(1, 1, nH, 1).cuda()
    x = (x + offset_x)/nW
    y = (y + offset_y)/nH
    return torch.cat([x.view(nB, nA, nH, nW, 1),y.view(nB, nA, nH, nW, 1)], dim=-1)


def transform_size(w, h, anchor_scales):
    nB, nA, nH, nW = w.size()
    return torch.cat([
                        (torch.exp(w)*anchor_scales[:, 0].view(1, nA, 1, 1).cuda()/nW).view(nB, nA, nH, nW, 1),
                        (torch.exp(h)*anchor_scales[:, 1].view(1, nA, 1, 1).cuda()/nH).view(nB, nA, nH, nW, 1)
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
            x_pred, y_pred, w_pred, h_pred, iou_pred, prob_pred = model(imgs)

        nB, nA, nH, nW = x_pred.size()
        xy_transform = transform_center(x_pred, y_pred)
        wh_transform = transform_size(w_pred, h_pred, anchor_scales)
        bbox_transform = torch.cat([xy_transform, wh_transform], dim=-1)
        bbox_corner = transform_center2corner(bbox_transform)
        bbox_corner = bbox_corner.cpu().numpy()
        iou_pred = iou_pred.cpu().numpy()
        prob_pred  = prob_pred.permute(0,1,3,4,2).contiguous().view(nB, nA, nH, nW, args.num_classes)     
        prob_pred = prob_pred.cpu().numpy()

        idxs = np.where(iou_pred > args.threshold)
        ious = iou_pred[idxs]
        ious = ious[:, np.newaxis]
        probs = prob_pred[idxs]
        classes = np.argmax(probs, axis=1)
        classes = classes[:, np.newaxis]
        bboxs = bbox_corner[idxs]
        np.clip(bboxs, 0, 1, out=bboxs)
        bboxs_iou_class = np.concatenate((bboxs, ious, classes), axis=-1)

        bboxs_iou_class_nmsd = nms(bboxs_iou_class, 0.6)
        nProposals = bboxs_iou_class_nmsd.shape[0]
        img = Image.open(filename[0])
        w, h = img.size
        draw = ImageDraw.Draw(img)
        for i in range(nProposals):
            x0 = bboxs_iou_class_nmsd[i][0]*w
            y0 = bboxs_iou_class_nmsd[i][1]*h
            x1 = bboxs_iou_class_nmsd[i][2]*w
            y1 = bboxs_iou_class_nmsd[i][3]*h
            draw.line([(x0, y0), (x1, y0)], fill='red', width=3)
            draw.line([(x1, y0), (x1, y1)], fill='red', width=3)
            draw.line([(x1, y1), (x0, y1)], fill='red', width=3)
            draw.line([(x0, y1), (x0, y0)], fill='red', width=3)
            draw.text((x0, y0), reverse_cls_dict[bboxs_iou_class_nmsd[i, 5]], fill='white')
        img.show()


def main():
    global args
    args = parser.parse_args()
    assert args.batch_size == 1
    anchor_scales = map(float, args.anchor_scales.split(','))
    anchor_scales = np.array(list(anchor_scales), dtype=np.float32).reshape(-1, 2)
    anchor_scales = torch.from_numpy(anchor_scales).cuda()

    data_transform = transforms.Compose(
            [
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    test_dataset = VOCdataset_single(test_jpg=args.test_jpg, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
					      num_workers=4,
                                              pin_memory=True,
                                              drop_last=False)

    tinynet = TinyYoloNet(args.num_anchors, args.num_classes)    
    tinynet.cuda()
    tinynet.load_weights('yolov2-tiny-voc.weights')
    # darknet = Darknet_19(3, args.num_anchors, args.num_classes)
    # darknet.cuda()
    # darknet.load_net('yolo-voc.weights.h5')
    if args.resume:
        if os.path.isfile(args.resume):
            print("load checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            tinynet.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint success: '{}'".format(args.resume))
        else:
            print("no checkpoint found at {}".format(args.resume))
    test(test_loader, tinynet, anchor_scales)


if __name__ == '__main__':
    main()
