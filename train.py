import argparse
import logging
import sys

import numpy as np
from libs.data import VOCdataset
from libs.net import Darknet_19
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from libs.tiny_net import TinyYoloNet
from libs import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pdb
import os
import math


parser = argparse.ArgumentParser(description='PyTorch YOLOv2')
parser.add_argument('--anchor_scales', type=str,
                    default='1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071',
                    help='anchor scales')
parser.add_argument('--resume', type=str, default=None,
                    help='path to latest checkpoint')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=350,
                    help='number of total epochs to run')
parser.add_argument('--lr', type=float, default=0.001,
                    help='base learning rate')
parser.add_argument('--num_classes', type=int, default=20,
                    help='number of classes')
parser.add_argument('--num_anchors', type=int, default=5,
                    help='number of anchors per cell')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight of l2 regularize')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size must be 1')
parser.add_argument('--iou_obj', type=float, default=5.0,
                    help='iou loss weight')
parser.add_argument('--iou_noobj', type=float, default=1.0,
                    help='iou loss weight')
parser.add_argument('--coord_obj', type=float, default=1.0,
                    help='coord loss weight with obj')
parser.add_argument('--prob_obj', type=float, default=1.0,
                    help='prob loss weight with obj')
parser.add_argument('--coord_noobj', type=float, default=0.1,
                    help='coord loss weight without obj')
parser.add_argument('--pretrained_model', type=str, default=None,
                    help='path to pretrained model')
parser.add_argument('--train_data', type=str, default=None,
                    help='path to train data')


logger = logging.getLogger()
fmt = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
file_handler = logging.FileHandler('train.log')
file_handler.setFormatter(fmt)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(fmt)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def variable_input_collate_fn(batch):
    data = list(zip(*batch))
    return [torch.stack(data[0], 0), data[1]]


def iou(bbox1, bbox2):
    if isinstance(bbox1, list) and isinstance(bbox2, list):
        bbox1_xmax = bbox1[0] + 0.5*bbox1[2]
        bbox1_xmin = bbox1[0] - 0.5*bbox1[2]
        bbox1_ymax = bbox1[1] + 0.5*bbox1[3]
        bbox1_ymin = bbox1[1] - 0.5*bbox1[3]
        bbox2_xmax = bbox2[0] + 0.5*bbox2[2]
        bbox2_xmin = bbox2[0] - 0.5*bbox2[2]
        bbox2_ymax = bbox2[1] + 0.5*bbox2[3]
        bbox2_ymin = bbox2[1] - 0.5*bbox2[3]
        tb = min(bbox1_xmax, bbox2_xmax) - max(bbox1_xmin, bbox2_xmin)
        lr = min(bbox1_ymax, bbox2_ymax) - max(bbox1_ymin, bbox2_ymin)
        if tb < 0 or lr < 0:
            return 0
        else:
            return (tb*lr)/(bbox1[2]*bbox1[3] + bbox2[2]*bbox2[3] - tb*lr)
    else:
        bbox1_xmax = bbox1[..., 0]+0.5*bbox1[..., 2]
        bbox1_xmin = bbox1[..., 0]-0.5*bbox1[..., 2]
        bbox1_ymax = bbox1[..., 1]+0.5*bbox1[..., 3]
        bbox1_ymin = bbox1[..., 1]-0.5*bbox1[..., 3]

        tb = np.minimum(bbox1_xmax, bbox2[0]+0.5*bbox2[2])-np.maximum(bbox1_xmin, bbox2[0]-0.5*bbox2[2])
        lr = np.minimum(bbox1_ymax, bbox2[1]+0.5*bbox2[3])-np.maximum(bbox1_ymin, bbox2[1]-0.5*bbox2[3])
        intersection = tb * lr
        intersection[np.where((tb < 0) | (lr < 0))] = 0
        return intersection / (bbox1[..., 2]*bbox1[..., 3] + bbox2[2]*bbox2[3] - intersection)


def build_target(bbox_pred, gt, anchor_scales, seen, threshold=0.6):
    bs, n, h, w, _ = bbox_pred.size()

    tx = np.zeros((bs, n, h, w), dtype=np.float32)
    ty = np.zeros((bs, n, h, w), dtype=np.float32)
    tw = np.zeros((bs, n, h, w), dtype=np.float32)
    th = np.zeros((bs, n, h, w), dtype=np.float32)
    prob_mask = np.zeros((bs, n, h, w), dtype=np.float32)
    iou_mask = np.ones((bs, n, h, w), dtype=np.float32) * np.sqrt(args.iou_noobj)
    iou_mask_ = np.ones((bs, n, h, w), dtype=np.float32) * np.sqrt(args.iou_noobj)
    target_iou = np.zeros((bs, n, h, w), dtype=np.float32)
    bbox_mask  = np.zeros((bs, n, h, w), dtype=np.float32)
    target_class = np.zeros((bs, n, h, w), dtype=np.float32)

    if seen < 12800:
        bbox_mask += np.sqrt(args.coord_noobj)
        tx.fill(0.5)
        ty.fill(0.5)
        tw.fill(0)
        th.fill(0)

    for b in range(bs):
        num_gts = len(gt[b])
        cur_pred = bbox_pred[b]
        max_ious = np.zeros((n, h, w), dtype=np.float32)
        for i in range(num_gts):
            gt_x = (gt[b][i][0]+gt[b][i][2])/2 * w
            gt_y = (gt[b][i][1]+gt[b][i][3])/2 * h
            gt_w = (gt[b][i][2]-gt[b][i][0]) * w
            gt_h = (gt[b][i][3]-gt[b][i][1]) * h
            max_ious = np.maximum(max_ious, iou(cur_pred, [gt_x, gt_y, gt_w, gt_h]))
        iou_mask[b][np.where(max_ious > threshold)] = 0  #dont use iou_mask[b][max_ious > threshold]

    nGT = 0
    nCorrect = 0
    for b in range(bs):
        num_gts = len(gt[b])
        for i in range(num_gts):
            nGT += 1
            gt_x = (gt[b][i][0]+gt[b][i][2])/2 * w
            gt_y = (gt[b][i][1]+gt[b][i][3])/2 * h
            gt_w = (gt[b][i][2]-gt[b][i][0]) * w
            gt_h = (gt[b][i][3]-gt[b][i][1]) * h
            cell_x, cell_y = int(gt_x), int(gt_y)
            best_iou = -1
            best_anchor = -1
            for a in range(n):
                aw = anchor_scales[a][0]
                ah = anchor_scales[a][1]
                cur_iou = iou([0,0,aw,ah], [0,0,gt_w,gt_h])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_anchor = a
            
            pred_bbox = list(bbox_pred[b, best_anchor, cell_y, cell_x])
            gt_bbox = [gt_x, gt_y, gt_w, gt_h]
            bbox_mask[b, best_anchor, cell_y, cell_x] = np.sqrt(args.coord_obj)
            iou_mask[b, best_anchor, cell_y, cell_x] = np.sqrt(args.iou_obj)
            prob_mask[b, best_anchor, cell_y, cell_x] = 1

            tx[b, best_anchor, cell_y, cell_x] = gt_x - cell_x
            ty[b, best_anchor, cell_y, cell_x] = gt_y - cell_y
            tw[b, best_anchor, cell_y, cell_x] = np.log(gt_w/anchor_scales[best_anchor][0])
            th[b, best_anchor, cell_y, cell_x] = np.log(gt_h/anchor_scales[best_anchor][1])
            target_class[b, best_anchor, cell_y, cell_x] = int(gt[b][i][4])
            tiou = iou(pred_bbox, gt_bbox)
            target_iou[b, best_anchor, cell_y, cell_x] = tiou

            if tiou > 0.5:
                nCorrect += 1
    return nGT, nCorrect, bbox_mask, prob_mask, iou_mask, tx, ty, tw, th, target_class, target_iou


def save_fn(state, filename='./yolov2_hflip.pth.tar'):
    torch.save(state, filename)


def train(train_loader, model, anchor_scales, epochs, opt):
    lr_scheduler = MultiStepLR(opt, milestones=[155,233], gamma=0.1)
    samples = len(train_loader.dataset)
    criterion = nn.MSELoss(size_average=False)
    seen = 0
    for epoch in range(args.start_epoch, epochs):
        lr_scheduler.step(epoch=epoch)
        bbox_loss_avg, prob_loss_avg, iou_loss_avg = 0.0, 0.0, 0.0
        model.train()

        for idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.cuda()
            opt.zero_grad()
            with torch.enable_grad():
                x_pred, y_pred, w_pred, h_pred, iou_pred, prob_pred = model(imgs)
            
            nB, nA, nH, nW = iou_pred.size()
            prob_pred = prob_pred.permute(0,1,3,4,2).contiguous().view(nB*nA*nH*nW, args.num_classes)

            with torch.no_grad():
                pred_boxes = torch.cuda.FloatTensor(nB, nA, nH, nW, 4)
                offset_x = torch.arange(nW).view(1, 1, 1, nW).cuda()
                offset_y = torch.arange(nH).view(1, 1, nH, 1).cuda()
                pred_boxes[..., 0] = x_pred + offset_x
                pred_boxes[..., 1] = y_pred + offset_y
                pred_boxes[..., 2] = torch.exp(w_pred)*anchor_scales[:, 0].view(1, nA, 1, 1).cuda()
                pred_boxes[..., 3] = torch.exp(h_pred)*anchor_scales[:, 1].view(1, nA, 1, 1).cuda()
                
                pred_boxes = pred_boxes.cpu()
                anchor_scales = anchor_scales.cpu()
                nGT, nCorrect, bbox_mask, prob_mask, iou_mask, tx, ty, tw, th, target_class, target_iou = \
                    build_target(pred_boxes, labels, anchor_scales, seen)


                bbox_mask = torch.from_numpy(bbox_mask).cuda()
                prob_mask = torch.from_numpy(prob_mask).cuda()
                iou_mask = torch.from_numpy(iou_mask).cuda()            
                tx = torch.from_numpy(tx).cuda()
                ty = torch.from_numpy(ty).cuda()
                tw = torch.from_numpy(tw).cuda()
                th = torch.from_numpy(th).cuda()
                target_class = torch.from_numpy(target_class).cuda()
                target_iou = torch.from_numpy(target_iou).cuda()
                prob_mask = (prob_mask == 1)
                target_class = target_class[prob_mask].view(-1).long()
                prob_mask = prob_mask.view(-1, 1).repeat(1, args.num_classes).cuda()
                prob_pred = prob_pred[prob_mask].view(-1, args.num_classes)
                nProposals = torch.sum(iou_pred > 0.25)

            with torch.enable_grad():
                x_loss = criterion(x_pred*bbox_mask, tx*bbox_mask) / 2.0
                y_loss = criterion(y_pred*bbox_mask, ty*bbox_mask) / 2.0
                w_loss = criterion(w_pred*bbox_mask, tw*bbox_mask) / 2.0
                h_loss = criterion(h_pred*bbox_mask, th*bbox_mask) / 2.0
                prob_loss = args.prob_obj * nn.CrossEntropyLoss(size_average=False)(prob_pred, target_class) / 2.0
                iou_loss = criterion(iou_pred*iou_mask, target_iou*iou_mask) / 2.0
                loss = x_loss + y_loss + w_loss + h_loss + prob_loss + iou_loss
            loss.backward()
            opt.step()
            seen += args.batch_size

            if idx % 10 == 0:
                logger.info('epoch:{} nGT:{} nCorrect:{} nProposals:{} x:{:.4f} y:{:.4f} w:{:.4f} h:{:.4f} iou:{:.4f} cls:{:.4f}'.format(
                    epoch, nGT, nCorrect, nProposals, x_loss.item(), y_loss.item(), w_loss.item(), h_loss.item(), iou_loss.item(), prob_loss.item()
                ))
        if epoch % 50 == 0 or epoch == epochs-1:
            save_fn({'epoch': epoch+1,
                     'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict()}, filename='./trained_models/yolov2_darknet19_hflip_crop_{}.pth.tar'.format(epoch))


def main():
    global args
    args = parser.parse_args()
    anchor_scales = map(float, args.anchor_scales.split(','))
    anchor_scales = np.array(list(anchor_scales), dtype=np.float32).reshape(-1, 2)
    anchor_scales = torch.from_numpy(anchor_scales).cuda()
    torch.backends.cudnn.benchmark = True

    train_transform = transforms.Compose([
                transforms.Resize((416, 416)),
                transforms.ColorJitter(brightness=0.5, saturation=0.5, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                      ])

    train_dataset = VOCdataset(usage='train', data_dir=args.train_data, jitter=0.2, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True,
                                               collate_fn=variable_input_collate_fn,
                                               drop_last=True)

    # net = TinyYoloNet(args.num_anchors, args.num_classes)
    net = Darknet_19(args.num_anchors, args.num_classes)       
    net.cuda()
    net.load_from_npz(args.pretrained_model, num_conv=18)
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr/args.batch_size,
                          weight_decay=args.weight_decay*args.batch_size)

    if args.resume:
        if os.path.isfile(args.resume):
            print("load checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("no checkpoint found at '{}'".format(args.resume))

    train(train_loader,
          net,
          anchor_scales,
          epochs=args.epochs,
          opt=optimizer)


if __name__ == '__main__':
    main()