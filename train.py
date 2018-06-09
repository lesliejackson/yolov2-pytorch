import argparse
import logging
import sys

import numpy as np
from libs.data import VOCdataset
from libs.net import Darknet_19
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR

import torch
import torch.nn as nn
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
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='base learning rate')
parser.add_argument('--num_classes', type=int, default=20,
                    help='number of classes')
parser.add_argument('--num_anchors', type=int, default=5,
                    help='number of anchors per cell')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight of l2 regularize')
parser.add_argument('--batch_size', type=int, default=25,
                    help='batch size must be 1')
parser.add_argument('--iou_obj', type=float, default=10.0,
                    help='iou loss weight')
parser.add_argument('--iou_noobj', type=float, default=1.0,
                    help='iou loss weight')
parser.add_argument('--coord_obj', type=float, default=1.0,
                    help='coord loss weight with obj')
parser.add_argument('--prob_obj', type=float, default=1.0,
                    help='prob loss weight with obj')
parser.add_argument('--coord_noobj', type=float, default=1.0,
                    help='coord loss weight without obj')
parser.add_argument('--pretrained_model', type=str, default=None,
                    help='path to pretrained model')


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
    bs, h, w, n, _ = bbox_pred.size()

    target_bbox = np.zeros((bs, h, w, n, 4), dtype=np.float32)
    prob_mask = np.zeros((bs, h, w, n, 1), dtype=np.float32)
    iou_mask = np.ones((bs, h, w, n), dtype=np.float32) * np.sqrt(args.iou_noobj)
    target_iou = np.zeros((bs, h, w, n), dtype=np.float32)
    bbox_mask  = np.zeros((bs, h, w, n, 1), dtype=np.float32)
    target_class = np.zeros((bs, h, w, n, args.num_classes), dtype=np.float32)

    if seen < 1000:
        bbox_mask += np.sqrt(args.coord_noobj)
        target_bbox[..., 0:2] += 0.5

    for b in range(bs):
        num_gts = len(gt[b])
        cur_pred = bbox_pred[b]
        max_ious = np.zeros((h, w, n), dtype=np.float32)
        for i in range(num_gts):
            gt_x = (gt[b][i][0]+gt[b][i][2])/2 * w
            gt_y = (gt[b][i][1]+gt[b][i][3])/2 * h
            gt_w = (gt[b][i][2]-gt[b][i][0]) * w
            gt_h = (gt[b][i][3]-gt[b][i][1]) * h
            # gt_x, gt_y, gt_w, gt_h = gt_x.item(), gt_y.item(), gt_w.item(), gt_h.item()
            # pdb.set_trace()
            max_ious = np.maximum(max_ious, iou(cur_pred, [gt_x, gt_y, gt_w, gt_h]))
        iou_mask[b][max_ious > threshold] = 0
    # pdb.set_trace()
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
                # pdb.set_trace()
                cur_iou = iou([0,0,aw,ah], [0,0,gt_w,gt_h])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_anchor = a
            
            pred_bbox = list(bbox_pred[b, cell_y, cell_x, best_anchor])
            gt_bbox = [gt_x, gt_y, gt_w, gt_h]
            # pdb.set_trace()
            bbox_mask[b, cell_y, cell_x, best_anchor] = np.sqrt(args.coord_obj)
            iou_mask[b, cell_y, cell_x, best_anchor] = np.sqrt(args.iou_obj)
            prob_mask[b, cell_y, cell_x, best_anchor] = 1

            tx, ty = gt_x-cell_x, gt_y-cell_y
            # pdb.set_trace()
            tw = np.log(gt_w/anchor_scales[best_anchor][0])
            th = np.log(gt_h/anchor_scales[best_anchor][1])
            target_bbox[b, cell_y, cell_x, best_anchor] = tx, ty, tw, th
            target_class[b, cell_y, cell_x, best_anchor, int(gt[b][i][4])] = 1
            tiou = iou(pred_bbox, gt_bbox)
            target_iou[b, cell_y, cell_x, best_anchor] = tiou
            if tiou > 0.5:
                nCorrect += 1
    return nGT, nCorrect, bbox_mask, prob_mask, iou_mask, target_bbox, target_class, target_iou


def save_fn(state, filename='./yolov2.pth.tar'):
    torch.save(state, filename)


def train(train_loader, eval_loader, model, anchor_scales, epochs, opt):
    lr_scheduler = MultiStepLR(opt, milestones=[70,150], gamma=0.1)
    samples = len(train_loader.dataset)
    criterion = nn.MSELoss(size_average=False)
    seen = 0
    lowest_loss = float('inf')
    for epoch in range(args.start_epoch, epochs):
        lr_scheduler.step(epoch=epoch)
        bbox_loss_avg, prob_loss_avg, iou_loss_avg = 0.0, 0.0, 0.0
        model.train()

        for idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.cuda()
            opt.zero_grad()
            # pdb.set_trace()
            with torch.enable_grad():
                bbox_pred, iou_pred, prob_pred = model(imgs)
            
            h = bbox_pred.size(1)
            w = bbox_pred.size(2)
            bbox_pred_np = bbox_pred.detach().cpu()
            offset_x = torch.arange(w).view(1, 1, w, 1, 1)
            offset_y = torch.arange(h).view(1, h, 1, 1, 1)
            # pdb.set_trace()
            bbox_pred_np[..., 0:1] += offset_x
            bbox_pred_np[..., 1:2] += offset_y
            bbox_pred_np[..., 2:3] = torch.exp(bbox_pred_np[..., 2:3])*anchor_scales[:, 0:1]
            bbox_pred_np[..., 3:4] = torch.exp(bbox_pred_np[..., 3:4])*anchor_scales[:, 1:2]
            # pdb.set_trace()
            with torch.no_grad():
                nGT, nCorrect, bbox_mask, prob_mask, iou_mask, target_bbox, target_class, target_iou = \
                    build_target(bbox_pred_np, labels, anchor_scales, seen)
            
            bbox_mask = torch.from_numpy(bbox_mask).cuda()
            prob_mask = torch.from_numpy(prob_mask).cuda()
            iou_mask = torch.from_numpy(iou_mask).cuda()            
            target_bbox = torch.from_numpy(target_bbox).cuda()
            target_class = torch.from_numpy(target_class).cuda()
            target_iou = torch.from_numpy(target_iou).cuda()

            with torch.enable_grad():
                bbox_loss = criterion(bbox_pred*bbox_mask, target_bbox*bbox_mask) / 2.0
                prob_loss = args.prob_obj * criterion(prob_pred*prob_mask, target_class*prob_mask) / 2.0
                iou_loss = criterion(iou_pred*iou_mask, target_iou*iou_mask) / 2.0
                loss = bbox_loss+prob_loss+iou_loss
            loss.backward()
            opt.step()
            # bbox_loss_avg += bbox_loss.item()
            # prob_loss_avg += prob_loss.item()
            # iou_loss_avg += iou_loss.item()
            seen += args.batch_size
        # pdb.set_trace()

            if idx % 10 == 0:
                logger.info('epoch:{} nGT:{} nCorrect:{} bbox loss:{:.4f} iou loss:{:.4f} prob loss:{:.4f}'.format(
                    epoch, nGT, nCorrect, bbox_loss.item(), iou_loss.item(), prob_loss.item()
                ))
        e_nGT, e_nCorrect, e_bbox_loss, e_iou_loss, e_prob_loss = evaluation(eval_loader, model, anchor_scales)
        logger.info('eval nGt:{} nCorrect:{} bbox loss:{:.4f} iou loss:{:.4f} prob loss:{:.4f}'.format(
            e_nGT, e_nCorrect, e_bbox_loss, e_iou_loss, e_prob_loss
        ))
        if e_bbox_loss + e_iou_loss + e_prob_loss < lowest_loss:
            lowest_loss = e_bbox_loss + e_iou_loss + e_prob_loss
            save_fn({'epoch': epoch+1,
                     'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict()})


def evaluation(eval_loader, model, anchor_scales):
    samples = len(eval_loader)
    criterion = nn.MSELoss(size_average=False)
    model.eval()
    bbox_loss_avg, prob_loss_avg, iou_loss_avg = 0.0, 0.0, 0.0

    epoch_nGT = epoch_nCorrect = 0
    for idx, (imgs, labels) in enumerate(eval_loader):
        imgs = imgs.cuda()
        with torch.no_grad():
            bbox_pred, iou_pred, prob_pred = model(imgs)
        
            h = bbox_pred.size(1)
            w = bbox_pred.size(2)
            bbox_pred_np = bbox_pred.detach().cpu()
            offset_x = torch.arange(w).view(1, 1, w, 1, 1)
            offset_y = torch.arange(h).view(1, h, 1, 1, 1)
            bbox_pred_np[..., 0:1] += offset_x
            bbox_pred_np[..., 1:2] += offset_y
            bbox_pred_np[..., 2:3] = torch.exp(bbox_pred_np[..., 2:3])*anchor_scales[:, 0:1]
            bbox_pred_np[..., 3:4] = torch.exp(bbox_pred_np[..., 3:4])*anchor_scales[:, 1:2]      
            
            nGT, nCorrect, bbox_mask, prob_mask, iou_mask, target_bbox, target_class, target_iou = \
                build_target(bbox_pred_np, labels, anchor_scales, 13000)
        
            epoch_nCorrect += nCorrect
            epoch_nGT += nGT
            bbox_mask = torch.from_numpy(bbox_mask).cuda()
            prob_mask = torch.from_numpy(prob_mask).cuda()
            iou_mask = torch.from_numpy(iou_mask).cuda()            
            target_bbox = torch.from_numpy(target_bbox).cuda()
            target_class = torch.from_numpy(target_class).cuda()
            target_iou = torch.from_numpy(target_iou).cuda()

            bbox_loss = criterion(bbox_pred*bbox_mask, target_bbox*bbox_mask) / 2.0
            prob_loss = args.prob_obj * criterion(prob_pred*prob_mask, target_class*prob_mask) / 2.0
            iou_loss = criterion(iou_pred*iou_mask, target_iou*iou_mask) / 2.0
            bbox_loss_avg += bbox_loss.item()
            prob_loss_avg += prob_loss.item()
            iou_loss_avg += iou_loss.item()

    return epoch_nGT, epoch_nCorrect, bbox_loss_avg/samples, iou_loss_avg/samples, prob_loss_avg/samples


def main():
    global args
    args = parser.parse_args()
    anchor_scales = map(float, args.anchor_scales.split(','))
    anchor_scales = np.array(list(anchor_scales), dtype=np.float32).reshape(-1, 2)
    anchor_scales = torch.from_numpy(anchor_scales)

    train_transform = transforms.Compose(
            [
                transforms.Resize((416, 416)),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=1.5, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    eval_transform = transforms.Compose(
            [
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    train_dataset = VOCdataset(usage='train', data_dir='train_data', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True,
                                               collate_fn=variable_input_collate_fn,
                                               drop_last=True)
    eval_dataset = VOCdataset(usage='eval', data_dir='eval_data', transform=eval_transform)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              collate_fn=variable_input_collate_fn,
                                              drop_last=False)
    darknet = Darknet_19(3, args.num_anchors, args.num_classes)
    darknet.cuda()
    optimizer = optim.SGD(darknet.parameters(),
                          lr=args.lr,
                          weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("load checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            darknet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("no checkpoint found at '{}'".format(args.resume))            
    else:
        darknet.load_from_npz(args.pretrained_model, num_conv=18)
        # pass
    train(train_loader,
          eval_loader,
          darknet,
          anchor_scales,
          epochs=args.epochs,
          opt=optimizer)


if __name__ == '__main__':
    main()