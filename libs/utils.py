import random
from PIL import Image, ImageDraw
import pdb
import torch


def random_horizon_flip(img, labels):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        for label in labels:
            label[0], label[2] = 1-label[2], 1-label[0]
    return img, labels


def random_crop(img, labels, jitter):
    w, h = img.size
    delta_w = int(w*jitter)
    delta_h = int(h*jitter)
    left = random.randint(0, delta_w)
    right = random.randint(0, delta_w)
    top = random.randint(0, delta_h)
    bottom = random.randint(0, delta_h)
    img = img.crop((left, top, w-right, h-bottom))
    factor_w = (w-left-right)/w
    factor_h = (h-top-bottom)/h
    
    croped_labels = []
    for label in labels:
        diff_x1 = label[0]/factor_w-left/w/factor_w
        diff_y1 = label[1]/factor_h-top/h/factor_h
        diff_x2 = label[2]/factor_w-left/w/factor_w
        diff_y2 = label[3]/factor_h-top/h/factor_h
        label[0] = min(0.999, max(0.001, diff_x1))
        label[1] = min(0.999, max(0.001, diff_y1))
        label[2] = min(0.999, max(0.001, diff_x2))
        label[3] = min(0.999, max(0.001, diff_y2))
        if label[0] == label[2] or label[1] == label[3]:
            continue
        croped_labels.append(label)
    return img, croped_labels


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.weight.requires_grad_(False)
    conv_model.bias.requires_grad_(False)
    conv_model.bias.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(conv_model.bias));   start = start + num_b
    conv_model.weight.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight)); start = start + num_w
    conv_model.weight.requires_grad_()
    conv_model.bias.requires_grad_()
    return start


def load_bn(buf, start, bn_model):
    num_b = bn_model.bias.numel()
    bn_model.bias.requires_grad_(False)
    bn_model.weight.requires_grad_(False)
    bn_model.bias.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(bn_model.bias));     start = start + num_b
    bn_model.weight.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(bn_model.weight));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(bn_model.running_mean));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(bn_model.running_var));   start = start + num_b
    bn_model.bias.requires_grad_()
    bn_model.weight.requires_grad_()
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.requires_grad_(False)
    bn_model.weight.requires_grad_(False)
    conv_model.weight.requires_grad_(False)
    bn_model.bias.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(bn_model.bias));     start = start + num_b
    bn_model.weight.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(bn_model.weight));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(bn_model.running_mean));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(bn_model.running_var));   start = start + num_b
    conv_model.weight.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight)); start = start + num_w
    bn_model.bias.requires_grad_()
    bn_model.weight.requires_grad_()
    conv_model.weight.requires_grad_()
    return start


# if __name__ == '__main__':
#     img = Image.open('d:/YOLOV2/eval_data/JPEGImages/000001.jpg')
#     labels = [[48/353, 240/500, 195/353, 371/500], [8/353, 12/500, 352/353, 498/500]]
#     img_, gts = random_crop(img, labels, 0.2)
#     # pdb.set_trace()
#     draw_ = ImageDraw.Draw(img_)
#     w,h = img_.size
#     for gt in gts:
#         x0 = gt[0]*w
#         y0 = gt[1]*h
#         x1 = gt[2]*w
#         y1 = gt[3]*h
#         draw_.line([(x0, y0), (x1, y0)], fill='red', width=3)
#         draw_.line([(x1, y0), (x1, y1)], fill='red', width=3)
#         draw_.line([(x1, y1), (x0, y1)], fill='red', width=3)
#         draw_.line([(x0, y1), (x0, y0)], fill='red', width=3)
#     img_.show()