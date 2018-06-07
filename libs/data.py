import os
import xml.etree.cElementTree as ET

import numpy as np
from PIL import Image

import torch
import torch.utils.data

import pdb


def make_dataset(usage, data_dir):
    """
    get data paths
    """
    if usage not in ['test', 'train', 'eval']:
        raise ValueError('unknown usage:{}'.format(usage))
    if usage == 'test':
        if not data_dir or not os.path.isdir(data_dir) or not os.path.exists(data_dir):
            raise ValueError('invalid test_dir:{}'.format(data_dir))
    imgs = []
    if usage in ['train', 'eval']:
        labels = []
        for rt, dirs, files in os.walk(data_dir + '/JPEGImages_small'):
            for file in files:
                imgs.append(os.path.join(rt, file))
        for rt, dirs, files in os.walk(data_dir + '/Annotations_small'):
            for file in files:
                labels.append(os.path.join(rt, file))
        imgs.sort(), labels.sort()
        num_samples = len(imgs)
        assert num_samples == len(labels)
        return imgs, labels
    else:
        for rt, dirs, files in os.walk(data_dir):
            for file in files:
                imgs.append(os.path.join(rt, file))
        imgs.sort()
        return imgs, None


class VOCdataset(torch.utils.data.Dataset):
    """
    Pascal VOC dataset
    """
    def __init__(self, usage, data_dir, transform=None):
        super(VOCdataset, self).__init__()
        self.transform = transform
        self.imgs, self.labels = make_dataset(usage, data_dir)
        self.classes = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
                        'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
                        'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12,
                        'motorbike': 13, 'person': 14, 'pottedplant': 15,
                        'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            xml_path = self.labels[index]
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_size = root.find('size')
            img_width = int(img_size.find('width').text)
            img_height = int(img_size.find('height').text)
            gt = []
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))/img_width
                xmax = int(float(bndbox.find('xmax').text))/img_width
                ymin = int(float(bndbox.find('ymin').text))/img_height
                ymax = int(float(bndbox.find('ymax').text))/img_height
                c = self.classes[obj.find('name').text]
                gt.append([xmin, ymin, xmax, ymax, c])
            return img, torch.from_numpy(np.array(gt, dtype=np.float32))
        return img, self.imgs[index]

    def __len__(self):
        return len(self.imgs)