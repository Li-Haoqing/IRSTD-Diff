import os
import sys
import pickle
import cv2
import random
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from PIL import Image, ImageOps, ImageFilter


def sample_resume(names, path=''):
    if path:
        done = [os.path.splitext(i)[0] for i in os.listdir(path)]
        new_names = [i for i in names if i not in done]
        return new_names
    else:
        return names


class NUDT_Dataset(Dataset):

    def __init__(self, args, data_path, mode='train'):

        base_dir = '../data/NUDT-SIRST/'

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test.txt'

        self.list_dir = os.path.join(base_dir, txtfile)
        self.imgs_dir = os.path.join(base_dir, 'images')
        self.label_dir = os.path.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.crop_size = args.image_size
        self.base_size = args.image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.36956465, 0.36938766, 0.37360018], [0.23812144, 0.2380954, 0.24050511]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = os.path.join(self.imgs_dir, name + '.png')
        label_path = os.path.join(self.label_dir, name + '.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        img, mask = self._testval_sync_transform(img, mask)

        img, mask = self.transform(img), transforms.ToTensor()(mask)

        if self.mode == 'train':
            return img, mask
        else:
            return img, mask, name
        # return img[0][None, ...], mask

    def __len__(self):
        return len(self.names)

    def _testval_sync_transform(self, img, mask):

        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask


class IRSTD_Dataset(Dataset):
    def __init__(self, args, data_path, mode='train', plane=False):

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test.txt'

        self.data_path = data_path

        self.list_dir = os.path.join(self.data_path, txtfile)
        self.imgs_dir = os.path.join(self.data_path, 'images')
        self.label_dir = os.path.join(self.data_path, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]
        # self.names = sample_resume(self.names, path='')

        self.mode = mode
        self.crop_size = args.image_size
        self.base_size = args.image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = os.path.join(self.imgs_dir, name + '.png')
        label_path = os.path.join(self.label_dir, name + '.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        img, mask = self._testval_sync_transform(img, mask)

        img, mask = self.transform(img), transforms.ToTensor()(mask)

        if self.mode == 'train':
            return img, mask
        else:
            return img, mask, name

    def __len__(self):
        return len(self.names)

    def _testval_sync_transform(self, img, mask):

        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask


class SIRST_Dataset(Dataset):
    def __init__(self, args, data_path, mode='train'):

        if mode == 'train':
            txtfile = 'idx_427/trainval.txt'
        elif mode == 'test':
            txtfile = 'idx_427/test.txt'

        self.data_path = data_path

        self.list_dir = os.path.join(self.data_path, txtfile)
        self.imgs_dir = os.path.join(self.data_path, 'images')
        self.label_dir = os.path.join(self.data_path, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.crop_size = args.image_size
        self.base_size = args.image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = os.path.join(self.imgs_dir, name + '.png')
        label_path = os.path.join(self.label_dir, name + '_pixels0' + '.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        img, mask = self._testval_sync_transform(img, mask)  # 测试时，仅缩放到base_size

        img, mask = self.transform(img), transforms.ToTensor()(mask)

        if self.mode == 'train':
            return img, mask
        else:
            return img, mask, name

    def __len__(self):
        return len(self.names)

    def _testval_sync_transform(self, img, mask):

        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask


class IRSTD_Dataset2(Dataset):
    def __init__(self, args, data_path, mode='train'):

        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test.txt'

        self.data_path = data_path

        self.list_dir = os.path.join(self.data_path, txtfile)
        self.imgs_dir = os.path.join(self.data_path, 'images')
        self.label_dir = os.path.join(self.data_path, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]
        # self.names = sample_resume(self.names, path='')

        self.mode = mode
        self.crop_size = args.image_size
        self.base_size = args.image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        img_path = os.path.join(self.imgs_dir, name + '.png')
        label_path = os.path.join(self.label_dir, name + '.png')

        img = Image.open(img_path)
        mask = Image.open(label_path).convert('RGB')

        img, mask = self._testval_sync_transform(img, mask)

        img, mask = self.transform(img)[0][None, :], transforms.ToTensor()(mask)

        if self.mode == 'train':
            return mask, img
        else:
            return mask, img, name

    def __len__(self):
        return len(self.names)

    def _testval_sync_transform(self, img, mask):

        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask
