import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import torch.utils.data as Data

from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from metrics import IoUMetric, PD_FA
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():

    parser = ArgumentParser(description='Implement of MY unet')

    parser.add_argument('--img-size', type=int, default=256, help='base image size')
    parser.add_argument('--output_path', type=str, default=r"../sample/NUDT_000000", help='output val_image path')

    parser.add_argument('--batch-size', type=int, default=1, help='batch_size')

    args = parser.parse_args()
    return args


class IRSTD1k_Data(Data.Dataset):
    def __init__(self, args):
        txtfile = 'test.txt'
        self.list_dir = os.path.join('../data/IRSTD-1k', txtfile)

        self.imgs_dir = args.output_path
        self.label_dir = '../data/IRSTD-1k/masks'

        self.names = []
        self.names_mask = []
        with open(self.list_dir, 'r') as f:
            self.names_mask += [line.strip() for line in f.readlines()]
        for name in self.names_mask:
            self.names.append(name)

        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        name_mask = self.names_mask[i]
        img_path = os.path.join(self.imgs_dir, name + '.png')
        label_path = os.path.join(self.label_dir, name_mask + '.png')

        img = Image.open(img_path)
        mask = Image.open(label_path)

        img, mask = self._testval_sync_transform(img, mask)

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img[0][None, :], mask

    def __len__(self):
        return len(self.names)

    def _testval_sync_transform(self, img, mask):

        base_size = self.img_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask


class SirstDataset(Data.Dataset):
    def __init__(self, args):
        txtfile = 'test.txt'
        self.list_dir = os.path.join('../data/NUAA-SIRST', txtfile)

        self.imgs_dir = args.output_path
        self.label_dir = '../data/NUAA-SIRST/masks'

        self.names = []
        self.names_mask = []
        with open(self.list_dir, 'r') as f:
            self.names_mask += [line.strip() for line in f.readlines()]
        for name in self.names_mask:
            self.names.append(name)

        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
        ])

    def __getitem__(self, i):

        name = self.names[i]

        name_mask = self.names_mask[i]

        img_path = os.path.join(self.imgs_dir, name + '.png')
        label_path = os.path.join(self.label_dir, name_mask + '_pixels0.png')

        img = Image.open(img_path)
        mask = Image.open(label_path)

        img, mask = self._testval_sync_transform(img, mask)

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img[0][None, :], mask

    def __len__(self):
        return len(self.names)

    def _testval_sync_transform(self, img, mask):

        base_size = self.img_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask


class NUDT_Data(Data.Dataset):
    def __init__(self, args):
        txtfile = 'test.txt'
        self.list_dir = os.path.join('../data/NUDT-SIRST', txtfile)

        self.imgs_dir = args.output_path
        self.label_dir = '../data/NUDT-SIRST/masks'

        self.names = []
        self.names_mask = []
        with open(self.list_dir, 'r') as f:
            self.names_mask += [line.strip() for line in f.readlines()]
        for name in self.names_mask:
            self.names.append(name)

        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.36956465, 0.36938766, 0.37360018], [0.23812144, 0.2380954, 0.24050511]),
        ])

    def __getitem__(self, i):

        name = self.names[i]
        name_mask = self.names_mask[i]
        img_path = os.path.join(self.imgs_dir, name + '.png')
        label_path = os.path.join(self.label_dir, name_mask + '.png')

        img = Image.open(img_path)
        mask = Image.open(label_path)

        img, mask = self._testval_sync_transform(img, mask)

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img[0][None, :], mask

    def __len__(self):
        return len(self.names)

    def _testval_sync_transform(self, img, mask):

        base_size = self.img_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask


class Val:

    def __init__(self, args):
        self.args = args

        # dataset
        # self.val_set = SirstDataset(args)
        # self.val_set = IRSTD1k_Data(args)
        self.val_set = NUDT_Data(args)
        self.val_data_loader = Data.DataLoader(self.val_set, batch_size=args.batch_size)

        self.iou_metric = IoUMetric()
        self.PD_FA = PD_FA(img_size=args.img_size)
        # self.ROC = ROCMetric2(1, bins=10)

    def test_model(self):
        self.iou_metric.reset()
        self.PD_FA.reset()

        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            self.iou_metric.update(data, labels)
            self.PD_FA.update(data, labels)
            # output2 = data.squeeze(0).permute(1, 2, 0)
            # labels2 = labels.squeeze(0)
            # self.ROC.update(output2, labels2)

            _, IoU = self.iou_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))
            # ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()

            # 进度条描述信息
            tbar.set_description('IoU:%f, Fa:%.10f, Pd:%.10f'
                                 % (IoU, Fa, Pd))
        # return IoU, Fa, Pd, ture_positive_rate, false_positive_rate
        return IoU, Fa, Pd


if __name__ == "__main__":
    args = parse_args()

    value = Val(args)
    IoU, Fa, Pd = value.test_model()
    print('IoU:{},\n Fa:{},\n Pd:{}'.format(IoU, Fa, Pd))
    # ROC(ture_positive_rate, false_positive_rate)
