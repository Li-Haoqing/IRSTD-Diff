import torch.nn.functional as F
import numpy as np
import torch

from skimage import measure
from torchvision import transforms


class IoUMetric:

    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)
        #
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (
                    np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.detach().numpy()
        target = target.detach().numpy()

        predict = (output > 0).astype('int64')  # P
        pixel_labeled = np.sum(target > 0)  # T
        pixel_correct = np.sum((predict == target) * (target > 0))  # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass
        predict = (output.detach().numpy() > 0).astype('int64')  # P
        target = target.numpy().astype('int64')  # T
        intersection = predict * (predict == target)  # TP

        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class ROCMetric:

    def __init__(self, bins, nclass=1):
        self.nclass = nclass
        self.bins = bins

        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)

    def update(self, outputs, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) * 42 / self.bins
            # score_thresh = iBin
            i_tp, i_pos, i_fp, i_neg, _, _ = cal_tp_pos_fp_neg(outputs, labels, score_thresh)

            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + np.spacing(1))
        fp_rates = self.fp_arr / (self.neg_arr + np.spacing(1))

        return tp_rates, fp_rates

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)


def cal_tp_pos_fp_neg(output, target, score_thresh):

    predict = (output.detach().numpy() > score_thresh).astype('int64')  # P # np[512,512,1]
    predict = transforms.ToTensor()(predict).numpy()  # tensor[1,512,512] -> np[1,512,512]
    target = target.detach().numpy()  # np[1,512,512]

    tp = np.sum((predict == target) * (target > 0))  # TP
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()  # FN
    pos = tp + fn
    neg = fp + tn
    return tp, pos, fp, neg, tn, fn


class PD_FA:

    def __init__(self, img_size=256):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0
        self.PD = 0
        self.target = 0
        self.img_size = img_size

    def update(self, preds, labels):

        preds, labels = preds.squeeze(1), labels.squeeze(1)
        predits = np.array((preds > 0).cpu()).astype('int64')
        labelss = np.array(labels.cpu()).astype('int64')  # P

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.FA += np.sum(self.dismatch)
        self.PD += len(self.distance_match)

    def get(self, img_num):

        Final_FA = self.FA / ((self.img_size * self.img_size) * img_num)
        Final_PD = self.PD / self.target

        return Final_FA, Final_PD

    def reset(self):
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0
        self.PD = 0
        self.target = 0
