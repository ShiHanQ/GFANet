import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        # PA：像素准确率
        # return all class overall pixel accuracy
        #  PA = Acc = (TP + TN) / (TP + TN + FP + FN)
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Spe_Sen(self):

        '''
           混淆矩阵     预测值
                     0     1
               0    TN     FP
        真实值
               1    FN     TP

        '''

        # sensitivity = TP / (TP + FN)
        # specificity = TN / (TN + FP)
        # sensitivity 、specificity 就是归一化混淆矩阵对角线上的值

        Spe, Sen = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)  # 1：行，0：列

        return Spe, Sen

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def Jaccard(self, result, reference):
        """
        Jaccard coefficient

        Computes the Jaccard coefficient between the binary objects in two images.

        Parameters
        ----------
        result: array_like
                Input data containing objects. Can be any type but will be converted
                into binary: background where 0, object everywhere else.
        reference: array_like
                Input data containing objects. Can be any type but will be converted
                into binary: background where 0, object everywhere else.

        Returns
        -------
        jc: float
            The Jaccard coefficient between the object(s) in `result` and the
            object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).

        Notes
        -----
        This is a real metric. The binary images can therefore be supplied in any order.
        """
        result = np.atleast_1d(result.astype(np.bool))
        reference = np.atleast_1d(reference.astype(np.bool))

        intersection = np.count_nonzero(result & reference)
        union = np.count_nonzero(result | reference)

        jaccard = float(intersection) / (float(union) + 1e-6)

        return jaccard

    def cal_subject_level_dice(self, prediction, target):
        prediction = np.transpose(prediction, [1, 2, 0])
        target = np.transpose(target, [1, 2, 0])
        '''
        step1: calculate the dice of each category
        step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
        :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
        :param target: the ground truth mask, a numpy array with shape of (h, w, d)
        :param class_num: total number of categories
        :return:
        '''
        class_num = self.num_class
        eps = 1e-10
        empty_value = -1.0
        dscs = empty_value * np.ones((class_num), dtype=np.float32)

        for i in range(0, class_num):
            if i not in target and i not in prediction:
                continue
            target_per_class = np.where(target == i, 1, 0).astype(np.float32)  # one-hot
            prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)  # one-hot
            tp = np.sum(prediction_per_class * target_per_class)
            fp = np.sum(prediction_per_class) - tp
            fn = np.sum(target_per_class) - tp
            dsc = 2 * tp / (2 * tp + fp + fn + eps)
            dscs[i] = dsc

        dscs = np.where(dscs == -1.0, np.nan, dscs)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subject_level_dice = np.nanmean(dscs[1:])  # 为了忽略该行的全空警告，将该行专门进行警告忽视

        return subject_level_dice

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape

        if gt_image.ndim == 3:  # 传入一个batch的图像
            for lp, lt in zip(pre_image, gt_image):
                self.confusion_matrix += self._generate_matrix(lt.flatten(), lp.flatten())
        else:  # 传入一张图像
            self.confusion_matrix += self._generate_matrix(gt_image.flatten(), pre_image.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

