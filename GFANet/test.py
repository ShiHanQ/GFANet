import argparse
import os
from models.layers import build_net
import numpy as np
import torch
import torch.utils.data as Data

from utils.metrics import Evaluator
from prepared_datasets.skin import dataloader

import sys
from utils.common import *
from utils.logger import Print_Logger
from collections import OrderedDict

from tqdm import tqdm


def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Train a Semantic Segmentation network')

    parser.add_argument('--folder', default=1, type=int,
                        help='which cross validation folder')
    parser.add_argument('--dataset', default='ISIC2018_Task1_342x256_RandomShuffle', dest='dataset', type=str,
                        help='choose the dataset, ISIC2018, ISIC2017, ISIC2016, PH2')
    parser.add_argument('--net', default='GFANet', type=str,
                        help='unet, CANet, fcn, etc')
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--val_batch_size', default=1, type=int,
                        help='input batch size for validating')
    parser.add_argument('--num_workers', default=16, type=int)

    args = parser.parse_args()

    return args


def load_data():
    """
    加载数据
    """

    data = np.load('./prepared_datasets/{}/ISIC2018_Task1_folder{}.npy'.format(args.dataset, args.folder),
                   allow_pickle=True).item()

    val, val_label = data['test'], data['test_label']

    val_num = len(val)

    val_dataset = dataloader(args, val, val_label, num_classes=args.num_class, mode='val')

    val_loader = Data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    print("using  {} images for test.".format(val_num))

    return val_loader


def test(epoch, device, validate_loader, evaluator, net):
    # validate
    net.eval()
    evaluator.reset()
    Dice = []
    Jaccard = []

    with torch.no_grad():
        for iter, val_data in tqdm(enumerate(validate_loader), total=len(validate_loader)):
            val_images, val_labels = val_data
            val_images, val_labels = val_images.to(device), val_labels.to(device)

            outputs = net(val_images)
            pred = outputs[4].data.cpu().numpy()
            target = val_labels.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            # 评估
            evaluator.add_batch(target, pred)
            # 指标
            dice = evaluator.cal_subject_level_dice(pred, target)
            Dice.append(dice)

            jaccard = evaluator.Jaccard(pred, target)
            Jaccard.append(jaccard)

        # Fast test during the training
        ACC = evaluator.Pixel_Accuracy()
        Dice = np.nanmean(Dice)
        Jaccard = np.nanmean(Jaccard)
        Spe, Sen = evaluator.Spe_Sen()

        log = OrderedDict([('ACC', ACC),
                           ('Dice', Dice),
                           ('Jaccard', Jaccard),
                           ('Spe', Spe),
                           ('Sen', Sen)])

        print('epoch: [{}]  ACC:{}  Dice:{}  Jaccard:{}  Spe:{}  Sen:{}'.format(
            epoch, log['ACC'], log['Dice'], log['Jaccard'], log['Spe'], log['Sen']))

    return None


def main():
    # 日志
    setpu_seed(2021)  # 设置一个随机数种子，会固定每次运行时的随机顺序，这样可以保证它的可复现性
    save_path = 'run/{}/{}_folder{}/'.format(args.dataset, args.net, args.folder)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    evaluator = Evaluator(args.num_class)

    # 数据加载
    validate_loader = load_data()

    # 网络模型
    model_name = './{}/{}/{}_folder{}.pt'.format(args.dataset, args.net, args.net, args.folder)
    checkpoints = torch.load(os.path.join('checkpoints', model_name), map_location=device)
    net = build_net(args.net)
    net.to(device)

    net.load_state_dict(checkpoints['state_dict'], strict=True)
    epoch = checkpoints['epoch']

    # validata
    test(epoch, device, validate_loader, evaluator, net)
    return None


if __name__ == '__main__':
    args = parse_args()

    main()
