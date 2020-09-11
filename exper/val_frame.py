import sys
sys.path.append('../')

import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import pdb
import os
from tqdm import tqdm
import time
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader
import shutil
import collections
from torch.optim import SGD
import my_optim
import torch.nn.functional as F
from models import *
from torch.autograd import Variable
import torchvision
from utils import AverageMeter
from utils import Metrics
from utils.save_atten import SAVE_ATTEN
from utils.LoadData import  data_loader
from utils.Restore import restore, full_restore
import cv2
import scipy.io as sio

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:',ROOT_DIR)

IMG_DIR=os.path.join(ROOT_DIR,'data','ILSVRC','Data','CLS-LOC','train')
SNAPSHOT_DIR=os.path.join(ROOT_DIR,'snapshot_bins')

train_list = os.path.join(ROOT_DIR,'datalist', 'ILSVRC', 'train_list.txt')
test_list = os.path.join(ROOT_DIR,'datalist','ILSVRC', 'val_list.txt')

# Default parameters
LR = 0.001
EPOCH = 21
DISP_INTERVAL = 20

def get_arguments():
    parser = argparse.ArgumentParser(description='SPG')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--train_list", type=str, default=train_list)
    parser.add_argument("--test_list", type=str, default=test_list)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=356)
    parser.add_argument("--crop_size", type=int, default=320)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--arch", type=str,default='vgg_v0')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--loss_local_factor", type=float, default=0.008)
    parser.add_argument("--loss_global_factor", type=float, default=0.01)
    parser.add_argument("--local_seed_num", type=int, default=3)
    parser.add_argument("--save_atten_dir", type=str)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes, args=args, threshold=args.threshold)

    model.cuda()

    if args.resume == 'True':
        # restore(args, model, optimizer)
        full_restore(args, model)

    model = torch.nn.DataParallel(model, range(args.num_gpu))

    return  model

def get_org_img(img):
    img = np.transpose(img, (1,2,0))
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    img = img*std_vals + mean_vals
    img = img*255
    return img

def val(args, model=None, current_epoch=0):
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1.reset()
    top5.reset()

    if model is None:
        model = get_model(args)
    model.eval()
    _, val_loader = data_loader(args, test_path=True)

    save_atten = SAVE_ATTEN(save_dir=args.save_atten_dir)

    global_counter = 0
    prob = None
    gt = None
    for idx, dat  in tqdm(enumerate(val_loader)):
        img_path, img, label_in = dat
        global_counter += 1
        if args.tencrop == 'True':
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)
            label_input = label_in.repeat(10, 1)
            label = label_input.view(-1)
        else:
            label = label_in

        img, label = img.cuda(), label.cuda()
        img_var, label_var = Variable(img), Variable(label)

        logits = model(img_var, label_var)

        logits0 = logits[0]
        logits0 = F.softmax(logits0, dim=1)
        if args.tencrop == 'True':
            logits0 = logits0.view(bs, ncrops, -1).mean(1)


        # Calculate classification results
        prec1_1, prec5_1 = Metrics.accuracy(logits0.cpu().data, label_in.long(), topk=(1,5))
        # prec3_1, prec5_1 = Metrics.accuracy(logits[1].data, label.long(), topk=(1,5))
        top1.update(prec1_1[0], img.size()[0])
        top5.update(prec5_1[0], img.size()[0])


        # save_atten.save_heatmap_segmentation(img_path, np_last_featmaps, label.cpu().numpy(),
        #                                      save_dir='./save_bins/heatmaps', size=(0,0), maskedimg=True)
        #

        np_last_featmaps = logits[2].cpu().data.numpy()
        np_last_featmaps = logits[-1].cpu().data.numpy()
        np_scores, pred_labels = torch.topk(logits0,k=args.num_classes,dim=1)
        pred_np_labels = pred_labels.cpu().data.numpy()
        save_atten.save_top_5_pred_labels(pred_np_labels[:,:5], img_path, global_counter)

        # pred_np_labels[:,0] = label.cpu().numpy() #replace the first label with gt label
        # save_atten.save_top_5_atten_maps(np_last_featmaps, pred_np_labels, img_path)



    print('Top1:', top1.avg, 'Top5:',top5.avg)



if __name__ == '__main__':
    args = get_arguments()
    import json
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
