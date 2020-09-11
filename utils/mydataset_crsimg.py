from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import torch
from PIL import Image
import random
import cv2
import math
import pdb



import pickle

class ReaderCrossImage(Dataset):
    def __init__(self, datalist_file, root_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_images = 0
        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.img_list, self.label_list = self.read_labeled_image_list(self.datalist_file)
        self.parse_classification_list(self.img_list, self.label_list)
        self.transform = transform
        self.image_list = []
        self.img_list_len = len(self.img_list)
        self.read_image_list = []

        batch_cls = min(40, self.num_classes)
        self.read_image_list.extend(self.get_random_img(batch_cls, 2))


    def set_length(self, len_val):
        self.img_list_len = len_val

    def __len__(self):
        return self.img_list_len

    def read_img(self, path):
        # img = cv2.imread(path)
        img = Image.open(path)
        if img is None:
            raise Exception("The given img is not exist! %s" % (path))
        return img

    def get_img_path(self, img_path):
        path_check = lambda x: x.strip()[1:] if x.strip().startswith('/') else x
        return os.path.join(self.root_dir, path_check(img_path))



    def read_labeled_image_list(self,data_list_file):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list_file, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.JPEG'
                        # image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = list(map(int, line[1:]))
            # img_name_list.append(os.path.join(data_dir, image))
            img_name_list.append(image)
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)

    def parse_classification_list(self, img_name_list, label_list):
        self.map_img_path = {}
        for img_name, onehot_label in zip(img_name_list, label_list):
            img_item = {'img_name':img_name, 'label': onehot_label}
            for label in np.nonzero(onehot_label)[0]:
                try:
                    self.map_img_path[label].append(img_item)
                except KeyError:
                    self.map_img_path[label] = [img_item, ]

        total_list = []
        for k in self.map_img_path.keys():
            total_list.extend([x['img_name'] for x in self.map_img_path[k]])
        print("%d images are for training."%(len(set(total_list))))

        self.num_classes = len(self.map_img_path.keys())

    def get_random_img(self, num_cls, num_img_each_cls):
        # print("cls:", num_cls, list(self.map_img_path.keys()))
        cls_list = random.sample(list(self.map_img_path.keys()), num_cls) # random sample without replacement
        img_list = []
        img_label_list = []
        img_cls_list = []
        for cls_nid in cls_list:
            sample_img_list = random.sample(self.map_img_path[cls_nid], num_img_each_cls) # random sample without replacement
            for single_item in sample_img_list:
                img_list.append(single_item['img_name'])
                img_label_list.append(single_item['label'])
                img_cls_list.extend([cls_nid,])

        return zip(img_list, img_label_list, img_cls_list)

    def __getitem__(self, idx):
        batch_cls = min(40, self.num_classes)
        self.read_image_list.extend(self.get_random_img(batch_cls, 2))

        img_name, onehot_label, label = self.read_image_list.pop(0)
        # img_path = self.image_list[idx]
        # img_id = self.get_img_id(img_name)
        img_dat = self.read_img(self.get_img_path(img_name))
        if self.transform is not None:
            img_dat = self.transform(img_dat)
            # fake_gt_mask = torch.from_numpy(fake_gt_mask.copy())

        # print(img_dat.shape, onehot_label.shape, label.shape)
        return img_dat, onehot_label, label



def get_name_id(name_path):
    name_id = name_path.strip().split('/')[-1]
    name_id = name_id.strip().split('.')[0]
    return name_id


if __name__ == '__main__':
    datalist = '../data/COCO14/list/train_onehot.txt';
    data_dir = '../data/COCO14/images'

    data = dataset(datalist, data_dir)

    img_mean = np.zeros((len(data), 3))
    img_std = np.zeros((len(data), 3))
    for idx in range(len(data)):
        img, _ = data[idx]
        numpy_img = np.array(img)
        per_img_mean = np.mean(numpy_img, axis=(0,1))/255.0
        per_img_std = np.std(numpy_img, axis=(0,1))/255.0

        img_mean[idx] = per_img_mean
        img_std[idx] = per_img_std

    print(np.mean(img_mean, axis=0), np.mean(img_std, axis=0))

