# imports
import os
import time

import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
# from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from skimage import io, color


"""
# DATA preparation
original size:
200 classes
train: 200*500
val: 10,000
test: 10,000
for train data, sample 50(10%)/5(1%) images from each class
(10%)/(1%)val data are used to tune
test data without labels
"""

# adopted from https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/tiny_imagenet.py
class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    # base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, base_folder, split='train', transform=None, target_transform=None, download=False, color_distortion=False, col = False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        os.makedirs(root, exist_ok=True)
        self.root = root
        self.base_folder = base_folder
        self.color_distortion = color_distortion
        self.col = col
        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)


    def __getitem__(self, index):
        if self.col:
          img_path, _ = self.data[index]
          color_img = io.imread(img_path)
          if len(color_img.shape)==3:
            gray_img = color.rgb2gray(color_img)
          else: # if the image is oroginally gray
            gray_img = color_img
            h, w = color_img.shape
            color_img = np.reshape(np.array(color_img), (h,w,1))
            color_img = color_img.repeat(3,2)

          h, w = gray_img.shape
          gray_img = np.reshape(np.array(gray_img), (h,w,1))

           # augmentation:
          if np.random.random()>0.7:
            color_img = np.flipud(color_img).copy()
            gray_img = np.flipud(gray_img).copy()
          if np.random.random()>0.7:
            color_img = np.fliplr(color_img).copy()
            gray_img = np.fliplr(gray_img).copy()

          # transform to tensor
          if self.transform is not None:
            gray_img = self.transform(gray_img)
          if self.target_transform is not None:
            color_img = self.target_transform(color_img)
          image = gray_img
          target = color_img
          # print('gray')

        else:
          img_path, target = self.data[index]
          image = self.loader(img_path)

          if self.transform is not None:
            if self.color_distortion:
              image = get_color_distortion(1)(image)
            image = self.transform(image)
          if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                image_names = os.listdir(cls_imgs_path)
                for imgname in sorted(image_names):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        image_names = os.listdir(imgs_path)
        for imgname in image_names:
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def TinyImageNet_data_loader(base_folder, batch_size, color_distortion=False, col=False):

  norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  # data augmentation to training data
  train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
  # data augmentation to val data
  test_transform = transforms.Compose([transforms.ToTensor(), norm])
  if not col:
    train_dataset = TinyImageNet('.', base_folder, split='train', download=True, transform=train_transform,color_distortion=color_distortion, col=col)
    val_dataset = TinyImageNet('.', base_folder, split='val', download=True, transform=test_transform, col=col)
  else:
    col_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = TinyImageNet('.', base_folder, split='train', download=True, transform=col_transform,target_transform=col_transform, col=col)
    val_dataset = TinyImageNet('.', base_folder, split='val', download=True, transform=col_transform, target_transform=col_transform, col=col)
   
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size,  shuffle=False)

  # print(len(val_dataset))
  # print(len(train_dataset))
  return train_dataloader, val_dataloader
def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum



class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]


