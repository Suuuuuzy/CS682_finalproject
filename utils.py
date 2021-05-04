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
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, percent=0.1):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        os.makedirs(root, exist_ok=True)
        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))
        self.percent = percent
        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx, self.percent)

    def _download(self):
        if not os.path.isfile(os.path.join(self.root, self.filename)):
          print('Downloading...')
          download_url(self.url, root=self.root, filename=self.filename)
          print('Extracting...')
          extract_archive(os.path.join(self.root, self.filename))
        else:
          print('Data zip already downloaded')

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
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


def make_dataset(root, base_folder, dirname, class_to_idx, percent=1):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                image_names = os.listdir(cls_imgs_path)
                image_number = len(image_names)
                smaple_image_names = random.sample(image_names, int(image_number*percent))
                for imgname in sorted(smaple_image_names):
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
        image_number = len(image_names)
        smaple_image_names = random.sample(image_names, int(image_number*percent))
        for imgname in smaple_image_names:
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images


def TinyImageNet_data_loader(batch_size):

  norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  # data augmentation to training data
  train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
  # data augmentation to val data
  test_transform = transforms.Compose([transforms.ToTensor(), norm])

  train_dataset = TinyImageNet('data', split='train', download=False, transform=train_transform, percent = 0.1)
  val_dataset = TinyImageNet('data', split='val', download=False, transform=test_transform)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size,  shuffle=False)

  # print(len(val_dataset))
  # print(len(train_dataset))
  return train_dataloader, val_dataloader

