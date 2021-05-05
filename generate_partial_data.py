import os
import random
import sys
import shutil
def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def generate_partial_data(dataset_path,new_dataset_path, percent):
    shutil.rmtree(new_dataset_path)
    os.makedirs(new_dataset_path, exist_ok=True)
    if 'tiny-imagenet-200' in dataset_path:
        shutil.copyfile(os.path.join(dataset_path, 'wnids.txt'), os.path.join(new_dataset_path, 'wnids.txt'))
        _, class_to_idx = find_classes(os.path.join(dataset_path, 'wnids.txt'))

    for dirname in ['train', 'val']:
        images = make_partial_dataset(dataset_path,new_dataset_path, dirname,class_to_idx, percent)


def make_partial_dataset(dataset_path,new_dataset_path,  dirname, class_to_idx, percent=0.01):
    images = []
    dir_path = os.path.join(dataset_path, dirname)
    new_dir_path = os.path.join(new_dataset_path, dirname)
    os.makedirs(new_dataset_path, exist_ok=True)
    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            new_cls_fpath = os.path.join(new_dir_path, fname)
            os.makedirs(new_cls_fpath, exist_ok=True)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                new_cls_imgs_path = os.path.join(new_cls_fpath, 'images')
                os.makedirs(new_cls_imgs_path, exist_ok=True)
                image_names = os.listdir(cls_imgs_path)
                image_number = len(image_names)
                images = random.sample(image_names, int(image_number*percent))
                print(len(images),'for each class in train set')
                for image in images:
                    shutil.copyfile(os.path.join(cls_imgs_path, image), os.path.join(new_cls_imgs_path, image))
                
    else:
        imgs_path = os.path.join(dir_path, 'images')
        new_imgs_path = os.path.join(new_dir_path, 'images')
        os.makedirs(new_imgs_path, exist_ok=True)

        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')
        shutil.copyfile(imgs_annotations, os.path.join(new_dir_path, 'wnids.txt'))

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        image_names = os.listdir(imgs_path)
        image_number = len(image_names)
        images = random.sample(image_names, int(image_number*percent))
        for image in images:
            shutil.copyfile(os.path.join(imgs_path, image), os.path.join(new_imgs_path, image))
    return images

dataset_path = sys.argv[1]
new_dataset_path = sys.argv[2]
percent = float(sys.argv[3])
if __name__=='__main__':
    generate_partial_data(dataset_path, new_dataset_path, percent)
