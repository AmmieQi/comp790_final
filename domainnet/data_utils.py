import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image

random.seed(0)


class DomainNetDataset(Dataset):

    def __init__(self, image_list, class_labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = class_labels

        self.image_list = image_list
        self.data_list = []
        self.transform = transform

        for image in image_list:

            class_name = image.split('/')[-2]

            datum = [image, self.labels[class_name]]

            self.data_list.append(datum)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):


        datum = self.data_list[idx]
        image, label = datum
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)

        return image, label

def split_data(data_path, split_im_list, K = None):

    images_to_keep = []
    class_id_dict = {}

    with open ("../data/DomainNet/real_test.txt") as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(' ')
            im_fp, class_id = line
            if int(class_id) > 49:
                continue
            class_name = im_fp.split('/')[1]
            if not class_name in class_id_dict:
                class_id_dict[class_name] = int(class_id)

    with open(split_im_list) as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(' ')
            im_fp, class_id = line

            if int(class_id) > 49:
                continue

            images_to_keep.append(im_fp)

    domain = data_path.split('/')[-1]
    all_images = []
    for classes in os.listdir(data_path):
        if not classes in class_id_dict:
            continue

        class_ct = 0
        for image in os.listdir(os.path.join(data_path, classes)):
            im_fp = os.path.join(data_path, classes, image)
            im_sub = os.path.join(domain, classes, image)
            if im_sub in images_to_keep:
                if K:
                    if class_ct >= K:
                        break
                    else:
                        all_images.append(im_fp)
                        class_ct += 1

                else:
                    all_images.append(im_fp)



    random.shuffle(all_images)

    return all_images, class_id_dict


def get_dataloaders(data_path, train_im_f, test_im_f, K = None):

    train, _ = split_data(data_path, train_im_f, K)
    test, class_labels = split_data(data_path, test_im_f)

    print ("Number of training images for {} is {}".format(data_path, len(train)))
    print ("Number of testing images for {} is {}".format(data_path, len(test)))

    normalize = transforms.Normalize(
        mean= [0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
    ])

    train_dataset = DomainNetDataset(train, class_labels, train_transform)
    test_dataset = DomainNetDataset(test, class_labels, valid_transform)


    return train_dataset, test_dataset
