import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image

random.seed(0)


class OfficeDataset(Dataset):

    def __init__(self, image_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = {'keyboard' : 0, 'pen': 1 , 'laptop_computer' : 2, 'mug' : 3, 'ruler' : 4,
                    'calculator':5, 'monitor':6, 'back_pack':7, 'punchers':8, 'trash_can':9,
                    'headphones':10, 'bottle':11, 'speaker':12, 'printer':13, 'letter_tray':14,
                    'mobile_phone':15, 'file_cabinet':16, 'tape_dispenser':17,
                    'projector':18, 'scissors':19, 'desk_lamp':20, 'mouse':21,
                    'bike':22, 'bike_helmet':23, 'bookcase':24, 'phone':25,
                    'paper_notebook':26, 'desktop_computer':27,
                    'ring_binder':28, 'desk_chair':29, 'stapler':30}

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

def split_data(root, data_type):

    classes = os.listdir(root)
    all_images = []

    if data_type == 'amazon':
        cutoff = 20
    else:
        cutoff = 8

    for x in classes:
        class_dir = os.path.join(root, x)
        for idx, images in enumerate(os.listdir(class_dir)):
            if idx >= cutoff:
                continue
            else:
                im_fp = os.path.join(class_dir, images)
                all_images.append(im_fp)

    random.shuffle(all_images)
    
    num_ims = len(all_images)
    val_split = int(.10*num_ims)
    test_idx, val_idx, train_idx = all_images[:val_split], all_images[val_split:val_split*2], all_images[val_split*2:]

    assert ((len(test_idx)+ len(val_idx)+ len(train_idx)) == num_ims)

    return test_idx, val_idx, train_idx

def get_dataloaders(root, data_type):

    test, val, train = split_data(root, data_type)

    normalize = transforms.Normalize(
        mean= [0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
    ])

    train_dataset = OfficeDataset(train, train_transform)
    test_dataset = OfficeDataset(test, valid_transform)
    val_dataset = OfficeDataset(val, valid_transform)

    return train_dataset, test_dataset, val_dataset
