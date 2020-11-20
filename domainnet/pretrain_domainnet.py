import os
import itertools
import random
import argparse
import torch
import numpy as np
import torchvision
import sys
sys.path.append("../")
from dataloaders import standard_dataloaders
from torch.utils.data import Dataset, DataLoader
import models
import data_utils
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAND_SEED = 1
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def validate(val_dataloader, encoder, classifier):

    correct = 0
    total = 0

    with torch.no_grad():

        for iter, (data, labels) in enumerate(val_dataloader):
            data = data.to(device)
            labels = labels.to(device)

            encoded = encoder(data)
            preds = classifier(encoded)

            _, predicted = torch.max(preds.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_type")
    parser.add_argument("--save_dir", type = str, default = '../weights/' )
    parser.add_argument("--data_path", type = str, default = '../data/DomainNet')
    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--num_epochs", type = int, default = 20)
    parser.add_argument("--num_workers", type = int, default = 16)
    #training helpers
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, '{}_pretrain'.format(args.data_type))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_path = os.path.join(args.data_path, args.data_type)
    test_ims = os.path.join(args.data_path, '{}_test.txt'.format(args.data_type))
    train_ims = os.path.join(args.data_path, '{}_train.txt'.format(args.data_type))

    data_train, data_test = data_utils.get_dataloaders(data_path, train_ims, test_ims)
    data_train = DataLoader(data_train, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    data_val = DataLoader(data_test, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    encoder = torchvision.models.resnet101(pretrained=True)
    encoder.fc = Identity()
    encoder.to(device)

    classifier = torch.nn.Linear(2048, 345)
    classifier.to(device)

    encoder = torch.nn.DataParallel(encoder)
    classifier = torch.nn.DataParallel(classifier)

    cls_criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.SGD(itertools.chain(
                                classifier.parameters(),
                                encoder.parameters()) ,
                                        lr = args.lr, momentum = 0.9, nesterov = True)

    best_accuracy = -1
    for epoch_iter in range(args.num_epochs):

        running_epoch_loss = 0

        for iter, (data, labels) in enumerate(tqdm.tqdm(data_train)):


            optimizer_cls.zero_grad()

            data = data.to(device)
            labels = labels.to(device)

            encoded = encoder(data)
            preds = classifier(encoded)

            cls_loss = cls_criterion(preds, labels)

            cls_loss.backward()
            optimizer_cls.step()

            running_epoch_loss += cls_loss.item()



        epoch_loss = running_epoch_loss / len(data_train)

        accuracy = validate(data_val, encoder, classifier)

        print ("|Epoch: {} | Epoch Loss: {} | Val Accuracy: {}|".format(epoch_iter+1, epoch_loss, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            encoder_f = 'best_{}_encoder_pretrain.pth'.format(args.data_type)
            cls_f = 'best_{}_cls_pretrain.pth'.format(args.data_type)

            encoder_path = os.path.join(save_dir, encoder_f)
            cls_path = os.path.join(save_dir, cls_f)

            torch.save(encoder.state_dict(), encoder_path)
            torch.save(classifier.state_dict(), cls_path)
