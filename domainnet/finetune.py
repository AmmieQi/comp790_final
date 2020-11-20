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

        for iter, (data, labels) in enumerate(tqdm.tqdm(val_dataloader)):
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

    parser.add_argument("--exp_type")
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--save_dir", type = str, default = '../weights/' )
    parser.add_argument("--data_path", type = str, default = '../data/DomainNet')
    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--K", type = int, default = 7)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--num_epochs", type = int, default = 50)
    parser.add_argument("--num_workers", type = int, default = 16)
    #training helpers
    args = parser.parse_args()

    pretrain_dir = os.path.join(args.save_dir, '{}_pretrain'.format(args.source))
    pretrain_cls_fp = os.path.join(pretrain_dir, 'best_{}_cls_pretrain.pth'.format(args.source))
    pretrain_encoder_fp = os.path.join(pretrain_dir, 'best_{}_encoder_pretrain.pth'.format(args.source))
    save_dir = '../weights/{}'.format(args.exp_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    encoder = torchvision.models.resnet101(pretrained=True)
    encoder.fc = Identity()
    classifier = torch.nn.Linear(2048, 50)


    encoder = torch.nn.DataParallel(encoder)
    classifier = torch.nn.DataParallel(classifier)

    encoder.load_state_dict(torch.load(pretrain_encoder_fp))

    encoder = encoder.to(device)
    classifier = classifier.to(device)

    cls_criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.SGD(itertools.chain(
                                classifier.parameters(),
                                encoder.parameters()) ,
                                        lr = args.lr, momentum = 0.9, nesterov = True)

    '''
    Get Target
    '''
    data_path = os.path.join(args.data_path, args.target)
    test_ims = os.path.join(args.data_path, '{}_test.txt'.format(args.target))
    train_ims = os.path.join(args.data_path, '{}_train.txt'.format(args.target))
    print (pretrain_dir, pretrain_cls_fp, test_ims, train_ims)
    target_data_train, target_data_val = data_utils.get_dataloaders(data_path, train_ims, test_ims, K = args.K)


    print (len(target_data_train), len(target_data_val))

    data_train = DataLoader(target_data_train, batch_size = args.batch_size, shuffle = True, drop_last = True)
    data_val = DataLoader(target_data_val, batch_size = 128, shuffle = False, drop_last = False, num_workers = args.num_workers)

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
            encoder_f = 'best_{}_encoder_pretrain.pth'.format(args.exp_type)
            cls_f = 'best_{}_cls_pretrain.pth'.format(args.exp_type)

            encoder_path = os.path.join(save_dir, encoder_f)
            cls_path = os.path.join(save_dir, cls_f)

            torch.save(encoder.state_dict(), encoder_path)
            torch.save(classifier.state_dict(), cls_path)
