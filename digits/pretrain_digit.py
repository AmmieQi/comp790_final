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
import models

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
    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--num_epochs", type = int, default = 20)
    parser.add_argument("--num_workers", type = int, default = 16)
    #training helpers
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, '{}_pretrain'.format(args.data_type))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.data_type == 'mnist':
        data_train = standard_dataloaders.mnist_dataloader(args.batch_size, train = True, shuffle = True)
        data_val = standard_dataloaders.mnist_dataloader(args.batch_size, train = False, shuffle = False)

    elif args.data_type == 'usps':
        data_train = standard_dataloaders.usps_dataloader(args.batch_size, train = True, shuffle = True)
        data_val = standard_dataloaders.usps_dataloader(args.batch_size, train = False, shuffle = False)

    elif args.data_type == 'svhn':
        data_train = standard_dataloaders.svhn_dataloader(args.batch_size, train = True, shuffle = True)
        data_val = standard_dataloaders.svhn_dataloader(args.batch_size, train = False, shuffle = False)
    else:
        print ("Invalid data type")
        exit()

    encoder = models.Encoder()
    # encoder = models.LeNetPP(num_classes = 10)
    encoder.to(device)

    classifier = models.Classifier(input_features = 84)
    classifier.to(device)

    cls_criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.SGD(itertools.chain(
                                classifier.parameters(),
                                encoder.parameters()) ,
                                        lr = args.lr, momentum = 0.9, nesterov = True)

    best_accuracy = -1
    for epoch_iter in range(args.num_epochs):

        running_epoch_loss = 0

        for iter, (data, labels) in enumerate(data_train):

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
