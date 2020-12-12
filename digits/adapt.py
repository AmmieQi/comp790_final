import os
import itertools
import random
import argparse
import sys
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append('../')
from Trainer import trainer
from dataloaders import custom_dataloaders
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

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def validate(val_dataloader, content_encoder, style_encoder, classifier, fusion, style_num, style_discrim, content_discrim):

    correct = 0
    total = 0

    content_encoder.eval()
    style_encoder.eval()
    style_discrim.eval()
    content_discrim.eval()

    #style num 1 -> mnist, 0 ->svhn
    binary_correct = 0
    binary_total = 0

    style_space_correct = 0

    with torch.no_grad():

        for iter, (data, labels) in enumerate(val_dataloader):
            data = data.to(device)

            labels = labels.to(device)

            encoded_style = style_encoder(data)
            encoded_content = content_encoder(data)
            encoded = fusion(encoded_style , encoded_content, style_num)

            #style discrim should be at 0.5 accuracy for encoded_content
            correct_style = [style_num] * len(encoded_content)
            style_pred_on_content = torch.sigmoid(style_discrim(encoded_content)).detach().cpu().numpy()
            style_pred_on_content[style_pred_on_content < 0.5] = 0
            style_pred_on_content[style_pred_on_content >= 0.5] = 1

            binary_correct += (style_pred_on_content == correct_style).sum().item()
            binary_total += labels.size(0)

            #content discrim. should be 0.1 accuracy for encoded_Style
            preds_style_space = content_discrim(encoded_style)
            _, preds_style_space = torch.max(preds_style_space.data, 1)
            style_space_correct += (preds_style_space == labels).sum().item()

            preds = classifier(encoded)

            _, predicted = torch.max(preds.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    bin_acc = binary_correct / binary_total
    style_space_acc = style_space_correct / total

    content_encoder.train()
    style_encoder.train()
    style_discrim.train()
    content_discrim.train()

    return accuracy, bin_acc, style_space_acc



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_type")
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--save_dir", type = str, default = '../weights/' )
    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.0003)
    parser.add_argument("--K", type = int, default = 7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--iters", type = int, default = 1000000)
    parser.add_argument("--num_workers", type = int, default = 16)
    #training helpers
    args = parser.parse_args()

    pretrain_dir = os.path.join(args.save_dir, '{}_pretrain'.format(args.source))
    pretrain_cls_fp = os.path.join(pretrain_dir, 'best_{}_cls_pretrain.pth'.format(args.source))
    pretrain_encoder_fp = os.path.join(pretrain_dir, 'best_{}_encoder_pretrain.pth'.format(args.source))
    save_dir = '../weights/{}'.format(args.exp_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    encoder = models.Encoder()
    # encoder = models.LeNetPP(10)
    cls_head = models.Classifier(84)
    encoder.load_state_dict(torch.load(pretrain_encoder_fp))
    fusion = models.Fusion()
    style_discrim = models.Style_Discrim(84)
    content_discrim = models.Content_Discrim(84)

    args.style_discriminator_loss = .2
    args.content_discriminator_loss = .1
    args.style_generator_loss = .2
    args.content_generator_loss =  .1

    # args.style_discriminator_loss = 1
    # args.content_discriminator_loss = 1
    # args.style_generator_loss = 1
    # args.content_generator_loss =  1

    args.cls_loss = 1.0

    print (args)

    model_trainer = trainer.Trainer(loss_weights = args, encoder = encoder, cls_head = cls_head, fusion = fusion,
                    style_discriminator = style_discrim, content_discriminator = content_discrim,
                    device = device, save_dir = save_dir, lr = args.lr)

    if args.source == 'mnist':
        source_data_train = custom_dataloaders.MNIST(root = '/app/fsl_da/data/MNIST', train = True,
                             download=False,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ]),
                            n_samples = None)
        source_data_val = custom_dataloaders.MNIST(root = '/app/fsl_da/data/MNIST', train = False,
                             download=False,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                 ]),
                            n_samples = None
                           )
    elif args.source == 'svhn':
        source_data_train = custom_dataloaders.SVHN('/app/fsl_da/data/SVHN', split='train', download=False,
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                           ]),
                           n_samples =None
                           )
        source_data_val = custom_dataloaders.SVHN('/app/fsl_da/data/SVHN', split='test', download=False,
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                           ]),
                           n_samples = None
                           )

    elif args.source == 'usps':
        source_data_train = custom_dataloaders.USPS('/app/fsl_da/data/usps', train = True, download=False,
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                           ]),
                           n_samples =None
                           )
        source_data_val = custom_dataloaders.USPS('/app/fsl_da/data/usps', train = False, download=False,
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                           ]),
                           n_samples = None
                           )
    else:
        print ("Incorrect source data name")
        exit()

    if args.target == 'mnist':
        target_data_train = custom_dataloaders.MNIST(root = '/app/fsl_da/data/MNIST', train = True,
                             download=False,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ]),
                            n_samples = args.K)

        target_data_val = custom_dataloaders.MNIST(root = '/app/fsl_da/data/MNIST', train = False,
                             download=False,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                 ]),
                            n_samples = None
                           )
    elif args.target == 'svhn':
        target_data_train = custom_dataloaders.SVHN('/app/fsl_da/data/SVHN', split='train', download=False,
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                           ]),
                           n_samples = args.K
                           )
        target_data_val = custom_dataloaders.SVHN('/app/fsl_da/data/SVHN', split='test', download=False,
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                           ]),
                           n_samples = None
                           )

    elif args.target == 'usps':
        target_data_train = custom_dataloaders.USPS('/app/fsl_da/data/usps', train = True, download=False,
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                           ]),
                           n_samples =args.K
                           )
        target_data_val = custom_dataloaders.USPS('/app/fsl_da/data/usps', train = False, download=False,
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.Grayscale(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5), (0.5)),
                           ]),
                           n_samples = None
                           )
    else:
        print ("Incorrect source data name")
        exit()


    print (len(source_data_train), len(target_data_train), len(source_data_val), len(target_data_val))

    source_data_train_dl = DataLoader(source_data_train,
            batch_size = args.batch_size, shuffle = True,
            drop_last = True, num_workers = args.num_workers)

    target_data_train_dl = DataLoader(target_data_train,
            batch_size = args.batch_size, shuffle = True,
            drop_last = True, num_workers = args.num_workers)

    source_data_val_dl = DataLoader(source_data_val, batch_size = 256,
            shuffle = False, drop_last = False, num_workers = args.num_workers)
    target_data_val_dl = DataLoader(target_data_val, batch_size = 256,
            shuffle = False, drop_last = False, num_workers = args.num_workers)

    source_train_cycle = cycle(source_data_train_dl)
    target_train_cycle = cycle(target_data_train_dl)

    best_target_acc = -1
    best_source_acc = -1


    for iter in range (args.iters):

        source_data, source_label = next(source_train_cycle)
        target_data, target_label = next(target_train_cycle)
        #[source data, source label, target data, target label]
        data_input = [source_data, source_label, target_data, target_label]
        model_trainer.pass_input(data_input)
        model_trainer.train()

        if (iter % 100) == 0:

            print ("Running Validation at Iter: {}".format(iter))

            source_val_accuracy, source_bin_acc, source_style_space_acc = validate(source_data_val_dl, model_trainer.content_encoder, model_trainer.style_encoder, model_trainer.cls_head, model_trainer.fusion, 1, model_trainer.style_discrim, model_trainer.content_discrim)
            target_val_accuracy, target_bin_acc, target_style_space_acc = validate(target_data_val_dl, model_trainer.content_encoder, model_trainer.style_encoder, model_trainer.cls_head, model_trainer.fusion, 0, model_trainer.style_discrim, model_trainer.content_discrim)
            if target_val_accuracy > best_target_acc:
                best_target_acc = target_val_accuracy
            if source_val_accuracy > best_source_acc:
                best_source_acc = source_val_accuracy
            print ("Source Val Acc: {} | Best Source Val Acc: {} | Target Val Acc: {} | Best Target Val Acc: {} |".format(source_val_accuracy, best_source_acc, target_val_accuracy, best_target_acc))
            print ("Source Bin Acc: {} | Target Bin Acc: {} | Source style space acc {} | Target style space acc {} |".format(source_bin_acc, target_bin_acc, source_style_space_acc, target_style_space_acc))
