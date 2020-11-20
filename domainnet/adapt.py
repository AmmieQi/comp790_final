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
import tqdm

sys.path.append('../')
from Trainer import trainer
from dataloaders import custom_dataloaders
import models
import data_utils

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

        for iter, (data, labels) in enumerate(tqdm.tqdm(val_dataloader)):
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
    parser.add_argument("--data_path", type = str, default = '../data/DomainNet')
    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.0003)
    parser.add_argument("--K", type = int, default = 7)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--iters", type = int, default = 5000)
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
    # encoder.to(device)

    classifier = torch.nn.Linear(2048, 50)
    # classifier.to(device)

    style_discrim = models.Style_Discrim(2048, 1024)
    # classifier.to(device)

    content_discrim = torch.nn.Linear(2048, 50)
    # content_discrim.to(device)

    fusion = models.Fusion()

    encoder = torch.nn.DataParallel(encoder)
    cls_head = torch.nn.DataParallel(classifier)
    fusion = torch.nn.DataParallel(fusion)
    style_discrim =  torch.nn.DataParallel(style_discrim)
    content_discrim =  torch.nn.DataParallel(content_discrim)

    encoder.load_state_dict(torch.load(pretrain_encoder_fp))

    args.style_discriminator_loss = .15
    args.content_discriminator_loss = .15
    args.style_generator_loss = .15
    args.content_generator_loss =  .15
    args.cls_loss = 1.0

    print (args)

    model_trainer = trainer.Trainer(loss_weights = args, encoder = encoder, cls_head = cls_head, fusion = fusion,
                    style_discriminator = style_discrim, content_discriminator = content_discrim,
                    device = device, save_dir = save_dir, lr = args.lr)

    '''
    Get Source
    '''
    data_path = os.path.join(args.data_path, args.source)
    test_ims = os.path.join(args.data_path, '{}_test.txt'.format(args.source))
    train_ims = os.path.join(args.data_path, '{}_train.txt'.format(args.source))

    source_data_train, source_data_val = data_utils.get_dataloaders(data_path, train_ims, test_ims, K = None)


    '''
    Get Target
    '''
    data_path = os.path.join(args.data_path, args.target)
    test_ims = os.path.join(args.data_path, '{}_test.txt'.format(args.target))
    train_ims = os.path.join(args.data_path, '{}_train.txt'.format(args.target))

    target_data_train, target_data_val = data_utils.get_dataloaders(data_path, train_ims, test_ims, K = args.K)


    print (len(source_data_train), len(target_data_train), len(source_data_val), len(target_data_val))

    source_data_train_dl = DataLoader(source_data_train, batch_size = args.batch_size, shuffle = True, drop_last = True, num_workers = args.num_workers)
    target_data_train_dl = DataLoader(target_data_train, batch_size = args.batch_size, shuffle = True, drop_last = True, num_workers = args.num_workers)

    source_data_val_dl = DataLoader(source_data_val, batch_size = 128, shuffle = False, drop_last = False, num_workers = args.num_workers)
    target_data_val_dl = DataLoader(target_data_val, batch_size = 128, shuffle = False, drop_last = False, num_workers = args.num_workers)

    source_train_cycle = cycle(source_data_train_dl)
    target_train_cycle = cycle(target_data_train_dl)

    best_target_acc = -1
    best_source_acc = -1

    pbar = tqdm.tqdm(total=100)

    for iter in range (args.iters):

        pbar.update()
        source_data, source_label = next(source_train_cycle)
        target_data, target_label = next(target_train_cycle)
        #[source data, source label, target data, target label]
        data_input = [source_data, source_label, target_data, target_label]
        model_trainer.pass_input(data_input)
        model_trainer.train()


        if (iter % 100) == 0:

            print ("Running Validation at Iter: {}".format(iter))

            # source_val_accuracy, source_bin_acc, source_style_space_acc = validate(source_data_val_dl, model_trainer.content_encoder, model_trainer.style_encoder, model_trainer.cls_head, model_trainer.fusion, 1, model_trainer.style_discrim, model_trainer.content_discrim)
            target_val_accuracy, target_bin_acc, target_style_space_acc = validate(target_data_val_dl, model_trainer.content_encoder, model_trainer.style_encoder, model_trainer.cls_head, model_trainer.fusion, 0, model_trainer.style_discrim, model_trainer.content_discrim)
            if target_val_accuracy > best_target_acc:
                best_target_acc = target_val_accuracy
                torch.save(model_trainer.content_encoder.state_dict(), os.path.join(save_dir, 'best_content_encoder.pt'))
                torch.save(model_trainer.style_encoder.state_dict(), os.path.join(save_dir, 'best_style_encoder.pt'))
                torch.save(model_trainer.cls_head.state_dict(), os.path.join(save_dir, 'best_cls_head.pt'))
                torch.save(model_trainer.fusion.state_dict(), os.path.join(save_dir, 'best_fusion.pt'))
                torch.save(model_trainer.style_discrim.state_dict(), os.path.join(save_dir, 'best_style_discrim.pt'))
                torch.save(model_trainer.content_discrim.state_dict(), os.path.join(save_dir, 'best_content_discrim.pt'))
                print ("Saving")
            # if source_val_accuracy > best_source_acc:
            #     best_source_acc = source_val_accuracy
            # print ("Source Val Acc: {} | Best Source Val Acc: {} | Target Val Acc: {} | Best Target Val Acc: {} |".format(source_val_accuracy, best_source_acc, target_val_accuracy, best_target_acc))
            # print ("Source Bin Acc: {} | Target Bin Acc: {} | Source style space acc {} | Target style space acc {} |".format(source_bin_acc, target_bin_acc, source_style_space_acc, target_style_space_acc))
            print ("Target val acc: {} | Best target val acc: {}".format(target_val_accuracy, best_target_acc))
            pbar.reset()
