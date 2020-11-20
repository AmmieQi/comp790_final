import os
import itertools
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataloaders import custom_dataloaders
import models



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

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_entropy_loss(out):
    return -torch.mean(torch.log(torch.nn.functional.softmax(out + 1e-6, dim=-1)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAND_SEED = 1
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
K = 7
NUM_ITERS = 50000
BATCH_SIZE = 32
LR = 0.0002
DISCRIM_LR = 0.0002

print( "Training for {} Iters".format(NUM_ITERS))

pretrain_weight =  'weights/mnist_pretrain/'
pretrain_cls_fp = os.path.join(pretrain_weight, 'best_mnist_cls_pretrain.pth')
pretrain_encoder_fp = os.path.join(pretrain_weight, 'best_mnist_encoder_pretrain.pth')
save_dir = 'weights/mnist_to_svhn'


mnist_data_train = custom_dataloaders.MNIST(root = './data/MNIST', train = True,
                     download=True,
                     transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                        ]),
                    n_samples = None
                   )
svhn_data_train = custom_dataloaders.SVHN('./data/SVHN', split='train', download=True,
                   transform=transforms.Compose([
                       transforms.Resize((28,28)),
                       transforms.Grayscale(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5), (0.5)),
                   ]),
                   n_samples = None
                   )

mnist_data_val = custom_dataloaders.MNIST(root = './data/MNIST', train = False,
                     download=True,
                     transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                         ]),
                    n_samples = K
                   )
svhn_data_val = custom_dataloaders.SVHN('./data/SVHN', split='test', download=True,
                   transform=transforms.Compose([
                       transforms.Resize((28,28)),
                       transforms.Grayscale(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5), (0.5)),
                   ]),
                   n_samples = None
                   )

print (len(mnist_data_train), len(svhn_data_train), len(mnist_data_val), len(svhn_data_val))

mnist_data_train_dl = DataLoader(mnist_data_train, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
svhn_data_train_dl = DataLoader(svhn_data_train, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

mnist_data_val_dl = DataLoader(mnist_data_val, batch_size = 256, shuffle = False, drop_last = False)
svhn_data_val_dl = DataLoader(svhn_data_val, batch_size = 256, shuffle = False, drop_last = False)

mnist_train_cycle = cycle(mnist_data_train_dl)
svhn_train_cycle = cycle(svhn_data_train_dl)

style_encoder = models.Encoder()
style_encoder.to(device)

content_encoder = models.Encoder()
content_encoder.to(device)

style_discrim = models.Style_Discrim()
style_discrim.to(device)

content_discrim = models.Content_Discrim()
content_discrim.to(device)

dcd = models.DCD(h_features=32,input_features=84)
dcd.to(device)

classifier = models.Classifier()
classifier.to(device)

fusion = models.Fusion()
fusion.to(device)

style_encoder.load_state_dict(torch.load(pretrain_encoder_fp))
content_encoder.load_state_dict(torch.load(pretrain_encoder_fp))
classifier.load_state_dict(torch.load(pretrain_cls_fp))


cls_criterion = torch.nn.CrossEntropyLoss(ignore_index = -1)
binary_criterion = torch.nn.BCEWithLogitsLoss()
triplet_criterion = torch.nn.TripletMarginLoss(margin = 1.0)

optimizer_style_discrim = torch.optim.Adam( style_discrim.parameters(), lr = DISCRIM_LR )
optimizer_content_discrim = torch.optim.Adam( content_discrim.parameters(), lr = DISCRIM_LR )
optimizer_dcd = torch.optim.Adam(dcd.parameters(), lr = DISCRIM_LR )
optimizer_cls = torch.optim.Adam(itertools.chain(
                            classifier.parameters(),
                            fusion.parameters(),
                            style_encoder.parameters(),
                            content_encoder.parameters()) ,
                                    lr = LR  )


epoch_approx = (len(svhn_data_train) // BATCH_SIZE)
best_svhn_acc = -1
best_mnist_acc = -1
running_content_discrim_loss = 0
running_style_discrim_loss = 0
running_adv_content_loss = 0
running_adv_style_loss  = 0
running_seq_loss = 0
running_rs_rc_loss = 0
running_rs_sc_loss = 0
running_ss_rc_loss = 0
running_ss_sc_loss = 0
running_dcd_loss = 0
running_dcd_adv_loss = 0

for iter in range(1, NUM_ITERS+1):

    optimizer_style_discrim.zero_grad()
    optimizer_content_discrim.zero_grad()
    # optimizer_dcd.zero_grad()
    optimizer_cls.zero_grad()

    mnist_data, mnist_label = next(mnist_train_cycle)
    svhn_data, svhn_label = next(svhn_train_cycle)

    mnist_data = mnist_data.to(device)
    mnist_label  = mnist_label.to(device)
    svhn_data = svhn_data.to(device)
    svhn_label = svhn_label.to(device)

    mnist_style = style_encoder(mnist_data)
    svhn_style = style_encoder(svhn_data)

    mnist_content = content_encoder(mnist_data)
    svhn_content = content_encoder(svhn_data)

    content_concat = torch.cat((mnist_content, svhn_content), 0)
    style_concat = torch.cat((mnist_style, svhn_style), 0)

    '''
    Step 1) Train Style Discriminator Adversarially
    '''
    #discrim
    style_mnist_labels = torch.tensor(np.ones((mnist_data.size(0))), requires_grad = False).float().to(device)
    style_svhn_labels = torch.tensor(np.zeros((svhn_data.size(0))), requires_grad = False).float().to(device)
    style_labels = torch.cat((style_mnist_labels, style_svhn_labels ), 0)
    style_discriminator_predictions = style_discrim(content_concat.detach()) #detach when training discriminator
    style_discriminator_loss = binary_criterion(style_discriminator_predictions, style_labels)

    #generator
    style_adversarial_labels = torch.cat((style_svhn_labels, style_mnist_labels), 0)
    style_adversarial_predictions = style_discrim(content_concat)
    style_adversarial_loss = binary_criterion(style_adversarial_predictions, style_adversarial_labels)


    '''
    Step 2) Train Content Discriminator Adversarially
    '''
    content_labels = torch.cat((mnist_label, mnist_label), 0)
    content_discriminator_predictions  = content_discrim(style_concat.detach())
    content_discriminator_loss = cls_criterion(content_discriminator_predictions, content_labels)

    #2) Adversarial
    content_adversarial_predictions = content_discrim(style_concat)
    content_adversarial_loss = get_entropy_loss(content_adversarial_predictions)


    '''
    Step 3) Form pairs
    '''
    rs_rc_encoded = fusion(mnist_style , mnist_content, 0)
    rs_sc_encoded = fusion(mnist_style , svhn_content, 0)
    ss_rc_encoded = fusion(svhn_style , mnist_content, 1)
    ss_sc_encoded = fusion(svhn_style , svhn_content, 1)

    rs_rc_labels = torch.tensor(np.full(rs_rc_encoded.size(0),0 ), requires_grad = False).to(device)
    rs_sc_labels = torch.tensor(np.full(rs_sc_encoded.size(0),1 ), requires_grad = False).to(device)
    ss_rc_labels = torch.tensor(np.full(ss_rc_encoded.size(0),2 ), requires_grad = False).to(device)
    ss_sc_labels = torch.tensor(np.full(ss_sc_encoded.size(0),3 ), requires_grad = False).to(device)

    '''
    Step 5) Prediction
    '''

    rs_rc_preds = classifier(rs_rc_encoded)
    rs_sc_preds = classifier(rs_sc_encoded)
    ss_rc_preds = classifier(ss_rc_encoded)
    ss_sc_preds = classifier(ss_sc_encoded)


    rs_rc_loss = cls_criterion(rs_rc_preds, mnist_label)
    ss_rc_loss = cls_criterion(ss_rc_preds, mnist_label)

    cls_loss = (rs_rc_loss +  ss_rc_loss ) / 2.

    running_content_discrim_loss += content_discriminator_loss.item()
    running_style_discrim_loss += style_discriminator_loss.item()
    running_adv_content_loss += content_adversarial_loss.item()
    running_adv_style_loss += style_adversarial_loss.item()
    running_seq_loss += cls_loss.item()
    running_rs_rc_loss += rs_rc_loss.item()
    running_rs_sc_loss += 0
    running_ss_rc_loss += ss_rc_loss.item()
    running_ss_sc_loss += 0
    running_dcd_loss += 0
    running_dcd_adv_loss += 0

    style_discriminator_loss *= .05
    content_discriminator_loss *= .05
    dcd_discrim_loss = 0
    style_adversarial_loss *= 0.05
    content_adversarial_loss *=  0.05
    dcd_adv_loss =  0
    cls_loss *= 1.0


    cls_loss = cls_loss + content_adversarial_loss + style_adversarial_loss #+ dcd_adv_loss

    # print (cls_loss.item(), content_adversarial_loss.item(), style_adversarial_loss.item(), dcd_adv_loss.item())

    style_discriminator_loss.backward(retain_graph = True)
    content_discriminator_loss.backward(retain_graph = True)
    # dcd_discrim_loss.backward(retain_graph = True)
    cls_loss.backward()

    optimizer_style_discrim.step()
    optimizer_content_discrim .step()
    # optimizer_dcd.step()
    optimizer_cls.step()





    if (iter % 100) == 0:

        print ("Running Validation at Iter: {}".format(iter))

        mnist_val_accuracy, mnist_bin_acc, mnist_style_space_acc = validate(mnist_data_val_dl, content_encoder, style_encoder, classifier, fusion, 1, style_discrim, content_discrim)
        svhn_val_accuracy, svhn_bin_acc, svhn_style_space_acc = validate(svhn_data_val_dl, content_encoder, style_encoder, classifier, fusion, 0, style_discrim, content_discrim)
        if svhn_val_accuracy > best_svhn_acc:
            best_svhn_acc = svhn_val_accuracy
        if mnist_val_accuracy > best_mnist_acc:
            best_mnist_acc = mnist_val_accuracy
        print ("Mnist Val Acc: {} | Best Mnist Val Acc: {} | SVHN Val Acc: {} | Best SVHN Val Acc: {} |".format(mnist_val_accuracy, best_mnist_acc, svhn_val_accuracy, best_svhn_acc))
        print ("Mnist Bin Acc: {} | SVHN Bin Acc: {} | Mnist style space acc {} | SVHN style space acc {} |".format(mnist_bin_acc, svhn_bin_acc, mnist_style_space_acc, svhn_style_space_acc))
        this_content_disc_loss = running_content_discrim_loss / 100
        this_style_disc_loss = running_style_discrim_loss / 100
        this_content_adv_loss = running_adv_content_loss / 100
        this_style_adv_loss = running_adv_style_loss / 100
        this_seq_loss = running_seq_loss / 100
        this_rs_rc_loss = running_rs_rc_loss / 100
        this_rs_sc_loss = running_rs_sc_loss / 100
        this_ss_rc_loss = running_ss_rc_loss / 100
        this_ss_sc_loss = running_ss_sc_loss / 100
        this_dcd_loss = running_dcd_loss / 100
        this_dcd_adv_loss = running_dcd_adv_loss / 100

        print ("|Content Discrim Loss {} | Style Discrim Loss {} |"
                     "Adv Content Loss {} | Adv Style Loss {} | Seq Loss {} |"
                     "rs_rc Loss {} | rs_sc Loss {} | ss_rc Loss {} | ss_ss Loss {} | "
                     "Joint Discrim Loss {} | Joint Adv Loss {} ". \
                    format(this_content_disc_loss, this_style_disc_loss,
                             this_content_adv_loss,
                             this_style_adv_loss, this_seq_loss, this_rs_rc_loss,
                             this_rs_sc_loss, this_ss_rc_loss, this_ss_sc_loss,
                             this_dcd_loss, this_dcd_adv_loss))
        print ("="*30)
        running_content_discrim_loss = 0
        running_style_discrim_loss = 0
        running_adv_content_loss = 0
        running_adv_style_loss = 0
        running_seq_loss = 0
        running_rs_rc_loss = 0
        running_rs_sc_loss = 0
        running_ss_rc_loss = 0
        running_ss_sc_loss = 0
        running_dcd_loss = 0
        running_dcd_adv_loss = 0
