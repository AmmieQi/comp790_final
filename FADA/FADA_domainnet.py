import torch
import FADA.dataloader_domainnet as dataloader_domainnet
from FADA.models import main_models
import numpy as np
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description='PyTorch FA DA')
parser.add_argument("--n_classes", type=int, default=7, help="Number of classes of ssl")

args = parser.parse_args()

use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)


def train_gh(num_epochs):
    # load data
    train_dataloader, _ = dataloader_domainnet.return_dataset('real_train.txt', 'train')
    test_dataloader, _ = dataloader_domainnet.return_dataset('real_test.txt', 'test')

    # define model
    classifier = main_models.Classifier_DomainNet()

    encoder = models.resnet101(pretrained=True)
    encoder.fc = main_models.Identity()

    classifier.to(device)
    encoder.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=0.05)

    for epoch in range(num_epochs):
        # train
        loss_mean = []
        for data, labels in train_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            y_pred = classifier(encoder(data))
            loss = loss_fn(y_pred, labels)
            loss_mean.append(loss.item())
            loss.backward()
            optimizer.step()
        # test accuracy
        acc = 0
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
        accuracy = round(acc / float(len(test_dataloader)), 3)
        print("step1----Epoch %d/%d  training loss:%.3f  testing accuracy: %.3f " % (
        epoch + 1, num_epochs, np.mean(loss_mean), accuracy))
        if epoch % 50 == 49:
            torch.save(encoder.state_dict(), "encoder_" + str(epoch + 1) + ".pt")
            torch.save(classifier.state_dict(), "classifier_" + str(epoch + 1) + ".pt")
    return encoder, classifier


def train_dcd(num_epochs, encoder, X_s, Y_s, X_t, Y_t, num_samples):
    discriminator = main_models.DCD_DomainNet(input_features=4096)
    discriminator.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.005)

    for epoch in range(num_epochs):
        # data
        groups, aa = dataloader_domainnet.sample_groups(X_s, Y_s, X_t, Y_t, seed=epoch, num_samples=num_samples)

        n_iters = 4 * len(groups[1])
        index_list = torch.randperm(n_iters)
        mini_batch_size = 16

        X1 = []
        X2 = []
        loss_mean = []
        ground_truths = []
        for index in range(n_iters):

            ground_truth = index_list[index] // len(groups[1])

            x1, x2 = groups[ground_truth][index_list[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            # select data for a mini-batch to train
            if (index + 1) % mini_batch_size == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_D.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_D.step()
                loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []
        print("step2----Epoch %d/%d loss:%.3f" % (epoch + 1, num_epochs, np.mean(loss_mean)))
    return discriminator


def train_all(encoder, discriminator, classifier, num_epochs, X_s, Y_s, X_t, Y_t, num_samples):
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer_g_h = torch.optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=0.005)
    optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=0.005)

    test_dataloader, _ = dataloader_domainnet.return_dataset('clipart_test.txt', 'test')

    for epoch in range(num_epochs):
        # training g and h , DCD is frozen

        groups, groups_y = dataloader_domainnet.sample_groups(X_s, Y_s, X_t, Y_t, seed=num_epochs + epoch,
                                                              num_samples=num_samples)
        G1, G2, G3, G4 = groups
        Y1, Y2, Y3, Y4 = groups_y
        groups_2 = [G2, G4]
        groups_y_2 = [Y2, Y4]

        n_iters = 2 * len(G2)
        index_list = torch.randperm(n_iters)

        n_iters_dcd = 4 * len(G2)
        index_list_dcd = torch.randperm(n_iters_dcd)

        mini_batch_size_g_h = 8  # data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_dcd = 16  # data contains G1,G2,G3,G4 so use 40 as mini_batch
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels = []
        for index in range(n_iters):

            ground_truth = index_list[index] // len(G2)
            x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]

            dcd_label = 0 if ground_truth == 0 else 2
            X1.append(x1)
            X2.append(x2)
            ground_truths_y1.append(y1)
            ground_truths_y2.append(y2)
            dcd_labels.append(dcd_label)

            if (index + 1) % mini_batch_size_g_h == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths_y1 = torch.LongTensor(ground_truths_y1)
                ground_truths_y2 = torch.LongTensor(ground_truths_y2)
                dcd_labels = torch.LongTensor(dcd_labels)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths_y1 = ground_truths_y1.to(device)
                ground_truths_y2 = ground_truths_y2.to(device)
                dcd_labels = dcd_labels.to(device)

                optimizer_g_h.zero_grad()

                encoder_X1 = encoder(X1)
                encoder_X2 = encoder(X2)

                X_cat = torch.cat([encoder_X1, encoder_X2], 1)
                y_pred_X1 = classifier(encoder_X1)
                y_pred_X2 = classifier(encoder_X2)
                y_pred_dcd = discriminator(X_cat)

                loss_X1 = loss_fn(y_pred_X1, ground_truths_y1)
                loss_X2 = loss_fn(y_pred_X2, ground_truths_y2)
                loss_dcd = loss_fn(y_pred_dcd, dcd_labels)

                loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd

                loss_sum.backward()
                optimizer_g_h.step()

                X1 = []
                X2 = []
                ground_truths_y1 = []
                ground_truths_y2 = []
                dcd_labels = []

        # training dcd ,g and h frozen
        X1 = []
        X2 = []
        ground_truths = []
        for index in range(n_iters_dcd):

            ground_truth = index_list_dcd[index] // len(groups[1])

            x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            if (index + 1) % mini_batch_size_dcd == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_d.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_d.step()
                X1 = []
                X2 = []
                ground_truths = []

        # testing
        acc = 0
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(test_dataloader)), 3)

        print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, num_epochs, accuracy))


if __name__ == '__main__':
    batch_size = 32
    num_epochs_step1 = 100
    num_epochs_step2 = 100
    num_epochs_step3 = 200
    num_samples = args.n_classes

    ### step 1: train feature extractor(g) and classifier(h) ###
    encoder, classifier = train_gh(num_epochs_step1)

    ### step 2: train DCD ###
    X_s, Y_s = dataloader_domainnet.sample_data()
    X_t, Y_t = dataloader_domainnet.create_target_samples(num_samples)

    discriminator = train_dcd(num_epochs_step2, encoder, X_s, Y_s, X_t, Y_t, num_samples)

    ### step 3: train all ###
    train_all(encoder, discriminator, classifier, num_epochs_step3, X_s, Y_s, X_t, Y_t, num_samples)
