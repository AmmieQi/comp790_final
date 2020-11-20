import torch
from models import main_models
import dataloader_character_mine
import pdb


def test_model(epoch, encoder, classifier, is_M2S):
    if is_M2S:
        test_dataloader = dataloader_character_mine.svhn_dataloader(train=False, batch_size=batch_size)
    else:
        test_dataloader = dataloader_character_mine.mnist_dataloader(train=False, batch_size=batch_size)
    # testing
    acc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier(encoder(data))
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
    accuracy = round(acc / float(len(test_dataloader)), 3)
    print("Epoch %d/%d  accuracy: %.3f " % (epoch + 1, num_epochs, accuracy))


def train_model(num_epochs, num_samples, is_M2S):
    # define model
    classifier = main_models.Classifier()
    encoder = main_models.Encoder()
    discriminator = main_models.DCD(input_features=168)
    classifier.to(device)
    encoder.to(device)
    discriminator.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_g_h = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)
    optimizer_d_h = torch.optim.Adam(list(discriminator.parameters()) + list(classifier.parameters()), lr=0.001)

    # get all source training data and "num_sample" of target training data
    X_s, Y_s, X_t, Y_t = dataloader_character_mine.get_train_data(num_samples, is_M2S)

    for epoch in range(num_epochs):
        # ---training g and h , DCD is frozen
        groups, groups_y = dataloader_character_mine.create_groups(X_s, Y_s, X_t, Y_t, seed=num_epochs + epoch)

        n_data = len(groups[0] + groups[1] + groups[2] + groups[3])
        n_iters = n_data
        index_list = torch.randperm(n_iters)

        n_iters_dcd = n_data
        index_list_dcd = torch.randperm(n_iters_dcd)    # shuffle

        mini_batch_size_g_h = 40
        mini_batch_size_dcd = 40
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels = []
        # extract all groups of data based on the shuffled order
        for index in range(n_iters):
            temp1 = index_list[index] // int(n_data / 2)
            temp2 = index_list[index] % int(n_data / 2)
            if temp1 == 0:
                if temp2 < len(groups[0]):
                    ground_truth = 0
                    x1, x2 = groups[ground_truth][index_list[index]]
                    y1, y2 = groups_y[ground_truth][index_list[index]]
                else:
                    ground_truth = 1
                    x1, x2 = groups[ground_truth][index_list[index] - len(groups[0])]
                    y1, y2 = groups_y[ground_truth][index_list[index] - len(groups[0])]
            else:
                if temp2 < len(groups[2]):
                    ground_truth = 2
                    x1, x2 = groups[ground_truth][index_list[index] - len(groups[0]) - len(groups[1])]
                    y1, y2 = groups_y[ground_truth][index_list[index] - len(groups[0]) - len(groups[1])]
                else:
                    ground_truth = 3
                    x1, x2 = groups[ground_truth][index_list[index] - len(groups[0]) - len(groups[1]) - len(groups[2])]
                    y1, y2 = groups_y[ground_truth][
                        index_list[index] - len(groups[0]) - len(groups[1]) - len(groups[2])]

            dcd_label = ground_truth
            X1.append(x1)
            X2.append(x2)
            ground_truths_y1.append(y1)
            ground_truths_y2.append(y2)
            dcd_labels.append(dcd_label)

            # train
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

        # ----training dcd ,g and h frozen
        X1 = []
        X2 = []
        ground_truths = []
        # get all group label for the shuffled data
        for index in range(n_iters_dcd):
            temp1 = index_list_dcd[index] // int(n_data / 2)
            temp2 = index_list_dcd[index] % int(n_data / 2)
            if temp1 == 0:
                if temp2 < len(groups[0]):
                    ground_truth = 0
                    x1, x2 = groups[ground_truth][index_list_dcd[index]]
                else:
                    ground_truth = 1
                    x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[0])]
            else:
                if temp2 < len(groups[2]):
                    ground_truth = 2
                    x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[0]) - len(groups[1])]
                else:
                    ground_truth = 3
                    x1, x2 = groups[ground_truth][
                        index_list_dcd[index] - len(groups[0]) - len(groups[1]) - len(groups[2])]

            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            # train
            if (index + 1) % mini_batch_size_dcd == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_d_h.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_d_h.step()
                X1 = []
                X2 = []
                ground_truths = []
        test_model(epoch, encoder, classifier, is_M2S)


if __name__ == '__main__':
    is_M2S = True

    batch_size = 64
    num_epochs = 300
    num_samples = 7

    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed(123)

    train_model(num_epochs, num_samples, is_M2S)
