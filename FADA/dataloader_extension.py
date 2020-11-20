import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def mnist_dataloader(batch_size=256, train=True):
    dataloader = DataLoader(
        datasets.MNIST('./data/mnist', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                           transforms.Normalize([0.5], [0.5])
                       ])),
        batch_size=batch_size, shuffle=True)

    return dataloader


def svhn_dataloader(batch_size=4, train=True):
    dataloader = DataLoader(
        datasets.SVHN('./data/SVHN', split=('train' if train else 'test'), download=True,
                      transform=transforms.Compose([
                          transforms.Resize((28, 28)),
                          transforms.Grayscale(),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])
                      ])),
        batch_size=batch_size, shuffle=False)

    return dataloader


def get_source_data(is_M2S):
    if is_M2S:
        source_dataset = datasets.MNIST('./data/mnist', train=True, download=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            transforms.Normalize([0.5], [0.5])
                                        ]))
    else:
        source_dataset = datasets.SVHN('./data/SVHN', split='train', download=False,
                                       transform=transforms.Compose([
                                           transforms.Resize((28, 28)),
                                           transforms.Grayscale(),
                                           transforms.ToTensor(),
                                           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           transforms.Normalize([0.5], [0.5])
                                       ]))
    X, Y = [], []
    i = 0
    while i < len(source_dataset):
        x, y = source_dataset[i]
        X.append(x)
        Y.append(y)
        i += 1
    return torch.stack(X, dim=0), torch.from_numpy(np.array(Y))


def get_target_data(num_samples, is_M2S):
    if is_M2S:
        target_dataset = datasets.SVHN('./data/SVHN', split='train', download=False,
                                       transform=transforms.Compose([
                                           transforms.Resize((28, 28)),
                                           transforms.Grayscale(),
                                           transforms.ToTensor(),
                                           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           transforms.Normalize([0.5], [0.5])
                                       ]))
    else:
        target_dataset = datasets.MNIST('./data/mnist', train=True, download=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            transforms.Normalize([0.5], [0.5])
                                        ]))
    X, Y = [], []
    classes = 10 * [num_samples]
    i = 0
    while True:
        if len(X) == num_samples * 10:
            break
        x, y = target_dataset[i]
        if classes[y] > 0:
            X.append(x)
            Y.append(y)
            classes[y] -= 1
        i += 1

    assert (len(X) == num_samples * 10)
    return torch.stack(X, dim=0), torch.from_numpy(np.array(Y))


def get_train_data(num_samples, is_M2S):
    X_s, Y_s = get_source_data(is_M2S)
    X_t, Y_t = get_target_data(num_samples, is_M2S)
    return X_s, Y_s, X_t, Y_t


def create_groups(X_s, Y_s, X_t, Y_t, seed=1):
    # change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    n = X_t.shape[0]  # 10*shot

    # shuffle order
    classes = torch.unique(Y_t)
    classes = classes[torch.randperm(len(classes))]  # shuffle 0~9

    class_num = classes.shape[0]  # 10
    shot = n // class_num  # 7

    def s_idxs(c):
        idx = torch.nonzero(Y_s.eq(int(c)))
        return idx[torch.randperm(len(idx))][:5420].squeeze()  # for all classes, 5421 is the min length of data

    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)  # shape: [10, 5421]
    target_matrix = torch.stack(target_idxs)  # shape: [10, 7]

    G1, G2, G3, G4 = [], [], [], []
    Y1, Y2, Y3, Y4 = [], [], [], []

    for i in range(10):
        for j in range(shot):
            # different domain, same class
            G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
            # different domain, different class
            G4.append((X_s[source_matrix[i % 10][j]], X_t[target_matrix[(i + 1) % 10][j]]))
            Y4.append((Y_s[source_matrix[i % 10][j]], Y_t[target_matrix[(i + 1) % 10][j]]))
    for i in range(10):
        for j in range(int(5420 / 2)):
            # same domain, same class
            G1.append((X_s[source_matrix[i][j * 2]], X_s[source_matrix[i][j * 2 + 1]]))
            Y1.append((Y_s[source_matrix[i][j * 2]], Y_s[source_matrix[i][j * 2 + 1]]))
            # same domain, different class
            G3.append((X_s[source_matrix[i % 10][j]], X_s[source_matrix[(i + 1) % 10][j]]))
            Y3.append((Y_s[source_matrix[i % 10][j]], Y_s[source_matrix[(i + 1) % 10][j]]))

    groups = [G1, G2, G3, G4]
    groups_y = [Y1, Y2, Y3, Y4]

    return groups, groups_y
