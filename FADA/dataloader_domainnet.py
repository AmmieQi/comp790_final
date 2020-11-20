import os
import torch
import numpy as np
from torchvision import transforms
from FADA.data_list import Imagelists_VISDA, return_classlist


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(file_name, file_type):
    base_path = './data/DomainNet'
    root = './data/DomainNet'
    image_set_file = os.path.join(base_path, file_name)

    crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            ResizeImage(256),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dataset = Imagelists_VISDA(image_set_file, root=root,
                               transform=data_transforms[file_type])

    class_list = return_classlist(image_set_file)
    print("%d classes in this dataset" % len(class_list))
    bs = 16
    if file_type == 'train':
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                                  num_workers=8, shuffle=True,
                                                  drop_last=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=min(bs, len(dataset)),
                                                  num_workers=8,
                                                  shuffle=True, drop_last=True)

    return data_loader, class_list


def sample_data():
    image_set_file = "./data/DomainNet/clipart_train.txt"
    dataset = Imagelists_VISDA(image_set_file, root='./data/DomainNet',
                               transform=transforms.Compose([
                                   ResizeImage(256),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ]))

    n = len(dataset)
    X = torch.Tensor(n, 3, 224, 224)
    Y = torch.LongTensor(n)
    inds = torch.randperm(len(dataset))
    for i, index in enumerate(inds):
        x, y = dataset[index]
        X[i] = x
        Y[i] = torch.tensor(y.astype(np.long))
    return X, Y


def create_target_samples(n=1):
    n_class = 50
    image_set_file = "./data/DomainNet/real_train.txt"
    dataset = Imagelists_VISDA(image_set_file, root='./data/DomainNet',
                               transform=transforms.Compose([
                                   ResizeImage(256),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ]))
    X, Y = [], []
    classes = n_class * [n]

    i = 0
    while True:
        if len(X) == n * n_class:
            break
        x, y = dataset[i]
        if classes[y] > 0:
            X.append(x)
            Y.append(y)
            classes[y] -= 1
        i += 1

    assert (len(X) == n * n_class)
    return torch.stack(X, dim=0), torch.from_numpy(np.array(Y))


def create_groups(X_s, Y_s, X_t, Y_t, seed=1, num_samples=1):
    n_class = 50
    # change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    n = X_t.shape[0]

    # shuffle order
    classes = torch.unique(Y_t)
    classes = classes[torch.randperm(len(classes))]

    class_num = classes.shape[0]
    shot = n // class_num

    def s_idxs(c):
        idx = torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot * 2].squeeze()

    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)

    target_matrix = torch.stack(target_idxs)

    G1, G2, G3, G4 = [], [], [], []
    Y1, Y2, Y3, Y4 = [], [], [], []

    for i in range(n_class):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j * 2]], X_s[source_matrix[i][j * 2 + 1]]))
            Y1.append((Y_s[source_matrix[i][j * 2]], Y_s[source_matrix[i][j * 2 + 1]]))
            G3.append((X_s[source_matrix[i % n_class][j]], X_s[source_matrix[(i + 1) % n_class][j]]))
            Y3.append((Y_s[source_matrix[i % n_class][j]], Y_s[source_matrix[(i + 1) % n_class][j]]))
            if num_samples > 1:
                G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
                Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
                G4.append((X_s[source_matrix[i % n_class][j]], X_t[target_matrix[(i + 1) % n_class][j]]))
                Y4.append((Y_s[source_matrix[i % n_class][j]], Y_t[target_matrix[(i + 1) % n_class][j]]))
            else:
                G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i]]))
                Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i]]))
                G4.append((X_s[source_matrix[i % n_class][j]], X_t[target_matrix[(i + 1) % n_class]]))
                Y4.append((Y_s[source_matrix[i % n_class][j]], Y_t[target_matrix[(i + 1) % n_class]]))

    groups = [G1, G2, G3, G4]
    groups_y = [Y1, Y2, Y3, Y4]

    # make sure we sampled enough samples
    for g in groups:
        assert (len(g) == n)
    return groups, groups_y


def sample_groups(X_s, Y_s, X_t, Y_t, seed=1, num_samples=1):
    print("Sampling groups")
    return create_groups(X_s, Y_s, X_t, Y_t, seed=seed, num_samples=num_samples)
