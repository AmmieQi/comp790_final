import torch.nn as nn
import torch.nn.functional as F
from FADA.models.BasicModule import BasicModule


class DCD(BasicModule):
    def __init__(self, h_features=84, input_features=168):
        super(DCD, self).__init__()
        self.fc1 = nn.Linear(input_features, h_features)
        self.fc2 = nn.Linear(h_features, h_features)
        self.fc3 = nn.Linear(h_features, 4)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = self.fc2(out)
        return F.softmax(self.fc3(out), dim=1)


class Classifier(BasicModule):
    def __init__(self, input_features=84):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_features, 10)

    def forward(self, input):
        return F.softmax(self.fc(input), dim=1)


class Classifier_DomainNet(BasicModule):
    def __init__(self, input_features=2048):
        super(Classifier_DomainNet, self).__init__()
        self.fc = nn.Linear(input_features, 345)

    def forward(self, input):
        return F.softmax(self.fc(input), dim=1)


class DCD_DomainNet(BasicModule):
    def __init__(self, h_features=2048, input_features=4096):
        super(DCD_DomainNet, self).__init__()
        self.fc1 = nn.Linear(input_features, h_features)
        self.fc2 = nn.Linear(h_features, h_features)
        self.fc3 = nn.Linear(h_features, 4)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = self.fc2(out)
        return F.softmax(self.fc3(out), dim=1)


# LeNet
class Encoder(BasicModule):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(0.5)
        self.gn1 = nn.GroupNorm(num_groups=6, num_channels=6)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=16)

    def forward(self, input):
        out = self.conv1(input)
        out = self.gn1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = self.gn2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# for modify ResNet
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
