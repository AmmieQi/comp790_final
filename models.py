import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder,self).__init__()

        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,84)
        self.dropout = nn.Dropout(0.5)
        self.gn1 = nn.GroupNorm(num_groups = 6, num_channels = 6)
        self.gn2 = nn.GroupNorm(num_groups = 16, num_channels = 16)

    def forward(self,input):

        out = self.conv1(input)
        out = self.gn1(out)
        out=F.relu(out)
        out=F.max_pool2d(out,2)
        out = self.conv2(out)
        out = self.gn2(out)
        out=F.relu(out)
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)

        out= F.relu(self.fc1(self.dropout(out)))
        out= self.fc2(self.dropout(out))

        return out

class LeNetPP(nn.Module):
    """
    LeNet++ as described in the Center Loss paper.
    https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/models.py
    """
    def __init__(self, num_classes):
        super(LeNetPP, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128*3*3, 64)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128*3*3)

        x = self.prelu_fc1(self.fc1(x))
        # y = self.fc2(x)

        return  x


class Style_Discrim(nn.Module):
    def __init__(self, input_features=84, h_features = 32):
        super(Style_Discrim, self).__init__()

        self.fc1=nn.Linear(input_features,h_features)
        self.fc2 = nn.Linear(h_features, h_features)
        self.fc3=nn.Linear(h_features,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,inputs):

        out=self.fc1(self.dropout(inputs))
        out = self.relu(out)
        return self.fc3(out).squeeze(dim= 1)


class Content_Discrim(nn.Module):
    def __init__(self,input_features=84, out_classes = 10):
        super(Content_Discrim,self).__init__()
        self.fc=nn.Linear(input_features, out_classes)

    def forward(self,input):
        # return F.softmax(self.fc(input),dim=1)
        out = self.fc(input)
        return out


class DCD(nn.Module):
    def __init__(self,h_features=64,input_features=128):
        super(DCD,self).__init__()

        self.fc1=nn.Linear(input_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):

        out=F.relu(self.fc1(inputs))

        return self.fc3(out)

class Classifier(nn.Module):
    def __init__(self,input_features=84):
        super(Classifier,self).__init__()
        self.fc=nn.Linear(input_features,10)

    def forward(self,input):
        # return F.softmax(self.fc(input),dim=1)
        return self.fc(input)


class FiLM(nn.Module):

    #https://github.com/pytorch/pytorch/pull/9177/files
    r"""Applies Feature-wise Linear Modulation to the incoming data as described in
    the paper `FiLM: Visual Reasoning with a General Conditioning Layer`_ .
    .. math::
        y_{n,c,*} = \gamma_{n,c} * x_{n,c,*} + \beta_{n,c},
    where :math:`\gamma_{n,c}` and :math:`\beta_{n,c}` are scalars and operations are
    broadcast over any additional dimensions of :math:`x`
    Shape:
        - Input: :math:`(N, C, *)` where :math:`*` means any number of additional
          dimensions
        - Gammas: :math:`(N, C)`
        - Betas: :math:`(N, C)`
        - Output: :math:`(N, C, *)`, same shape as the input
    Examples::
        >>> m = nn.FiLM()
        >>> input = torch.randn(128, 20, 4, 4)
        >>> gammas = torch.randn(128, 20)
        >>> betas = torch.randn(128, 20)
        >>> output = m(input, gammas, betas)
        >>> print(output.size())
    .. _`FiLM: Visual Reasoning with a General Conditioning Layer`:
        https://arxiv.org/abs/1709.07871
    """
    def __init__(self):
        super(FiLM, self).__init__()

    def film(input, gammas, betas):
        r"""Applies Feature-wise Linear Modulation to the incoming data.
        See :class:`~torch.nn.FiLM` for details.
        """
        for _ in range(input.dim() - 2):
            gammas = gammas.unsqueeze(-1)
            betas = betas.unsqueeze(-1)
        return gammas * input + betas

    def forward(self, input, gammas, betas):
        return self.film(input, gammas, betas)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    feat_mean = torch.mean(feat)
    feat_std = torch.std(feat)

    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, mean = 0, std = 1 ):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()

    # noise = torch.randn(style_feat.size()) * std + mean
    # noise = noise.to(device)
    # style_feat = style_feat + noise
    style_mean, style_std = calc_mean_std(style_feat)

    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)



    return normalized_feat * style_std.expand(size) + style_mean.expand(size)



class Fusion(nn.Module):
    def __init__(self, in_channels = 84):
        super(Fusion, self).__init__()

        self.film = nn.Linear(in_channels, in_channels*2)
        self.style_0_in = nn.LayerNorm(in_channels)
        self.style_1_in = nn.LayerNorm(in_channels)



    def forward(self, style, content, style_num, alpha = 1.0):

        # film = self.film(style)
        # film = film.view(style.shape[0], 2, -1)
        # gamma, beta = film[:,0,:], film[:,1,:]
        # out = gamma * content + beta

        # out = style + content
        # if style_num == 0:
        #     out = self.style_0_in(out)
        # else:
        #     out = self.style_1_in(out)
        out = adaptive_instance_normalization(content, style)

        return out
