import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv


class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out


class NormedLinear_new(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear_new, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class Encoder(nn.Module):
    def __init__(self, x_dim, num_cls):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        hid_dim = 128
        self.conv1 = nn.Linear(x_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        self.relu = nn.ReLU()
        self.linear = NormedLinear(hid_dim, num_cls)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x)
        feat = x
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        out_feat = x
        out = self.linear(x)
        return out, feat, out_feat


class Encoder_modi(nn.Module):
    def __init__(self, x_dim, num_cls):
        super(Encoder_modi, self).__init__()
        self.x_dim = x_dim
        self.conv2 = SAGEConv(x_dim, x_dim)
        self.linear = NormedLinear(x_dim, num_cls)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        feat = x
        x = self.conv2(x, edge_index)
        out_feat = x
        out = self.linear(x)
        return out, feat, out_feat


class Encoder_modified(nn.Module):
    def __init__(self, x_dim, shared_classes, total_classes):
        super(Encoder_modified, self).__init__()
        self.x_dim = x_dim
        self.conv2 = SAGEConv(self.x_dim, self.x_dim)
        self.linear1 = NormedLinear(self.x_dim, shared_classes)
        self.linear2 = NormedLinear(self.x_dim, total_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv2(x, edge_index)
        out_feat = x1
        out1 = self.linear1(x1)
        out2 = self.linear2(x1)
        return out1, out2, out_feat


class Encoder_modified2(nn.Module):
    def __init__(self, x_dim, shared_classes, total_classes):
        super(Encoder_modified2, self).__init__()
        self.x_dim = x_dim
        self.conv2 = SAGEConv(self.x_dim, self.x_dim)
        self.linear1 = NormedLinear_new(self.x_dim, shared_classes)
        self.linear2 = NormedLinear_new(self.x_dim, total_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv2(x, edge_index)
        out_feat = x1
        out1 = self.linear1(x1)
        out2 = self.linear2(x1)
        return out1, out2, out_feat


class FCNet(nn.Module):
    def __init__(self, x_dim, num_cls):
        super(FCNet, self).__init__()
        self.linear = NormedLinear(x_dim, num_cls)

    def forward(self, data):
        x = data.x
        out = self.linear(x)
        return out, x, x


class Encoder2(nn.Module):
    def __init__(self, x_dim, num_cls):
        super(Encoder2, self).__init__()
        self.x_dim = x_dim
        hid_dim = 128
        self.conv1 = nn.Linear(x_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        self.relu = nn.ReLU()
        self.linear = NormedLinear(hid_dim, num_cls)

    def forward(self, x, edge_index):
        x = self.conv1(x)
        feat = x
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        # out_feat = x
        out = self.linear(x)
        return feat, out


class Encoder_new(nn.Module):
    def __init__(self, x_dim, hid_dim):
        super(Encoder_new, self).__init__()
        self.conv1 = nn.Linear(x_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        out_feat = x
        return out_feat


def buildNetwork(layers, activation="relu", noise=False, batchnorm=False):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if noise:
            net.append(GaussianNoise())
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if batchnorm:
            net.append(nn.BatchNorm1d(layers[i]))
    return nn.Sequential(*net)


class scAuto(nn.Module):
    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[], activation="relu", tau=1.0):
        super(scAuto, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.activation = activation
        self.tau = tau
        self.encoder = buildNetwork([self.input_dim] + encodeLayer, activation=activation, batchnorm=False)
        self.decoder = buildNetwork([self.z_dim] + decodeLayer, activation=activation, batchnorm=False)
        self._enc_mu = nn.Linear(encodeLayer[-1], self.z_dim)
        self._dec_mean = nn.Linear(decodeLayer[-1], self.input_dim)
        self._dec_mask = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(z)
        mean = self._dec_mean(h)
        mask = self._dec_mask(h)
        return z, mean, mask


class Classifier(nn.Module):
    def __init__(self, hid_dim, num_cls):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hid_dim, hid_dim)
        self.linear2 = NormedLinear_new(hid_dim, num_cls)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out


class Classifier_GCN(nn.Module):
    def __init__(self, hid_dim, num_cls):
        super(Classifier_GCN, self).__init__()
        self.conv1 = SAGEConv(hid_dim, hid_dim)
        self.linear1 = NormedLinear_new(hid_dim, num_cls)

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index)
        out = self.linear1(out)
        return out


class Classifier_GCN2(nn.Module):
    def __init__(self, hid_dim, shared_classes, total_classes):
        super(Classifier_GCN2, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = SAGEConv(hid_dim, hid_dim)
        self.linear_s = NormedLinear_new(hid_dim, shared_classes)
        self.linear_t = NormedLinear_new(hid_dim, total_classes)

    def forward(self, x, edge_index):
        x = self.relu(x)
        out1 = self.conv1(x, edge_index)
        out_s = self.linear_s(out1)
        out_t = self.linear_t(out1)
        return out1, out_s, out_t


class Projector(nn.Module):
    def __init__(self, hid_dim):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(hid_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out











