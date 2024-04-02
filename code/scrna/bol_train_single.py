import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from layers import ZINBLoss, MeanAct, DispAct, GaussianNoise
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
from bol_preprocess import *
import argparse
import random
from itertools import cycle
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd
from augmentation import *
from collections import Counter


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)
    return res


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def auxilarly_dis(pred):
    weight = (pred ** 2) / torch.sum(pred, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def entropy(x):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


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


class Prototype(nn.Module):
    def __init__(self, num_classes, input_size, tau=0.05):
        super(Prototype, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tau = tau
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tau
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class scOpen1(nn.Module):
    def __init__(self, input_dim, z_dim, shared_classes, total_classes, num_batches, encodeLayer=[], decodeLayer=[], activation="relu", tau=1.0):
        super(scOpen1, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.shared_classes = shared_classes
        self.total_classes = total_classes
        self.num_batches = num_batches
        self.activation = activation
        self.tau = tau
        self.encoder = buildNetwork([self.input_dim] + encodeLayer, activation=activation, noise=True, batchnorm=False)
        self.decoder = buildNetwork([self.z_dim + self.num_batches] + decodeLayer, activation=activation, batchnorm=False)
        self._enc_mu = nn.Linear(encodeLayer[-1], self.z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), nn.Sigmoid())
        self._dec_mask = nn.Sequential(nn.Linear(decodeLayer[-1], self.input_dim), nn.Sigmoid())
        self.source_classifier = Prototype(self.shared_classes, self.z_dim, self.tau)
        self.target_classifier = Prototype(self.total_classes, self.z_dim, self.tau)

    def forward(self, x, batch):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(torch.cat([z, batch], dim=1))
        mean = self._dec_mean(h)
        disp = self._dec_disp(h)
        pi = self._dec_pi(h)
        mask = self._dec_mask(h)
        out_s = self.source_classifier(z)
        out_t = self.target_classifier(z)
        return z, mean, disp, pi, mask, out_s, out_t


def extractor(model, test_loader, device):
    model.eval()
    test_embedding = []
    test_output = []
    test_label = []
    test_index = []
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_t, label_t, index_t, batch_t = data[0].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
            z_t, _, _, _, _, out_s, out_t = model(x_t, batch_t)
            test_embedding.append(z_t.detach())
            test_output.append(out_t.detach())
            test_label.append(label_t)
            test_index.append(index_t)
    test_embedding = torch.cat(test_embedding, dim=0)
    test_output = torch.cat(test_output, dim=0)
    test_label = torch.cat(test_label)
    test_index = torch.cat(test_index)
    _, test_indexes = torch.sort(test_index, descending=False)
    test_embedding = test_embedding[test_indexes]
    test_output = test_output[test_indexes]
    test_label = test_label[test_indexes]
    return test_embedding, test_output, test_label


def cluster_match(train_clusters, test_clusters):
    cluster_similarity = torch.matmul(F.normalize(train_clusters), F.normalize(test_clusters).t())
    cluster_mapping = torch.argmax(cluster_similarity, dim=1)
    return cluster_mapping.cpu().numpy()


def bipartite_match(train_embeddings, train_clusters, test_embeddings, test_clusters):
    cluster_similarity = torch.matmul(F.normalize(train_clusters), F.normalize(test_clusters).t())
    test_for_train = torch.argmax(cluster_similarity, dim=1)
    train_for_test = torch.argmax(cluster_similarity, dim=0)
    train_mapping = -torch.ones_like(test_for_train)
    test_mapping = -torch.ones_like(train_for_test)
    for i in range(len(train_mapping)):
        if i == train_for_test[test_for_train[i]]:
            train_mapping[i] = test_for_train[i]
    for j in range(len(test_mapping)):
        if j == test_for_train[train_for_test[j]]:
            test_mapping[j] = train_for_test[j]

    train_similarity_from_train = torch.matmul(F.normalize(train_embeddings), F.normalize(train_clusters).t())
    train_preds_from_train = torch.argmax(train_similarity_from_train, dim=1)
    train_similarity_from_test = torch.matmul(F.normalize(train_embeddings), F.normalize(test_clusters).t())
    train_preds_from_test = torch.argmax(train_similarity_from_test, dim=1)
    test_similarity_from_test = torch.matmul(F.normalize(test_embeddings), F.normalize(test_clusters).t())
    test_preds_from_test = torch.argmax(test_similarity_from_test, dim=1)
    test_similarity_from_train = torch.matmul(F.normalize(test_embeddings), F.normalize(train_clusters).t())
    test_preds_from_train = torch.argmax(test_similarity_from_train, dim=1)
    train_preds = -torch.ones_like(train_preds_from_test)
    test_preds = -torch.ones_like(test_preds_from_train)

    for k in range(len(train_preds)):
        if test_mapping[train_preds_from_test[k]] != -1:
            if train_preds_from_train[k] == test_mapping[train_preds_from_test[k]]:
                train_preds[k] = train_preds_from_test[k]
    for l in range(len(test_preds)):
        if train_mapping[test_preds_from_train[l]] != -1:
            if test_preds_from_test[l] == train_mapping[test_preds_from_train[l]]:
                test_preds[l] = test_preds_from_train[l]

    total_score = 0.
    total_match_num = 0
    for m in range(len(train_mapping)):
        if train_mapping[m] != -1:
            total_score += len(train_preds == train_mapping[m]) / len(train_preds_from_train == m)
            total_match_num += 1
    for n in range(len(test_mapping)):
        if test_mapping[n] != -1:
            total_score += len(test_preds == test_mapping[n]) / len(test_preds_from_test == n)
            total_match_num += 1
    consensus_score = total_score / total_match_num
    return train_preds, test_preds, consensus_score


def test_new2(model, labeled_num, device, test_loader, cluster_mapping, epoch):
    model.eval()
    idxs = np.array([])
    preds = np.array([])
    preds_open = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_t, label_t, index_t, batch_t = data[0].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
            z, _, _, _, _, output_s, output_t = model(x_t, batch_t)
            output_c = torch.cat([output_s, output_t], dim=1)
            conf_c, pred_c = output_c.max(1)
            conf_t, pred_t = output_t.max(1)
            targets = np.append(targets, label_t.cpu().numpy())
            idxs = np.append(idxs, index_t.cpu().numpy())
            preds = np.append(preds, pred_c.cpu().numpy())
            preds_open = np.append(preds_open, pred_t.cpu().numpy())
    cluster_mapping = cluster_mapping + labeled_num
    for i in range(len(cluster_mapping)):
        preds[preds == cluster_mapping[i]] = i
    targets = targets.astype(int)
    idxs = idxs.astype(int)
    preds = preds.astype(int)
    preds_open = preds_open.astype(int)
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    overall_acc2 = cluster_acc(preds_open, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    seen_acc2 = cluster_acc(preds_open[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_acc2 = cluster_acc(preds_open[unseen_mask], targets[unseen_mask])
    preds = preds[np.argsort(idxs)]
    preds_open = preds_open[np.argsort(idxs)]
    return overall_acc, seen_acc, unseen_acc, overall_acc2, seen_acc2, unseen_acc2, preds, preds_open


def dataset_spliting(X, cellname, class_set, common_number, source_private_number, labeled_ratio=0.5, random_seed=8888):
    shared_class_set = class_set[:common_number]
    source_private_class_set = class_set[common_number:(common_number + source_private_number)]
    target_private_class_set = class_set[(common_number + source_private_number):]
    source_class_set = shared_class_set + source_private_class_set
    target_class_set = shared_class_set + target_private_class_set

    source_index = []
    target_index = []
    np.random.seed(random_seed)

    for i in range(X.shape[0]):
        if cellname[i] in shared_class_set:
            if np.random.rand() < labeled_ratio:
                source_index.append(i)
            else:
                target_index.append(i)
        elif cellname[i] in source_private_class_set:
            source_index.append(i)
        elif cellname[i] in target_private_class_set:
            target_index.append(i)

    source_X = X[source_index]
    source_cellname = cellname[source_index]
    target_X = X[target_index]
    target_cellname = cellname[target_index]

    source_Y = np.array([0] * len(source_cellname))
    target_Y = np.array([0] * len(target_cellname))
    for i in range(len(shared_class_set)):
        source_Y[source_cellname == shared_class_set[i]] = i
        target_Y[target_cellname == shared_class_set[i]] = i
    for j in range(len(shared_class_set), len(source_class_set)):
        source_Y[source_cellname == class_set[j]] = j
    for k in range(len(source_class_set), len(class_set)):
        target_Y[target_cellname == class_set[k]] = k
    source_batch = np.zeros_like(source_Y)
    target_batch = np.ones_like(target_Y)
    return source_X, source_cellname, source_Y, source_batch, target_X, target_cellname, target_Y, target_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scUDA')
    parser.add_argument('--random-seed', type=int, default=8888, metavar='S')
    parser.add_argument('--gpu-id', default='0', type=int)
    parser.add_argument('--number', default=0, type=int)
    parser.add_argument('--num', default=0, type=int)
    parser.add_argument('--ra', type=float, default=0.5)
    parser.add_argument('--removal', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--highly-genes', type=int, default=2000)
    parser.add_argument('--noi', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--deta', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--pretrain', type=int, default=600)
    parser.add_argument('--midtrain', type=int, default=800)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--thres', type=float, default=5.0)
    parser.add_argument('--quan', type=float, default=0.8)
    parser.add_argument('--structure', type=int, default=1)
    parser.add_argument("--rw", action='store_true', help='random walk or not')

    args = parser.parse_args()
    args.quantile = args.quan
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    adv_method = 'CDAN+E' # 'CDAN+E' # 'CDAN', 'DANN'

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu", args.gpu_id)

    filename_set = ["Quake_10x", "Quake_Smart-seq2"]

    result_list = []

    for i in range(args.num, args.num + 1):
        filename = filename_set[i]
        dataname = filename
        X, cell_name, gene_name = read_real_with_genes(filename, batch=False)
        class_set = class_splitting_single(filename)
        if dataname == "Cao":
            shared_classes = 6
            source_private_number = 4
            target_private_number = 6
        if dataname == "Quake_10x":
            shared_classes = 12
            source_private_number = 12
            target_private_number = 12
        if dataname == "Quake_Smart-seq2":
            shared_classes = 15
            source_private_number = 15
            target_private_number = 15
        if dataname == "Wagner":
            shared_classes = 5
            source_private_number = 4
            target_private_number = 5
        if dataname == "Zeisel_2018":
            shared_classes = 6
            source_private_number = 5
            target_private_number = 6


        source_classes = shared_classes + source_private_number
        target_classes = shared_classes + target_private_number
        labeled_ratio = args.ra # 0.5

        source_X, source_cellname, source_Y, source_batch, target_X, target_cellname, target_Y, target_batch \
            = dataset_spliting(X, cell_name, class_set, shared_classes, source_private_number,
                               labeled_ratio=labeled_ratio, random_seed=args.random_seed)
        print("The shape of source data is {}, and the shape of target data is {}".format(source_X.shape, target_X.shape))

        shared_classes_set = class_set[:shared_classes]
        novel_classes_set = class_set[(shared_classes + source_private_number):]

        X = np.concatenate((source_X, target_X), axis=0)
        Y = np.concatenate((source_Y, target_Y))
        cell_name = np.concatenate((source_cellname, target_cellname))
        batch_label = np.concatenate((source_batch, target_batch))
        count_X = X.astype(np.int)
        if X.shape[1] == args.highly_genes:
            args.highly_genes = None

        adata = sc.AnnData(X)
        adata.var["gene_id"] = gene_name
        adata.obs["batch"] = batch_label
        adata.obs["celltype"] = Y
        adata.obs["cellname"] = cell_name
        adata = normalize(adata, highly_genes=args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
        X = adata.X.astype(np.float32)
        Y = np.array(adata.obs["celltype"])
        cell_name = np.array(adata.obs["cellname"])
        batch_label = np.array(adata.obs["batch"])
        gene_name = np.array(adata.var["gene_id"])
        print("after preprocessing, the gene dimension is {}".format(len(gene_name)))

        if args.highly_genes != None:
            high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
            count_X = count_X[:, high_variable]
        else:
            select_genes = np.array(adata.var.index, dtype=np.int)
            select_cells = np.array(adata.obs.index, dtype=np.int)
            count_X = count_X[:, select_genes]
            count_X = count_X[select_cells]
        assert X.shape == count_X.shape
        size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)

        batch_matrix = OneHotEncoder().fit_transform(batch_label.reshape(-1, 1)).toarray()

        source_x = X[batch_label == 0]
        source_raw_x = count_X[batch_label == 0]
        source_y = Y[batch_label == 0]
        source_sf = size_factor[batch_label == 0]
        source_b = batch_label[batch_label == 0]
        source_batch = batch_matrix[batch_label == 0]
        source_cellname = cell_name[batch_label == 0]

        target_x = X[batch_label == 1]
        target_raw_x = count_X[batch_label == 1]
        target_y = Y[batch_label == 1]
        target_sf = size_factor[batch_label == 1]
        target_b = batch_label[batch_label == 1]
        target_batch = batch_matrix[batch_label == 1]
        target_cellname = cell_name[batch_label == 1]
        num_batches = 2

        if source_x.shape[0] < args.batch_size or target_x.shape[0] < args.batch_size:
            args.batch_size = min(source_x.shape[0], target_x.shape[0])

        if args.structure == 0:
            model = scOpen1(X.shape[1], 32, source_classes, target_classes, num_batches, encodeLayer=[256, 64],
                            decodeLayer=[64, 256], activation="relu", tau=args.tau)
        else:
            model = scOpen1(X.shape[1], 128, source_classes, target_classes, num_batches, encodeLayer=[512, 256],
                            decodeLayer=[256, 512], activation="relu", tau=args.tau)
        model = model.to(device)

        freq = Counter(source_y)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_y]
        sampler = WeightedRandomSampler(source_weights, len(source_y))

        source_x = torch.tensor(source_x)
        source_raw_x = torch.tensor(source_raw_x)
        source_sf = torch.tensor(source_sf)
        source_y = torch.tensor(source_y)
        source_batch = torch.tensor(source_batch).float()
        target_x = torch.tensor(target_x)
        target_raw_x = torch.tensor(target_raw_x)
        target_sf = torch.tensor(target_sf)
        target_y = torch.tensor(target_y)
        target_batch = torch.tensor(target_batch).float()

        source_dataset = TensorDataset(source_x, source_raw_x, source_sf, source_y, torch.arange(source_x.shape[0]), source_batch)
        source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
        target_dataset = TensorDataset(target_x, target_raw_x, target_sf, target_y, torch.arange(target_x.shape[0]), target_batch)
        target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_dataset = TensorDataset(source_x, source_raw_x, source_sf, source_y, torch.arange(source_x.shape[0]), source_batch)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataset = TensorDataset(target_x, target_raw_x, target_sf, target_y, torch.arange(target_x.shape[0]), target_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

        best_overall_acc = 0.
        best_seen_acc = 0.
        best_unseen_acc = 0.

        best_overall_acc2 = 0.
        best_seen_acc2 = 0.
        best_unseen_acc2 = 0.

        bce = nn.BCELoss()
        ce = nn.CrossEntropyLoss()

        for epoch in range(args.epochs + 1):
            if epoch % args.interval == 0:
                train_embeddings, train_outputs, train_targets = extractor(model, train_dataloader, device)
                test_embeddings, test_outputs, test_targets = extractor(model, test_dataloader, device)
                if epoch <= args.pretrain:
                    kmeans_ = KMeans(n_clusters=target_classes, init="k-means++", random_state=args.random_seed).fit(
                        F.normalize(test_embeddings).cpu().numpy())
                    test_label, cluster_centers = torch.from_numpy(kmeans_.labels_).to(device), torch.from_numpy(
                        kmeans_.cluster_centers_).to(device)

                    if epoch == args.pretrain:
                        test_mask = torch.zeros_like(test_label).float()
                        for i in range(target_classes):
                            test_distance = torch.matmul(F.normalize(test_embeddings[test_label == i]), F.normalize(cluster_centers).t())
                            intra_distance = test_distance[:, i]
                            inter_distance = torch.max(test_distance[:, torch.arange(test_distance.shape[1]) != i], dim=1)[0]
                            test_score = intra_distance / inter_distance
                            test_mask[test_label == i] = torch.where(test_score >= torch.quantile(test_score, args.quantile),
                                                                     torch.ones_like(test_score), torch.zeros_like(test_score))
                    test_label = test_label.to(train_targets.dtype)

                    label_centers = torch.zeros(source_classes, cluster_centers.shape[1]).to(device)
                    for i in range(source_classes):
                        label_centers[i] = torch.mean(train_embeddings[train_targets == i], dim=0)
                    state_dict = model.state_dict()
                    state_dict['source_classifier.fc.weight'] = F.normalize(label_centers)
                    state_dict['target_classifier.fc.weight'] = F.normalize(cluster_centers)
                    model.load_state_dict(state_dict)
                    cluster_mappings = cluster_match(label_centers, cluster_centers)
                    overall_acc, seen_acc, unseen_acc, overall_acc2, seen_acc2, unseen_acc2, target_preds, _ = test_new2(model,
                                                                                                        source_classes,
                                                                                                        device,
                                                                                                        test_dataloader,
                                                                                                        cluster_mappings,
                                                                                                        epoch)

                    if epoch == args.pretrain:
                        train_pseudo_labels, test_pseudo_labels, consensus_score = bipartite_match(train_embeddings,
                                                                                                   label_centers,
                                                                                                   test_embeddings,
                                                                                                   cluster_centers)
                        train_pseudo_labels = train_pseudo_labels.to(device)
                        test_pseudo_labels = test_pseudo_labels.to(device)

                else:
                    test_label = torch.max(test_outputs, dim=1)[1].to(train_targets.dtype)
                    test_mask = torch.zeros_like(test_label).float()
                    label_centers = model.source_classifier.fc.weight.data
                    cluster_centers = model.target_classifier.fc.weight.data
                    for i in range(target_classes):
                        if torch.sum(test_label == i) >= 5:
                            test_distance = torch.matmul(F.normalize(test_embeddings[test_label == i]), F.normalize(cluster_centers).t())
                            intra_distance = test_distance[:, i]
                            inter_distance = torch.max(test_distance[:, torch.arange(test_distance.shape[1]) != i], dim=1)[0]
                            test_score = intra_distance / inter_distance
                            test_mask[test_label == i] = torch.where(test_score >= torch.quantile(test_score, args.quantile),
                                                                     torch.ones_like(test_score), torch.zeros_like(test_score))
                    cluster_mappings = cluster_match(label_centers, cluster_centers)
                    overall_acc, seen_acc, unseen_acc, overall_acc2, seen_acc2, unseen_acc2, target_preds, _ = test_new2(model,
                                                                                                        source_classes,
                                                                                                        device,
                                                                                                        test_dataloader,
                                                                                                        cluster_mappings,
                                                                                                        epoch)

                    train_pseudo_labels, test_pseudo_labels, consensus_score = bipartite_match(train_embeddings,
                                                                                               label_centers,
                                                                                               test_embeddings,
                                                                                               cluster_centers)
                    train_pseudo_labels = train_pseudo_labels.to(device)
                    test_pseudo_labels = test_pseudo_labels.to(device)

                if overall_acc > best_overall_acc:
                    best_overall_acc = overall_acc
                    best_seen_acc = seen_acc
                    best_unseen_acc = unseen_acc
                if overall_acc2 > best_overall_acc2:
                    best_overall_acc2 = overall_acc2
                    best_seen_acc2 = seen_acc2
                    best_unseen_acc2 = unseen_acc2
                print("Currently, we have the overall acc is {:.4f}, seen acc is {:.4f}, unseen acc is {:.4f}.".format(best_overall_acc, best_seen_acc, best_unseen_acc))
                print("Currently, we have the overall acc2 is {:.4f}, seen acc2 is {:.4f}, unseen acc2 is {:.4f}.".format(best_overall_acc2, best_seen_acc2, best_unseen_acc2))

            source_dataloader_iter = cycle(source_dataloader)
            recon_losses = AverageMeter('recon_loss', ':.4e')
            ce_losses = AverageMeter('ce_loss', ':.4e')
            align_losses = AverageMeter('align_loss', ':.4e')
            cluster_losses = AverageMeter('pse_loss', ':.4e')
            model.train()

            for batch_idx, (x_t, raw_x_t, sf_t, y_t, index_t, batch_t) in enumerate(target_dataloader):
                (x_s, raw_x_s, sf_s, y_s, index_s, batch_s) = next(source_dataloader_iter)
                x_s, raw_x_s, sf_s, y_s, index_s, batch_s = x_s.to(device), raw_x_s.to(device), \
                                                            sf_s.to(device), y_s.to(device), \
                                                            index_s.to(device), batch_s.to(device)
                x_t, raw_x_t, sf_t, index_t, batch_t = x_t.to(device), raw_x_t.to(device), \
                                                       sf_t.to(device), index_t.to(device), batch_t.to(device)

                labeled_len = len(y_s)
                x_all = torch.cat([x_s, x_t], dim=0)
                mask_all_ = mask_generator(args.noi, x_all.cpu().numpy())
                mask_all, x_all2 = pretext_generator(mask_all_, x_all.cpu().numpy())
                mask_all = torch.from_numpy(mask_all).to(torch.float32).to(device)
                x_all2 = torch.from_numpy(x_all2).to(torch.float32).to(device)
                x_transform1 = x_all
                x_transform2 = x_all2

                z_s1, mean_s1, disp_s1, pi_s1, mask_s1, output_ss1, output_st1 = model(x_transform1[:labeled_len],
                                                                                       batch_s)
                z_s2, mean_s2, disp_s2, pi_s2, mask_s2, output_ss2, output_st2 = model(x_transform2[:labeled_len],
                                                                                       batch_s)
                z_t1, mean_t1, disp_t1, pi_t1, mask_t1, output_ts1, output_tt1 = model(x_transform1[labeled_len:],
                                                                                       batch_t)
                z_t2, mean_t2, disp_t2, pi_t2, mask_t2, output_ts2, output_tt2 = model(x_transform2[labeled_len:],
                                                                                       batch_t)

                recon_loss = ZINBLoss().to(device)(x=raw_x_s, mean=mean_s1, disp=disp_s1, pi=pi_s1, scale_factor=sf_s) \
                             + ZINBLoss().to(device)(x=raw_x_t, mean=mean_t1, disp=disp_t1, pi=pi_t1, scale_factor=sf_t)
                recon_loss = recon_loss - torch.mean(
                    mask_all[:labeled_len] * torch.log(mask_s2 + 1e-8) + (1. - mask_all[:labeled_len]) * torch.log(
                        1. - mask_s2 + 1e-8)) \
                             - torch.mean(
                    mask_all[labeled_len:] * torch.log(mask_t2 + 1e-8) + (1. - mask_all[labeled_len:]) * torch.log(
                        1. - mask_t2 + 1e-8))

                if epoch >= args.pretrain:
                    with torch.no_grad():
                        feat_detach1 = z_t1.detach()
                        distance1 = torch.matmul(F.normalize(feat_detach1, dim=1), F.normalize(feat_detach1, dim=1).t())
                        _, idx_near1 = torch.topk(distance1, dim=-1, largest=True, k=2)
                        t_idx_near1 = idx_near1[:, 1]

                        feat_detach2 = z_t2.detach()
                        distance2 = torch.matmul(F.normalize(feat_detach2, dim=1), F.normalize(feat_detach2, dim=1).t())
                        _, idx_near2 = torch.topk(distance2, dim=-1, largest=True, k=2)
                        t_idx_near2 = idx_near2[:, 1]

                    labeled_onehot = F.one_hot(train_targets, num_classes = source_classes)[index_s].float()
                    unlabeled_onehot = F.one_hot(test_label, num_classes = target_classes)[index_t].float()
                    unlabeled_mask = test_mask[index_t].float()

                    output_t1 = torch.cat([output_ts1, output_tt1], dim=1)
                    output_t2 = torch.cat([output_ts2, output_tt2], dim=1)

                    output_ss1 = F.softmax(output_ss1, dim=1)
                    output_ss2 = F.softmax(output_ss2, dim=1)

                    output_ts1 = F.softmax(output_ts1, dim=1)
                    output_ts2 = F.softmax(output_ts2, dim=1)

                    output_st1 = F.softmax(output_st1, dim=1)
                    output_st2 = F.softmax(output_st2, dim=1)

                    output_tt1 = F.softmax(output_tt1, dim=1)
                    output_tt2 = F.softmax(output_tt2, dim=1)

                    ce_loss = -torch.mean(torch.sum(labeled_onehot * torch.log(torch.clamp(output_ss1, min=1e-8)), dim=1)) \
                              - torch.mean(torch.sum(labeled_onehot * torch.log(torch.clamp(output_ss2, min=1e-8)), dim=1)) \
                              - torch.mean(unlabeled_mask * torch.sum(unlabeled_onehot * torch.log(torch.clamp(output_tt1, min=1e-8)), dim=1)) \
                              - torch.mean(unlabeled_mask * torch.sum(unlabeled_onehot * torch.log(torch.clamp(output_tt2, min=1e-8)), dim=1))

                    cluster_loss = -torch.mean(
                        torch.log(torch.clamp(F.sigmoid(torch.sum(output_t1[t_idx_near1] * output_t1, dim=1)), min=1e-8))) - \
                                   torch.mean(
                                       torch.log(torch.clamp(F.sigmoid(torch.sum(output_t2[t_idx_near2] * output_t2, dim=1)), min=1e-8))) - \
                                   torch.mean(
                                       torch.log(torch.clamp(F.sigmoid(torch.sum(output_t1[t_idx_near1] * output_t2, dim=1)), min=1e-8))) - \
                                   torch.mean(
                                       torch.log(torch.clamp(F.sigmoid(torch.sum(output_t2[t_idx_near2] * output_t1, dim=1)), min=1e-8)))

                    train_pseudo_labels_batch = train_pseudo_labels[index_s]
                    test_pseudo_labels_batch = test_pseudo_labels[index_t]
                    if len(train_pseudo_labels_batch[train_pseudo_labels_batch != -1]) != 0:
                        output_st1 = output_st1[train_pseudo_labels_batch != -1]
                        output_st2 = output_st2[train_pseudo_labels_batch != -1]
                        train_pseudo_labels_batch = train_pseudo_labels_batch[train_pseudo_labels_batch != -1]
                        train_pseudo_labels_batch_onehot = F.one_hot(train_pseudo_labels_batch, num_classes = target_classes).float()
                        align_loss = -torch.mean(torch.sum(train_pseudo_labels_batch_onehot * torch.log(torch.clamp(output_st1, min=1e-8)), dim=1)) \
                          - torch.mean(torch.sum(train_pseudo_labels_batch_onehot * torch.log(torch.clamp(output_st2, min=1e-8)), dim=1))
                    if len(test_pseudo_labels_batch[test_pseudo_labels_batch != -1]) != 0:
                        output_ts1 = output_ts1[test_pseudo_labels_batch != -1]
                        output_ts2 = output_ts2[test_pseudo_labels_batch != -1]
                        test_pseudo_labels_batch = test_pseudo_labels_batch[test_pseudo_labels_batch != -1]
                        test_pseudo_labels_batch_onehot = F.one_hot(test_pseudo_labels_batch, num_classes = source_classes).float()
                        align_loss += -torch.mean(torch.sum(test_pseudo_labels_batch_onehot * torch.log(torch.clamp(output_ts1, min=1e-8)), dim=1)) \
                          - torch.mean(torch.sum(test_pseudo_labels_batch_onehot * torch.log(torch.clamp(output_ts2, min=1e-8)), dim=1))

                if epoch < args.pretrain:
                    loss = recon_loss
                else:
                    loss = recon_loss + ce_loss + cluster_loss + align_loss

                recon_losses.update(recon_loss.item(), args.batch_size)
                if epoch >= args.pretrain:
                    ce_losses.update(ce_loss.item(), args.batch_size)
                    cluster_losses.update(cluster_loss.item(), args.batch_size)
                    align_losses.update(align_loss.item(), args.batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch >= args.pretrain:
                print("Training {}/{}, zinb loss: {:.4f}, ce loss: {:.4f}, "
                      "cluster loss: {:.4f}, align loss: {:.4f}".format(epoch, args.epochs, recon_losses.avg,
                                                    ce_losses.avg, cluster_losses.avg, align_losses.avg))
            else:
                print("Training {}/{}, zinb loss: {:.4f}".format(epoch, args.epochs, recon_losses.avg))

















