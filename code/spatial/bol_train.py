import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from utils import entropy
import numpy as np
from itertools import cycle
import copy
from torch_geometric.data import ClusterData, ClusterLoader
import scanpy as sc
from anndata import AnnData
from scipy.optimize import linear_sum_assignment
from ssKM_protos_initialization import ssKM_protos_init
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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


class BOL:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        args.input_dim = dataset.unlabeled_data.x.shape[-1]
        self.model = models.Encoder_modified(args.input_dim, args.shared_classes, args.total_classes)
        self.model = self.model.to(args.device)


    def bipartite_match(self, train_embeddings, train_clusters, test_embeddings, test_clusters):
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

        return train_preds, test_preds


    def train_epoch_ours(self, args, model, device, dataset, optimizer, epoch):
        """ Train for 1 epoch."""
        model.train()
        bce = nn.BCELoss()
        ce = nn.CrossEntropyLoss()

        ce_losses = 0.
        cluster_losses = 0.
        align_losses = 0.
        reg_losses = 0.

        labeled_graph, unlabeled_graph = dataset.labeled_data, dataset.unlabeled_data
        labeled_data = ClusterData(labeled_graph, num_parts=100, recursive=False)
        labeled_loader = ClusterLoader(labeled_data, batch_size=1, shuffle=True, num_workers=1) # 1
        unlabeled_data = ClusterData(unlabeled_graph, num_parts=100, recursive=False)
        unlabeled_loader = ClusterLoader(unlabeled_data, batch_size=1, shuffle=True, num_workers=1) # 1
        label_loader_iter = cycle(labeled_loader)

        for batch_idx, unlabeled_x in enumerate(unlabeled_loader):
            labeled_x = next(label_loader_iter)
            labeled_x, unlabeled_x = labeled_x.to(device), unlabeled_x.to(device)

            optimizer.zero_grad()

            labeled_output_ss, labeled_output_st, labeled_feat = model(labeled_x)
            unlabeled_output_ts, unlabeled_output_tt, unlabeled_feat = model(unlabeled_x)

            labeled_len = len(labeled_output_ss)
            unlabeled_len = len(unlabeled_output_ts)
            batch_size = len(labeled_output_ss) + len(unlabeled_output_ts)

            total_output_s = torch.cat([labeled_output_ss, unlabeled_output_ts], dim=0)
            total_output_t = torch.cat([labeled_output_st, unlabeled_output_tt], dim=0)
            pui = torch.mm(F.normalize(total_output_s.t(), p=2, dim=1), F.normalize(total_output_t, p=2, dim=0))
            reg_loss = ce(pui, torch.arange(pui.size(0)).to(device))
            pui2 = torch.mm(F.normalize(total_output_t.t(), p=2, dim=1), F.normalize(total_output_t, p=2, dim=0))
            reg_loss += ce(pui2, torch.arange(pui2.size(0)).to(device))

            labeled_prob_ss = F.softmax(labeled_output_ss, dim=1)
            labeled_onehot = F.one_hot(labeled_x.y, num_classes=args.shared_classes).float()
            ce_loss = -torch.mean(torch.sum(labeled_onehot * torch.log(torch.clamp(labeled_prob_ss, min=1e-8)), dim=1))

            unlabeled_onehot = F.one_hot(unlabeled_x.test_label, num_classes=args.total_classes).float()
            unlabeled_mask = unlabeled_x.test_mask.float()
            unlabeled_prob_tt = F.softmax(unlabeled_output_tt, dim=1)
            ce_loss += -torch.mean(unlabeled_mask * torch.sum(unlabeled_onehot * torch.log(torch.clamp(unlabeled_prob_tt, min=1e-8)), dim=1))

            cluster_loss = -torch.mean((1.0 - unlabeled_mask) * torch.sum(auxilarly_dis(unlabeled_prob_tt) * torch.log(torch.clamp(unlabeled_prob_tt, min=1e-8)), dim=1))

            train_clusters = self.model.linear1.weight.data.t().detach()
            test_clusters = self.model.linear2.weight.data.t().detach()
            train_embeddings = labeled_feat.detach()
            test_embeddings = unlabeled_feat.detach()
            train_pseudo_labels_batch, test_pseudo_labels_batch = self.bipartite_match(train_embeddings, train_clusters, test_embeddings, test_clusters)
            align_loss = torch.tensor([0.]).to(device)
            if len(train_pseudo_labels_batch[train_pseudo_labels_batch != -1]) != 0:
                labeled_prob_st = F.softmax(labeled_output_st, dim=1)
                labeled_prob_st = labeled_prob_st[train_pseudo_labels_batch != -1]
                train_pseudo_labels_batch = train_pseudo_labels_batch[train_pseudo_labels_batch != -1]
                train_pseudo_labels_batch_onehot = F.one_hot(train_pseudo_labels_batch, num_classes=args.total_classes).float()
                align_loss = -torch.mean(torch.sum(train_pseudo_labels_batch_onehot * torch.log(torch.clamp(labeled_prob_st, min=1e-8)), dim=1))
            if len(test_pseudo_labels_batch[test_pseudo_labels_batch != -1]) != 0:
                unlabeled_prob_ts = F.softmax(unlabeled_output_ts, dim=1)
                unlabeled_prob_ts = unlabeled_prob_ts[test_pseudo_labels_batch != -1]
                test_pseudo_labels_batch = test_pseudo_labels_batch[test_pseudo_labels_batch != -1]
                test_pseudo_labels_batch_onehot = F.one_hot(test_pseudo_labels_batch, num_classes=args.shared_classes).float()
                align_loss += -torch.mean(torch.sum(test_pseudo_labels_batch_onehot * torch.log(torch.clamp(unlabeled_prob_ts, min=1e-8)), dim=1))

            loss = ce_loss + cluster_loss + reg_loss + align_loss

            optimizer.zero_grad()
            ce_losses += ce_loss.item()
            cluster_losses += cluster_loss.item()
            align_losses += align_loss.item()
            reg_losses += reg_loss.item()
            loss.backward()
            optimizer.step()

        print('In epoch {}, CE Loss: {:.4f}, Cluster Loss: {:.4f}, Align Loss: {:.4f}, Regular Loss: {:.4f}'.format(epoch, ce_losses / (batch_idx + 1),
                                                                                              cluster_losses / (batch_idx + 1),
                                                                                              align_losses / (batch_idx + 1), reg_losses / (batch_idx + 1)))


    def cluster_match(selff, train_clusters, test_clusters):
        cluster_similarity = torch.matmul(F.normalize(train_clusters), F.normalize(test_clusters).t())
        cluster_mapping = torch.argmax(cluster_similarity, dim=1)
        return cluster_mapping.cpu().numpy()


    def pred(self, epoch):
        self.model.eval()

        label_centers = self.model.linear1.weight.data.t()
        cluster_centers = self.model.linear2.weight.data.t()
        cluster_mappings = self.cluster_match(label_centers, cluster_centers)

        preds = np.array([])
        preds_open = np.array([])
        targets = np.array([])

        with torch.no_grad():
            _, unlabeled_graph = self.dataset.labeled_data, self.dataset.unlabeled_data
            unlabeled_graph_cp = copy.deepcopy(unlabeled_graph)
            unlabeled_graph_cp = unlabeled_graph_cp.to(self.args.device)
            output_s, output_t, _ = self.model(unlabeled_graph_cp)

            output_c = torch.cat([output_s, output_t], dim=1)
            conf_c, pred_c = output_c.max(1)
            conf_t, pred_t = output_t.max(1)
            targets = np.append(targets, unlabeled_graph_cp.y.cpu().numpy())
            preds = np.append(preds, pred_c.cpu().numpy())
            preds_open = np.append(preds_open, pred_t.cpu().numpy())

        cluster_mapping = cluster_mappings + self.args.shared_classes
        for i in range(len(cluster_mapping)):
            preds[preds == cluster_mapping[i]] = i

        targets = targets.astype(int)
        preds = preds.astype(int)
        preds_open = preds_open.astype(int)
        seen_mask = targets < self.args.shared_classes
        unseen_mask = ~seen_mask

        overall_acc = cluster_acc(preds, targets)
        seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
        unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])

        overall_acc2 = cluster_acc(preds_open, targets)
        seen_acc2 = cluster_acc(preds_open[seen_mask], targets[seen_mask])
        unseen_acc2 = cluster_acc(preds_open[unseen_mask], targets[unseen_mask])
        print('In the old {}-th epoch, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch,
                                                                                                           overall_acc,
                                                                                                           seen_acc,
                                                                                                           unseen_acc))
        print('In the old {}-th epoch, Test overall acc2 {:.4f}, seen acc2 {:.4f}, unseen acc2 {:.4f}'.format(epoch,
                                                                                                              overall_acc2,
                                                                                                              seen_acc2,
                                                                                                              unseen_acc2))
        return preds_open


    def SSKM(self, args, train_embeddings, train_targets, test_embeddings, test_targets, distortion_metric="eucl"):
        all_embeddings = np.concatenate((train_embeddings, test_embeddings), axis=0)
        all_targets = np.concatenate((train_targets, test_targets))
        all_mask_labs_ = np.array([0] * len(train_targets) + [1] * len(test_targets))
        all_mask_labs = all_mask_labs_ < 1
        all_mask_clss = all_targets < args.shared_classes
        print("PIM init prototypes estimation using ssKM...")
        prototypes = ssKM_protos_init(all_embeddings, args.total_classes, all_targets[all_mask_labs],
                                      all_mask_labs, all_mask_clss, args.nbr, args.device, # args.device_name,
                                      "protos_file_name", distortion_metric=distortion_metric) # "cosine_sim"

        if distortion_metric == "cosine_sim":
            normed_prototypes = preprocessing.normalize(np.asarray(prototypes), axis=1, norm='l2')
            normed_test_embeddings = preprocessing.normalize(test_embeddings, axis=1, norm='l2')
            test_scores = np.matmul(normed_test_embeddings, np.transpose(normed_prototypes))
            test_preds = np.argmax(test_scores, axis=1)

            test_preds = torch.tensor(test_preds)
            test_mask = torch.zeros_like(test_preds).float()
            test_embeddings = torch.tensor(test_embeddings)
            prototypes = torch.tensor(prototypes)
            for i in range(args.total_classes):
                if torch.sum(test_preds == i) >= 10:
                    test_distance = torch.matmul(F.normalize(test_embeddings[test_preds == i]),
                                                 F.normalize(prototypes).t())
                    intra_distance = test_distance[:, i]
                    inter_distance = torch.max(test_distance[:, torch.arange(test_distance.shape[1]) != i], dim=1)[0]
                    test_score = intra_distance / inter_distance
                    test_mask[test_preds == i] = torch.where(test_score >= torch.quantile(test_score, args.quantile),
                                                             torch.ones_like(test_score), torch.zeros_like(test_score))
            test_preds = test_preds.numpy()
            test_mask = test_mask.numpy()

        elif distortion_metric == "eucl":
            test_scores = pairwise_distances(X=test_embeddings, Y=np.asarray(prototypes))
            test_preds = np.argmin(test_scores, axis=1)

            test_preds = torch.tensor(test_preds)
            test_mask = torch.zeros_like(test_preds).float()
            test_embeddings = torch.tensor(test_embeddings)
            prototypes = torch.tensor(prototypes)
            for i in range(args.total_classes):
                if torch.sum(test_preds == i) >= 10:
                    test_distance = torch.sum(torch.square(test_embeddings[test_preds == i].unsqueeze(1) - prototypes), dim=2)
                    intra_distance = test_distance[:, i]
                    inter_distance = torch.min(test_distance[:, torch.arange(test_distance.shape[1]) != i], dim=1)[0]
                    test_score = intra_distance / inter_distance
                    test_mask[test_preds == i] = torch.where(test_score <= torch.quantile(test_score, 1.0 - args.quantile),
                                                             torch.ones_like(test_score), torch.zeros_like(test_score))

            test_preds = test_preds.numpy()
            test_mask = test_mask.numpy()

        return test_preds, test_mask


    def train_supervised(self, args, model, device, dataset, optimizer, epoch):
        model.train()
        ce = nn.CrossEntropyLoss()
        sum_loss = 0

        labeled_graph = dataset.labeled_data
        labeled_data = ClusterData(labeled_graph, num_parts=100, recursive=False)
        labeled_loader = ClusterLoader(labeled_data, batch_size=1, shuffle=True, num_workers=1)

        for batch_idx, labeled_x in enumerate(labeled_loader):
            labeled_x = labeled_x.to(device)
            optimizer.zero_grad()
            output, _, _ = model(labeled_x)
            loss = ce(output, labeled_x.y)
            optimizer.zero_grad()
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Loss: {:.6f}'.format(sum_loss / (batch_idx + 1)))


    def pred_fc(self, model):
        model.eval()
        preds = np.array([])
        confs = np.array([])
        targets = np.array([])
        with torch.no_grad():
            _, unlabeled_graph = self.dataset.labeled_data, self.dataset.unlabeled_data
            unlabeled_graph_cp = copy.deepcopy(unlabeled_graph)
            unlabeled_graph_cp = unlabeled_graph_cp.to(self.args.device)
            output, _, _ = model(unlabeled_graph_cp)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())
            targets = np.append(targets, unlabeled_graph_cp.y.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
        preds = preds.astype(int)

        targets = targets.astype(int)
        preds = preds.astype(int)
        seen_mask = targets < self.args.shared_classes
        seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
        print('Currently, Test seen acc {:.4f}'.format(seen_acc))


    def find_novel(self, model):
        model.eval()
        with torch.no_grad():
            unlabeled_graph = self.dataset.unlabeled_data
            unlabeled_graph_cp = copy.deepcopy(unlabeled_graph)
            unlabeled_graph_cp = unlabeled_graph_cp.to(self.args.device)
            output, _, _ = model(unlabeled_graph_cp)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            entr = -torch.sum(prob * torch.log(prob), 1)

        conf = conf.cpu().numpy()
        pred = pred.cpu().numpy()
        entr = entr.cpu().numpy()
        mask = np.zeros_like(conf)
        unknown_guess = entr > np.median(entr)
        original_x = self.dataset.unlabeled_data.x.numpy()
        kmeans_ = KMeans(n_clusters=self.args.total_classes - self.args.shared_classes, init="k-means++",
                         random_state=self.args.seed).fit(original_x[unknown_guess])
        cluster_label, cluster_centers = kmeans_.labels_, kmeans_.cluster_centers_
        pred[unknown_guess] = cluster_label + self.args.shared_classes
        for i in range(self.args.total_classes):
            if i < self.args.shared_classes:
                cur_index = pred == i
                cur_conf = conf[cur_index]
                cur_mask = np.where(cur_conf >= np.quantile(cur_conf, self.args.quantile),
                                    np.ones_like(cur_conf), np.zeros_like(cur_conf))
                mask[cur_index] = cur_mask
            else:
                cur_index = pred == i
                i = i - self.args.shared_classes
                cur_x = original_x[cur_index]
                distance = pairwise_distances(cur_x, cluster_centers)
                intra_distance = distance[:, i]
                inter_distance = np.min(distance[:, np.arange(distance.shape[1]) != i], axis=1)
                score = intra_distance / inter_distance
                cur_mask = np.where(score <= np.quantile(score, 1.0 - self.args.quantile),
                                    np.ones_like(score), np.zeros_like(score))
                mask[cur_index] = cur_mask
        return pred, mask


    def extractor(self, model):
        model.eval()
        with torch.no_grad():
            labeled_graph, unlabeled_graph = self.dataset.labeled_data, self.dataset.unlabeled_data
            unlabeled_graph_cp = copy.deepcopy(unlabeled_graph)
            unlabeled_graph_cp = unlabeled_graph_cp.to(self.args.device)
            _, _, unlabeled_output = model(unlabeled_graph_cp)
            unlabeled_output = unlabeled_output.cpu().numpy()
            unlabeled_y = unlabeled_graph_cp.y.cpu().numpy()

            labeled_graph_cp = copy.deepcopy(labeled_graph)
            labeled_graph_cp = labeled_graph_cp.to(self.args.device)
            _, _, labeled_output = model(labeled_graph_cp)
            labeled_output = labeled_output.cpu().numpy()
            labeled_y = labeled_graph_cp.y.cpu().numpy()
        return labeled_output, labeled_y, unlabeled_output, unlabeled_y


    def train(self):
        clusters, masks = self.SSKM(self.args, self.dataset.labeled_data.x.numpy(), self.dataset.labeled_data.y.numpy(),
                                    self.dataset.unlabeled_data.x.numpy(), self.dataset.unlabeled_data.y.numpy())

        self.dataset.unlabeled_data.test_label = torch.tensor(clusters).to(self.dataset.unlabeled_data.y.dtype)
        self.dataset.unlabeled_data.test_mask = torch.tensor(masks).to(self.dataset.unlabeled_data.y.dtype)
        # Set the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

        for epoch in range(1000 + 1):
            self.train_epoch_ours(self.args, self.model, self.args.device, self.dataset, optimizer, epoch)
            test_preds = self.pred(epoch)
            print("the unique test preds are {}".format(np.unique(test_preds)))

