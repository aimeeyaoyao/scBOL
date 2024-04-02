import torch
import torch.nn as nn
import os
import os.path
import torch.nn.functional as F
from ssKM_protos_initialization import ssKM_protos_init
import numpy as np
from scipy.optimize import linear_sum_assignment


def from_numpy_to_torch(np_array, torch_device):
    return torch.from_numpy(np_array).to(torch_device)


def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()


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


def prepare_save_dir(args, filename):
    """ Create saving directory."""
    runner_name = os.path.basename(filename).split(".")[0]
    model_dir = './experiments/{}/{}/'.format(runner_name, args.name)
    args.savedir = model_dir
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    return args


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


class MarginLoss(nn.Module):
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)


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


def test1(model, classifier, labeled_num, device, test_loader, epoch):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_t, label_t, edge_index_t = data.x.to(device), data.y.to(device), data.edge_index.to(device)
            z_t = model(x_t, edge_index_t)
            output = classifier(z_t)
            conf, pred = output.max(1)
            targets = np.append(targets, label_t.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    print('In the test1 {}-th epoch, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, overall_acc,
                                                                                                     seen_acc,
                                                                                                     unseen_acc))
    return overall_acc, seen_acc, unseen_acc


def test2(model, projector, labeled_num, total_num, device, device_name, train_loader, test_loader, epoch):
    model.eval()

    train_embeddings = []
    train_targets = []
    with torch.no_grad():
        for _, data in enumerate(train_loader):
            x_s, label_s, edge_index_s = data.x.to(device), data.y.to(device), data.edge_index.to(device)
            z_s = model(x_s, edge_index_s)
            output_s = projector(z_s)
            train_embeddings.append(output_s.detach())
            train_targets.append(label_s)
    train_embeddings = F.normalize(torch.cat(train_embeddings, dim=0), dim=-1).cpu().numpy()
    train_targets = torch.cat(train_targets).cpu().numpy()

    test_embeddings = []
    test_targets = []
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x_t, label_t, edge_index_t = data.x.to(device), data.y.to(device), data.edge_index.to(device)
            z_t = model(x_t, edge_index_t)
            output_t = projector(z_t)
            test_embeddings.append(output_t.detach())
            test_targets.append(label_t)
    test_embeddings = F.normalize(torch.cat(test_embeddings, dim=0), dim=-1).cpu().numpy()
    test_targets = torch.cat(test_targets).cpu().numpy()

    all_embeddings = np.concatenate((train_embeddings, test_embeddings), axis=0)
    all_targets = np.concatenate((train_targets, test_targets))
    all_mask_labs_ = np.array([0] * len(train_targets) + [1] * len(test_targets))
    all_mask_labs = all_mask_labs_ < 1
    all_mask_clss = all_targets < labeled_num

    print("PIM init prototypes estimation using ssKM...")
    prototypes = ssKM_protos_init(all_embeddings,
                                  total_num,
                                  all_targets[all_mask_labs],
                                  all_mask_labs,
                                  all_mask_clss,
                                  total_num,
                                  device_name,
                                  "protos_file_name")
    from sklearn import preprocessing
    normed_prototypes = preprocessing.normalize(np.asarray(prototypes), axis=1, norm='l2')
    test_scores = np.matmul(test_embeddings, np.transpose(normed_prototypes))
    test_preds = np.argmax(test_scores, axis=1)
    test_targets = test_targets.astype(int)
    test_preds = test_preds.astype(int)
    seen_mask = test_targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(test_preds, test_targets)
    seen_acc = accuracy(test_preds[seen_mask], test_targets[seen_mask])
    unseen_acc = cluster_acc(test_preds[unseen_mask], test_targets[unseen_mask])
    print('In the test2 {}-th epoch, Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, overall_acc,
                                                                                                     seen_acc,
                                                                                                     unseen_acc))
    return overall_acc, seen_acc, unseen_acc, prototypes


def build_mask(gene_num, masked_percentage, device):
    mask = torch.cat([torch.ones(int(gene_num * masked_percentage), dtype=bool),
                      torch.zeros(gene_num - int(gene_num * masked_percentage), dtype=bool)])
    shuffle_index = torch.randperm(gene_num)
    return mask[shuffle_index].to(device)


def random_mask(data, mask_percentage, apply_mask_prob, device):
    ### data cell by gene
    s = np.random.uniform(0, 1)
    if s < apply_mask_prob:
        mask = build_mask(data.shape[1], mask_percentage, device)
        data[:, mask] = 0
    return data


def random_gaussian_noise(data, noise_percentage, sigma, apply_noise_prob, device):
    ### data cell by gene
    s = np.random.uniform(0, 1)
    if s < apply_noise_prob:
        mask = build_mask(data.shape[1], noise_percentage, device)
        noise = torch.randn(int(data.shape[1] * noise_percentage)) * sigma
        data[:, mask] += noise.to(device)
    return data


def random_swap(data, swap_percentage, apply_swap_prob, device):
    ### data cell by gene
    s = np.random.uniform(0, 1)
    if s < apply_swap_prob:
        swap_instances = int(data.shape[1] * swap_percentage / 2)
        swap_pair = torch.randint(data.shape[1], size=(swap_instances, 2)).to(device)
        data[:, swap_pair[:, 0]], data[:, swap_pair[:, 1]] = data[:, swap_pair[:, 1]], data[:, swap_pair[:, 0]]
    return data


def instance_crossover(data, cross_percentage, apply_cross_prob, device):
    ### data cell by gene
    s = np.random.uniform(0, 1)
    if s < apply_cross_prob:
        cross_idx = torch.randint(data.shape[0], size=(1, )).to(device)
        cross_instance = data[cross_idx]
        mask = build_mask(data.shape[1], cross_percentage, device)
        data[:, mask] = cross_instance[:, mask]
    return data


def mask_generator(p_m, x):
    mask = np.random.binomial(1, p_m, x.shape)
    return mask


def pretext_generator(m, x):
    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def transformation(data, mask_percentage=0.1, apply_mask_prob=0.5, noise_percentage=0.1, sigma=0.5, apply_noise_prob=0.5,
                   swap_percentage=0.1, apply_swap_prob=0.5, cross_percentage=0.1, apply_cross_prob=0.5, device=None):
    data = random_mask(data, mask_percentage, apply_mask_prob, device)
    data = random_gaussian_noise(data, noise_percentage, sigma, apply_noise_prob, device)
    # data = random_swap(data, swap_percentage, apply_swap_prob, device)
    # data = instance_crossover(data, cross_percentage, apply_cross_prob, device)
    return data

