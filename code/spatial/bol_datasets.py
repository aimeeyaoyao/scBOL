import torch
import numpy as np
from builtins import range
from torch_geometric.data import InMemoryDataset, Data
from sklearn.metrics import pairwise_distances
import pandas as pd
from collections import Counter
import scanpy.api as sc


def normalize(adata, highly_genes = None, size_factors=True, normalize_input=True, logtrans_input=True):
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def get_hubmap_edge_index(pos, regions, distance_thres):
    # construct edge indexes when there is region information
    edge_list = []
    regions_unique = np.unique(regions)
    for reg in regions_unique:
        locs = np.where(regions == reg)[0]
        pos_region = pos[locs, :]
        dists = pairwise_distances(pos_region)
        dists_mask = dists < distance_thres
        np.fill_diagonal(dists_mask, 0)
        region_edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
        for (i, j) in region_edge_list:
            edge_list.append([locs[i], locs[j]])
    return edge_list


def get_tonsilbe_edge_index(pos, distance_thres):
    # construct edge indexes in one region
    edge_list = []
    dists = pairwise_distances(pos)
    dists_mask = dists < distance_thres
    np.fill_diagonal(dists_mask, 0)
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
    return edge_list


def get_tonsilbe_edge_index_new(pos, distance_thres):
    # construct edge indexes in one region
    dists = pairwise_distances(pos)
    dists_mask = dists < distance_thres
    dists_mask = dists_mask + 0.
    np.fill_diagonal(dists_mask, 0)
    return dists_mask


def get_stereo_edge_index(pos, K):
    dists = pairwise_distances(pos)
    dists_topk = np.argpartition(dists, K+1, axis=1)[:, K+1]
    dists_thres = dists[np.arange(dists.shape[0]), dists_topk]
    dists_mask = dists < dists_thres
    np.fill_diagonal(dists_mask, 0)
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
    return edge_list


def load_hubmap_data_ours_new(labeled_file, distance_thres, sample_rate, seen_cell_type_set, novel_cell_type_set):
    total_cell_type_set = seen_cell_type_set + novel_cell_type_set
    train_df = pd.read_csv(labeled_file)
    train_df = train_df.loc[np.logical_and(train_df['tissue'] == 'CL', train_df['donor'] == 'B004')]
    train_X = train_df.iloc[:, 1:49].values  # node features, indexes depend on specific datasets
    train_pos = train_df.iloc[:, -6:-4].values  # x,y coordinates, indexes depend on specific datasets
    train_regions = train_df['unique_region'].values
    train_cell_name = train_df['cell_type_A'].values  # class information
    print("for hubmap dataset, we have {} cells".format(len(train_cell_name)))

    all_cell_type_dict = {}
    all_inverse_dict = {}
    for i, cell_type in enumerate(total_cell_type_set):
        all_cell_type_dict[cell_type] = i
        all_inverse_dict[i] = cell_type

    labeled_index = []
    unlabeled_index = []
    shared_classes = len(seen_cell_type_set)
    total_classes = len(total_cell_type_set)
    np.random.seed(8888)
    for i in range(train_X.shape[0]):
        if train_cell_name[i] in seen_cell_type_set and np.random.rand() < sample_rate:
            labeled_index.append(i)
        elif train_cell_name[i] in total_cell_type_set:
            unlabeled_index.append(i)

    labeled_X = train_X[labeled_index]
    labeled_cell_name = train_cell_name[labeled_index]
    labeled_pos = train_pos[labeled_index]
    labeled_regions = train_regions[labeled_index]
    unlabeled_X = train_X[unlabeled_index]
    unlabeled_cell_name = train_cell_name[unlabeled_index]
    unlabeled_pos = train_pos[unlabeled_index]
    unlabeled_regions = train_regions[unlabeled_index]

    labeled_y = []
    unlabeled_y = []
    for cellname1 in labeled_cell_name:
        labeled_y.append(all_cell_type_dict[cellname1])
    for cellname2 in unlabeled_cell_name:
        unlabeled_y.append(all_cell_type_dict[cellname2])
    labeled_y = np.array(labeled_y)
    unlabeled_y = np.array(unlabeled_y)

    labeled_edges = get_hubmap_edge_index(labeled_pos, labeled_regions, distance_thres)
    unlabeled_edges = get_hubmap_edge_index(unlabeled_pos, unlabeled_regions, distance_thres)
    return labeled_X, labeled_y, unlabeled_X, unlabeled_y, labeled_edges, unlabeled_edges, all_inverse_dict, \
           shared_classes, total_classes, labeled_pos, unlabeled_pos


def load_seqfish_data_ours_new(distance_thres, sample_rate, seen_cell_type_set, novel_cell_type_set):
    total_cell_type_set = seen_cell_type_set + novel_cell_type_set
    train_X = np.array(pd.read_csv("../scrna/st/seqFISH_filter_labeled_gene_expression.csv", header=0, index_col=0))
    train_pos = np.array(pd.read_csv("../scrna/st/seqFISH_filter_labeled_spatial_coordinate.csv", header=0, index_col=0))
    train_infor = pd.read_csv("../scrna/st/seqFISH_filter_labeled_adata_obs.csv", header=0, index_col=0)
    train_cell_name = train_infor["celltype_mapped_refined"].values
    train_regions = train_infor["region"].values
    selected_index = []
    for i in range(train_X.shape[0]):
        if train_cell_name[i] in total_cell_type_set:
            selected_index.append(i)
    train_X = train_X[selected_index]
    train_pos = train_pos[selected_index]
    train_cell_name = train_cell_name[selected_index]
    train_regions = train_regions[selected_index]

    train_adata = sc.AnnData(train_X)
    train_adata = normalize(train_adata, highly_genes=150, size_factors=True, normalize_input=False, logtrans_input=True)
    train_X = train_adata.X.astype(np.float32)

    all_cell_type_dict = {}
    all_inverse_dict = {}
    for i, cell_type in enumerate(total_cell_type_set):
        all_cell_type_dict[cell_type] = i
        all_inverse_dict[i] = cell_type

    labeled_index = []
    unlabeled_index = []
    shared_classes = len(seen_cell_type_set)
    total_classes = len(total_cell_type_set)
    np.random.seed(8888)
    for i in range(train_X.shape[0]):
        if train_cell_name[i] in seen_cell_type_set and np.random.rand() < sample_rate:
            labeled_index.append(i)
        elif train_cell_name[i] in total_cell_type_set:
            unlabeled_index.append(i)

    labeled_X = train_X[labeled_index]
    labeled_cell_name = train_cell_name[labeled_index]
    labeled_pos = train_pos[labeled_index]
    labeled_regions = train_regions[labeled_index]
    unlabeled_X = train_X[unlabeled_index]
    unlabeled_cell_name = train_cell_name[unlabeled_index]
    unlabeled_pos = train_pos[unlabeled_index]
    unlabeled_regions = train_regions[unlabeled_index]

    labeled_y = []
    unlabeled_y = []
    for cellname1 in labeled_cell_name:
        labeled_y.append(all_cell_type_dict[cellname1])
    for cellname2 in unlabeled_cell_name:
        unlabeled_y.append(all_cell_type_dict[cellname2])
    labeled_y = np.array(labeled_y)
    unlabeled_y = np.array(unlabeled_y)

    labeled_edges = get_hubmap_edge_index(labeled_pos, labeled_regions, distance_thres)
    unlabeled_edges = get_hubmap_edge_index(unlabeled_pos, unlabeled_regions, distance_thres)
    return labeled_X, labeled_y, unlabeled_X, unlabeled_y, labeled_edges, unlabeled_edges, all_inverse_dict, \
           shared_classes, total_classes, labeled_pos, unlabeled_pos


class GraphDataset(InMemoryDataset):
    def __init__(self, labeled_X, labeled_y, unlabeled_X, unlabeled_y, labeled_edges, unlabeled_edges, transform=None,):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root, transform)
        self.labeled_data = Data(x=torch.FloatTensor(labeled_X), edge_index=torch.LongTensor(labeled_edges).T,
                                 y=torch.LongTensor(labeled_y))
        self.unlabeled_data = Data(x=torch.FloatTensor(unlabeled_X), edge_index=torch.LongTensor(unlabeled_edges).T,
                                   y=torch.LongTensor(unlabeled_y))

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.labeled_data, self.unlabeled_data

