import os
gpu_option = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import argparse
from utils import prepare_save_dir
from bol_train import BOL
import numpy as np
import os
import torch
from bol_datasets import GraphDataset, load_hubmap_data_ours_new, load_seqfish_data_ours_new
import random


def main():
    parser = argparse.ArgumentParser(description='STELLAR')
    parser.add_argument('--number', default=1, type=int)
    parser.add_argument('--gpu-id', default='2', type=int)
    parser.add_argument('--seed', type=int, default=8888, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='STELLAR')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-seeds', type=int, default=4)
    parser.add_argument('--lab-pri', type=int, default=1)
    parser.add_argument('--sample-rate', type=float, default=0.5)
    # parser.add_argument('--shared-ratio', type=float, default=0.5)
    parser.add_argument('--quantile', type=float, default=0.8)
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--distance_thres', default=50, type=int)
    parser.add_argument('--nbr', default=30, type=int)
    parser.add_argument('--savedir', type=str, default='./')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # args.device = torch.device("cuda:{}".format(args.gpu_id) if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    # args.device_name = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.num_seed_class = args.num_seeds

    data_list = ['Hubmap', 'seqfish']
    args.dataset = data_list[args.number]

    # Seed the run and create saving directory
    args.name = '_'.join([args.dataset, args.name])
    args = prepare_save_dir(args, __file__)

    if args.dataset == 'Hubmap':
        seen_cell_type_set = ['Macrophage', 'Endothelial', 'Nerve'] ###
        novel_cell_type_set = ['Enterocyte', 'SmoothMuscle', 'Plasma']
        labeled_X, labeled_y, unlabeled_X, unlabeled_y, labeled_edges, unlabeled_edges, inverse_dict, shared_classes, total_classes, labeled_pos, unlabeled_pos \
            = load_hubmap_data_ours_new('./data/B004_training_dryad.csv', args.distance_thres, args.sample_rate,
                                        seen_cell_type_set, novel_cell_type_set)
        dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, unlabeled_y, labeled_edges, unlabeled_edges)
    elif args.dataset == 'seqfish':
        seen_cell_type_set = ['Cranial mesoderm', 'Endothelium', 'Splanchnic mesoderm']
        novel_cell_type_set = ['Cardiomyocytes', 'Surface ectoderm', 'Neural crest']
        labeled_X, labeled_y, unlabeled_X, unlabeled_y, labeled_edges, unlabeled_edges, inverse_dict, shared_classes, total_classes, labeled_pos, unlabeled_pos \
            = load_seqfish_data_ours_new(args.distance_thres, args.sample_rate,
                                          seen_cell_type_set, novel_cell_type_set)
        dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, unlabeled_y, labeled_edges, unlabeled_edges)

    print("the shared cell type number is {}, and the total cell type number is {}".format(shared_classes, total_classes))
    args.shared_classes = shared_classes
    args.total_classes = total_classes
    args.inverse_dict = inverse_dict
    args.labeled_pos = labeled_pos
    args.unlabeled_pos = unlabeled_pos
    scbol = BOL(args, dataset)
    scbol.train()


if __name__ == '__main__':
    main()

