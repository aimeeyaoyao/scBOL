from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import bol_utils as utils
import numpy as np
import h5py
import scipy as sp
import pandas as pd
import scanpy.api as sc
from sklearn.metrics.cluster import contingency_matrix
import anndata
from collections import Counter


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=utils.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=utils.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def read_real_with_genes(filename, batch=True):
    data_path = "../scrna/data/" + filename + "/data.h5"
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_ontology_class"])
    gene_name = np.array(list(var.index))
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"
    if batch == True:
        if "dataset_name" in obs.keys():
            batch_name = np.array(obs["dataset_name"])
        else:
            batch_name = np.array(obs["study"])
        return X, cell_name, batch_name, gene_name
    else:
        return X, cell_name, gene_name


def class_splitting_cross(filename, source_name, target_name):
    seen_classes = []
    novel_classes = []
    if filename == "ALIGNED_Homo_sapiens_Pancreas":
        if source_name == "Muraro" and target_name == "Baron_human":
            seen_classes = ['pancreatic A cell', 'pancreatic acinar cell', 'pancreatic ductal cell', 'type B pancreatic cell']
            novel_classes = ['endothelial cell', 'pancreatic D cell', 'pancreatic PP cell', 'pancreatic stellate cell']
    if filename == "ALIGNED_Homo_sapiens_Placenta":
        if source_name == "Vento-Tormo_Smart-seq2" and target_name == "Vento-Tormo_10x":
            seen_classes = ['T cell', 'decidual natural killer cell', 'macrophage', 'monocyte']
            novel_classes = ['natural killer cell', 'placental villous trophoblast', 'stromal cell', 'trophoblast cell']
    if filename == "ALIGNED_Mus_musculus_Mammary_Gland":
        if source_name == "Quake_Smart-seq2_Mammary_Gland" and target_name == "Quake_10x_Mammary_Gland":
            seen_classes = ['basal cell', 'endothelial cell', 'luminal epithelial cell of mammary gland', 'stromal cell']
            novel_classes = ['B cell', 'T cell', 'macrophage']
    if filename == "ALIGNED_Mus_musculus_Small_Intestine":
        if source_name == "Haber_10x_largecell" and target_name == "Haber_10x_region":
            seen_classes = ['brush cell', 'enterocyte of epithelium of small intestine', 'enteroendocrine cell']
            novel_classes = ['paneth cell', 'small intestine goblet cell', 'stem cell']
    if filename == "ALIGNED_Mus_musculus_Trachea":
        if source_name == "Plasschaert" and target_name == "Montoro_10x":
            seen_classes = ['basal cell of epithelium of trachea', 'club cell']
            novel_classes = ['brush cell of trachea', 'ciliated columnar cell of tracheobronchial tree']
    return seen_classes, novel_classes


def class_splitting_single(dataname):
    class_set = []
    if dataname == "Quake_10x": # 36
        class_set = ['B cell', 'T cell', 'alveolar macrophage', 'basal cell', 'basal cell of epidermis', 'bladder cell',
                     'bladder urothelial cell', 'blood cell', 'endothelial cell', 'epithelial cell', 'fibroblast',
                     'granulocyte', 'granulocytopoietic cell', 'hematopoietic precursor cell', 'hepatocyte',
                     'immature T cell', 'keratinocyte', 'kidney capillary endothelial cell', 'kidney collecting duct epithelial cell',
                     'kidney loop of Henle ascending limb epithelial cell', 'kidney proximal straight tubule epithelial cell',
                     'late pro-B cell', 'leukocyte', 'luminal epithelial cell of mammary gland', 'lung endothelial cell',
                     'macrophage', 'mesenchymal cell', 'mesenchymal stem cell', 'monocyte', 'natural killer cell',
                     'neuroendocrine cell', 'non-classical monocyte', 'proerythroblast', 'promonocyte', 'skeletal muscle satellite cell',
                     'stromal cell']
    if dataname == "Quake_Smart-seq2": # 45
        class_set = ['B cell', 'Slamf1-negative multipotent progenitor cell', 'T cell', 'astrocyte of the cerebral cortex',
                     'basal cell', 'basal cell of epidermis', 'bladder cell', 'bladder urothelial cell', 'blood cell',
                     'endothelial cell', 'enterocyte of epithelium of large intestine', 'epidermal cell', 'epithelial cell',
                     'epithelial cell of large intestine', 'epithelial cell of proximal tubule', 'fibroblast', 'granulocyte',
                     'hematopoietic precursor cell', 'hepatocyte', 'immature B cell', 'immature T cell', 'keratinocyte',
                     'keratinocyte stem cell', 'large intestine goblet cell', 'late pro-B cell', 'leukocyte',
                     'luminal epithelial cell of mammary gland', 'lung endothelial cell', 'macrophage', 'mesenchymal cell',
                     'mesenchymal stem cell', 'mesenchymal stem cell of adipose', 'microglial cell', 'monocyte', 'myeloid cell',
                     'naive B cell', 'neuron', 'oligodendrocyte', 'oligodendrocyte precursor cell', 'pancreatic A cell',
                     'pro-B cell', 'skeletal muscle satellite cell', 'skeletal muscle satellite stem cell', 'stromal cell',
                     'type B pancreatic cell']
    if dataname == "Cao": # 16
        class_set = ['GABAergic neuron', 'cholinergic neuron', 'ciliated olfactory receptor neuron', 'coelomocyte', 'epidermal cell',
                     'germ line cell', 'glial cell', 'interneuron', 'muscle cell', 'nasopharyngeal epithelial cell', 'neuron',
                     'seam cell', 'sensory neuron', 'sheath cell', 'socket cell (sensu Nematoda)', 'visceral muscle cell']
    if dataname == "Wagner": # 14
        class_set = ['early embryonic cell', 'ectodermal cell', 'embryonic cell', 'endodermal cell', 'epiblast cell', 'epidermal cell',
                     'erythroid progenitor cell', 'lateral mesodermal cell', 'mesodermal cell', 'midbrain dopaminergic neuron',
                     'neural crest cell', 'neurecto-epithelial cell', 'neuronal stem cell', 'spinal cord interneuron']
    if dataname == "Zeisel_2018": # 17
        class_set = ['CNS neuron (sensu Vertebrata)', 'astrocyte', 'cerebellum neuron', 'dentate gyrus of hippocampal formation granule cell',
                     'endothelial cell of vascular tree', 'enteric neuron', 'ependymal cell', 'glial cell', 'inhibitory interneuron',
                     'microglial cell', 'neuroblast', 'oligodendrocyte', 'peptidergic neuron', 'pericyte cell', 'peripheral sensory neuron',
                     'perivascular macrophage', 'vascular associated smooth muscle cell']
    return class_set


def read_simu(data_path, cross=False):
    data_mat = h5py.File("../scrna/Simulation_scalability_data/" + data_path)
    x = np.array(data_mat["X"])
    y = np.array(data_mat["Y"])
    if cross:
        batch = np.array(data_mat["B"])
        return x, y, batch
    else:
        return x, y


def normalize(adata, highly_genes = None, size_factors=True, normalize_input=True, logtrans_input=True):
    sc.pp.filter_genes(adata, min_counts=10)
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

