import scanpy as sc
import anndata
import matplotlib.pyplot as plt

import torch
import numpy as np
from args_parser import get_parser
from model.mars import MARS
from model.experiment_dataset import ExperimentDataset
from data.maca_facs import MacaData
import warnings
from model.utils import plot_x_batch, plot_x_class

warnings.filterwarnings('ignore')


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(batch_key=None):
    """Init dataset"""

    test_maca = MacaData('../tabula-muris-senis-facs_mars.h5ad', annotation_type='cell_ontology_class_reannotated')

    print(test_maca.adata)
    test_maca.adata = test_maca.preprocess_data(test_maca.adata)
    tissues = list(set(test_maca.adata.obs['tissue']))
    tissues = sorted(tissues)

    test_data = []
    pretrain_data = []

    for tissue in tissues[:3]:
        tiss = test_maca.get_tissue_data(tissue)

        # extract 'batch'
        if batch_key == None:
            batch = []
        else:
            assert batch_key in tiss.obs.keys(), "{} is not a valid key for in adata.obs".format(batch_key)
            batch = np.array(tiss.obs[batch_key], dtype=np.int64)

        y_test = np.array(tiss.obs['truth_labels'], dtype=np.int64)

        test_data.append(ExperimentDataset(tiss.X.toarray(), tiss.obs_names,
                                           tiss.var_names, batch, tissue, y_test))
        pretrain_data.append(ExperimentDataset(tiss.X.toarray(), tiss.obs_names,
                                               tiss.var_names, batch, tissue))

        plot_x_batch(tiss)
        #plot_x_class(tiss)
    IDs_to_celltypes = {v: k for k, v in test_maca.celltype_id_map.items()}

    return test_data, pretrain_data, IDs_to_celltypes


def main():
    '''
    Initialize everything and train
    '''
    params = get_parser().parse_args()
    print(params)

    if torch.cuda.is_available() and not params.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    params.device = device

    init_seed(params)
    test_data, pretrain_data, cell_type_name_map = init_dataset(batch_key='batch')

    avg_score_direct = np.zeros((len(test_data), 5))

    for idx, unlabeled_data in enumerate(test_data):

        print(unlabeled_data.metadata)

        if unlabeled_data.metadata == 'Brain_Myeloid':
            continue

        # leave one tissue out
        labeled_data = test_data[:idx] + test_data[idx + 1:]

        n_clusters = len(np.unique(unlabeled_data.y))
        mars = MARS(n_clusters, params, labeled_data, unlabeled_data, pretrain_data[idx], n_hidden=128, n_latent=32)
        adata = anndata.AnnData(np.array([x.numpy() for x in unlabeled_data.x]))
        adata.obs['class'] = unlabeled_data.y
        adata.obsm["latent"] = mars.pretrain_train()
        adata.obs['batch'] = unlabeled_data.batch

        sc.pp.neighbors(adata, use_rep="latent", n_neighbors=15)
        sc.tl.umap(adata)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc.pl.umap(adata, color=["batch"], ax=ax, show=True)
        plt.show()

if __name__ == '__main__':
    main()

