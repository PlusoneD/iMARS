
import torch
import numpy as np
from args_parser import get_parser
from model.mars import MARS
from model.experiment_dataset import ExperimentDataset
from data.maca_facs import MacaData
import warnings
import os
import pandas as pd
from anndata import read_h5ad
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
    test_data = []
    pretrain_data = []
    annotations = []
    file_names = os.listdir('../dataset')
    for idx,file_name in enumerate(file_names):
        tiss = read_h5ad('../dataset/' + file_name)
        tissue = file_name[24:-5]

        tiss[pd.isna(tiss.X)] = 0

        ann = list(tiss.obs['cell_ontology_class_reannotated'])
        annotations += ann
        annotations_set = sorted(set(annotations))
        mapping = {a:idx for idx,a in enumerate(annotations_set)}
        truth_labels = [mapping[a] for a in ann]
        tiss.obs['truth_labels'] = pd.Categorical(values=truth_labels)

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

    IDs_to_celltypes = {v: k for k, v in mapping.items()}

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
        adata, landmarks, scores = mars.train(evaluation_mode=True)
        mars.name_cell_types(adata, landmarks, cell_type_name_map)

        # adata.write(params.MODEL_DIR+tissue+'/'+tissue+'.h5ad')

        avg_score_direct[idx, 0] = scores['accuracy']
        avg_score_direct[idx, 1] = scores['f1_score']
        avg_score_direct[idx, 2] = scores['nmi']
        avg_score_direct[idx, 3] = scores['adj_rand']
        avg_score_direct[idx, 4] = scores['adj_mi']

        print('{}: Acc {}, F1_score {}, NMI {}, Adj_Rand {}, Adj_MI {}'.format(unlabeled_data.metadata,
                                                                               scores['accuracy'], scores['f1_score'],
                                                                               scores['nmi'],
                                                                               scores['adj_rand'], scores['adj_mi']))

    avg_score_direct = np.mean(avg_score_direct, axis=0)
    print('\nAverage: Acc {}, F1_score {}, NMI {}, Adj_Rand {}, Adj_MI {}\n'.format(avg_score_direct[0],
                                                                                    avg_score_direct[1],
                                                                                    avg_score_direct[2],
                                                                                    avg_score_direct[3],
                                                                                    avg_score_direct[4]))


if __name__ == '__main__':
    main()

