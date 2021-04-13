# coding=utf-8
import torch.utils.data as data
import numpy as np
import torch

from data.utils import _compute_mean_and_var_of_batch


'''
Class representing dataset for an single-cell experiment.
'''

IMG_CACHE = {}


class ExperimentDataset(data.Dataset):
    
    
    def __init__(self, x, cells, genes, batch, metadata, y=[]):
        '''
        x: numpy array of gene expressions of cells (rows are cells)
        cells: cell IDs in the order of appearance
        genes: gene IDs in the order of appearance
        batch: batch identifier
        metadata: experiment identifier
        y: numeric labels of cells (empty list if unknown)
        '''
        super(ExperimentDataset, self).__init__()
        
        self.nitems = x.shape[0]
        if len(y)>0:
            print("== Dataset: Found %d items " % x.shape[0])
            print("== Dataset: Found %d classes" % len(np.unique(y)))
                
        if type(x)==torch.Tensor:
            self.x = x
        else:
            shape = x.shape[1]
            self.x = [torch.from_numpy(inst).view(shape).float() for inst in x]

        if len(batch) == 0:
            batch = np.zeros(len(self.x), dtype=np.int64)
        self.batch = tuple(batch.tolist())

        if len(y)==0:
            y = np.zeros(len(self.x), dtype=np.int64)
        self.y = tuple(y.tolist())

        self.xIDs = cells
        self.yIDs = genes
        self.metadata = metadata

        self.local_l_mean, self.local_l_var = _compute_mean_and_var_of_batch(x, self.batch)


    def __getitem__(self, idx):
        return self.x[idx].squeeze(), self.batch[idx], self.local_l_mean[idx], self.local_l_var[idx],\
               self.y[idx], self.xIDs[idx]

    def __len__(self):
        return self.nitems
    
    def get_dim(self):
        return self.x[0].shape[0]
    


