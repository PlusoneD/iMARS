import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse

logger = logging.getLogger(__name__)

def _compute_mean_and_var(
    data: Union[sp_sparse.spmatrix, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    sum_counts = data.sum(axis=1)
    masked_log_sum = np.ma.log(sum_counts)
    if np.ma.is_masked(masked_log_sum):
        logger.warning(
            "This dataset has some empty cells, this might fail inference."
            "Data should be filtered with `scanpy.pp.filter_cells()`"
        )
    log_counts = masked_log_sum.filled(0)
    local_mean = (np.mean(log_counts).reshape(-1, 1)).astype(np.float32)
    local_var = (np.var(log_counts).reshape(-1, 1)).astype(np.float32)
    return local_mean, local_var


def _compute_mean_and_var_of_batch(
    data: Union[sp_sparse.spmatrix, np.ndarray],
    batch: Tuple,
):
    local_means = np.zeros((data.shape[0], 1))
    local_vars = np.zeros((data.shape[0], 1))
    for i_batch in np.unique(batch):
        idx_batch = batch == i_batch
        idx_data = data[idx_batch]
        (local_means[idx_batch], local_vars[idx_batch]) = _compute_mean_and_var(idx_data)

    return local_means, local_vars
