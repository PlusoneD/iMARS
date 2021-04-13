try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import anndata
import pandas as pd
from collections import OrderedDict, defaultdict
import numpy as np
import scanpy.api as sc
from scipy.spatial import distance

from .loss import reconstruction_loss, loss_test, loss_task, loss_distribution
from .vae import VAE
from .utils import init_data_loaders
from model.landmarks import init_landmarks, compute_landmk_tr
from .utils import euclidean_dist
from model.metrics import compute_scores


class MARS:
    """
    Parameters
    ----------
    n_clusters
        number of clusters in the unlabeled meta-dataset
    params
        parameters of the MARS model
    labeled_data
        list of labeled datasets. Each dataset needs to be instance of CellDataset.
    unlabeled_data
        unlabeled dataset. Instance of CellDataset.
    pretrain_data
        dataset for pretraining MARS. Instance of CellDataset. If not specified, unlabeled_data will be used.
    val_split 
        percentage of data to use for train/val split (default: 1, meaning no validation set)
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.(default:1)
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    tau
        regularizer for inter-cluster distance
    """
    
    def __init__(
        self,
        n_clusters,
        params,
        labeled_data,
        unlabeled_data, 
        pretrain_data=None,
        val_split: float = 1.0, 
        n_hidden:int = 1000, 
        n_latent: int = 100,
        n_layers: int = 1, 
        dropout_rate: float = 0.1, 
        dispersion: Literal["gene", "gene-batch"] = "gene-batch",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        tau=0.2
    ):
        train_load, test_load, pretrain_load, val_load = init_data_loaders(labeled_data, unlabeled_data, 
                                                                           pretrain_data, params.pretrain_batch, 
                                                                           val_split)
        self.train_loader = train_load
        self.test_loader = test_load
        self.pretrain_loader = pretrain_load
        self.val_loader = val_load
        self.labeled_metadata = [data.metadata for data in labeled_data]
        self.unlabeled_metadata = unlabeled_data.metadata
        self.genes = unlabeled_data.yIDs
        x_dim = self.test_loader.dataset.get_dim()

        batch = set(pretrain_data.batch)
        unlabeled_batch = set(unlabeled_data.batch)
        batch.union(unlabeled_batch)
        for labeled in labeled_data:
            labeled_batch = set(labeled.batch)
            batch = batch.union(labeled_batch)
        n_batch = len(batch)

        self.n_clusters = n_clusters
        self.device = params.device
        self.epochs = params.epochs
        self.epochs_pretrain = params.epochs_pretrain
        self.pretrain_flag = params.pretrain
        self.model_file = params.model_file
        self.lr = params.learning_rate
        self.lr_gamma = params.lr_scheduler_gamma
        self.step_size = params.lr_scheduler_step
        self.tau = tau

        self.model = VAE(
            device=self.device,
            n_input=x_dim,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
        )


    def pretrain_train(self):
        optim = torch.optim.Adam(params=list(self.model.parameters()), lr=self.lr)
        print('Pretraining..')
        for _ in range(100):  # self.epochs_pretrain):
            for _, batch in enumerate(self.pretrain_loader):
                x, batch_index, local_l_mean, local_l_var, y, id = batch
                x = x.to(self.device)
                batch_index = batch_index.to(self.device)
                paramter = self.model(x, batch_index)
                loss = reconstruction_loss(x, local_l_mean, local_l_var, paramter)
                optim.zero_grad()
                loss.backward()
                optim.step()

        for _, batch in enumerate(self.pretrain_loader):
            x, batch_index, local_l_mean, local_l_var, y, id = batch
            x = x.to(self.device)
            batch_index = batch_index.to(self.device)
            z = self.model.get_latents(x, batch_index.view(len(batch_index), 1))
        return z.detach().numpy()




    def pretrain(self, optim):
        """
            Pretraining model with VAE
            optim: optimizer
        """
        print('Pretraining..')
        for _ in range(100): #self.epochs_pretrain):
            for _, batch in enumerate(self.pretrain_loader):
                x, batch_index, local_l_mean, local_l_var, y, id = batch
                x, batch_index, local_l_mean, local_l_var, y = \
                x.to(self.device), batch_index.to(self.device), local_l_mean.to(self.device), local_l_var.to(self.device), y.to(self.device)
                self.model = self.model.to(self.device)
                paramter = self.model(x, batch_index)
                loss = reconstruction_loss(x, local_l_mean, local_l_var, paramter)
                loss = loss.to(self.device)
                optim.zero_grad()
                loss.backward()
                optim.step()

    def train(self, evaluation_mode=True, save_all_embeddings=True):
        tr_iter = [iter(dl) for dl in self.train_loader]

        if self.val_loader is not None:
            val_iter = [iter(dl) for dl in self.val_loader]

        optim_pretrain = torch.optim.Adam(params=list(self.model.parameters()), lr=self.lr)
        if self.pretrain_flag:
            self.pretrain(optim_pretrain)
        else:
            self.model.load_state_dict(torch.load(self.MODEL_FILE))

        test_iter = iter(self.test_loader)
        landmk_tr, landmk_test = init_landmarks(self.n_clusters, self.train_loader, self.test_loader, self.model, self.device)
        optim = torch.optim.Adam(list(self.model.z_encoder.parameters()) + list(self.model.l_encoder.parameters()),
                                 self.lr)
        optim_lndmk_test = torch.optim.Adam(landmk_test, self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=self.lr_gamma, step_size=self.step_size)

        best_acc = 0
        for epoch in range(1, self.epochs+1):
            self.model.train()
            loss_tr, acc_tr, landmk_tr, landmk_test = self.do_epoch(tr_iter, test_iter, optim, optim_lndmk_test, landmk_tr, landmk_test)

            if epoch == self.epochs:
                print('\n=== Epoch: {} ==='.format(epoch))
                print('Train acc: {}'.format(acc_tr))
            if self.val_loader is None:
                continue
            self.model.eval()

            with torch.no_grad():
                loss_val, acc_val = self.do_val_epoch(val_iter, landmk_tr)
                if acc_val > best_acc:
                    print('Saving model...')
                    best_acc = acc_val
                    best_state = self.model.state_dict()
                    #torch.save(model.state_dict(), self.model_file)
                postfix = ' (Best)' if acc_val >= best_acc else ' (Best: {})'.format(best_acc)
                print('Val loss: {}, acc: {}{}'.format(loss_val, acc_val, postfix))
            lr_scheduler.step()

        if self.val_loader is None:
            best_state = self.model.state_dict()  # best is last

        landmk_all = landmk_tr + [torch.stack(landmk_test).squeeze()]

        adata_test, eval_results = self.assign_labels(landmk_all[-1], evaluation_mode)

        adata = self.save_result(tr_iter, adata_test, save_all_embeddings)

        if evaluation_mode:
            return adata, landmk_all, eval_results
        return landmk_all

    def do_epoch(self, tr_iter, test_iter, optim, optim_landmk_test, landmk_tr, landmk_test):
        """
        One training epoch.
        tr_iter: iterator over labeled meta-data
        test_iter: iterator over unlabeled meta-data
        optim: optimizer for embedding
        optim_landmk_test: optimizer for test landmarks
        landmk_tr: landmarks of labeled meta-data from previous epoch
        landmk_test: landmarks of unlabeled meta-data from previous epoch
        """

        for param in self.model.parameters():
            param.requires_grad = False

        for landmk in landmk_test:
            landmk.require_grad=False

        optim_landmk_test.zero_grad()

        #update centroids
        task_idx = torch.randperm(len(tr_iter))

        for task in task_idx:
            task = int(task)
            x, batch_index, local_l_mean, local_l_var, y, id =next(tr_iter[task])
            x, batch_index, y = x.to(self.device), batch_index.to(self.device), y.to(self.device)
            z = self.model.get_latents(x, batch_index)
            curr_landmk_tr = compute_landmk_tr(z, y, landmk_tr[task], tau=self.tau)
            landmk_tr[task] = curr_landmk_tr.data  # save landmarks

        for landmk in landmk_test:
            landmk.requires_grad=True

        x, batch_index, _, _, y, _ = next(test_iter)
        x, batch_index, y = x.to(self.device), batch_index.to(self.device), y.to(self.device)
        z = self.model.get_latents(x, batch_index)
        loss, args_count = loss_test(z, torch.stack(landmk_test).squeeze(), self.tau)

        loss = loss.to(self.device)
        loss.backward()
        optim_landmk_test.step()

        #update embedding
        for param in self.model.parameters():
            param.requires_grad =True
        for landmk in landmk_test:
            landmk.require_grad=False

        optim.zero_grad()
        total_acc = 0
        total_loss = 0
        n_tasks = 0
        mean_acc = 0

        task_idx = torch.randperm(len(tr_iter))

        for task in task_idx:
            task = int(task)
            x, batch_index, local_l_mean, local_l_var, y, id = next(tr_iter[task])
            x, batch_index,local_l_mean, local_l_var, y = \
                x.to(self.device), batch_index.to(self.device),local_l_mean.to(self.device), local_l_var.to(self.device), y.to(self.device)
            paramter = self.model(x, batch_index)
            klloss = loss_distribution(local_l_mean, local_l_var, paramter)
            z = self.model.get_latents(x, batch_index)
            loss, acc = loss_task(z, landmk_tr[task], y, criterion="dist")
            total_loss += (loss + klloss)
            total_acc += acc.item()
            n_tasks += 1

        '''
        for task in task_idx:
            task = int(task)
            x, batch_index, local_l_mean, local_l_var, y, id = next(tr_iter[task])
            x, batch_index,local_l_mean, local_l_var, y = \
                x.to(self.device), batch_index.to(self.device),local_l_mean.to(self.device), local_l_var.to(self.device), y.to(self.device)
            paramter = self.model(x, batch_index)
            reloss = reconstruction_loss(x, local_l_mean, local_l_var, paramter)
            z = self.model.get_latents(x, batch_index)
            loss, acc = loss_task(z, landmk_tr[task], y, criterion="dist")
            total_loss += (loss + reloss)
            total_acc += acc.item()
            n_tasks += 1
        '''
        if n_tasks>0:
            mean_acc = total_acc / n_tasks

        # test part
        x, batch_index, _, _, y, _ = next(test_iter)
        x, batch_index, y = x.to(self.device), batch_index.to(self.device), y.to(self.device)
        z = self.model.get_latents(x, batch_index)
        loss,_ = loss_test(z, torch.stack(landmk_test).squeeze(), self.tau)
        total_loss += loss
        n_tasks += 1

        mean_loss = total_loss / n_tasks
        mean_loss = mean_loss.to(self.device)
        mean_loss.backward()
        optim.step()
        return mean_loss, mean_acc, landmk_tr, landmk_test


    def assign_labels(self,landmk_test, evaluation_mode):
        torch.no_grad()
        self.model.eval() # eval mode

        test_iter = iter(self.test_loader)
        x_test, batch_index, _, _, y_true, cells = next(test_iter)
        x_test, batch_index, y_true = x_test.to(self.device), batch_index.to(self.device), y_true.to(self.device)
        z = self.model.get_latents(x_test, batch_index)

        dists = euclidean_dist(z, landmk_test)
        y_pred = torch.min(dists, 1)[1]

        adata = self.pack_anndata(x_test, cells, z, y_true, y_pred)

        eval_results = None
        if evaluation_mode:
            eval_results = compute_scores(y_true, y_pred)

        return adata, eval_results

    def pack_anndata(self, x_input, cells, mean, gtruth=[], estimated=[]):
        adata = anndata.AnnData(x_input.data.cpu().numpy())
        adata.obs_names = cells
        adata.var_names = self.genes
        if len(estimated) != 0:
            adata.obs['MARS_labels'] = pd.Categorical(values=estimated.cpu().numpy())
        if len(gtruth) != 0:
            adata.obs['truth_labels'] = pd.Categorical(values=gtruth.cpu().numpy())
        adata.uns['MARS_embedding'] = mean.data.cpu().numpy()
        return adata

    def save_result(self, tr_iter, adata_test, save_all_embeddings):
        """Saving embeddings from labeled and unlabeled dataset, ground truth labels and
        predictions to joint anndata object."""
        adata_all = []

        if save_all_embeddings:
            for task in range(len(tr_iter)):  # saving embeddings from labeled dataset
                task = int(task)
                x, batch_index, _, _, y, cells = next(tr_iter[task])
                x, batch_index, y = x.to(self.device), batch_index.to(self.device), y.to(
                    self.device)
                z = self.model.get_latents(x, batch_index)

                adata_all.append(self.pack_anndata(x, cells, z, gtruth=y))

        adata_all.append(adata_test)

        if save_all_embeddings:
            adata = adata_all[0].concatenate(adata_all[1:], batch_key='experiment',
                                             batch_categories=self.labeled_metadata + [self.unlabeled_metadata])
        else:
            adata = adata_all[0]

        adata.obsm['MARS_embedding'] = np.concatenate([a.uns['MARS_embedding'] for a in adata_all])
        #adata.write('result_adata.h5ad')

        return adata

    def name_cell_types(self, adata, landmk_all, cell_name_mappings, top_match=5, umap_reduce_dim=True, n_dim=10):
        experiments = list(OrderedDict.fromkeys(list(adata.obs['experiment'])))

        encoded_tr = []
        landmk_tr = []
        landmk_tr_labels = []
        for idx, exp in enumerate(experiments[:-1]):
            tiss = adata[adata.obs['experiment'] == exp, :]

            if exp == self.unlabeled_metadata:
                raise ValueError("Error: Unlabeled dataset needs to be last one in the input anndata object.")

            encoded_tr.append(tiss.obsm['MARS_embedding'])
            landmk_tr.append(landmk_all[idx])
            landmk_tr_labels.append(np.unique(tiss.obs['truth_labels']))

        tiss = adata[adata.obs['experiment'] == self.unlabeled_metadata, :]
        y_pred_test = tiss.obs['MARS_labels']
        uniq_y_test = np.unique(y_pred_test)
        encoded_test = tiss.obsm['MARS_embedding']

        landmk_tr_labels = np.concatenate(landmk_tr_labels)
        encoded_tr = np.concatenate(encoded_tr)
        landmk_tr = np.concatenate([p.cpu() for p in landmk_tr])
        if umap_reduce_dim:
            encoded_extend = np.concatenate((encoded_tr, encoded_test, landmk_tr))
            adata = anndata.AnnData(encoded_extend)
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
            sc.tl.umap(adata, n_components=n_dim)
            encoded_extend = adata.obsm['X_umap']
            n1 = len(encoded_tr)
            n2 = n1 + len(encoded_test)
            encoded_tr = encoded_extend[:n1, :]
            encoded_test = encoded_extend[n1:n2, :]
            landmk_tr = encoded_extend[n2:, :]

        interp_names = defaultdict(list)
        for y_test in uniq_y_test:
            print('\nCluster label: {}'.format(str(y_test)))
            idx = np.where(y_pred_test == y_test)
            subset_encoded = encoded_test[idx[0], :]
            mean = np.expand_dims(np.mean(subset_encoded, axis=0), 0)

            sigma = self.estimate_sigma(subset_encoded)

            prob = np.exp(-np.power(distance.cdist(mean, landmk_tr, metric='euclidean'),2) / (2*sigma*sigma))
            prob = np.squeeze(prob, 0)
            normalizat = np.sum(prob)
            if normalizat==0:
                print('Unassigned')
                interp_names[y_test].append('unassigned')
                continue

            prob = np.divide(prob, normalizat)
            uniq_tr = np.unique(landmk_tr_labels)
            prob_unique = []
            for cell_type in uniq_tr:
                prob_unique.append(np.sum(prob[np.where(landmk_tr_labels==cell_type)]))

            sorted = np.argsort(prob_unique, axis=0)
            best = uniq_tr[sorted[-top_match:]]
            sortededv = np.sort(prob_unique, axis=0)
            sortedv = sortededv[-top_match:]
            for idx, b in enumerate(best):
                interp_names[y_test].append((cell_name_mappings[b], sortedv[idx]))
                print("{}:{}".format(cell_name_mappings[b], sortedv[idx]))
        return interp_names

    def estimate_sigma(self,dataset):
        nex = dataset.shape[0]
        dst = []
        for i in range(nex):
            for j in range(i+1, nex):
                dst.append(distance.euclidean(dataset[i,:], dataset[j,:]))
        return np.std(dst)
