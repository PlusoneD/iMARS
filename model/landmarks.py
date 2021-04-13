
import numpy as np
import torch
from sklearn.cluster import KMeans


def init_landmarks(n_clusters, tr_load, test_load, model, device, mode="kmeans", pretrain=True):
    """
    Initialization of landmarks of the labeled and unlabeled datasets.
    :param n_clusters:
    :param tr_load:
    :param test_load:
    :param model:
    :param device:
    :param mode:
    :param pretrain:
    :return:
    """
    lndmk_tr = [torch.zeros(size=(len(np.unique(tr.dataset.y)), model.n_latent), requires_grad=True, device=device) for tr in tr_load]
    lndmk_test = [torch.zeros(size=(1, model.n_latent), requires_grad=True, device=device) for _ in range(n_clusters)]
    kmeans_init_tr = [init_step(tr.dataset, model, device, pretrained=pretrain, mode=mode) for tr in tr_load]
    kmeans_init_test = init_step(test_load.dataset, model, device, pretrained=pretrain, mode=mode, n_clusters=n_clusters)
    with torch.no_grad():
        [lndmk.copy_(kmeans_init_tr[idx]) for idx, lndmk in enumerate(lndmk_tr)]
        [lndmk_test[i].copy_(kmeans_init_test[i, :]) for i in range(kmeans_init_test.shape[0])]
    return lndmk_tr, lndmk_test

def init_step(dataset, model, device, pretrained, mode='kmeans', n_clusters=None):
    """
    Initialization of landmarks with k-means or k-means++ given dataset.
    :param dataset:
    :param model:
    :param device:
    :param pretrained:
    :param mode:
    :param n_clusters:
    :return:
    """
    if n_clusters == None:
        n_clusters = len(np.unique(dataset.y))

    n_examples = len(dataset.x)

    X = torch.stack([dataset.x[i] for i in range(n_examples)])
    batch_index = torch.Tensor(dataset.batch)

    X = X.to(device)
    batch_index = batch_index.to(device)
    z = model.get_latents(X, batch_index)

    # 方案一：重写KMeans
    # 方案二：用均值代替;
    kmeans = KMeans(n_clusters, random_state=0).fit(z.data.cpu().numpy())
    lndmk_encoded = torch.tensor(kmeans.cluster_centers_, device=device)

    return lndmk_encoded
        # 方案三：用latent




def compute_landmk_tr(z, target, prev_landmarks=None, tau=0.2):
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c:target.eq(c).nonzero(), uniq))

    landmarks_mean = torch.stack([z[idx_class].mean(0) for idx_class in class_idxs]).squeeze()

    if prev_landmarks is None or tau==0:
        return landmarks_mean

    suma = prev_landmarks.sum(0)
    n_lndmk = prev_landmarks.shape[0]
    lndmk_dist_part = (tau/(n_lndmk-1)) * torch.stack([suma-p for p in prev_landmarks])
    landmarks = 1/(1-tau) * (landmarks_mean-lndmk_dist_part)

    return landmarks