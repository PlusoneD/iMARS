import torch
from torch.distributions import Normal, Poisson, LogNormal
from torch.distributions import kl_divergence as kl
from .distribution import (
    NegativeBinomial,
    ZeroInflatedNegativeBinomial,
)
from .utils import euclidean_dist
from torch.nn import functional as F

def get_reconstruction_loss(x, px_rate, px_r, px_dropout,gene_likelihood, **kwargs) -> torch.Tensor:
    if gene_likelihood == "zinb":
        reconst_loss = -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout).log_prob(x).sum(dim=-1)
    elif gene_likelihood == "nb":
        reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
    elif gene_likelihood == "possion":
        reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
    return reconst_loss


def reconstruction_loss(x, local_l_mean, local_l_var, parameter, use_observed_lib_size=True, gene_likelihood="zinb"):

    qz_m = parameter["qz_m"]
    qz_v = parameter["qz_v"]
    ql_m = parameter["ql_m"]
    ql_v = parameter["ql_v"]
    px_rate = parameter["px_rate"]
    px_r = parameter["px_r"]
    px_dropout = parameter["px_dropout"]

    # norm(0,I)
    mean = torch.zeros_like(qz_m)
    scale = torch.ones_like(qz_v)
    kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)

    kl_divergence_l = kl(LogNormal(ql_m, torch.sqrt(ql_v)), LogNormal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

    reconst_loss = get_reconstruction_loss(x, px_rate, px_r, px_dropout, gene_likelihood)

    loss = len(x) * torch.mean(reconst_loss + kl_divergence_z + kl_divergence_l)
    return loss

def loss_test_basic(z, prototypes):
    # 修改为分布

    dists = euclidean_dist(z, prototypes)

    min_dist = torch.min(dists, 1)

    y_hat = min_dist[1]
    args_uniq = torch.unique(y_hat, sorted=True)
    args_count = torch.stack([(y_hat==x_u).sum() for x_u in args_uniq])
    min_dist = min_dist[0]
    loss_val = torch.stack([min_dist[y_hat==idx_class].mean(0) for idx_class in args_uniq]).mean()
    return loss_val, args_count

def loss_test(z, prototypes, tau):
    loss_val_test, args_count = loss_test_basic(z, prototypes)

    if tau > 0:
        dists = euclidean_dist(prototypes, prototypes)
        n_proto = prototypes.shape[0]
        loss_val2 = -torch.sum(dists) / (n_proto * n_proto - n_proto)
        loss_val_test += tau * loss_val2

    return loss_val_test, args_count


def loss_task(z, prototypes, target, criterion="dist"):
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c:target.eq(c).nonzero(), uniq))

    # prepare targets so they start from 0, 1
    for idx, v in enumerate(uniq):
        target[target==v] = idx

    dists = euclidean_dist(z, prototypes)

    if criterion=="NNLoss":
        loss = torch.nn.NLLLoss()
        log_p_y = F.log_softmax(-dists, dim=1)
        loss_val = loss(log_p_y, target)
        _, y_hat = log_p_y.max(1)
    elif criterion=="dist":
        loss_val = torch.stack([dists[idx_example, idx_proto].mean(0) for idx_proto, idx_example in enumerate(class_idxs)]).mean()
        y_hat = torch.max(-dists, 1)[1]

    acc_val = y_hat.eq(target.squeeze()).float().mean()
    return loss_val, acc_val


def loss_distribution(local_l_mean, local_l_var, parameter):
    qz_m = parameter["qz_m"]
    qz_v = parameter["qz_v"]
    ql_m = parameter["ql_m"]
    ql_v = parameter["ql_v"]

    # norm(0,I)
    mean = torch.zeros_like(qz_m)
    scale = torch.ones_like(qz_v)
    kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
    kl_divergence_l = kl(LogNormal(ql_m, torch.sqrt(ql_v)), LogNormal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

    loss = torch.mean(kl_divergence_z) + torch.mean(kl_divergence_l)
    return loss