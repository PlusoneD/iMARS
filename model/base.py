from typing import Iterable
from torch import nn as nn
import torch
from torch.distributions import Normal
from .compose import FCLayers


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

def identity(x):
    return x

# Encoder
class Encoder(nn.Module):
    """
       Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

       Uses a fully-connected neural network of ``n_hidden`` layers.

       Parameters
       ----------
       n_input
           The dimensionality of the input (data space)
       n_output
           The dimensionality of the output (latent space)
       n_cat_list
           A list containing the number of categories
           for each category of interest. Each category will be
           included using a one-hot encoding
       n_layers
           The number of fully-connected hidden layers
       n_hidden
           The number of nodes per hidden layer
       dropout_rate
           Dropout rate to apply to each of the hidden layers
       distribution
           Distribution of z
       **kwargs
           Keyword args for :class:`~scvi.core.modules._base.FCLayers`
       """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
        **kwargs,
    ):
        super().__init__()

        self.distribuion = distribution
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

    def forward(self, x: torch.Tensor, *cat_list: int):
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent


class Decoder(nn.Module):
    """
        Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.
        Uses a fully-connected neural network of ``n_hidden`` layers.

        Parameters
        ----------
        n_input
            The dimensionality of the input (latent space)
        n_output
            The dimensionality of the output (data space)
        n_cat_list
            A list containing the number of categories
            for each category of interest. Each category will be
            included using a one-hot encoding
        n_layers
            The number of fully-connected hidden layers
        n_hidden
            The number of nodes per hidden layer
        dropout_rate
            Dropout rate to apply to each of the hidden layers
        inject_covariates
            Whether to inject covariates in each layer, or just the first (default).
        use_batch_norm
            Whether to use batch norm in layers
        use_layer_norm
            Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        # The decoder return values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value exp(12) to avoid nans
        px_rate = library * px_scale
        return px_scale, px_rate, px_dropout




