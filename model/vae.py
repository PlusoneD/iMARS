try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Dict
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


from .base import Encoder, Decoder
from .compose import one_hot
torch.backends.cudnn.benchmark = True


# VAE model
class VAE(nn.Module):
    """
    Variational auto-encoder model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches

    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    """

    def __init__(
        self,
        device,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_observed_lib_size: bool = True,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood

        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch'"
                "], but input was {}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-dimensional data
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_cat_list=[n_batch] if encode_covariates else None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        # l encoder goes from n_input-dimensional data to 1-dimensional library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_cat_list=[n_batch] if encode_covariates else None,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        #decoder goes from n_latent-dimensional space to n_input-dimensional data
        self.decoder = Decoder(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )



    def forward(self, x, batch_index=None, y=None)-> Dict[str, torch.Tensor]:
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)

        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, batch_index.view(len(batch_index), 1))
        ql_m, ql_v, library_encoded = self.l_encoder(x_, batch_index.view(len(batch_index), 1))

        #if not self.use_observed_lib_size:
        library = library_encoded

        px_scale, px_rate, px_dropout = self.decoder(z, library, batch_index.view(len(batch_index), 1), y)

        px_r = F.linear(one_hot(batch_index.view(len(batch_index), 1), self.n_batch), self.px_r)
        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def get_latents(self, x, batch_index, y=None) -> torch.Tensor:
        """
        Returns the result of ``sample_from_posterior_z`` inside a list.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)

        Returns
        -------
        type
            one element list of tensor
        """
        return self.sample_from_posterior_z(x, batch_index, y)

    def sample_from_posterior_z(
            self, x, batch_index, y=None, give_mean=True, n_samples=5000
    ) -> torch.Tensor:
        """
        Samples the tensor of latent values from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        give_mean
            is True when we want the mean of the posterior  distribution rather than sampling (Default value = False)
        n_samples
            how many MC samples to average over for transformed mean (Default value = 5000)

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, batch_index.view(len(batch_index), 1), y)

        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.z_encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return z

    def sample_from_posterior_l(
            self, x, batch_index, give_mean=False
    ) -> torch.Tensor:
        """
        Samples the tensor of library sizes from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        give_mean
            Return mean or sample

        Returns
        -------
        type
            tensor of shape ``(batch_size, 1)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x, batch_index.view(len(batch_index), 1))
        if give_mean is False:
            library = library
        else:
            library = torch.distributions.LogNormal(ql_m, ql_v.sqrt()).mean
        return library

    def get_sample_scale(
            self, x, batch_index, y=None
    ) -> torch.Tensor:
        """
        Returns the tensor of predicted frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        n_samples
            number of samples (Default value = 1)
        transform_batch
            int of batch to transform samples into (Default value = None)

        Returns
        -------
        type
            tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        """
        return self.forward(
            x,
            batch_index=batch_index,
            y=y,
        )["px_scale"]

    def get_sample_rate(
            self, x, batch_index=None, y=None
    ) -> torch.Tensor:
        """
        Returns the tensor of means of the negative binomial distribution.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        n_samples
            number of samples (Default value = 1)
        transform_batch
            int of batch to transform samples into (Default value = None)

        Returns
        -------
        type
            tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        """
        return self.forward(
            x,
            batch_index=batch_index,
            y=y,
        )["px_rate"]



