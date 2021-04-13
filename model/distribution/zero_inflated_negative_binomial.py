import warnings
from typing import Optional, Union, Tuple
from torch.distributions.utils import broadcast_all, lazy_property, probs_to_logits, logits_to_probs
import torch

from .negative_binomial import NegativeBinomial
from .utils import log_zinb_positive

class ZeroInflatedNegativeBinomial(NegativeBinomial):
    r"""
        Zero-inflated negative binomial distribution.
    """
    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        zi_logits: Optional[torch.Tensor] = None,
        validate_args: bool = False,
     ):
        super().__init__(
            total_count = total_count,
            probs = probs,
            logits = logits,
            mu = mu,
            theta = theta,
            validate_args = validate_args,
        )

        self.zi_logits, self.mu, self.theta = broadcast_all(zi_logits, mu, theta)


    @property
    def mean(self):
        pi = self.zi_probs
        return (1 - pi) * self.mu

    @property
    def variance(self):
        raise NotImplementedError

    @lazy_property
    def zi_logits(self) -> torch.Tensor:
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self) -> torch.Tensor:
        return logits_to_probs(self.zi_logits, is_binary=True)

    def sample(self, sample_shape: Union[torch.Size, Tuple] = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            samp = super().sample(sample_shape = sample_shape)
            is_zero = torch.rand_like(samp) <= self.zi_probs
            samp[is_zero] = 0.0
            return samp

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn("The value argument must be within the support of the distribution",
                UserWarning)
        return log_zinb_positive(value, self.mu, self.theta, self.zi_logits, eps=1e-8)


