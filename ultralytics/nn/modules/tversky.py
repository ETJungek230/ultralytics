# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tversky similarity and projection layers adapted from mdoumbouya/tversky.

The implementation is kept inside Ultralytics so models do not require the external
``tversky`` package. Batched transpose operations use ``transpose`` for ONNX export.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    """Transpose the final two dimensions using an ONNX-supported operation."""
    return x.transpose(-2, -1)


def intersection_measure(dot_a_m, dot_b_m, ind_a, ind_b, reduction):
    """Compute the Tversky intersection measure from masked feature projections."""
    if reduction == "product":
        return dot_a_m @ _transpose_last_two(dot_b_m)

    inter_a = dot_a_m @ _transpose_last_two(ind_b.to(dtype=dot_a_m.dtype))
    inter_b = ind_a.to(dtype=dot_b_m.dtype) @ _transpose_last_two(dot_b_m)

    if reduction == "min":
        return torch.minimum(inter_a, inter_b)
    if reduction == "max":
        return torch.maximum(inter_a, inter_b)
    if reduction == "mean":
        return (inter_a + inter_b) / 2
    if reduction == "gmean":
        return torch.exp((torch.log(inter_a) + torch.log(inter_b)) / 2.0)
    if reduction == "softmin":
        return -torch.logsumexp(-torch.stack((inter_a, inter_b), dim=0), dim=0)
    raise ValueError(f"Unknown intersection reduction: {reduction!r}")


def difference_measure(dot_a_m, dot_b_m, ind_a, ind_b, reduction):
    """Compute active features of A that are not shared with B."""
    if reduction == "ignorematch":
        return dot_a_m @ _transpose_last_two((~ind_b).to(dtype=dot_a_m.dtype))
    if reduction == "substractmatch":
        dot_a_ind_b = dot_a_m @ _transpose_last_two(ind_b.to(dtype=dot_a_m.dtype))
        dot_b_ind_a = ind_a.to(dtype=dot_b_m.dtype) @ _transpose_last_two(dot_b_m)
        ind_gt = dot_a_ind_b > dot_b_ind_a
        return (
            dot_a_m @ _transpose_last_two((~ind_b).to(dtype=dot_a_m.dtype))
            + dot_a_ind_b * ind_gt
            - dot_b_ind_a * ind_gt
        )
    raise ValueError(f"Unknown difference reduction: {reduction!r}")


def _project(x, fbank, threshold):
    """Project inputs onto the feature bank and return scores, masked scores, and indicators."""
    dot = x @ fbank.transpose(-2, -1)
    ind = dot > threshold
    return dot, dot * ind, ind


def similarity_contrast_model(
    x, y, fbank, threshold, alpha, beta, theta, normalize, intersection_reduction, difference_reduction
):
    """Compute the contrast-form Tversky similarity logits."""
    if normalize:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

    _, dot_x_m, ind_x = _project(x, fbank, threshold)
    _, dot_y_m, ind_y = _project(y, fbank, threshold)
    inter = intersection_measure(dot_x_m, dot_y_m, ind_x, ind_y, intersection_reduction)
    diff_xy = difference_measure(dot_x_m, dot_y_m, ind_x, ind_y, difference_reduction)
    diff_yx = difference_measure(dot_y_m, dot_x_m, ind_y, ind_x, difference_reduction)
    return theta * inter - alpha * diff_xy - beta * _transpose_last_two(diff_yx)


def similarity_ratio_model(
    x, y, fbank, threshold, alpha, beta, theta, normalize, intersection_reduction, difference_reduction
):
    """Compute the ratio-form Tversky similarity scores."""
    if normalize:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

    _, dot_x_m, ind_x = _project(x, fbank, threshold)
    _, dot_y_m, ind_y = _project(y, fbank, threshold)
    inter = intersection_measure(dot_x_m, dot_y_m, ind_x, ind_y, intersection_reduction)
    diff_xy = difference_measure(dot_x_m, dot_y_m, ind_x, ind_y, difference_reduction)
    diff_yx = difference_measure(dot_y_m, dot_x_m, ind_y, ind_x, difference_reduction)
    return inter / (inter + alpha * diff_xy + beta * _transpose_last_two(diff_yx) + 1)


class TverskyProjection(nn.Module):
    """Project input embeddings to class similarity logits using learned prototypes."""

    def __init__(
        self,
        embedding_dim,
        class_count,
        fbank_size,
        similarity_model,
        normalize,
        intersection_reduction="product",
        difference_reduction="ignorematch",
    ):
        super().__init__()
        self.feature_bank = nn.Embedding(fbank_size, embedding_dim)
        self.prototypes = nn.Embedding(class_count, embedding_dim)
        nn.init.uniform_(self.feature_bank.weight)
        nn.init.uniform_(self.prototypes.weight)
        self.theta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.threshold = 0.0
        self.similarity_model = similarity_model
        self.normalize = normalize
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction

    def similarity(self, x, y, alpha=None, beta=None, theta=None, fbank=None):
        """Compute similarity using the configured Tversky formulation."""
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        theta = self.theta if theta is None else theta
        fbank = self.feature_bank.weight if fbank is None else fbank
        fn = {"contrast": similarity_contrast_model, "ratio": similarity_ratio_model}.get(self.similarity_model)
        if fn is None:
            raise ValueError(f"Unknown similarity_model: {self.similarity_model!r}")
        return fn(
            x,
            y,
            fbank,
            self.threshold,
            alpha,
            beta,
            theta,
            self.normalize,
            self.intersection_reduction,
            self.difference_reduction,
        )

    def forward(self, x, prototypes=None, fbank=None):
        """Return class similarity scores, preserving an optional batch/position shape."""
        fbank = self.feature_bank.weight if fbank is None else fbank
        prototypes = self.prototypes.weight if prototypes is None else prototypes
        original_shape = None
        if x.ndim == 3:
            original_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
        x = self.similarity(x, prototypes, fbank=fbank)
        if original_shape is not None:
            x = x.reshape(original_shape[0], original_shape[1], -1)
        return x


class TverskySimilarity(nn.Module):
    """Compare two embedding sets using a learned feature bank."""

    def __init__(
        self,
        embedding_dim,
        fbank_size,
        similarity_model,
        normalize,
        intersection_reduction="product",
        difference_reduction="ignorematch",
    ):
        super().__init__()
        self.feature_bank = nn.Embedding(fbank_size, embedding_dim)
        nn.init.uniform_(self.feature_bank.weight)
        self.theta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.threshold = 0.0
        self.similarity_model = similarity_model
        self.normalize = normalize
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction

    def forward(self, x, y, alpha=None, beta=None, theta=None):
        """Return pairwise Tversky similarity scores."""
        fn = {"contrast": similarity_contrast_model, "ratio": similarity_ratio_model}.get(self.similarity_model)
        if fn is None:
            raise ValueError(f"Unknown similarity_model: {self.similarity_model!r}")
        return fn(
            x,
            y,
            self.feature_bank.weight,
            self.threshold,
            self.alpha if alpha is None else alpha,
            self.beta if beta is None else beta,
            self.theta if theta is None else theta,
            self.normalize,
            self.intersection_reduction,
            self.difference_reduction,
        )
