# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


def sample(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform sampling on a probability distribution. This is faster than top_p when p = 1.

    Args:
        probs (torch.Tensor): probability distribution tensor.
    Returns:
        next_tokens (torch.Tensor): sampled indices
        next_tokens_prob (torch.Tensor): prob of samples (not logprob!)
    """
    next_tokens = torch.multinomial(probs, num_samples=1)
    next_tokens_prob = torch.gather(probs, -1, next_tokens)
    return next_tokens.squeeze(-1), next_tokens_prob.squeeze(-1)


def top_p(probs: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): probability distribution tensor.
        p (float or torch.Tensor): probability threshold for top-p sampling.
                                   expected shape: (batch_size, 1)
    Returns:
        next_tokens (torch.Tensor): sampled indices
        next_tokens_prob (torch.Tensor): prob of samples (not logprob!)

    Note:
        Top-p sampling selects the smallest set whose cumulative
        probability mass exceeds the threshold p. The distribution
        is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[probs_sum - probs_sort > p] = 0.0
    probs_sort /= probs_sort.sum(-1, keepdim=True)
    sample = torch.multinomial(probs_sort, num_samples=1)
    next_tokens = torch.gather(probs_idx, -1, sample)
    next_tokens_prob = torch.gather(probs_sort, -1, sample)
    return next_tokens.squeeze(-1), next_tokens_prob.squeeze(-1)
