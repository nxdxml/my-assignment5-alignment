import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, sequence_length, vocab_size)
containing unnormalized logits.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length). The entropy for each next-token
prediction.
    """    
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits

    exp_logits = torch.exp(stable_logits) # b s v

    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True) # b s 1

    probs = exp_logits / sum_exp

    log_sum_exp = torch.log(sum_exp.squeeze(-1)) + max_logits.squeeze(-1)

    expected_logits = torch.sum(probs * logits, dim = -1)

    entropy = log_sum_exp - expected_logits

    return entropy