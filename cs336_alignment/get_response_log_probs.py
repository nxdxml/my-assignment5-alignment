import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from cs336_alignment.compute_entropy import compute_entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute log p(y | x) for each token, and optionally token-level entropy.
    Assumes input_ids and labels are same shape and aligned.
    """

    logits = model(input_ids).logits  # (B, S, V)

    # Compute log-softmax over vocabulary
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, S, V)

    # Extract per-token log p(y | x<t>) from label positions
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, S)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        token_entropy = compute_entropy(logits)  # (B, S)
        result["token_entropy"] = token_entropy

    return result
