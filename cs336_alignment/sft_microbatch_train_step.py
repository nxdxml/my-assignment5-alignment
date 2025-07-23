import torch

from cs336_alignment.masked_normalize import masked_normalize

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    执行一次 SFT 微批训练步骤：
    - 根据 mask 计算 masked loss
    - 除以 gradient_accumulation_steps 做梯度缩放
    - 调用 loss.backward()
    - 返回 loss 和元信息（例如 token 数）
    """
    # 1 -log 作为损失(最大化log-prob所以==最小化-log)
    batch_size, _ = policy_log_probs.shape
    neg_loss_likehood = -policy_log_probs

    # 2 mask归一化求和
    masked_loss = masked_normalize(
        tensor=neg_loss_likehood,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None, # 对所有
    )
    # 3 梯度缩放，因为会累积多个step的梯度
    loss = masked_loss / gradient_accumulation_steps / batch_size
    loss.backward()

    return loss ,{
        "microbatch_loss": masked_loss.detach(),      # 不参与 autograd 的原始 loss
        "num_tokens": response_mask.sum()             # 当前 microbatch 中 response token 数
    }
