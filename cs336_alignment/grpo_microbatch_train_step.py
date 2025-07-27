import torch
from typing import Literal


from cs336_alignment.utils import masked_mean
from cs336_alignment.grpo_loss import compute_policy_gradient_loss

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,              # 当前策略输出的 log 概率 (batch_size, seq_len)
    response_mask: torch.Tensor,                 # 掩码：1 表示响应 token，0 表示 prompt 或 padding (batch_size, seq_len)
    gradient_accumulation_steps: int,            # 梯度累计步数
    # 这个变量 只能是某几个固定字符串之一，否则会报类型错误
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],

    raw_rewards: torch.Tensor | None = None,     # 用于 no_baseline 的 raw reward (batch_size, 1)
    advantages: torch.Tensor | None = None,      # 用于 reinforce_with_baseline / grpo_clip 的 advantage (batch_size, 1)
    old_log_probs: torch.Tensor | None = None,   # 旧策略的 log_probs，用于 grpo_clip (batch_size, seq_len)
    cliprange: float | None = None,              # grpo_clip 中的 ε
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行 GRPO 的一个微批次（microbatch）训练步骤：
    1. 计算策略梯度损失（支持三种模式）
    2. 用 response mask 做 masked 平均
    3. 按照梯度累计步数缩放 loss
    4. 反向传播 loss
    """

    # 1选择并计算策略梯度损失
    loss_matrix, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )  # 返回每个 token 的 loss，形状 (batch_size, seq_len)

    # 2用 response_mask 做 masked mean（只对回答 token 求平均）
    loss_scalar = masked_mean(loss_matrix, response_mask, dim=None)  # 输出 scalar

    # 3缩放 loss（用于梯度累计）
    loss_scalar = loss_scalar / gradient_accumulation_steps

    # 4反向传播
    loss_scalar.backward()

    # 5返回 loss 及附加元信息（如被裁剪比例等）
    # 创建一个 不带梯度 的张量副本（requires_grad=False）
    # 防止被再次backward
    return loss_scalar.detach(), metadata
