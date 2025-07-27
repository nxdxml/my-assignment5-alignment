import torch


import torch
from typing import Literal

# 请确保你已正确实现并导入以下两个函数：
# - compute_naive_policy_gradient_loss
# - compute_grpo_clip_loss
# 或者修改 import 路径以适配你的项目结构

from your_module import compute_naive_policy_gradient_loss, compute_grpo_clip_loss

def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    根据传入的 loss_type（策略梯度损失类型），调用对应的策略梯度损失计算函数。

    参数：
        policy_log_probs (torch.Tensor):
            当前策略（正在训练的模型）输出的每个 token 的 log 概率。
            形状为 (batch_size, sequence_length)

        loss_type (str):
            策略梯度的类型，可选：
            - "no_baseline": 使用原始 reward（未归一化）作为 advantage
            - "reinforce_with_baseline": 使用归一化后的 advantage（如 group-normalized reward）
            - "grpo_clip": 使用 GRPO-Clip 策略（包含 ratio 裁剪）

        raw_rewards (torch.Tensor):
            原始奖励，形状为 (batch_size, 1)，仅在 "no_baseline" 模式下使用。

        advantages (torch.Tensor):
            已计算好的 advantage，形状为 (batch_size, 1)，用于 "reinforce_with_baseline" 和 "grpo_clip"

        old_log_probs (torch.Tensor):
            rollout 阶段旧策略生成的 log 概率，形状为 (batch_size, sequence_length)，仅用于 "grpo_clip"

        cliprange (float):
            策略裁剪参数 epsilon（如 0.2），控制更新幅度，仅用于 "grpo_clip"

    返回：
        Tuple，包括：
        - loss (torch.Tensor): 每个 token 的最终损失，形状为 (batch_size, sequence_length)
        - metadata (dict): 包含额外统计信息（如是否发生裁剪）或空字典
    """

    if loss_type == "no_baseline":
        # 使用原始奖励作为 advantage，直接计算 naive 策略梯度损失
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        )
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        # 使用归一化奖励作为 advantage（如 group-normalized reward）
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        )
        return loss, {}

    elif loss_type == "grpo_clip":
        # 使用 GRPO-Clip 损失函数，需要额外提供旧策略 log 概率 和 cliprange
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
        return loss, metadata

    else:
        raise ValueError(f"不支持的 loss_type 类型: {loss_type}")

