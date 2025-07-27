import torch


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个 token 的策略梯度损失，其中 raw_rewards_or_advantages 是原始奖励或者已经归一化的优势值。
    L_t = - A * log p_theta(o_t)
    参数:
        raw_rewards_or_advantages: torch.Tensor, 形状 (batch_size, 1)，表示每个 rollout 的标量奖励或优势。
        policy_log_probs: torch.Tensor, 形状 (batch_size, sequence_length)，表示每个 token 的对数概率。
        
    返回:
        torch.Tensor, 形状 (batch_size, sequence_length)，每个 token 的策略梯度损失
        （最终会在训练循环中对 batch 和序列维度进行聚合）。
    """

    advantages = raw_rewards_or_advantages.expand_as(policy_log_probs) # b s
    loss = - advantages * policy_log_probs

    return loss

