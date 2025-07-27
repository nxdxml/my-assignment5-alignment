import torch
from typing import Callable

# TODO 不懂RL GPT生成学习下路数再手搓
def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    计算 GRPO 中的组归一化奖励（advantage），支持：
    - 标准 GRPO：减去均值再除以标准差
    - Dr.GRPO 简化版：只减均值，不除以标准差

    参数：
        reward_fn: 评估函数，输入模型输出和标准答案，返回 dict，包括 reward 值
        rollout_responses: 模型生成的回答列表（按组展开）
        repeated_ground_truths: 标准答案列表，每个答案重复 group_size 次
        group_size: 每组有几个回答（通常是每个 prompt 生成多个回答）
        advantage_eps: 防止除零的平滑项
        normalize_by_std: 是否按标准差归一化（True = 标准 GRPO，False = Dr.GRPO）

    返回：
        advantages: torch.Tensor，组归一化后的奖励（advantage）
        raw_rewards: torch.Tensor，未归一化的原始 reward
        metadata: dict，日志指标（均值、最大值等）
    """
    assert len(rollout_responses) == len(repeated_ground_truths), "长度不一致"

    rollout_batch_size = len(rollout_responses)
    raw_rewards_list = []

    # Step 1: 计算每个样本的 raw reward
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, gt)
        raw_rewards_list.append(reward_info["reward"])

    # 转为 float32 tensor
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    advantages = torch.empty_like(raw_rewards)

    num_groups = rollout_batch_size // group_size

    # Step 2: 对每个 group 执行归一化
    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size

        group_rewards = raw_rewards[start:end]
        group_mean = group_rewards.mean()

        if normalize_by_std:
            # 标准 GRPO：减去均值并除以标准差（加上平滑项）
            group_std = group_rewards.std(unbiased=True)
            safe_std = torch.clamp(group_std, min=advantage_eps)  # 防止除以 0
            group_adv = (group_rewards - group_mean) / safe_std
        else:
            # Dr.GRPO：只减均值
            group_adv = group_rewards - group_mean

        advantages[start:end] = group_adv

    # Step 3: 构造日志数据
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std(unbiased=False).item(),
        "reward_min": raw_rewards.min().item(),
        "reward_max": raw_rewards.max().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std(unbiased=False).item(),
    }

    return advantages, raw_rewards, metadata


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

    # advantages = raw_rewards_or_advantages.expand_as(policy_log_probs) # b s
    advantages = raw_rewards_or_advantages.to(policy_log_probs.device).expand_as(policy_log_probs)

    loss = - advantages * policy_log_probs

    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算每个 token 的 GRPO-Clip 策略梯度损失。

    参数说明:
        advantages (torch.Tensor): 
            形状为 (batch_size, 1)，表示每条输出的 advantage（通常是 group-normalized reward）。
            所有 token 将共享该 scalar，并进行广播。

        policy_log_probs (torch.Tensor): 
            形状为 (batch_size, sequence_length)，表示当前策略（正在训练的模型）生成每个 token 的对数概率。

        old_log_probs (torch.Tensor): 
            形状为 (batch_size, sequence_length)，表示旧策略（比如 rollout 阶段）生成的 token 对数概率。

        cliprange (float): 
            裁剪范围 epsilon，例如 0.2，表示允许策略概率变化的比例上下限。
            用于防止策略过度更新，提升稳定性。

    返回值:
        Tuple，其中包含：
        - loss (torch.Tensor): 
            形状为 (batch_size, sequence_length)，每个 token 的 clip 后的策略梯度损失（已加负号，便于最小化）。
        - metadata (dict[str, torch.Tensor]): 
            记录额外调试信息，例如哪些 token 被 clip 过（is_clipped）。

    公式参考:
        L_t = - min(r_t * A, clip(r_t, 1 - ε, 1 + ε) * A)
        其中 r_t = exp(log_prob_new - log_prob_old)
    """
    # 1. 将 advantage 广播到所有 token：形状变成 (batch_size, sequence_length)
    advs = advantages.expand_as(policy_log_probs) # b s

    # 2. 计算概率比 r_t = π_new / π_old（在 log 空间做减法后取 exp）
    r_t = torch.exp(policy_log_probs - old_log_probs) # b s

    # 3. 未裁剪的目标值：r_t * A
    t0 = r_t * advs

    # 4. 裁剪后的目标值：clip(r_t, 1-ε, 1+ε) * A
    t1 = torch.clamp(r_t, 1 - cliprange, 1 + cliprange) * advs

    # 5. 取两者最小值，作为最终损失（注意加负号用于最小化）
    loss = -torch.min(t0, t1)

    # 6. 标记哪些 token 被裁剪了（用于可视化或调试）
    megadata = {}
    is_clamp = (t0 != t1).float()
    megadata["is_clamp"] = is_clamp
    # print(megadata)
    return loss, megadata


def compute_policy_gradient_loss(
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

