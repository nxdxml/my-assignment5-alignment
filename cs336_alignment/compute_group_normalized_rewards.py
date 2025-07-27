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
