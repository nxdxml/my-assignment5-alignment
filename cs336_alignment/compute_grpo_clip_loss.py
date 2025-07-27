
import torch


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


