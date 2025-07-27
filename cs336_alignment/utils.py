import torch

import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask
    summed = torch.sum(masked_tensor, dim=dim)

    return summed / normalize_constant


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    计算 tensor 在某个维度上的“掩码平均值”，仅考虑 mask == 1 的元素。

    参数：
        tensor (torch.Tensor):
            要计算平均值的张量（可以是任意形状）

        mask (torch.Tensor):
            与 tensor 形状相同的布尔或 0/1 掩码，表示哪些元素被包含在平均值中。
            mask == 1 的位置会参与平均，mask == 0 的会被忽略。

        dim (int | None):
            要在哪个维度上计算平均值：
            - 如果指定维度（例如 dim=1），则输出形状与 `tensor.mean(dim)` 一致；
            - 如果为 None，则在所有元素上计算全局平均（只对被 mask 包含的元素）。

    返回：
        torch.Tensor:
            被 mask 限定后的平均值。其形状符合 `tensor.mean(dim)` 的语义。
    """
    # 1先将 mask 转为 float，以参与数学运算（1.0 表示有效，0.0 表示忽略）
    mask = mask.to(dtype=tensor.dtype)
    # 2-1如果不指定 dim，就在所有元素上求 masked mean
    # 2-2在指定维度上分别计算加权和与有效元素个数
    if dim is None:
        tol = (tensor * mask).sum()
        count = mask.sum()
        ret = tol / count.clamp(min=1.0)
    else:
        tol = (tensor * mask).sum(dim=dim)
        count = mask.sum(dim=dim)
        ret = tol / count.clamp(min=1.0)
        ret[count==0] = float("nan")
    return ret
