from transformers import PreTrainedTokenizerBase
from torch import Tensor
import torch
import torch.nn.functional as F
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch = len(prompt_strs)
    max_prompt_and_output_lens = 0
    p_tokens_list = []
    o_tokens_list = []
    for p_str, o_str in zip(prompt_strs, output_strs):
        p_tokens = tokenizer.encode(p_str)
        o_tokens = tokenizer.encode(o_str)
        prompt_lens = len(p_tokens)
        output_lens = len(o_tokens)
        prompt_and_output_lens = prompt_lens + output_lens
        max_prompt_and_output_lens = max(max_prompt_and_output_lens, prompt_and_output_lens)
        p_tokens_list.append(p_tokens)
        o_tokens_list.append(o_tokens)

    inputs_id = torch.full((batch, max_prompt_and_output_lens - 1), tokenizer.pad_token_id, dtype=torch.int64)
    labels = torch.full((batch, max_prompt_and_output_lens - 1), tokenizer.pad_token_id, dtype=torch.int64)
    response_mask = torch.full((batch, max_prompt_and_output_lens - 1), False, dtype=torch.bool)

    for i, (p_tokens, o_tokens) in enumerate(zip(p_tokens_list, o_tokens_list)):
        concat_tensor = torch.full((max_prompt_and_output_lens,), tokenizer.pad_token_id, dtype=torch.int64)
        p_o_tokens = p_tokens + o_tokens
        t = len(p_o_tokens)
        concat_tensor[:t] = torch.tensor(p_o_tokens)
        inputs_id[i] = concat_tensor[:-1]
        labels[i] = concat_tensor[1:]
        # p p p o o 3 2
        #   0 0 1 1
        response_mask[i][len(p_tokens) - 1 : len(p_tokens) - 1 + len(o_tokens)] = True
    return {
        "input_ids":inputs_id,
        "labels":labels,
        "response_mask":response_mask,
    }
