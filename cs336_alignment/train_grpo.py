import os
import json
import re
import torch
import mlflow
from typing import Literal
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_compute_group_normalized_rewards,
    run_grpo_microbatch_train_step,
)

# ------------------ 数据处理 ------------------

def load_gsm8k(path: str, limit: int = None) -> tuple[list[str], list[str]]:
    questions, answers = [], []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            data = json.loads(line)
            questions.append(data["question"])
            answers.append(data["answer"])
    return questions, answers

def format_gsm8k_prompt(question: str) -> str:
    return f"""You are a helpful and honest assistant. Solve the following problem step by step. Then give the final answer in the format:

Final Answer: [your numeric answer]

Question: {question}

Answer:"""

def parse_gsm8k_response(output: str) -> str | None:
    match = re.search(r"Final Answer:\s*(\-?\d+\.?\d*)", output)
    if match:
        return match.group(1)
    nums = re.findall(r'-?\d+\.?\d*', output)
    return nums[-1] if nums else None

# ------------------ 主训练逻辑 ------------------

def grpo_train_loop_main():
    # ========= 模型 & 路径 =========
    model_path = "/home/dl/projects/Qwen2-Math-1.5B"
    save_root = "/home/dl/projects/my-assignment5-alignment/cs336_alignment/checkpoint/1.5B-20250728-grpo"
    os.makedirs(save_root, exist_ok=True)

    # ========= 超参数 =========
    rollout_batch_size = 8
    group_size = 2
    gradient_accumulation_steps = 2
    learning_rate = 1e-5
    cliprange = 0.2
    advantage_eps = 1e-6
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    use_std_normalization = True
    device = "cuda:0"
    num_steps = 100
    save_every = 10

    # ========= 初始化模型 =========
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    policy.train()
    policy.to(device)
    optimizer = AdamW(policy.parameters(), lr=learning_rate)

    # ========= 加载数据 =========
    data_path = "/home/dl/projects/my-assignment5-alignment/data/gsm8k/test.jsonl"
    all_questions, all_answers = load_gsm8k(data_path)

    def gsm8k_reward_fn(pred: str, target: str) -> dict[str, float]:
        pred_ans = parse_gsm8k_response(pred)
        target_ans = parse_gsm8k_response(target)
        format_reward = 1.0 if pred_ans is not None else 0.0
        answer_reward = 1.0 if pred_ans == target_ans else 0.0
        return {
            "reward": format_reward * answer_reward,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        }

    # ========= 启动 MLflow =========
    mlflow.set_experiment("grpo-qwen-train")
    with mlflow.start_run(run_name="qwen1.5b-gsm8k-run"):
        mlflow.log_params({
            "learning_rate": learning_rate,
            "loss_type": loss_type,
            "cliprange": cliprange,
            "group_size": group_size,
            "grad_accum_steps": gradient_accumulation_steps,
            "batch_size": rollout_batch_size,
            "num_steps": num_steps,
        })

        for step in range(1, num_steps + 1):
            # ----- 采样一个 batch -----
            idx = torch.randperm(len(all_questions))[:rollout_batch_size]
            questions = [all_questions[i] for i in idx]
            answers = [all_answers[i] for i in idx]
            prompts = [format_gsm8k_prompt(q) for q in questions]

            # ----- 模型生成 -----
            tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.no_grad():
                generated_ids = policy.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id
                )
            generated_texts = tokenizer.batch_decode(
                generated_ids[:, tokenized["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # ----- log prob 计算 -----
            batch = run_tokenize_prompt_and_output(prompts, generated_texts, tokenizer)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            log_probs_batch = run_get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            policy_log_probs = log_probs_batch["log_probs"]
            old_log_probs = policy_log_probs.clone()

            # ----- 奖励函数 -----
            normed_rewards, raw_rewards, reward_meta = run_compute_group_normalized_rewards(
                reward_fn=gsm8k_reward_fn,
                rollout_responses=generated_texts,
                repeated_ground_truths=answers,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=use_std_normalization
            )
            advantages = normed_rewards.unsqueeze(1).to(device)
            raw_rewards = raw_rewards.unsqueeze(1).to(device)

            # ----- 训练一步 -----
            loss, metadata = run_grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                raw_rewards=raw_rewards if loss_type == "no_baseline" else None,
                advantages=advantages if loss_type != "no_baseline" else None,
                old_log_probs=old_log_probs if loss_type == "grpo_clip" else None,
                cliprange=cliprange if loss_type == "grpo_clip" else None
            )

            clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # ----- 日志 -----
            reward_mean = raw_rewards.mean().item()
            mlflow.log_metric("loss", loss.item(), step=step)
            mlflow.log_metric("reward_mean", reward_mean, step=step)
            mlflow.log_metric("reward_max", raw_rewards.max().item(), step=step)
            print(f"[Step {step}] Loss: {loss.item():.4f} | Reward mean: {reward_mean:.2f}")

            # ----- 保存模型 -----
        save_path = save_root
        policy.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        mlflow.log_artifacts(save_path)
        print(f"模型已保存到: {save_path}")

# ------------------ 入口 ------------------

if __name__ == "__main__":
    grpo_train_loop_main()
