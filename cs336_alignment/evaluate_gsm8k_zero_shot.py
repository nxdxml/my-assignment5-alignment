
import json
from typing import Callable, List
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.gsm8k import gsm8k_reward_fn
def load_math_validation(path: str) -> tuple[list[str], list[str]]:
    """
    从指定路径加载 MATH 验证集数据。

    参数:
        path (str): JSONL 文件路径

    返回:
        questions (list[str]): 原始的样本列表（每个元素是 JSON 对象）
        answers (list[str]): 每个样本对应的 ground truth 答案
    """
    # 1. 打开文件
    # 2. 遍历每一行，用 json.loads() 加载为字典
    # 3. 将整个样本加入 samples 列表
    # 4. 提取其中的 "answer" 字段加入 ground_truths 列表
    questions = []
    answers = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["question"])
            answers.append(data["answer"])

    return questions, answers

def evaluate_vllm(
    vllm_model: LLM,
    # reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    answers : List[str],
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    # 1调用模型生成
    generations = vllm_model.generate(prompts=prompts,sampling_params=eval_sampling_params)
    # generations 是一个列表，每个元素通常长这样：
    # {
    #     "prompt": "...",
    #     "outputs": [ { "text": "Assistant: <think> ... </think> <answer>42</answer>" } ]
    # }

    results = []
    # 2用reward_fn进行评估
    for i in range(len(generations)):
        output = generations[i]
        prompt = output.prompt
        gen_text = output.outputs[0].text

        # reward = reward_fn(prompt, answers[i])

        gt_answer = parse_gsm8k_response(answers[i])
        predicted_answer = parse_gsm8k_response(gen_text)
        # 格式是否符合：只要成功提取数字就算格式正确
        format_reward = 1.0 if predicted_answer is not None else 0.0
        # 正确性判断
        answer_reward = 1.0 if predicted_answer == gt_answer else 0.0
        reward = {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "reward": format_reward * answer_reward
        }


        results.append({
            "prompt":prompt,
            "gen":gen_text,
            "gt_raw":answers[i],
            "gt_parse":gt_answer,
            "gen_parse":predicted_answer,
            "reward":reward,
        })


    # 3统计三种 reward 情况
    correct_both = 0
    format_only = 0
    all_wrong = 0

    for r in results:
        f = r["reward"]["format_reward"]
        a = r["reward"]["answer_reward"]
        if f == 1 and a == 1:
            correct_both += 1
        elif f == 1 and a == 0:
            format_only += 1
        else:
            all_wrong += 1

    # 打印统计信息
    print("Format + Answer Correct:", correct_both)
    print("Format Correct Only:", format_only)
    print("Both Wrong:", all_wrong)

    # 4保存
    with open("/home/dl/projects/my-assignment5-alignment/cs336_alignment/checkpoint/gsm8k_eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

def parse_gsm8k_response(model_output: str) -> str | None:
    import re

    # 优先找 Final Answer 格式
    match = re.search(r"Final Answer:\s*(\-?\d+\.?\d*)", model_output)
    if match:
        return match.group(1)

    # fallback：找最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', model_output)
    return numbers[-1] if numbers else None

def main():
    # 1从 /data/a5-alignment/MATH/validation.jsonl 加载 MATH 验证集
    # 每一行是一个 JSON 对象，格式类似 {'question': ..., 'answer': ...}
    path = "/home/dl/projects/my-assignment5-alignment/data/gsm8k/test.jsonl"
    questions, answers = load_math_validation(path=path)
    # 正确性验证
    # print(f"问题{questions[:3]}->{len(questions)}  答案{answers[:3]}->{len(answers)}")

    # 2编写一个函数，将每道题包装成一个适合 LLM 的 prompt
    # 你需要找到或导入 cs336_alignment.drgrpo_prompts.r1_zero_prompt(template)
    # 返回字符串 prompt 列表，对应每个问题
    # with open("/home/dl/projects/my-assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt", encoding='utf-8') as f:
    #     r1_zero_prompt_format = f.read()
    # with open("/home/dl/projects/my-assignment5-alignment/cs336_alignment/prompts/gsm8k.prompt", encoding='utf-8') as f:
    #     gsm8k_zero_prompt_format = f.read()
    
    # print(r1_zero_prompt_format)
    # prompts = [gsm8k_zero_prompt_format.format(question=q) for q in questions]

    
    prompts = []
    for question in questions:
        prompt = f"""You are a helpful and honest assistant. Solve the following problem step by step. Then give the final answer in the format:

Final Answer: [your numeric answer]

Question: {question}

Answer:"""
        prompts.append(prompt)




    # print(prompts[:1])


    # 3编写函数 evaluate_vllm（如题目所给），其中：
    # - vllm_model 是已经加载好的模型
    # - prompts 是上一步生成的 prompt 列表
    # - eval_sampling_params 指定采样参数（如 temperature=0.7 等）
    # - reward_fn 是评估函数
    # 使用 vllm_model.generate(prompts, sampling_params) 获取模型输出   
    # llm = LLM("/home/dl/projects/Qwen2-Math-1.5B")
    # 微调后
    # llm = LLM("/home/dl/projects/my-assignment5-alignment/cs336_alignment/checkpoint/")
    # llm = LLM("/home/dl/projects/my-assignment5-alignment/cs336_alignment/checkpoint/1.5B-20250725")
    llm = LLM("/home/dl/projects/my-assignment5-alignment/cs336_alignment/checkpoint/1.5B-20250728-grpo")
    sampling_params = SamplingParams(
        temperature=1.0, # 控制生成文本的“随机性”或“创造力”。越大越随机
        top_p=1.0, # 模型会从概率总和达到 top_p 的词中采样
        max_tokens=1024, # 生成最大长度
        stop='\n',
    )
    evaluate_vllm(vllm_model=llm, 
        # reward_fn=gsm8k_reward_fn,
        eval_sampling_params=sampling_params,
        prompts=prompts,
        answers=answers,
    )



if __name__ == "__main__":
    main()