from vllm import LLM, SamplingParams

# prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=1.0, # 控制生成文本的“随机性”或“创造力”。越大越随机
    top_p=1.0, # 模型会从概率总和达到 top_p 的词中采样
    max_tokens=1024, # 生成最大长度
    stop='\n',
)

llm = LLM("/home/dl/projects/Qwen2-Math-1.5B")

outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

for output in outputs:
    prompt = output.prompt
    gen_text = output.outputs[0].text
    # !r调用 repr(x)，能看到特殊字符,引号
    print(f"Prompt: {prompt!r}, Generated text: {gen_text!r}")


"""
Prompt: 'Hello, my name is', Generated text: " Kellie, I'm very new to this forum so I'll do my best to make this question understandable."
Prompt: 'The president of the United States is', Generated text: ' elected by the people and the two baskets of seats are filled by an equal number of electors from the states that recognize the election.'
Prompt: 'The capital of France is', Generated text: ' Paris.'
Prompt: 'The future of AI is', Generated text: ' an exciting area of research, involving the development and application of advanced algorithms to enable machines to learn from experience, adapt to new situations, and solve complex problems. One such technique in AI is studying how neural networks evolve over time, represented as graphs.'
"""