from vllm import LLM, SamplingParams

prompt = "Who were the first five presidents of the USA?"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

llm = LLM(model="/home/debbie/Dev/vllm_test/LLMs/Mistral-7B-Instruct-v0.2-AWQ", quantization="awq", dtype="auto",enforce_eager=True)
outputs = llm.generate(prompt, sampling_params)
print("-----------------------------------------------")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
print("-----------------------------------------------")
