from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "SAMPLE_KEY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def call_ai(prompt, model="/mnt/Dragon/Vbox_SF/HuggingFaceLocal//gemma-2b-it/"):
    messages = [{"role": "user", "content": prompt}]
    temperature = 0.6 # this is the degree of randomness of the model's output
    max_tokens = 1024

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        'body': response.choices[0].message.content,
        'context': {
            'Model': response.model,
            'temperature': temperature,
            'ChatGPT token usage': response.usage,
        }
    }

#load prompt from text file
with open('/home/debbie/Dev/llm_localbench_data/PT-SL02_v022024.txt', 'r') as f:
    prompt = f.read()
    f.close()

response = call_ai(prompt)
print('Response:\n%s ' %response['body'])
print('\nContext:\n%s ' %response['context'])


#### Server invokes:

#Mistral:
# python -m vllm.entrypoints.openai.api_server --model $HF_LOCAL_MODEL_PATH/hf/AWQ/Mistral-7B-Instruct-v0.2-AWQ/ --max-model-len 8192 --enforce-eager

#Gemma:
# python -m vllm.entrypoints.openai.api_server --model $HF_LOCAL_MODEL_PATH/hf/gemma-2b-it/

#Llama 2 7B:
# python -m vllm.entrypoints.openai.api_server --model $HF_LOCAL_MODEL_PATH/hf/AWQ/llama-2-7b-chat-awq/

#Llama 2 13B:
# python -m vllm.entrypoints.openai.api_server --model $HF_LOCAL_MODEL_PATH/hf/AWQ/llama-2-13b-chat-awq/ --enforce-eager