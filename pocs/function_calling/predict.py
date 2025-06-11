import vllm
from transformers import AutoTokenizer

model_path = "/home/mircea/models/Llama-3.2-1B-Instruct"
model = vllm.LLM(model=model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [{"role": "user", "content": "What is the capital of France?"}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

response = model.generate(formatted_prompt)
print(response)
print(response[0].outputs[0].text)

