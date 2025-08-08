import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "openai/gpt-oss-20b"
# model_id = "openai/gpt-oss-120b"


messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

use_pipeline = False
if use_pipeline:
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
else:
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(inputs, max_new_tokens=256)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
