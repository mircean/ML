from typing import Dict, List

import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

# constants
hf_token = "xxx"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
max_seq_length = 2048
batch_size = 4
one_batch = True
epochs = 1


class Collator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # For models like LLaMA

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        completions = [item["completion"] for item in batch]

        # Concatenate prompt and completion
        inputs = [
            prompt + completion for prompt, completion in zip(prompts, completions)
        ]

        # Tokenize the concatenated inputs
        tokenized_inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
            add_special_tokens=True,  # Ensure special tokens are added
        )

        # Create labels by copying input_ids
        labels = tokenized_inputs["input_ids"].clone()

        # Compute the length of each prompt
        prompt_lengths = [
            len(self.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        ]

        # Mask out the labels for the prompt tokens
        for i, prompt_length in enumerate(prompt_lengths):
            labels[
                i, :prompt_length
            ] = -100  # -100 is the ignore index in PyTorch's CrossEntropyLoss

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels,
        }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_xlam_function_calling, dataset_alpaca):
        self.tokenizer = tokenizer
        self.dataset_xlam_function_calling = dataset_xlam_function_calling
        self.dataset_alpaca = dataset_alpaca

    def __len__(self):
        return len(self.dataset_xlam_function_calling) + len(self.dataset_alpaca)

    def __getitem__(self, idx):
        if idx < len(self.dataset_xlam_function_calling):
            item = self.dataset_xlam_function_calling[idx]
            tools, query, answers = item["tools"], item["query"], item["answers"]

            system_prompt = (
                f"You are a helpful assistant with access to the following tools or function calls. Your task is to produce a sequence of tools or function calls necessary to generate response to the user utterance. Use the following tools or function calls as required:\n{tools}",
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant"},
            ]

            completion = answers + "<|eot_id|>"
        else:
            item = self.dataset_alpaca[idx - len(self.dataset_xlam_function_calling)]
            # i_instruction = item["text"].index("### Instruction:")
            i_output = item["text"].index("### Response:")

            messages = [
                {"role": "user", "content": item["text"][:i_output]},
                {"role": "assistant"},
            ]

            completion = item["output"] + "<|eot_id|>"

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "completion": completion}


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, model_max_length=1e10, use_fast=True, token=hf_token
)
# TODO: set padding_side?
# tokenizer.padding_side = "left"
# TODO: add pad_token_id?
# if tokenizer.pad_token_id is None:
#     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#     tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

# Initialize the collator
collator = Collator(tokenizer=tokenizer)

# Example dataset
# raw_dataset = [
#     {"prompt": "What is the capital of France? ", "completion": "The capital of France is Paris."},
#     {"prompt": 'Who wrote "To Kill a Mockingbird"? ', "completion": 'Harper Lee wrote "To Kill a Mockingbird".'},
#     # Add more data samples as needed
# ]
dataset_xlam_function_calling = load_dataset(
    "Salesforce/xlam-function-calling-60k", split="train", token=hf_token
)
dataset_alpaca = load_dataset("tatsu-lab/alpaca", split="train")
# if one_batch:
#     raw_dataset = raw_dataset.select(range(batch_size))

dataset = Dataset(tokenizer, dataset_xlam_function_calling, dataset_alpaca)

# Create DataLoader
data_loader = DataLoader(
    dataset, batch_size=batch_size, collate_fn=collator, shuffle=True
)

# Initialize the model
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Iterate over DataLoader
model.train()
for epoch in range(epochs):
    for batch in (pbar := tqdm.tqdm(data_loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        # print(f"Epoch {epoch} Iter {i} Loss: {loss.item():.4f}")
        pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        loss = outputs.loss
        loss.backward()
        optimizer.step()
