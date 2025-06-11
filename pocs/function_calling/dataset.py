import numpy as np
import torch
import torchtune.data
from datasets import load_dataset

# constants
hf_token = "xxx"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.dataset_xlam_function_calling = load_dataset(
            "Salesforce/xlam-function-calling-60k", split="train", token=hf_token
        )
        self.dataset_alpaca = load_dataset("tatsu-lab/alpaca", split="train")

    def __len__(self):
        return len(self.dataset_xlam_function_calling) + len(self.dataset_alpaca)

    def __getitem__(self, idx):
        if idx < len(self.dataset_xlam_function_calling):
            item = self.dataset_xlam_function_calling[idx]
            tools, query, answers = item["tools"], item["query"], item["answers"]

            system_prompt = f"You are a helpful assistant with access to the following tools or function calls. Your task is to produce a sequence of tools or function calls necessary to generate response to the user utterance. Use the following tools or function calls as required:\n{tools}"
            messages = []
            messages.append(
                torchtune.data.Message(
                    role="system", content=system_prompt, masked=True
                )
            )
            messages.append(
                torchtune.data.Message(role="user", content=query, masked=True)
            )
            messages.append(
                torchtune.data.Message(role="assistant", content=answers, masked=False)
            )
        else:
            item = self.dataset_alpaca[idx - len(self.dataset_xlam_function_calling)]
            i_output = item["text"].index("### Response:")
            messages = []
            messages.append(
                torchtune.data.Message(
                    role="user", content=item["text"][:i_output], masked=True
                )
            )
            messages.append(
                torchtune.data.Message(
                    role="assistant", content=item["output"], masked=False
                )
            )

        sample = self.tokenizer({"messages": messages})

        tokens, mask = sample["tokens"], sample["mask"]
        labels = list(
            np.where(mask, torchtune.data.CROSS_ENTROPY_IGNORE_IDX, tokens)
        )  # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        assert len(tokens) == len(labels)

        return {"tokens": tokens, "labels": labels}
