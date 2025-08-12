import re

from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify

# from peft import LoraConfig, get_peft_model
# from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# model_id = "Qwen/Qwen2-0.5B-Instruct"
model_id = "Qwen/Qwen3-4B-Instruct-2507"


dataset_id = "AI-MO/NuminaMath-TIR"
# just 5% of the data
# train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:5%]"])
train_dataset, test_dataset = load_dataset(dataset_id, split=["train", "test"])

print(train_dataset)
print(train_dataset[0])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


train_dataset = train_dataset.map(make_conversation, load_from_cache_file=False)
test_dataset = test_dataset.map(make_conversation, load_from_cache_file=False)

train_dataset = train_dataset.remove_columns(["messages", "problem"])
print(train_dataset)


# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     # torch_dtype=torch.bfloat16,
# )

# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj"],
# )

# model = get_peft_model(model, lora_config)

# model.print_trainable_parameters()


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards


def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions, strict=False):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


# Configure training arguments using GRPOConfig
output_dir = "/data-mircea/models/" + model_id
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    # gradient_accumulation_steps=16,
    num_train_epochs=100,
    bf16=True,
    # Parameters that control de data preprocessing
    # max_completion_length=64,  # default: 256
    num_generations=4,  # default: 8
    # max_prompt_length=128,  # default: 512
    # Parameters related to reporting and saving
    report_to=["wandb"],
    logging_steps=10,
    # push_to_hub=True,
    save_strategy="steps",
    save_steps=1000,
)

# 4B accelerate
training_args.per_device_train_batch_size = 12

training_args.use_vllm = True

# trainer = GRPOTrainer(model=model, reward_funcs=[format_reward, accuracy_reward], args=training_args, train_dataset=train_dataset)
trainer = GRPOTrainer(model=model_id, reward_funcs=[format_reward, accuracy_reward], args=training_args, train_dataset=train_dataset)

trainer.train()
