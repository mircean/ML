from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")
# dataset[0]

model = "Qwen/Qwen2-0.5B-Instruct"
# model = "Qwen/Qwen3-4B-Instruct-2507"
# model = "Qwen/Qwen3-30B-A3B-Instruct-2507"


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


output_dir = "/data-mircea/models/" + model
training_args = GRPOConfig(output_dir=output_dir, logging_steps=10)
# 0.5B 1 GPU
# 4b accelerate
training_args.num_generations = 2
training_args.per_device_train_batch_size = 4
# 30B accelerate - CUDA OOM
# training_args.num_generations = 2
# training_args.per_device_train_batch_size = 1

# training_args.gradient_accumulation_steps = 2
# training_args.max_prompt_length = 512
# training_args.max_completion_length = 256
training_args.bf16 = True

training_args.use_vllm = True

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
