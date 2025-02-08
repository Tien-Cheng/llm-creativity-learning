import random
from typing import List, Callable
import click
from datasets import load_from_disk
import dspy
import wandb
from transformers import TrainerCallback
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from src.config import Settings
from src.rewards.structure import (
    strict_format_reward,
    soft_format_reward,
    xml_count_reward,
)
from src.rewards.antislop import antislop_penalty_reward
from src.rewards.content import (
    grammar_reward,
    emotional_impact_reward,
    originality_reward,
    coherence_reward,
)

# Patch GRPO before all functions
PatchFastRL("GRPO", FastLanguageModel)


def setup_dspy(settings: Settings):
    """Initialize DSPy language model for reward computation."""
    lm = dspy.LM(
        settings.dspy_model_name,
        api_key=settings.dspy_api_key,
        api_base=settings.dspy_api_base,
    )
    dspy.configure(lm=lm)


def get_system_prompt(num_chapters: int) -> str:
    """Generate system prompt with chapter-specific instructions."""
    chapter_instructions = []
    for i in range(num_chapters):
        chapter_instructions.append(f"""
For chapter {i+1}:
<think>
[Your thought process about chapter {i+1}]
</think>
<story chapter="{i+1}">
[Your story for chapter {i+1}]
</story>""")

    return f"""
Respond in the following format:
<chapters>{num_chapters}</chapters>

{'\n'.join(chapter_instructions)}
"""


def prepare_dataset(dataset, settings: Settings):
    """Prepare dataset for GRPO training with random chapter counts and length validation."""
    # Limit dataset size if specified
    if settings.max_train_samples:
        dataset = dataset.select(range(settings.max_train_samples))

    def add_prompts(example):
        # Limit number of chapters based on prompt length
        prompt_length = len(example["prompt"].split())
        max_possible_chapters = min(
            settings.max_chapters,
            max(
                1, (settings.max_completion_length - prompt_length) // 500
            ),  # Rough estimate of tokens per chapter
        )
        num_chapters = random.randint(settings.min_chapters, max_possible_chapters)

        system_prompt = get_system_prompt(num_chapters)
        user_prompt = f'<writing_prompt>{example["prompt"]}</writing_prompt>'

        # Validate total prompt length
        total_prompt = system_prompt + user_prompt
        if len(total_prompt.split()) > settings.max_prompt_length:
            # Truncate user prompt if needed while preserving XML structure
            max_prompt_words = settings.max_prompt_length - len(system_prompt.split())
            truncated_prompt = " ".join(example["prompt"].split()[:max_prompt_words])
            user_prompt = f"<writing_prompt>{truncated_prompt}</writing_prompt>"

        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        }

    return dataset.map(add_prompts)


REWARD_FUNCTIONS = {
    "structure": [strict_format_reward, soft_format_reward, xml_count_reward],
    "antislop": [antislop_penalty_reward],
    "grammar": [grammar_reward],
    "emotion": [emotional_impact_reward],
    "originality": [originality_reward],
    "coherence": [coherence_reward],
}


def get_reward_functions(settings: Settings) -> List[Callable]:
    """Get list of reward functions based on settings."""
    reward_funcs = []
    for func_name in settings.reward_functions:
        if func_name in REWARD_FUNCTIONS:
            # Apply weight to each reward function
            funcs = REWARD_FUNCTIONS[func_name]
            weight = settings.reward_weights.get(func_name, 1.0)
            for f in funcs:
                # Create a closure to capture the current f and weight
                def weighted_reward(func=f, w=weight):
                    def wrapper(*args, **kwargs):
                        result = func(*args, **kwargs)
                        # Handle both single values and lists
                        if isinstance(result, list):
                            return [w * x for x in result]
                        return w * result

                    return wrapper

                reward_funcs.append(weighted_reward())
    return reward_funcs


def setup_model(settings: Settings):
    """Initialize model and tokenizer with settings."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings.model_name,
        load_in_4bit=True,
        max_seq_length=settings.max_seq_length,
        # fast_inference=True,
        gpu_memory_utilization=0.5,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        use_gradient_checkpointing="unsloth",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
    )

    return model, tokenizer


class WandbEvalCallback(TrainerCallback):
    """Custom callback for logging evaluation samples to wandb."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Log evaluation samples to wandb at specified frequency."""
        self.step += 1
        if (
            self.settings.wandb_enabled
            and self.settings.wandb_log_eval_samples
            and self.step % self.settings.wandb_eval_samples_freq == 0
        ):
            # Get evaluation samples from trainer state
            if hasattr(state, "eval_samples") and state.eval_samples:
                for i, sample in enumerate(state.eval_samples):
                    wandb.log(
                        {
                            f"eval_sample_{i}/prompt": sample["prompt"],
                            f"eval_sample_{i}/completion": sample["completion"],
                            f"eval_sample_{i}/rewards": sample["rewards"],
                            "step": self.step,
                        }
                    )


def setup_trainer(
    model, tokenizer, train_dataset, eval_dataset, reward_funcs, settings: Settings
):
    """Set up GRPO trainer with settings and memory optimizations."""
    # Initialize wandb if enabled
    if settings.wandb_enabled:
        wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            name=settings.wandb_run_name,
            config={
                "model_name": settings.model_name,
                "max_seq_length": settings.max_seq_length,
                "max_prompt_length": settings.max_prompt_length,
                "max_completion_length": settings.max_completion_length,
                "num_train_epochs": settings.num_train_epochs,
                "max_steps": settings.max_steps,
                "max_train_samples": settings.max_train_samples,
                "min_chapters": settings.min_chapters,
                "max_chapters": settings.max_chapters,
                "reward_functions": settings.reward_functions,
                "reward_weights": settings.reward_weights,
            },
        )
    training_args = GRPOConfig(
        report_to="wandb" if settings.wandb_enabled else "none",
        output_dir="outputs",
        # Learning rate and optimization
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
        optim="adamw_8bit",
        # Memory optimizations
        gradient_accumulation_steps=1,  # Accumulate gradients to simulate larger batch size
        gradient_checkpointing=True,  # Trade computation for memory
        per_device_train_batch_size=1,
        # use_vllm=True,
        # Precision settings
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        bf16_full_eval=is_bfloat16_supported(),
        fp16_full_eval=not is_bfloat16_supported(),
        # Generation settings
        num_generations=1,
        # Training duration
        num_train_epochs=settings.num_train_epochs if settings.max_steps is None else 1,
        max_steps=settings.max_steps,
        # Length constraints
        max_prompt_length=settings.max_prompt_length,
        max_completion_length=settings.max_completion_length,
    )

    # Create trainer with wandb callback
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbEvalCallback(settings)] if settings.wandb_enabled else None,
    )

    return trainer


@click.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--model-name", type=str, help="Model name/path")
@click.option("--output-dir", type=str, default="outputs", help="Output directory")
@click.option("--min-chapters", type=int, help="Minimum number of chapters")
@click.option("--max-chapters", type=int, help="Maximum number of chapters")
@click.option("--max-steps", type=int, help="Maximum number of training steps")
@click.option(
    "--max-train-samples", type=int, help="Maximum number of training samples to use"
)
def train(
    config,
    model_name,
    output_dir,
    min_chapters,
    max_chapters,
    max_steps,
    max_train_samples,
):
    """Train model using GRPO with creative writing reward functions."""
    # Load settings
    settings = Settings()
    if model_name:
        settings.model_name = model_name
    if min_chapters:
        settings.min_chapters = min_chapters
    if max_chapters:
        settings.max_chapters = max_chapters
    if max_steps:
        settings.max_steps = max_steps
    if max_train_samples:
        settings.max_train_samples = max_train_samples

    # Initialize DSPy LM for reward computation
    setup_dspy(settings)

    # Load dataset and split into train/eval
    full_dataset = load_from_disk("data/processed/writing_prompts")
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = prepare_dataset(split_dataset["train"], settings)
    eval_dataset = prepare_dataset(split_dataset["test"], settings)

    # Initialize model and trainer
    model, tokenizer = setup_model(settings)
    reward_funcs = get_reward_functions(settings)
    trainer = setup_trainer(
        model, tokenizer, train_dataset, eval_dataset, reward_funcs, settings
    )

    try:
        # Train
        trainer.train()

        # Save
        if output_dir:
            model.save_pretrained_merged(output_dir, tokenizer)
    finally:
        # Close wandb run if enabled
        if settings.wandb_enabled:
            wandb.finish()


if __name__ == "__main__":
    train()
