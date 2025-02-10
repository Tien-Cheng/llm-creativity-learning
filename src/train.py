import logging
import random
from typing import List, Callable

import click
import dspy
import wandb
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

from src.config import Settings
from src.rewards.antislop import antislop_penalty_reward
from src.rewards.content import (
    coherence_reward,
    emotional_impact_reward,
    grammar_reward,
    originality_reward,
)
from src.rewards.structure import (
    soft_format_reward,
    strict_format_reward,
    xml_count_reward,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

loggers = ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM", "httpx"]

for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL + 1)


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

    return dataset.map(add_prompts, cache_file_name=settings.dataset_cache_dir)


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
        gpu_memory_utilization=settings.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        use_gradient_checkpointing="unsloth"
        if settings.gradient_checkpointing
        else None,
        target_modules=settings.peft_target_modules,
    )

    return model, tokenizer


def setup_trainer(model, tokenizer, train_dataset, reward_funcs, settings: Settings):
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
                "learning_rate": settings.learning_rate,
                "warmup_ratio": settings.warmup_ratio,
                "weight_decay": settings.weight_decay,
                "gradient_accumulation_steps": settings.gradient_accumulation_steps,
                "per_device_train_batch_size": settings.per_device_train_batch_size,
            },
        )

    training_args = GRPOConfig(
        report_to="wandb" if settings.wandb_enabled else "none",
        output_dir="outputs",
        # Learning rate and optimization
        learning_rate=settings.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=settings.weight_decay,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
        optim="adamw_8bit",
        # Memory optimizations
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        gradient_checkpointing=settings.gradient_checkpointing,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        # Precision settings
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        bf16_full_eval=is_bfloat16_supported(),
        fp16_full_eval=not is_bfloat16_supported(),
        # Generation settings
        num_generations=1,
        # use_vllm=True,
        # Training duration
        num_train_epochs=settings.num_train_epochs if settings.max_steps is None else 1,
        max_steps=settings.max_steps,
        # Evaluation settings
        evaluation_strategy="no",
        # Checkpointing
        save_strategy=settings.save_strategy,
        save_steps=settings.save_steps,
        save_total_limit=settings.save_total_limit,
        # Length constraints
        max_prompt_length=settings.max_prompt_length,
        max_completion_length=settings.max_completion_length,
    )

    # Create trainer with callbacks
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )
    # trainer.add_callback(LogCompletionsCallback(trainer=trainer, num_prompts=5))
    # trainer.add_callback(RichProgressCallback())

    return trainer


@click.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--model-name", type=str, help="Model name/path")
@click.option("--output-dir", type=str, default="outputs", help="Output directory")
# Dataset options
@click.option("--dataset-id", type=str, help="Hugging Face dataset ID")
@click.option(
    "--dataset-use-local",
    is_flag=True,
    help="Use local dataset instead of Hugging Face",
)
@click.option("--dataset-local-path", type=str, help="Path to local dataset")
@click.option(
    "--max-train-samples", type=int, help="Maximum number of training samples to use"
)
# Chapter settings
@click.option("--min-chapters", type=int, help="Minimum number of chapters")
@click.option("--max-chapters", type=int, help="Maximum number of chapters")
# Training configuration
@click.option(
    "--max-steps", type=int, help="Maximum number of training steps", default=-1
)
@click.option("--learning-rate", type=float, help="Learning rate")
@click.option("--warmup-ratio", type=float, help="Warmup ratio")
@click.option("--weight-decay", type=float, help="Weight decay")
@click.option("--batch-size", type=int, help="Per device train batch size")
@click.option(
    "--gradient-accumulation-steps", type=int, help="Gradient accumulation steps"
)
# Model settings
@click.option(
    "--gpu-memory-utilization", type=float, help="GPU memory utilization (0.0-1.0)"
)
@click.option(
    "--gradient-checkpointing", is_flag=True, help="Enable gradient checkpointing"
)
# Evaluation & Checkpointing
@click.option(
    "--evaluation-strategy",
    type=click.Choice(["no", "steps", "epoch"]),
    help="Evaluation strategy",
)
@click.option("--eval-steps", type=int, help="Number of steps between evaluations")
@click.option(
    "--save-strategy", type=click.Choice(["no", "steps", "epoch"]), help="Save strategy"
)
@click.option("--save-steps", type=int, help="Number of steps between saves")
@click.option(
    "--save-total-limit", type=int, help="Maximum number of checkpoints to keep"
)
# Weights & Biases settings
@click.option("--wandb-project", type=str, help="Weights & Biases project name")
@click.option("--wandb-entity", type=str, help="Weights & Biases entity/username")
@click.option("--wandb-run-name", type=str, help="Weights & Biases run name")
@click.option("--no-wandb", is_flag=True, help="Disable Weights & Biases logging")
def train(
    config,
    model_name,
    output_dir,
    dataset_id,
    dataset_use_local,
    dataset_local_path,
    max_train_samples,
    min_chapters,
    max_chapters,
    max_steps,
    learning_rate,
    warmup_ratio,
    weight_decay,
    batch_size,
    gradient_accumulation_steps,
    gpu_memory_utilization,
    gradient_checkpointing,
    evaluation_strategy,
    eval_steps,
    save_strategy,
    save_steps,
    save_total_limit,
    wandb_project,
    wandb_entity,
    wandb_run_name,
    no_wandb,
):
    """Train model using GRPO with creative writing reward functions."""
    # Load settings
    settings = Settings()

    # Override settings with command line arguments
    if model_name:
        settings.model_name = model_name
    if dataset_id:
        settings.dataset_id = dataset_id
    if dataset_use_local:
        settings.dataset_use_local = True
    if dataset_local_path:
        settings.dataset_local_path = dataset_local_path
    if max_train_samples:
        settings.max_train_samples = max_train_samples
    if min_chapters:
        settings.min_chapters = min_chapters
    if max_chapters:
        settings.max_chapters = max_chapters
    if max_steps:
        settings.max_steps = max_steps
    if learning_rate:
        settings.learning_rate = learning_rate
    if warmup_ratio:
        settings.warmup_ratio = warmup_ratio
    if weight_decay:
        settings.weight_decay = weight_decay
    if batch_size:
        settings.per_device_train_batch_size = batch_size
    if gradient_accumulation_steps:
        settings.gradient_accumulation_steps = gradient_accumulation_steps
    if gpu_memory_utilization:
        settings.gpu_memory_utilization = gpu_memory_utilization
    if gradient_checkpointing:
        settings.gradient_checkpointing = True
    if evaluation_strategy:
        settings.evaluation_strategy = evaluation_strategy
    if eval_steps:
        settings.eval_steps = eval_steps
    if save_strategy:
        settings.save_strategy = save_strategy
    if save_steps:
        settings.save_steps = save_steps
    if save_total_limit:
        settings.save_total_limit = save_total_limit
    if wandb_project:
        settings.wandb_project = wandb_project
    if wandb_entity:
        settings.wandb_entity = wandb_entity
    if wandb_run_name:
        settings.wandb_run_name = wandb_run_name
    if no_wandb:
        settings.wandb_enabled = False

    # Initialize DSPy LM for reward computation
    setup_dspy(settings)

    # Load dataset
    logger.info(
        f"Loading dataset from {'local path' if settings.dataset_use_local else 'Hugging Face Hub'}"
    )
    if settings.dataset_use_local:
        full_dataset = load_from_disk(settings.dataset_local_path)
    else:
        from datasets import load_dataset

        full_dataset = load_dataset(settings.dataset_id)
        if isinstance(full_dataset, dict):
            # If dataset has splits, use the 'train' split
            if "train" in full_dataset:
                full_dataset = full_dataset["train"]
            else:
                # Use the first split if 'train' not found
                full_dataset = full_dataset[list(full_dataset.keys())[0]]

    # Prepare training dataset
    train_dataset = prepare_dataset(full_dataset, settings)

    # Initialize model and trainer
    model, tokenizer = setup_model(settings)
    reward_funcs = get_reward_functions(settings)
    trainer = setup_trainer(model, tokenizer, train_dataset, reward_funcs, settings)

    try:
        # Resume from checkpoint if specified
        checkpoint_dir = settings.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint_dir)

        # Save
        if output_dir:
            model.save_pretrained_merged(output_dir, tokenizer)
    finally:
        # Close wandb run if enabled
        if settings.wandb_enabled:
            wandb.finish()


if __name__ == "__main__":
    train()
