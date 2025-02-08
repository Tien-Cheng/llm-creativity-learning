from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    # DSPy LM settings (for reward computation)
    dspy_model_name: str = "openai/gpt-4o"  # Full model name with provider prefix
    dspy_api_key: str  # API key for the provider
    dspy_api_base: str | None = None  # Provider's API base URL

    # Training model settings
    model_name: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    max_seq_length: int = 4096  # Reduced to prevent memory issues
    max_prompt_length: int = 512  # Keep original prompt length
    max_completion_length: int = 2048  # Reduced for better stability

    # Training settings
    num_train_epochs: int = 1
    max_steps: int | None = None  # Override num_train_epochs if set
    max_train_samples: int | None = None  # Limit dataset size for testing
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 1

    # Model settings
    gpu_memory_utilization: float = 0.5
    peft_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    gradient_checkpointing: bool = True

    # Dataset settings
    dataset_id: str = "tiencheng/writing_prompts_exp_1"  # Hugging Face dataset ID
    dataset_use_local: bool = False  # If True, load from local path instead of Hub
    dataset_local_path: str = "data/processed/writing_prompts"  # Local dataset path if dataset_use_local is True
    validation_split: float = 0.1
    dataset_cache_dir: str | None = None

    # Evaluation settings
    evaluation_strategy: str = "steps"  # "no", "steps", or "epoch"
    eval_steps: int = 100  # Evaluate every N steps if evaluation_strategy="steps"
    save_strategy: str = "steps"  # "no", "steps", or "epoch"
    save_steps: int = 500  # Save checkpoint every N steps if save_strategy="steps"
    save_total_limit: int | None = 3  # Maximum number of checkpoints to keep

    # Chapter settings
    min_chapters: int = 1
    max_chapters: int = 5

    # Reward settings
    reward_functions: list[str] = [
        "structure",
        "antislop",
        "grammar",
        "emotion",
        "originality",
        "coherence",
    ]
    reward_weights: Dict[str, float] = {
        "structure": 1.0,
        "antislop": 0.5,
        "grammar": 1.0,
        "emotion": 1.0,
        "originality": 1.0,
        "coherence": 1.0,
    }

    # Weights & Biases settings
    wandb_enabled: bool = True
    wandb_project: str = "llm-creativity-learning"
    wandb_entity: str | None = None  # Your wandb username/organization
    wandb_run_name: str | None = None
    wandb_log_eval_samples: bool = True
    wandb_eval_samples_freq: int = 100  # Log eval samples every N steps
    wandb_watch: str = "gradients"  # "gradients", "parameters", "all", or "false"
    wandb_log_memory: bool = True

    # Resume training settings
    resume_from_checkpoint: str | None = None  # Path to checkpoint directory


settings = Settings()
