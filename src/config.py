from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    # DSPy LM settings (for reward computation)
    dspy_model_name: str = (
        "openai/gpt-4-0125-preview"  # Full model name with provider prefix
    )
    dspy_api_key: str  # API key for the provider
    dspy_api_base: str | None = None  # Provider's API base URL

    # Training model settings
    model_name: str = "unsloth/Qwen2.5-7B-Instruct"
    max_seq_length: int = 4096  # Reduced to prevent memory issues
    max_prompt_length: int = 512  # Keep original prompt length
    max_completion_length: int = 2048  # Reduced for better stability

    # Training settings
    num_train_epochs: int = 1
    max_steps: int | None = None  # Override num_train_epochs if set
    max_train_samples: int | None = None  # Limit dataset size for testing

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


settings = Settings()
