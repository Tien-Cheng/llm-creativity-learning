import logging
import os
from datasets import load_from_disk

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dataset information
DATASET_PATH = "data/processed/writing_prompts"
REPO_ID = "tiencheng/writing_prompts_exp_1"


def update_metadata(dataset):
    """Update dataset metadata with proper description and information."""
    # Get dataset statistics
    total_prompts = len(dataset)
    reddit_prompts = len(
        [1 for source in dataset["source"] if source == "reddit_writing_prompts"]
    )
    fanfic_prompts = len(
        [1 for source in dataset["source"] if source == "fanfiction_prompts"]
    )

    # Create description
    description = f"""# Writing Prompts Dataset

This dataset is a collection of creative writing prompts from multiple sources, designed for training language models in creative writing tasks.

## Dataset Description

- Total prompts: {total_prompts:,}
- Reddit Writing Prompts: {reddit_prompts:,} ({reddit_prompts/total_prompts*100:.1f}%)
- Fanfiction Prompts: {fanfic_prompts:,} ({fanfic_prompts/total_prompts*100:.1f}%)

### Sources
1. Reddit Writing Prompts: Writing prompts collected from r/WritingPrompts
2. Fanfiction Prompts: Writing prompts from various fanfiction communities

### Features
- prompt: The writing prompt text
- source: The origin of the prompt ("reddit_writing_prompts" or "fanfiction_prompts")

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{REPO_ID}")

# Example: Print first prompt
print(dataset[0]['prompt'])
print(f"Source: {dataset[0]['source']}")
```
"""

    # Update dataset info
    dataset.info.description = description
    dataset.info.homepage = f"https://huggingface.co/datasets/{REPO_ID}"

    return dataset


def main():
    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not found")

    try:
        # Load dataset
        logger.info(f"Loading dataset from {DATASET_PATH}")
        dataset = load_from_disk(DATASET_PATH)

        # Update metadata
        logger.info("Updating dataset metadata")
        dataset = update_metadata(dataset)

        # Push to hub (this automatically handles authentication)
        logger.info(f"Pushing dataset to {REPO_ID}")
        dataset.push_to_hub(
            REPO_ID,
            token=token,
            private=False,
        )

        logger.info(
            f"Successfully uploaded dataset to https://huggingface.co/datasets/{REPO_ID}"
        )

    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise


if __name__ == "__main__":
    main()
