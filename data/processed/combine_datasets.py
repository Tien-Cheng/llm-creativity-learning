from datasets import Dataset, concatenate_datasets
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # List of datasets to combine
    datasets_to_combine = [
        ("reddit_writing_prompts", "data/raw/reddit_writing_prompts/data"),
        ("fanfiction_prompts", "data/raw/fanfiction_prompts/data"),
    ]

    all_datasets = []
    total_prompts = 0

    # Load and process each dataset
    for name, path in datasets_to_combine:
        logger.info(f"Loading {name}...")
        try:
            ds = Dataset.load_from_disk(path)

            # Verify the dataset has a 'prompt' column
            if "prompt" not in ds.column_names:
                logger.error(f"Dataset {name} does not have a 'prompt' column")
                continue

            # Add source column
            ds = ds.add_column("source", [name] * len(ds))

            logger.info(f"Loaded {len(ds)} prompts from {name}")
            all_datasets.append(ds)
            total_prompts += len(ds)

        except Exception as e:
            logger.error(f"Error loading {name}: {e}")

    if not all_datasets:
        logger.error("No datasets were loaded successfully")
        return

    # Combine datasets
    logger.info("Combining datasets...")
    combined_dataset = concatenate_datasets(all_datasets)

    # Save combined dataset
    output_path = "data/processed/writing_prompts"
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Saving combined dataset with {len(combined_dataset)} prompts...")
    combined_dataset.save_to_disk(output_path)

    # Log statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total prompts: {total_prompts}")
    for name, _ in datasets_to_combine:
        count = len([1 for source in combined_dataset["source"] if source == name])
        logger.info(f"- {name}: {count} prompts ({count/total_prompts*100:.1f}%)")


if __name__ == "__main__":
    main()
