from datasets import load_dataset, Dataset
import re


def clean_prompt(text):
    # Remove specific tags we want to filter out
    if any(
        tag in text for tag in ["[MP]", "[PM]", "[META]", "[OT]", "[POETRY]", "[SF]"]
    ):
        return None

    # Strip any remaining tags (e.g., [WP], [EU], etc.)
    cleaned = re.sub(r"\[(.*?)\]", "", text)
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else None


def main():
    ds = load_dataset("nothingiisreal/Reddit-Dirty-And-WritingPrompts")

    # Select only the 'prompt' feature and convert to list
    prompts = ds["train"]["prompt"]

    # Clean prompts and filter out None values
    cleaned_prompts = []
    seen = set()  # For tracking unique prompts

    for prompt in prompts:
        cleaned = clean_prompt(prompt)
        if cleaned and cleaned not in seen:
            cleaned_prompts.append(cleaned)
            seen.add(cleaned)

    # Create new dataset with unique, cleaned prompts
    new_ds = Dataset.from_dict({"prompt": cleaned_prompts})

    # Save the dataset
    new_ds.save_to_disk("data/raw/reddit_writing_prompts/data")


if __name__ == "__main__":
    main()
