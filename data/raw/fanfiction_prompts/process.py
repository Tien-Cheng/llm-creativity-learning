#!/usr/bin/env python3
import os
from datasets import load_dataset, Dataset
from litellm import completion
import asyncio
import json
from datetime import datetime
import logging
from typing import Dict, List, Any
import time
import random
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/raw/fanfiction_prompts/processing.log"),
    ],
)
logger = logging.getLogger(__name__)


def quality_filter(summary: str, min_length: int = 20, max_length: int = 500) -> bool:
    """
    Filter summaries based on quality criteria.

    Args:
        summary: The fanfiction summary to check
        min_length: Minimum character length
        max_length: Maximum character length

    Returns:
        bool: True if the summary passes quality checks
    """
    if not min_length <= len(summary) <= max_length:
        return False

    if not summary[0].isupper() or not summary.rstrip("...").endswith((".", "!", "?")):
        return False

    low_quality = ["wip", "tbd", "to be continued", "summary inside", "please read"]
    if any(indicator in summary.lower() for indicator in low_quality):
        return False

    words = summary.split()
    if len(words) < 5 or len(set(words)) < 4:
        return False

    return True


async def generate_prompts(summary: str, category: str) -> str:
    """
    Generate a writing prompt from a fanfiction summary.
    Randomly chooses between generic and fandom-specific style.

    Args:
        summary: The original fanfiction summary
        category: The fandom categories

    Returns:
        str: The generated writing prompt
    """
    # Randomly choose between generic and fandom-specific prompt
    is_generic = random.choice([True, False])

    if is_generic:
        messages = [
            {
                "role": "system",
                "content": """You are an expert at converting fanfiction summaries into creative writing prompts.
                Create a concise, engaging writing prompt (1-2 sentences) that:
                1. Captures the core narrative idea
                2. Removes any fandom-specific references
                3. Uses clear, compelling language
                4. Follows r/WritingPrompts style (short, hooks reader immediately)
                The prompt should work in any universe and not mention it's based on fanfiction.""",
            },
            {
                "role": "user",
                "content": f"Category: {category}\nSummary: {summary}\n\nConvert this into a short, compelling writing prompt that could work in any universe.",
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": """You are an expert at converting fanfiction summaries into creative writing prompts.
                Create a concise, engaging writing prompt (1-2 sentences) that:
                1. Captures the core narrative idea
                2. Maintains key fandom elements and crossovers
                3. Uses clear, compelling language
                4. Follows r/WritingPrompts style (short, hooks reader immediately)
                The prompt should excite fans of these universes.""",
            },
            {
                "role": "user",
                "content": f"Category: {category}\nSummary: {summary}\n\nConvert this into a short, compelling fanfiction writing prompt.",
            },
        ]

    try:
        response = completion(
            model="openrouter/amazon/nova-micro-v1",
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        return None


async def process_batch(items: List[Dict], stats: Dict[str, Any]) -> List[str]:
    """
    Process a batch of fanfiction summaries.

    Args:
        items: List of fanfiction entries
        stats: Dictionary to track statistics

    Returns:
        List of generated prompts
    """
    prompts = []
    for item in items:
        try:
            prompt = await generate_prompts(item["summary"], item["category"])
            if prompt:
                prompts.append(prompt)
                stats["successful"] += 1
                stats["cost"] += 0.0000219  # One prompt per summary
        except Exception as e:
            logger.error(f"Error processing {item['summary']}: {e}")
            stats["failed"] += 1
    return prompts


async def main():
    # Verify environment variables
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    # Load dataset
    logger.info("Loading dataset...")
    ds = load_dataset("mrzjy/fanfiction_meta")

    # Analyze dataset
    logger.info(f"Dataset size: {len(ds['train'])} entries")

    # Get category distribution
    categories = Counter(ds["train"]["category"])
    logger.info("\nTop 10 categories:")
    for cat, count in categories.most_common(10):
        logger.info(f"- {cat}: {count} entries")

    # Convert to list for random sampling
    all_items = [
        {
            "summary": item["summary"],
            "category": item["category"],
            "rating": item["rating"],
        }
        for item in ds["train"]
    ]

    # Randomly sample entries (temporarily set to 100 for testing)
    sample_size = min(100, len(all_items))
    logger.info(
        f"\nRandomly sampling {sample_size} entries from {len(all_items)} total entries..."
    )
    sampled_items = random.sample(all_items, sample_size)

    # Filter items
    logger.info("\nFiltering sampled entries...")
    filtered_items = []
    filter_stats = {"length": 0, "format": 0, "quality": 0, "passed": 0}

    for item in sampled_items:
        summary = item["summary"]
        # Track why items are filtered
        if len(summary) < 20 or len(summary) > 500:
            filter_stats["length"] += 1
        elif not summary[0].isupper() or not summary.rstrip("...").endswith(
            (".", "!", "?")
        ):
            filter_stats["format"] += 1
        elif any(
            indicator in summary.lower()
            for indicator in [
                "wip",
                "tbd",
                "to be continued",
                "summary inside",
                "please read",
            ]
        ):
            filter_stats["quality"] += 1
        else:
            words = summary.split()
            if len(words) >= 5 and len(set(words)) >= 4:
                filtered_items.append(item)
                filter_stats["passed"] += 1
            else:
                filter_stats["quality"] += 1

    filtered_count = len(filtered_items)
    logger.info("\nFilter statistics:")
    logger.info(f"- Failed length check: {filter_stats['length']} items")
    logger.info(f"- Failed format check: {filter_stats['format']} items")
    logger.info(f"- Failed quality check: {filter_stats['quality']} items")
    logger.info(f"- Passed all checks: {filter_stats['passed']} items")
    logger.info(
        f"\nFiltered {sample_size} -> {filtered_count} items ({filtered_count/sample_size*100:.1f}% pass rate)"
    )

    # Estimate total cost
    estimated_cost = filtered_count * 0.0000219
    logger.info(f"Estimated cost for {filtered_count} items: ${estimated_cost:.6f}")

    proceed = input(f"\nProceed with processing {filtered_count} items (y/n)? ")
    if proceed.lower() != "y":
        logger.info("Aborting...")
        return

    # Process in batches
    batch_size = 50
    all_prompts = []
    stats = {"successful": 0, "failed": 0, "cost": 0.0}

    # Create output directory
    os.makedirs("data/raw/fanfiction_prompts/data", exist_ok=True)

    try:
        # Process in smaller batches
        for i in range(0, len(filtered_items), batch_size):
            batch_items = filtered_items[i : i + batch_size]
            prompts = await process_batch(batch_items, stats)
            all_prompts.extend(prompts)

            # Log progress
            logger.info(
                f"Processed {i+len(batch_items)}/{len(filtered_items)} items. "
                f"Success: {stats['successful']}, Failed: {stats['failed']}, "
                f"Cost: ${stats['cost']:.6f}"
            )

            # Save progress
            if all_prompts:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(
                    f"data/raw/fanfiction_prompts/data/prompts_{timestamp}.json", "w"
                ) as f:
                    json.dump(all_prompts, f, indent=2)

            # Rate limiting
            time.sleep(1)  # Basic rate limiting

        # Convert to HuggingFace dataset format
        dataset = Dataset.from_dict({"prompt": all_prompts})

        # Save as HuggingFace dataset
        dataset.save_to_disk("data/raw/fanfiction_prompts/data")

        # Save final statistics
        stats["timestamp"] = datetime.now().isoformat()
        with open("data/raw/fanfiction_prompts/data/stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("Processing complete. Final stats:")
        logger.info(f"Total successful: {stats['successful']}")
        logger.info(f"Total failed: {stats['failed']}")
        logger.info(f"Total cost: ${stats['cost']:.6f}")
        logger.info(f"Dataset size: {len(dataset)} prompts")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        # Save what we have so far
        if all_prompts:
            with open(
                "data/raw/fanfiction_prompts/data/prompts_error_recovery.json", "w"
            ) as f:
                json.dump(all_prompts, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
