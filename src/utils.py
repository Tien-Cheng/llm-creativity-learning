"""
Utility functions for:
- Extracting XML from input and ouput
"""


def extract_story_only(
    completion: str,
) -> str:
    """
    Extract story content from XML tags in the completion.
    Stories are enclosed in <story> tags, potentially with chapter attributes.

    Args:
        completion: The raw completion string containing XML tags

    Returns:
        str: The extracted story content with XML tags removed
    """
    import re

    # Extract content between <story> tags (with optional chapter attributes)
    story_pattern = r'<story(?:\s+chapter="\d+")?>(.+?)</story>'
    stories = re.findall(story_pattern, completion, re.DOTALL)

    # Combine all story segments
    combined_story = " ".join(stories)

    # Remove any remaining XML tags (like <think>)
    clean_story = re.sub(r"<[^>]+>.*?</[^>]+>", "", combined_story)

    return clean_story.strip()
