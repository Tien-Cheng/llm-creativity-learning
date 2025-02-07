"""
Reward functions that reward model if it:
-> Encloses story inside <story> tags
-> Uses <think> tags
-> If in the prompt, we instruct it to produce n number of chapters, we reward the model if it produces n <story> tag pairs and n <think> tag pairs, where the xml tags include the chapter number as attributes e.g <story chapter="1"></story>

References: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
"""

import re
from typing import List


def extract_chapter_count(prompt) -> int:
    """Extract number of chapters from XML tags in the prompt."""
    # Get the last message (user's prompt)
    last_message = prompt[-1]["content"]

    # Extract chapter count from <chapters> tag
    if match := re.search(r"<chapters>(\d+)</chapters>", last_message):
        return int(match.group(1))

    # Default to 1 chapter if not specified
    return 1


def strict_format_reward(prompts, completions, **kwargs) -> List[float]:
    """
    Strict reward function that checks if the completion follows exact format requirements:
    - Has correct number of story and think tags
    - Story tags have proper chapter attributes
    - Tags are properly nested and formatted
    """
    num_chapters = extract_chapter_count(prompts[0])
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for response in responses:
        # Check for correct number of story and think tags
        story_matches = re.findall(
            r'<story chapter="(\d+)">(.*?)</story>', response, re.DOTALL
        )
        think_matches = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)

        if len(story_matches) != num_chapters or len(think_matches) != num_chapters:
            scores.append(0.0)
            continue

        # Check if chapter numbers are sequential and unique
        chapters = [int(n) for n, _ in story_matches]
        if chapters != list(range(1, num_chapters + 1)):
            scores.append(0.5)
            continue

        # Check proper nesting of tags
        valid_sequence = True
        for i in range(num_chapters):
            chapter_pattern = f'<think>.*?</think>.*?<story chapter="{i+1}">.*?</story>'
            if not re.search(chapter_pattern, response, re.DOTALL):
                valid_sequence = False
                break

        scores.append(1.0 if valid_sequence else 0.5)

    return scores


def soft_format_reward(prompts, completions, **kwargs) -> List[float]:
    """
    Soft reward function that gives partial credit for:
    - Having story tags
    - Having think tags
    - Having chapter attributes
    """
    num_chapters = extract_chapter_count(prompts[0])
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for response in responses:
        score = 0.0

        # Check for story tags with chapters
        story_matches = re.findall(
            r'<story chapter="\d+">(.*?)</story>', response, re.DOTALL
        )
        if len(story_matches) == num_chapters:
            score += 0.4
        elif story_matches:
            score += 0.2

        # Check for think tags
        think_matches = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        if len(think_matches) == num_chapters:
            score += 0.3
        elif think_matches:
            score += 0.15

        # Check for sequential chapter numbering
        chapter_nums = re.findall(r'chapter="(\d+)"', response)
        if len(chapter_nums) == num_chapters:
            try:
                if sorted([int(n) for n in chapter_nums]) == list(
                    range(1, num_chapters + 1)
                ):
                    score += 0.3
                else:
                    score += 0.15
            except ValueError:
                pass

        scores.append(score)

    return scores


def xml_count_reward(prompts, completions, **kwargs) -> List[float]:
    """
    Reward function that counts XML tags and gives partial credit.
    """
    num_chapters = extract_chapter_count(prompts[0])
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    for response in responses:
        score = 0.0

        # Check story tags
        story_opens = response.count("<story")
        story_closes = response.count("</story>")
        if story_opens == story_closes == num_chapters:
            score += 0.2
        elif story_opens > 0 and story_closes > 0:
            score += 0.1

        # Check think tags
        think_opens = response.count("<think>")
        think_closes = response.count("</think>")
        if think_opens == think_closes == num_chapters:
            score += 0.2
        elif think_opens > 0 and think_closes > 0:
            score += 0.1

        # Check chapter attributes
        chapter_attrs = len(re.findall(r'chapter="\d+"', response))
        if chapter_attrs == num_chapters:
            score += 0.2
        elif chapter_attrs > 0:
            score += 0.1

        # Check for proper tag nesting
        valid_pairs = 0
        for i in range(1, num_chapters + 1):
            if re.search(
                f'<think>.*?</think>.*?<story chapter="{i}">.*?</story>',
                response,
                re.DOTALL,
            ):
                valid_pairs += 1
        score += 0.4 * (valid_pairs / num_chapters)

        scores.append(score)

    return scores
