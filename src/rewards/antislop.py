"""
Reward functions that penalize the model for generating slop content

Modified from: https://github.com/sam-paech/antislop-sampler/blob/main/src/slop_index.py
"""

import re

from json import load
from pathlib import Path

from joblib import Parallel, delayed

from src.utils import extract_story_only


def load_and_preprocess_slop_words(top_n: int = 600) -> dict[str, float]:
    """
    Modified from: https://github.com/sam-paech/antislop-sampler/blob/main/src/slop_index.py
    """
    with open(Path(__file__).parent / "slop_phrase_prob_adjustments.json", "r") as f:
        slop_phrases = load(f)
    phrase_weighting = [1.0 - prob_adjustment for _, prob_adjustment in slop_phrases]
    max_score = max(phrase_weighting)
    scaled_weightings = [score / max_score for score in phrase_weighting]
    return {
        word.lower(): score
        for (word, _), score in zip(slop_phrases[:top_n], scaled_weightings[:top_n])
    }


def calculate_slop_score_chunk(args) -> float:
    """
    Modified from: https://github.com/sam-paech/antislop-sampler/blob/main/src/slop_index.py
    """
    text, slop_words_chunk = args
    return sum(
        score * len(re.findall(r"\b" + re.escape(word) + r"\b", text))
        for word, score in slop_words_chunk.items()
    )


def split_into_chunks(slop_words: dict[str, float], num_chunks: int = 12):
    """
    Modified from: https://github.com/sam-paech/antislop-sampler/blob/main/src/slop_index.py
    """
    slop_words_items = list(slop_words.items())
    chunk_size = len(slop_words_items) // num_chunks
    if chunk_size == 0:
        chunk_size = 1
    return [
        dict(slop_words_items[i : i + chunk_size])
        for i in range(0, len(slop_words_items), chunk_size)
    ]


def calculate_slop_index(
    extracted_text, n_jobs: int = -1
):  # -1 means use all available cores
    slop_words = load_and_preprocess_slop_words()
    slop_words_chunks = split_into_chunks(slop_words, n_jobs)

    if not extracted_text:
        slop_index = 0.0
    else:
        # Parallelize the calculation using joblib with proper backend
        slop_scores = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(calculate_slop_score_chunk)((extracted_text, chunk))
            for chunk in slop_words_chunks
        )

        slop_score = sum(slop_scores)
        total_words = len(extracted_text.split())
        slop_index = (slop_score / total_words) * 1000 if total_words > 0 else 0
    return slop_index


def antislop_penalty_reward(completions, **kwargs) -> list[float]:
    """Reward function that penalizes proportional to the slop index"""
    responses = [
        extract_story_only(completion[0]["content"]) for completion in completions
    ]

    return [-calculate_slop_index(text) for text in responses]
