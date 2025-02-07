"""
Reward models that evaluate the quality of a text
"""

import dspy
import re
from typing import List
from src.utils import extract_story_only


class GrammarAndSyntax(dspy.Signature):
    """Evaluate the grammatical correctness and syntactical structure of creative writing.

    Scoring Guidelines:
    1. Sentence Structure (0-2 points)
        - Variety in sentence length and structure
        - Proper use of subordinate clauses
        - Effective paragraph transitions

    2. Grammar Rules (0-2 points)
        - Subject-verb agreement
        - Proper tense usage
        - Correct pronoun usage

    3. Punctuation (0-2 points)
        - Appropriate use of commas, semicolons, and periods
        - Proper quotation mark usage
        - Consistent formatting

    4. Word Choice (0-2 points)
        - Appropriate vocabulary level
        - Consistent register/tone
        - No redundant words

    5. Technical Accuracy (0-2 points)
        - No run-on sentences
        - No sentence fragments
        - Proper modifier placement

    Analysis Process:
    1. First, scan for basic grammatical errors
    2. Then evaluate sentence structure and variety
    3. Check punctuation usage and consistency
    4. Assess vocabulary and word choice
    5. Finally, look for any technical writing issues
    """

    prompt: str = dspy.InputField(
        desc="The original creative writing prompt that the writer responded to"
    )
    response: str = dspy.InputField(
        desc="The complete creative writing response to be evaluated"
    )
    score: int = dspy.OutputField(
        desc="A score from 0-10 where 10 represents perfect grammar and syntax. Score must be justified based on the scoring guidelines."
    )


class EmotionalImpact(dspy.Signature):
    """Evaluate how effectively the writing evokes emotional responses and connects with readers.

    Scoring Guidelines:
    1. Emotional Depth (0-2 points)
        - Complex character emotions
        - Genuine emotional stakes
        - Avoidance of melodrama

    2. Show vs Tell (0-2 points)
        - Emotions conveyed through actions/dialogue
        - Physical manifestations of feelings
        - Subtle emotional undertones

    3. Reader Connection (0-2 points)
        - Universal emotional themes
        - Relatable situations/reactions
        - Emotional authenticity

    4. Atmosphere (0-2 points)
        - Mood-appropriate descriptions
        - Emotional pacing
        - Sensory details that enhance feeling

    5. Resonance (0-2 points)
        - Lasting emotional impact
        - Thematic depth
        - Emotional payoff

    Analysis Steps:
    1. Identify the core emotional themes
    2. Analyze how emotions are portrayed
    3. Evaluate authenticity and depth
    4. Assess reader engagement
    5. Consider lasting impact

    Examples of Strong Emotional Writing:
    - Physical descriptions of emotional states
    - Dialogue that reveals feeling without stating it
    - Metaphors that capture emotional experience
    - Situations that naturally evoke emotion

    Examples of Weak Emotional Writing:
    - Directly stating emotions ("She was sad")
    - Melodramatic reactions
    - Clichéd emotional situations
    - Lack of emotional buildup
    """

    response: str = dspy.InputField(
        desc="Full creative writing piece to analyze for emotional impact. Consider both obvious and subtle emotional elements."
    )
    score: int = dspy.OutputField(
        desc="Score from 0-10 based on emotional impact scoring guidelines. Must justify score with specific examples from the text."
    )


class OriginalityAndCreativity(dspy.Signature):
    """Evaluate the uniqueness and creative merit of the writing.

    Scoring Process:
    1. Analyze Common Elements (0-2 points)
        - Identify standard tropes/clichés
        - Note conventional plot structures
        - List familiar character types

    2. Assess Creative Elements (0-2 points)
        - Unique twists on familiar ideas
        - Novel concepts or approaches
        - Innovative narrative techniques

    3. Evaluate Execution (0-2 points)
        - Fresh metaphors/similes
        - Unique descriptive language
        - Original dialogue patterns

    4. Consider Voice (0-2 points)
        - Distinctive writing style
        - Unique perspective
        - Fresh narrative voice

    5. Judge Overall Innovation (0-2 points)
        - Creative risk-taking
        - Genre-blending elements
        - Unexpected elements that work

    Red Flags (Deduct Points):
    - Overused phrases/metaphors
    - Predictable plot developments
    - Stock characters without depth
    - Familiar story arcs without twists

    Strong Originality Indicators:
    - Surprising but logical plot turns
    - Fresh perspectives on common themes
    - Innovative narrative structures
    - Unique character dynamics
    """

    prompt: str = dspy.InputField(
        desc="Original writing prompt to understand the creative constraints and opportunities presented"
    )
    response: str = dspy.InputField(
        desc="Writing to be evaluated for originality. Consider both broad creative choices and specific language use."
    )
    score: int = dspy.OutputField(
        desc="Originality score from 0-10. Must cite specific creative choices that earned or lost points."
    )


class CoherenceAndLogicalFlow(dspy.Signature):
    """Evaluate the logical progression and structural integrity of the writing.

    Structure Analysis:
    1. Narrative Flow (0-2 points)
        - Clear beginning/middle/end
        - Smooth scene transitions
        - Logical story progression

    2. Cause and Effect (0-2 points)
        - Events follow logically
        - Character actions make sense
        - Clear motivations

    3. Pacing (0-2 points)
        - Appropriate story rhythm
        - Well-timed reveals
        - Balanced scene lengths

    4. Information Flow (0-2 points)
        - Clear exposition
        - No plot holes
        - Proper setup/payoff

    5. Thematic Coherence (0-2 points)
        - Consistent themes
        - Unity of purpose
        - Satisfying resolution

    Step-by-Step Evaluation:
    1. Map the story's structure
    2. Track cause/effect relationships
    3. Analyze pacing decisions
    4. Check information revelation
    5. Assess thematic consistency

    Common Issues to Flag:
    - Plot holes or logical inconsistencies
    - Unmotivated character actions
    - Sudden shifts in tone/style
    - Missing crucial information
    - Unresolved plot threads
    """

    response: str = dspy.InputField(
        desc="Complete writing piece to analyze for coherence and flow. Consider both macro-structure and scene-level progression."
    )
    score: int = dspy.OutputField(
        desc="Coherence score from 0-10. Must explain score by analyzing story structure, pacing, and logical progression."
    )


grammar_evaluator = dspy.ChainOfThought(GrammarAndSyntax)
emotion_evaluator = dspy.ChainOfThought(EmotionalImpact)
originality_evaluator = dspy.ChainOfThought(OriginalityAndCreativity)
coherence_evaluator = dspy.ChainOfThought(CoherenceAndLogicalFlow)


def extract_writing_prompt(prompt) -> str:
    """Extract writing prompt from XML tags in the prompt message."""
    last_message = prompt[-1]["content"]
    if match := re.search(
        r"<writing_prompt>(.*?)</writing_prompt>", last_message, re.DOTALL
    ):
        return match.group(1).strip()
    return last_message


def grammar_reward(prompts, completions, **kwargs) -> List[float]:
    """Reward function using GrammarAndSyntax DSPy signature"""
    writing_prompt = extract_writing_prompt(prompts[0])
    responses = [
        extract_story_only(completion[0]["content"]) for completion in completions
    ]
    return [
        grammar_evaluator(prompt=writing_prompt, response=r).score / 10.0
        for r in responses
    ]


def emotional_impact_reward(completions, **kwargs) -> List[float]:
    """Reward function using EmotionalImpact DSPy signature"""
    responses = [
        extract_story_only(completion[0]["content"]) for completion in completions
    ]
    return [emotion_evaluator(response=r).score / 10.0 for r in responses]


def originality_reward(prompts, completions, **kwargs) -> List[float]:
    """Reward function using OriginalityAndCreativity DSPy signature"""
    writing_prompt = extract_writing_prompt(prompts[0])
    responses = [
        extract_story_only(completion[0]["content"]) for completion in completions
    ]
    return [
        originality_evaluator(prompt=writing_prompt, response=r).score / 10.0
        for r in responses
    ]


def coherence_reward(completions, **kwargs) -> List[float]:
    """Reward function using CoherenceAndLogicalFlow DSPy signature"""
    responses = [
        extract_story_only(completion[0]["content"]) for completion in completions
    ]
    return [coherence_evaluator(response=r).score / 10.0 for r in responses]
