from typing import Dict


def build_prompt(prompt_record: Dict, sentence_record: Dict) -> str:
    """
    Build the final prompt text for one sentence.

    Inputs:
        prompt_record:
            One line from prompts.jsonl. Must include a template field.
        sentence_record:
            One line from sentences.jsonl. Must include text.

    Outputs:
        A fully formatted prompt string.

    Purpose:
        Keeps prompt construction consistent across OpenAI, Claude and Gemini.
        The same prompt text should be stored in raw_responses.jsonl for
        reproducibility.
    """
    raise NotImplementedError
