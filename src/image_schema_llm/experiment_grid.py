from typing import Any, Dict, Iterator, List


def build_experiment_grid(
    models: List[Dict[str, Any]],
    prompts: List[Dict[str, Any]],
    conditions: List[Dict[str, Any]],
    sentences: List[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """
    Build the full model × prompt × condition × sentence × repetition grid.

    Inputs:
        models:
            Loaded records from models.jsonl.
        prompts:
            Loaded records from prompts.jsonl.
        conditions:
            Loaded records from conditions.jsonl.
        sentences:
            Loaded records from sentences.jsonl.

    Outputs:
        Iterator of experiment jobs.

    Each yielded job should contain:
        - model_id
        - provider
        - model_name
        - prompt_id
        - prompt_type
        - condition_id
        - temperature
        - top_p
        - max_output_tokens
        - sentence_id
        - sentence_text
        - repetition_index

    Purpose:
        Defines the complete experimental permutation structure.
    """
    raise NotImplementedError


def make_run_key(job: Dict[str, Any]) -> str:
    """
    Create a stable unique key for a single experimental permutation.

    Inputs:
        job: One experiment job dictionary.

    Outputs:
        String key, for example:
        openai_gpt|p_naive_v1|c_temp_0|s0001|0

    Purpose:
        Used to check whether a permutation has already completed.
        This is essential for restart/resume behaviour.
    """
    raise NotImplementedError
