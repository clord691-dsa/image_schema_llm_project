from typing import Dict, List


def compute_schema_accuracy(
    gold_records: List[Dict],
    parsed_records: List[Dict],
) -> Dict:
    """
    Compute image-schema identification accuracy.

    Inputs:
        gold_records:
            Gold annotations from gold_annotations.jsonl.
        parsed_records:
            Parsed model outputs from parsed_responses.jsonl.

    Outputs:
        Dictionary containing accuracy scores.

    Purpose:
        Evaluates whether models identify the correct primary schema.
    """
    raise NotImplementedError


def compute_literal_metaphorical_f1(
    gold_records: List[Dict],
    parsed_records: List[Dict],
) -> Dict:
    """
    Compute performance for literal/metaphorical classification.

    Inputs:
        gold_records:
            Gold literal/metaphorical labels.
        parsed_records:
            Model-produced labels.

    Outputs:
        Dictionary containing precision, recall and F1.

    Purpose:
        Tests whether models can distinguish literal spatial language from
        metaphorical spatial language.
    """
    raise NotImplementedError


def compute_role_level_scores(
    gold_records: List[Dict],
    parsed_records: List[Dict],
) -> Dict:
    """
    Compute role-level evaluation scores.

    Inputs:
        gold_records:
            Gold schema-role annotations.
        parsed_records:
            Model role annotations.

    Outputs:
        Dictionary of role-level precision, recall and F1.

    Purpose:
        Evaluates whether the model identifies trajector, container, source,
        path, goal, obstacle and force roles correctly.
    """
    raise NotImplementedError

