"""Simple transparent lexical metrics for manually captured RAG evaluations.

These are screening metrics, not substitutes for expert human review or an LLM judge.
"""

import re

WORD_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*", re.I)


def _terms(text: str) -> set[str]:
    return {word.lower().strip(".,;:!?()[]{}") for word in WORD_RE.findall(text) if len(word) > 1}


def retrieval_hit(expected_evidence: str, passages: list[str]) -> float:
    """Evidence coverage across retrieved passages (1 means all evidence terms appear)."""
    expected = _terms(expected_evidence)
    observed = _terms(" ".join(passages))
    return len(expected & observed) / len(expected) if expected else 0.0


def answer_relevance(reference_answer: str, answer: str) -> float:
    """Recall of reference answer terms in the generated answer."""
    reference = _terms(reference_answer)
    response = _terms(answer)
    return len(reference & response) / len(reference) if reference else 0.0


def faithfulness_coverage(answer: str, passages: list[str]) -> float:
    """Fraction of informative answer terms that also occur in retrieved passages."""
    answer_terms = _terms(answer)
    context_terms = _terms(" ".join(passages))
    return len(answer_terms & context_terms) / len(answer_terms) if answer_terms else 0.0


def evaluate_case(case: dict, result: dict) -> dict[str, float | str]:
    passages = result.get("retrieved_passages", [])
    answer = result.get("answer", "")
    return {
        "question": case["question"],
        "retrieval_evidence_coverage": round(retrieval_hit(case["expected_evidence"], passages), 3),
        "answer_relevance": round(answer_relevance(case["reference_answer"], answer), 3),
        "faithfulness_coverage": round(faithfulness_coverage(answer, passages), 3),
    }
