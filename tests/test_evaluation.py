from src.evaluation.metrics import answer_relevance, faithfulness_coverage, retrieval_hit


def test_retrieval_evidence_coverage():
    assert retrieval_hit("MiniLM embedding model", ["DocuMind uses the MiniLM embedding model."]) == 1


def test_answer_relevance_and_faithfulness():
    answer = "The local MiniLM model creates document embeddings."
    assert answer_relevance("MiniLM model creates document embeddings", answer) == 1
    assert faithfulness_coverage(answer, ["The local MiniLM model creates document embeddings."]) == 1
