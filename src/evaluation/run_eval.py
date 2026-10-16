"""Score manually captured answers: python -m src.evaluation.run_eval eval_results.json"""

import json
import sys
from pathlib import Path

from src.evaluation.metrics import evaluate_case


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluation.run_eval <results.json>")
        return 2
    samples = json.loads(Path(__file__).with_name("samples.json").read_text(encoding="utf-8"))
    results = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    by_question = {item["question"]: item for item in results}
    rows = [evaluate_case(case, by_question.get(case["question"], {})) for case in samples]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    for metric in ("retrieval_evidence_coverage", "answer_relevance", "faithfulness_coverage"):
        mean = sum(row[metric] for row in rows) / max(len(rows), 1)
        print(f"mean_{metric}: {mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
