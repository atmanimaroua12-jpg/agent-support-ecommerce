"""Étape du pipeline DVC : évalue la recherche documentaire seule (déterministe, sans LLM)."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_support.evaluation import evaluate_retrieval  # noqa: E402
from agent_support.rag import RagIndex, load_chunks  # noqa: E402
from agent_support.versioning import corpus_hash  # noqa: E402

chunks, report = load_chunks(ROOT / "data" / "corpus")
index = RagIndex(chunks)
questions = yaml.safe_load((ROOT / "eval" / "retrieval_questions.yaml").read_text(encoding="utf-8"))
metrics = {**evaluate_retrieval(index, questions, k=1), **evaluate_retrieval(index, questions, k=3),
           "n_questions": len(questions), "n_chunks": len(chunks),
           "n_documents_actifs": sum(r["indexe"] for r in report), "corpus_hash": corpus_hash(ROOT / "data" / "corpus")}
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))