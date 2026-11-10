"""Traces opérationnelles : une ligne JSON par question, avec les données personnelles masquées.

On trace les ÉVÉNEMENTS OBSERVABLES (outils appelés, statuts, durées, tokens, coût, issue),
pas le raisonnement interne du modèle (cahier des charges, §5.4).
"""
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .guardrails import mask_obj, mask_pii


def estimate_cost(llm_cfg: dict, input_tokens: int, output_tokens: int) -> float:
    """Coût estimé en dollars, à partir du tarif par million de tokens indiqué dans la config."""
    price = llm_cfg.get("price_usd_per_million", {})
    cost = input_tokens * price.get("input", 0) + output_tokens * price.get("output", 0)
    return round(cost / 1_000_000, 6)


def build_trace(request_id: str, question: str, customer_id: str | None, response, meta: dict) -> dict:
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **meta,                                         # version de l'agent, modèle, version du prompt
        "customer_id": customer_id,
        "question": mask_pii(question[:300]),           # aperçu tronqué et masqué
        "outcome": response.outcome,
        "blocked_reason": response.blocked_reason,
        "tool_calls": [mask_obj(asdict(call)) for call in response.tool_calls],
        "n_tool_calls": len(response.tool_calls),
        "n_tool_errors": sum(call.status != "ok" for call in response.tool_calls),
        "iterations": response.iterations,
        "escalation_proposed": response.draft is not None,
        "sources": response.sources,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
        "error": response.error,
    }


class Tracer:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_traces(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
