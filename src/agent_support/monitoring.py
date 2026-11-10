"""Indicateurs de monitoring calculés à partir des traces (utilisés par /metrics et par le tableau de bord)."""
from collections import Counter


def percentile(values: list[float], p: float) -> float:
    """Percentile simple (p entre 0 et 100), sans dépendance externe."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))
    return ordered[index]


def compute_metrics(traces: list[dict]) -> dict:
    n = len(traces)
    if n == 0:
        return {"n_requests": 0}
    outcomes = Counter(t["outcome"] for t in traces)
    answered = [t for t in traces if t["outcome"] != "blocked"]   # questions réellement envoyées au LLM
    latencies = [t["latency_ms"] for t in answered]
    n_calls = sum(t["n_tool_calls"] for t in traces)
    n_tool_errors = sum(t["n_tool_errors"] for t in traces)
    tools = Counter(call["name"] for t in traces for call in t["tool_calls"])
    total_cost = sum(t["cost_usd"] for t in traces)
    return {
        "n_requests": n,
        "outcomes": dict(outcomes),
        "error_rate": round((outcomes["llm_error"] + outcomes["max_iterations"]) / n, 3),
        "blocked_rate": round(outcomes["blocked"] / n, 3),
        "blocked_reasons": dict(Counter(t["blocked_reason"] for t in traces if t["blocked_reason"])),
        "latency_mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "tool_calls_total": n_calls,
        "tool_calls_per_request": round(n_calls / len(answered), 2) if answered else 0.0,
        "tool_error_rate": round(n_tool_errors / n_calls, 3) if n_calls else 0.0,
        "tool_usage": dict(tools),
        "escalation_rate": round(sum(t["escalation_proposed"] for t in traces) / n, 3),
        "tokens_total": sum(t["input_tokens"] + t["output_tokens"] for t in traces),
        "cost_total_usd": round(total_cost, 6),
        "cost_per_request_usd": round(total_cost / n, 6),
        "agent_versions": dict(Counter(t.get("agent_version", "?") for t in traces)),
    }