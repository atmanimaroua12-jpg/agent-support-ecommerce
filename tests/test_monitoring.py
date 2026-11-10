from agent_support.monitoring import compute_metrics, percentile


def trace(outcome="completed", latency=1000.0, tools=(), errors=0, blocked=None, draft=False, cost=0.001):
    return {"outcome": outcome, "latency_ms": latency, "tool_calls": [{"name": n} for n in tools],
            "n_tool_calls": len(tools), "n_tool_errors": errors, "blocked_reason": blocked,
            "escalation_proposed": draft, "input_tokens": 100, "output_tokens": 10, "cost_usd": cost}


def test_percentile():
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([], 95) == 0.0


def test_metriques():
    m = compute_metrics([
        trace(tools=["search_docs"]),
        trace(tools=["get_order_status"], errors=1, latency=3000.0),
        trace(tools=["create_ticket"], draft=True),
        trace(outcome="llm_error", latency=500.0),
        trace(outcome="blocked", latency=1.0, blocked="prompt_injection", cost=0.0),
    ])
    assert m["n_requests"] == 5 and m["error_rate"] == 0.2 and m["blocked_rate"] == 0.2
    assert m["tool_error_rate"] == round(1 / 3, 3) and m["escalation_rate"] == 0.2
    assert m["latency_p95_ms"] == 3000.0   # les requêtes bloquées ne faussent pas la latence


def test_aucune_trace():
    assert compute_metrics([]) == {"n_requests": 0}