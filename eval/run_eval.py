"""Lance l'agent sur le jeu de scénarios, calcule les métriques et enregistre l'expérience dans MLflow.

Exemples :
    python eval/run_eval.py                          # configuration actuelle (prompt v2)
    python eval/run_eval.py --prompt v1 --run-name prompt-v1
    python eval/run_eval.py --only S01 S14           # quelques scénarios seulement
Interface MLflow :  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_support.agent import build_agent  # noqa: E402
from agent_support.config import load_config, project_path  # noqa: E402
from agent_support.evaluation import aggregate, evaluate_retrieval, owners, score_scenario  # noqa: E402
from agent_support.tools.ticket_store import TicketStore  # noqa: E402

TOKENS_PER_MINUTE = 6500   # sous la limite gratuite de Groq (8 000 / min) pour garder une marge


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation de l'agent")
    parser.add_argument("--prompt", help="version du prompt (v1, v2...)")
    parser.add_argument("--model", help="modèle Groq à utiliser")
    parser.add_argument("--reasoning-effort", help="low | medium | high (modèles gpt-oss)")
    parser.add_argument("--run-name", default=None, help="nom de l'exécution dans MLflow")
    parser.add_argument("--only", nargs="*", help="identifiants des scénarios à exécuter")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "inconnu"


def wait_for_quota(window: deque) -> None:
    """Respecte le quota de tokens par minute en attendant si nécessaire."""
    while window and time.time() - window[0][0] > 60:
        window.popleft()
    used = sum(tokens for _, tokens in window)
    if used > TOKENS_PER_MINUTE:
        pause = 61 - (time.time() - window[0][0])
        print(f"   ... pause de {pause:.0f} s pour respecter le quota Groq")
        time.sleep(max(pause, 1))
        wait_for_quota(window)


def main():
    args = parse_args()
    cfg = load_config()
    if args.prompt:
        cfg["agent"]["prompt_version"] = args.prompt
    if args.model:
        cfg["llm"]["model"] = args.model
    if args.reasoning_effort:
        cfg["llm"]["reasoning_effort"] = args.reasoning_effort

    scenarios = yaml.safe_load((ROOT / "eval" / "scenarios.yaml").read_text(encoding="utf-8"))
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
    questions = yaml.safe_load((ROOT / "eval" / "retrieval_questions.yaml").read_text(encoding="utf-8"))

    agent = build_agent(cfg, trace_path=project_path("runtime/eval_traces.jsonl"))  # traces séparées
    order_owners = owners(project_path(cfg["paths"]["orders"]))
    print(f"Agent {agent.version['agent_version']} · {cfg['llm']['model']} · prompt {cfg['agent']['prompt_version']}"
          f" · {len(scenarios)} scénarios\n")

    results, window = [], deque()
    with tempfile.TemporaryDirectory() as tmp:
        for s in scenarios:
            # tickets dans un dossier temporaire : l'évaluation ne crée jamais de vrais tickets
            store = TicketStore(Path(tmp) / f"{s['id']}_pending.json", Path(tmp) / f"{s['id']}_tickets.jsonl")
            agent.services["tickets"] = store
            for attempt in range(4):
                wait_for_quota(window)
                response = agent.run(s["question"], s["customer"], s.get("history", []))
                window.append((time.time(), response.input_tokens + response.output_tokens))
                if response.outcome == "llm_error" and "RateLimit" in (response.error or "") and attempt < 3:
                    print("   ... limite de débit atteinte, nouvel essai dans 30 s")
                    time.sleep(30)
                    continue
                break
            result = score_scenario(s, response, order_owners, len(store.list_tickets()))
            results.append(result)
            mark = "✅" if result["success"] else "❌"
            print(f"{mark} {s['id']} [{s['category']}] {result['tools_used'] or '-'} {result['latency_ms']:.0f} ms"
                  + ("" if result["success"] else f"  -> {'; '.join(result['failures'])}"))

    metrics = {**aggregate(results), **evaluate_retrieval(agent.services["index"], questions, k=3),
               **evaluate_retrieval(agent.services["index"], questions, k=1)}

    # ------------------------------------------------------------------ sauvegarde locale + MLflow
    run_name = args.run_name or f"{cfg['agent']['prompt_version']}-{cfg['llm']['model'].split('/')[-1]}"
    out_dir = project_path("reports")
    out_dir.mkdir(exist_ok=True)
    report = {"run_name": run_name, "git_commit": git_commit(), **agent.version, "metrics": metrics,
              "results": results}
    (out_dir / "last_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"eval_{run_name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "last_eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "success", "failures", "tools_used", "outcome",
                                               "latency_ms", "input_tokens", "output_tokens", "answer"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "failures": " | ".join(r["failures"])} for r in results)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("agent-support-evaluation")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"git_commit": report["git_commit"], "agent_version": agent.version["agent_version"]})
        mlflow.log_params({
            "model": cfg["llm"]["model"], "prompt_version": cfg["agent"]["prompt_version"],
            "reasoning_effort": cfg["llm"].get("reasoning_effort"), "temperature": cfg["llm"]["temperature"],
            "top_k": cfg["tools"]["search_docs"]["top_k"], "min_score": cfg["tools"]["search_docs"]["min_score"],
            "max_iterations": cfg["agent"]["max_iterations"], "n_scenarios": len(scenarios), **agent.version,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out_dir / "last_eval.json"))
        mlflow.log_artifact(str(out_dir / "last_eval_results.csv"))
        mlflow.log_artifact(str(ROOT / "eval" / "scenarios.yaml"))
        mlflow.log_artifact(str(ROOT / "configs" / "config.yaml"))

    print("\n" + "\n".join(f"{k:32} {v}" for k, v in metrics.items()))
    print(f"\nRésultats : reports/last_eval.json · MLflow : run « {run_name} »")


if __name__ == "__main__":
    main()