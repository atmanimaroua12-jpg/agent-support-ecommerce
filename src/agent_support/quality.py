"""Quality gates : vérifie qu'une version de l'agent respecte les seuils avant d'être promue."""
import json
from dataclasses import dataclass
from pathlib import Path

from .rag import load_chunks


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def check_metrics(metrics: dict, rules: dict) -> list[Check]:
    checks = []
    for name, rule in rules.items():
        value = metrics.get(name)
        if value is None:
            checks.append(Check(name, False, "métrique absente du rapport"))
            continue
        ok = ("min" not in rule or value >= rule["min"]) and ("max" not in rule or value <= rule["max"])
        bound = f">= {rule['min']}" if "min" in rule else f"<= {rule['max']}"
        checks.append(Check(name, ok, f"{value} (attendu {bound})"))
    return checks


def check_data(corpus_dir: str | Path, orders_path: str | Path, min_active_documents: int) -> list[Check]:
    checks = []
    try:  # load_chunks refuse tout document dont les métadonnées sont incomplètes
        chunks, report = load_chunks(corpus_dir)
        active = sum(r["indexe"] for r in report)
        checks.append(Check("corpus_metadonnees", True, f"{len(report)} documents, métadonnées complètes"))
        checks.append(Check("corpus_documents_actifs", active >= min_active_documents,
                            f"{active} actifs (attendu >= {min_active_documents})"))
    except ValueError as exc:
        checks.append(Check("corpus_metadonnees", False, str(exc)))

    db = json.loads(Path(orders_path).read_text(encoding="utf-8"))
    required = {"customer_id", "status", "order_date", "estimated_delivery", "items"}
    problems = [oid for oid, o in db["orders"].items()
                if not required <= o.keys() or o["customer_id"] not in db["customers"]]
    checks.append(Check("commandes_schema", not problems,
                        f"{len(db['orders'])} commandes" + (f", invalides : {problems}" if problems else "")))
    return checks


def check_version(report: dict, current_version: dict) -> Check:
    """Le rapport doit porter sur EXACTEMENT la version à promouvoir (même prompt, config, outils, corpus)."""
    ok = report.get("agent_version") == current_version["agent_version"]
    if ok:
        return Check("version_evaluee", True, f"agent_version {current_version['agent_version']}")
    changed = [k for k in ("config_hash", "prompt_hash", "tools_hash", "corpus_hash")
               if report.get(k) != current_version[k]]
    return Check("version_evaluee", False, f"rapport {report.get('agent_version')} ≠ version actuelle "
                 f"{current_version['agent_version']} (modifié : {', '.join(changed)}) : relancer l'évaluation")