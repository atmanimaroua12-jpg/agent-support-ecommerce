"""Quality gates et promotion de version.

Vérifier la version actuelle (utilisé par la CI) :
    python scripts/quality_gate.py --report reports/eval_prompt-v3.json
Promouvoir une version si tous les critères passent (met à jour configs/config.yaml) :
    python scripts/quality_gate.py --report reports/eval_prompt-v3.json --promote v3
Code de sortie : 0 si tout passe, 1 sinon (la CI échoue).
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_support.config import load_config, project_path  # noqa: E402
from agent_support.quality import check_data, check_metrics, check_version  # noqa: E402
from agent_support.versioning import version_info  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality gates")
    parser.add_argument("--report", default="reports/last_eval.json", help="rapport produit par eval/run_eval.py")
    parser.add_argument("--promote", metavar="PROMPT", help="version de prompt à promouvoir si tout passe")
    args = parser.parse_args()

    gates = yaml.safe_load((ROOT / "configs" / "quality_gates.yaml").read_text(encoding="utf-8"))
    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    cfg = load_config()
    if args.promote:
        cfg["agent"]["prompt_version"] = args.promote
    corpus = project_path(cfg["paths"]["corpus"])
    current = version_info(cfg, corpus)

    checks = [check_version(report, current),
              *check_metrics(report["metrics"], gates["evaluation"]),
              *check_data(corpus, project_path(cfg["paths"]["orders"]), gates["data"]["min_active_documents"])]

    print(f"Quality gates · rapport {args.report} · prompt {cfg['agent']['prompt_version']}\n")
    for c in checks:
        print(f"{'✅' if c.ok else '❌'} {c.name:28} {c.detail}")
    passed = all(c.ok for c in checks)
    print(f"\n{'VALIDÉ' if passed else 'REFUSÉ'} : {sum(c.ok for c in checks)}/{len(checks)} critères respectés")

    if passed and args.promote:
        path = ROOT / "configs" / "config.yaml"
        text = re.sub(r"(^\s*prompt_version:\s*)\S+", rf"\g<1>{args.promote}", path.read_text(encoding="utf-8"),
                      count=1, flags=re.MULTILINE)
        path.write_text(text, encoding="utf-8")
        manifest = {"promoted_at": datetime.now(timezone.utc).isoformat(), "prompt_version": args.promote,
                    "model": cfg["llm"]["model"], **current, "evaluation_report": args.report,
                    "evaluation_git_commit": report.get("git_commit"),
                    "metrics": {c.name: c.detail for c in checks}}
        (ROOT / "reports" / "promoted_version.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n🚀 Version {current['agent_version']} (prompt {args.promote}) promue : configs/config.yaml mis à "
              "jour, manifeste reports/promoted_version.json écrit.")
    elif not passed and args.promote:
        print("\n⛔ Promotion refusée : configs/config.yaml n'est pas modifié.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())