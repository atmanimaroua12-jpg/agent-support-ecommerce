"""Empreinte de version de l'agent.

Le comportement d'un agent dépend du prompt, de la configuration, des schémas d'outils et du corpus.
On calcule un hash de ces 4 éléments : si l'un change, la version change. Cette empreinte est écrite
dans chaque trace (et plus tard dans MLflow) pour savoir quelle version a produit quelle réponse.
"""
import hashlib
import json
from pathlib import Path

from .prompts import get_system_prompt
from .tools.catalog import build_registry


def short_hash(obj) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def corpus_hash(corpus_dir: str | Path) -> str:
    files = sorted(Path(corpus_dir).glob("*.md"))
    return short_hash({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files})


def version_info(cfg: dict, corpus_dir: str | Path) -> dict:
    behaviour = {key: cfg[key] for key in ("llm", "agent", "tools")}
    parts = {
        "config_hash": short_hash(behaviour),
        "prompt_hash": short_hash(get_system_prompt(cfg["agent"]["prompt_version"])),
        "tools_hash": short_hash(build_registry().schemas()),
        "corpus_hash": corpus_hash(corpus_dir),
    }
    return {"agent_version": short_hash(parts), **parts}
