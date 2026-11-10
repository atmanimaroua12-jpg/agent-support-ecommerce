"""Chargement de la configuration YAML."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # racine du projet


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Transforme un chemin de la config (relatif à la racine) en chemin absolu."""
    return ROOT / relative