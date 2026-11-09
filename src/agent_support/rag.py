"""RAG : ingestion du corpus, découpage par section, index TF-IDF et recherche."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_METADATA = ["doc_id", "titre", "version", "date", "type", "statut", "sensibilite"]
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Mots trop fréquents qui n'aident pas à trouver le bon passage (« de », « pour », « je »...).
STOP_WORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "cet", "cette", "comment", "dans", "de", "des", "du", "elle", "en",
    "est", "et", "il", "ils", "je", "j", "l", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "mes", "moi",
    "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "quel", "quelle",
    "quels", "quelles", "qui", "sa", "se", "ses", "si", "son", "sont", "sur", "ta", "te", "tes", "toi", "ton",
    "tu", "un", "une", "vos", "votre", "vous", "y", "d", "c", "s", "n", "m", "t", "ai", "as", "avez", "ont",
    "suis", "etes", "etre", "avoir", "fait", "faire", "puis", "peux", "peut", "quand", "combien",
}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    titre: str
    section: str
    version: str
    text: str


def parse_document(path: Path) -> tuple[dict, str]:
    """Sépare les métadonnées (front matter) du texte et vérifie qu'elles sont complètes."""
    raw = path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        raise ValueError(f"{path.name} : métadonnées (front matter) absentes")
    meta = yaml.safe_load(match.group(1)) or {}
    missing = [key for key in REQUIRED_METADATA if key not in meta]
    if missing:
        raise ValueError(f"{path.name} : métadonnées manquantes {missing}")
    meta = {key: str(value) for key, value in meta.items()}
    return meta, raw[match.end():]


def split_sections(text: str) -> list[tuple[str, str]]:
    """Découpe le texte en (titre de section, contenu) à chaque titre '## '."""
    sections, title, lines = [], None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if title and "\n".join(lines).strip():
                sections.append((title, "\n".join(lines).strip()))
            title, lines = line[3:].strip(), []
        elif title:  # on ignore le titre principal '# ...' placé avant la 1re section
            lines.append(line)
    if title and "\n".join(lines).strip():
        sections.append((title, "\n".join(lines).strip()))
    return sections


def load_chunks(corpus_dir: str | Path) -> tuple[list[Chunk], list[dict]]:
    """Lit tous les documents, n'indexe que les documents actifs, renvoie (chunks, rapport)."""
    chunks, report = [], []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        meta, body = parse_document(path)
        indexed = meta["statut"] == "actif"
        report.append({"fichier": path.name, "doc_id": meta["doc_id"], "statut": meta["statut"],
                       "indexe": indexed})
        if not indexed:
            continue
        for i, (section, text) in enumerate(split_sections(body)):
            chunks.append(Chunk(chunk_id=f"{meta['doc_id']}#{i}", doc_id=meta["doc_id"], titre=meta["titre"],
                                section=section, version=meta["version"], text=text))
    return chunks, report


def normalize(text: str) -> str:
    """Minuscules, sans accents, sans ponctuation, sans mots vides."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(w for w in re.findall(r"[a-z0-9]+", text) if w not in STOP_WORDS)


class RagIndex:
    """Index vectoriel unique : TF-IDF sur des morceaux de 3 à 5 lettres, similarité cosinus.

    Les morceaux de lettres rapprochent « remboursé » et « remboursement » sans dictionnaire.
    """

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("aucun document actif à indexer")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([normalize(f"{c.section} {c.text}") for c in chunks])

    def search(self, query: str, top_k: int = 3, min_score: float = 0.25) -> list[tuple[Chunk, float]]:
        query_vector = self.vectorizer.transform([normalize(query)])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        best = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in best if scores[i] >= min_score]