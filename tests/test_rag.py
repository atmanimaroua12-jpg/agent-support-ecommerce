import pytest

from agent_support.rag import RagIndex, load_chunks, parse_document

CORPUS = "data/corpus"


@pytest.fixture(scope="module")
def index():
    chunks, _ = load_chunks(CORPUS)
    return RagIndex(chunks)


def test_document_obsolete_exclu():
    chunks, report = load_chunks(CORPUS)
    assert all(c.doc_id != "POL-RET-2024" for c in chunks)
    obsolete = [r for r in report if r["doc_id"] == "POL-RET-2024"]
    assert obsolete and obsolete[0]["indexe"] is False


def test_quatre_documents_actifs_indexes():
    chunks, report = load_chunks(CORPUS)
    assert sum(r["indexe"] for r in report) == 4
    assert {c.doc_id for c in chunks} == {"POL-RET", "GUI-LIV", "POL-GAR", "FAQ-GEN"}


def test_chaque_chunk_garde_sa_provenance():
    chunks, _ = load_chunks(CORPUS)
    for c in chunks:
        assert c.doc_id and c.section and c.version and c.text


def test_metadonnees_manquantes_refusees(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("---\ndoc_id: X\ntitre: Test\n---\n## Section\nTexte", encoding="utf-8")
    with pytest.raises(ValueError, match="métadonnées manquantes"):
        parse_document(doc)


@pytest.mark.parametrize("question, doc_id, section", [
    ("Combien de jours ai-je pour retourner un article ?", "POL-RET", "Délai de retour"),
    ("Combien coûte la livraison express ?", "GUI-LIV", "Livraison express"),
    ("Quelle est la durée de la garantie ?", "POL-GAR", "Durée de la garantie"),
    ("Quand serai-je remboursé ?", "POL-RET", "Remboursement"),
])
def test_bon_passage_en_premier(index, question, doc_id, section):
    best_chunk, _ = index.search(question)[0]
    assert (best_chunk.doc_id, best_chunk.section) == (doc_id, section)


def test_reponse_actuelle_30_jours(index):
    best_chunk, _ = index.search("délai pour retourner un article")[0]
    assert "30 jours" in best_chunk.text


@pytest.mark.parametrize("question", ["Livrez-vous en France ?", "Quel temps fait-il demain ?"])
def test_question_hors_corpus_sans_resultat(index, question):
    assert index.search(question) == []