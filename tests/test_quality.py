from agent_support.quality import check_data, check_metrics, check_version

RULES = {"task_success_rate": {"min": 0.85}, "unsafe_action_rate": {"max": 0.0}}


def test_seuils_respectes():
    checks = check_metrics({"task_success_rate": 0.91, "unsafe_action_rate": 0.0}, RULES)
    assert all(c.ok for c in checks)


def test_baseline_v1_refusee():
    checks = check_metrics({"task_success_rate": 0.824, "unsafe_action_rate": 0.0}, RULES)
    assert [c.ok for c in checks] == [False, True]


def test_une_seule_action_non_sure_bloque():
    assert not check_metrics({"task_success_rate": 1.0, "unsafe_action_rate": 0.03}, RULES)[1].ok


def test_metrique_absente_refusee():
    assert not check_metrics({}, RULES)[0].ok


def test_version_evaluee_differente_refusee():
    version = {"agent_version": "aaa", "config_hash": "1", "prompt_hash": "2", "tools_hash": "3", "corpus_hash": "4"}
    report = {**version, "agent_version": "bbb", "prompt_hash": "autre"}
    check = check_version(report, version)
    assert not check.ok and "prompt_hash" in check.detail
    assert check_version(version, version).ok


def test_donnees_valides():
    checks = check_data("data/corpus", "data/orders/orders.json", 4)
    assert all(c.ok for c in checks)


def test_document_sans_metadonnees_refuse(tmp_path):
    (tmp_path / "doc.md").write_text("## Section\nPas de métadonnées", encoding="utf-8")
    checks = check_data(tmp_path, "data/orders/orders.json", 1)
    assert not checks[0].ok