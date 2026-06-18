from groundstation_train.train import registry


def test_register_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(registry.config, "ARTIFACTS_DIR", tmp_path)
    entry = registry.register(
        stage="sft",
        checkpoint_dir=str(tmp_path / "ckpt"),
        base_model="test/base",
        metrics={"ops_decision_accuracy": 0.9},
    )
    assert entry["stage"] == "sft"
    assert entry["base_model"] == "test/base"
    hist = registry.history()
    assert len(hist) == 1
    assert hist[0]["metrics"]["ops_decision_accuracy"] == 0.9


def test_register_with_wandb_run_is_resilient(tmp_path, monkeypatch, capsys):
    # Passing a wandb_run with no wandb installed must not fail the run; the
    # artifact step is best-effort and degrades to a printed notice.
    monkeypatch.setattr(registry.config, "ARTIFACTS_DIR", tmp_path)
    entry = registry.register(
        stage="dpo",
        checkpoint_dir=str(tmp_path / "ckpt"),
        base_model="test/base",
        wandb_run=object(),
    )
    assert entry["stage"] == "dpo"


def test_history_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(registry.config, "ARTIFACTS_DIR", tmp_path)
    assert registry.history() == []
