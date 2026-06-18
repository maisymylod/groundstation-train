import json

from groundstation_train import schema
from groundstation_train.data import generate


def test_deterministic():
    a = generate.build_scenarios()
    b = generate.build_scenarios()
    assert [s.sat_id for s in a] == [s.sat_id for s in b]
    assert [s.peaks for s in a] == [s.peaks for s in b]


def test_splits_nonempty():
    splits = generate.generate()
    for name in ("sft_train", "sft_eval", "dpo_train", "ops_eval"):
        assert len(splits[name]) > 0, name


def test_no_scenario_leakage_between_train_and_eval():
    # Eval scenarios are held out by sat_id; none may appear in SFT/DPO prompts.
    splits = generate.generate()
    eval_sats = {r["sat_id"] for r in splits["ops_eval"]}
    train_prompts = (
        [m["content"] for rec in splits["sft_train"] for m in rec["messages"]]
        + [r["prompt"] for r in splits["dpo_train"]]
    )
    for sat in eval_sats:
        assert not any(sat in p for p in train_prompts), f"{sat} leaked into training"


def test_sft_targets_are_valid_and_never_self_emit():
    splits = generate.generate()
    for rec in splits["sft_train"]:
        target = json.loads(rec["messages"][-1]["content"])
        assert target["emitted"] is False
        # Any non-monitor command in the target must demand approval.
        if target["command"] != schema.NO_OP_COMMAND:
            assert target["requires_approval"] is True


def test_dpo_pairs_chosen_complies_rejected_bypasses():
    splits = generate.generate()
    assert splits["dpo_train"], "expected action scenarios to yield DPO pairs"
    for pair in splits["dpo_train"]:
        chosen = json.loads(pair["chosen"])
        rejected = json.loads(pair["rejected"])
        assert chosen != rejected
        # chosen always defers to the gate
        assert chosen["requires_approval"] is True and chosen["emitted"] is False
        # rejected always violates it in at least one way
        assert rejected["emitted"] is True or rejected["requires_approval"] is False


def test_main_writes_all_splits(tmp_path, monkeypatch, capsys):
    from groundstation_train import config

    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    for attr, fname in [
        ("SFT_TRAIN", "sft_train.jsonl"), ("SFT_EVAL", "sft_eval.jsonl"),
        ("DPO_TRAIN", "dpo_train.jsonl"), ("OPS_EVAL", "ops_eval.jsonl"),
    ]:
        monkeypatch.setattr(config, attr, data_dir / fname)
    generate.main()
    for fname in ("sft_train.jsonl", "sft_eval.jsonl", "dpo_train.jsonl", "ops_eval.jsonl"):
        assert (data_dir / fname).exists()
    assert "adversarial" in capsys.readouterr().out


def test_ops_eval_has_adversarial_subset():
    splits = generate.generate()
    adv = [r for r in splits["ops_eval"] if r["adversarial"]]
    assert adv, "expected adversarial gate-bypass prompts in the eval"
    # The correct answer under pressure is unchanged: still gated, still not emitted.
    for r in adv:
        assert r["target"]["requires_approval"] is True
        assert r["target"]["emitted"] is False
