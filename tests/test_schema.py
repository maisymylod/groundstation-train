from groundstation_train import schema


def test_nominal_is_monitor_no_approval():
    d = schema.gate_decision(schema.NOMINAL, {})
    assert d["command"] == schema.NO_OP_COMMAND
    assert d["requires_approval"] is False
    assert d["may_emit_without_approval"] is True


def test_thermal_above_threshold_is_sev1_safe_mode_gated():
    peaks = {"temp_battery_c": 40.0}  # over the 35 C critical threshold
    d = schema.gate_decision(schema.THERMAL_RUNAWAY, peaks)
    assert d["severity"] == 1
    assert d["command"] == "enter_safe_mode"
    assert d["requires_approval"] is True
    assert d["may_emit_without_approval"] is False


def test_thermal_below_threshold_holds_at_monitor():
    peaks = {"temp_battery_c": 22.0}  # warming but under threshold
    d = schema.gate_decision(schema.THERMAL_RUNAWAY, peaks)
    assert d["severity"] == 2
    assert d["command"] == schema.NO_OP_COMMAND
    assert d["requires_approval"] is False


def test_drift_and_dropout_action_above_threshold():
    drift = schema.gate_decision(schema.DRIFT, {"attitude_error_deg": 0.9})
    assert drift["command"] == "reset_adcs" and drift["requires_approval"] is True
    dropout = schema.gate_decision(schema.DROPOUT, {"packet_loss_pct": 40.0})
    assert dropout["command"] == "failover_downlink" and dropout["requires_approval"] is True


def test_drift_and_dropout_monitor_below_threshold():
    drift = schema.gate_decision(schema.DRIFT, {"attitude_error_deg": 0.2})
    assert drift["command"] == schema.NO_OP_COMMAND
    dropout = schema.gate_decision(schema.DROPOUT, {"packet_loss_pct": 5.0})
    assert dropout["command"] == schema.NO_OP_COMMAND


def test_only_monitor_may_emit_without_approval():
    # Any command that requires approval must not be emittable without it.
    for label in schema.CLASSES:
        for peaks in ({"temp_battery_c": 40.0, "attitude_error_deg": 0.9, "packet_loss_pct": 40.0}, {}):
            d = schema.gate_decision(label, peaks)
            if d["command"] != schema.NO_OP_COMMAND:
                assert d["requires_approval"] is True
                assert d["may_emit_without_approval"] is False
