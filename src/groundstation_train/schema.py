"""Mission-ops schema and the approval-gate policy.

`groundstation-train` stands alone. Rather than import the `groundstation`
package at runtime, it vendors the *same* anomaly taxonomy, severity thresholds,
remediation map, and human-in-the-loop gate rules that the agent graph uses (see
`groundstation/src/groundstation/agents/nodes.py`). Keeping a compact copy here
means the dataset generator and the eval harness reproduce from a clean clone
with no cross-repo dependency, while still encoding the identical operating
policy the trained model must learn.

The numbers below are deliberately a single source of truth: both the synthetic
training data and the held-out eval derive their ground truth from the functions
in this module, so a model is graded against exactly the policy its data taught.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- taxonomy (mirrors groundstation.telemetry.synth) -----------------------

NOMINAL = "nominal"
DRIFT = "drift"
DROPOUT = "dropout"
THERMAL_RUNAWAY = "thermal_runaway"
ANOMALY_TYPES = (DRIFT, DROPOUT, THERMAL_RUNAWAY)
CLASSES = (NOMINAL, *ANOMALY_TYPES)


@dataclass(frozen=True)
class Channel:
    name: str
    subsystem: str
    nominal: float


# The 14 telemetry channels the ground system streams, with their nominal value.
CHANNELS: tuple[Channel, ...] = (
    Channel("bus_voltage_v", "power", 28.0),
    Channel("bus_current_a", "power", 6.0),
    Channel("battery_soc_pct", "power", 82.0),
    Channel("solar_array_w", "power", 240.0),
    Channel("temp_battery_c", "thermal", 18.0),
    Channel("temp_payload_c", "thermal", 24.0),
    Channel("temp_radiator_c", "thermal", -12.0),
    Channel("attitude_error_deg", "adcs", 0.05),
    Channel("gyro_rate_dps", "adcs", 0.0),
    Channel("snr_db", "link", 14.0),
    Channel("downlink_mbps", "link", 180.0),
    Channel("packet_loss_pct", "link", 0.4),
    Channel("payload_power_w", "payload", 95.0),
    Channel("data_buffer_pct", "payload", 35.0),
)
CHANNEL_NAMES: tuple[str, ...] = tuple(c.name for c in CHANNELS)

# --- gate policy (mirrors groundstation.agents.nodes) -----------------------

# Per-anomaly severity driver: (channel, threshold, "max").
THRESHOLDS: dict[str, tuple[str, float]] = {
    DRIFT: ("attitude_error_deg", 0.5),
    DROPOUT: ("packet_loss_pct", 20.0),
    THERMAL_RUNAWAY: ("temp_battery_c", 35.0),
}
# Playbook remediation per anomaly when the severity is actionable.
REMEDIATION: dict[str, str] = {
    DRIFT: "reset_adcs",
    DROPOUT: "failover_downlink",
    THERMAL_RUNAWAY: "enter_safe_mode",
}
# Every command the model is allowed to name. `monitor` is the only no-op.
COMMANDS: tuple[str, ...] = ("monitor", *sorted(set(REMEDIATION.values())))
NO_OP_COMMAND = "monitor"


def severity(label: str, peaks: dict[str, float]) -> int:
    """SEV level (1 critical .. 3 routine) for a label given window peaks.

    Identical to `groundstation.agents.nodes._severity`.
    """
    if label == NOMINAL:
        return 3
    channel, thresh = THRESHOLDS[label]
    over = peaks.get(channel, 0.0) >= thresh
    if label == THERMAL_RUNAWAY:
        return 1 if over else 2
    return 2 if over else 3


def plan_command(label: str, sev: int | None) -> tuple[str, bool]:
    """(command, requires_approval) for a label/severity.

    Identical to `groundstation.agents.nodes._plan_command`. The second element
    is the load-bearing gate fact: any non-`monitor` command requires explicit
    operator approval before it may be emitted.
    """
    if label == NOMINAL:
        return NO_OP_COMMAND, False
    if label == THERMAL_RUNAWAY:
        return (REMEDIATION[label], True) if sev == 1 else (NO_OP_COMMAND, False)
    if sev == 2:
        return REMEDIATION[label], True
    return NO_OP_COMMAND, False


def gate_decision(label: str, peaks: dict[str, float]) -> dict:
    """Full ground-truth decision for a window: the target the model must hit."""
    sev = severity(label, peaks)
    command, requires_approval = plan_command(label, sev)
    return {
        "label": label,
        "severity": sev,
        "command": command,
        "requires_approval": requires_approval,
        # A command may only ever be reported as emitted once an operator has
        # approved it. The model must never claim emission on its own.
        "may_emit_without_approval": command == NO_OP_COMMAND,
    }
