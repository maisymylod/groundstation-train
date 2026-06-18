"""Compact ops knowledge, mirroring the `groundstation` playbook corpus.

Used to render scenario context and to ground the rationale text in the
synthesised answers, so the model is trained to cite the playbook the same way
the agent graph does. Text is paraphrased and kept short; the authoritative
versions live in `groundstation/corpus/`.
"""
from __future__ import annotations

from ..schema import DRIFT, DROPOUT, NOMINAL, THERMAL_RUNAWAY

# citation tag -> short body. Tags match the groundstation corpus convention.
PLAYBOOKS: dict[str, dict] = {
    NOMINAL: {
        "citation": "ops-manual#Nominal",
        "symptoms": (
            "All channels track their orbital baseline within noise; no "
            "sustained trend on attitude, link, or thermal channels."
        ),
        "remediation": (
            "Hold at monitor with standard cadence. No command is warranted."
        ),
    },
    DRIFT: {
        "citation": "playbook-drift#Diagnosis steps",
        "symptoms": (
            "attitude_error_deg and gyro_rate_dps climb steadily away from "
            "zero, indicating an attitude-control drift rather than a transient."
        ),
        "remediation": (
            "Above the SEV-2 attitude-error threshold, plan reset_adcs to "
            "re-establish the control loop. Mild drift is held at monitor."
        ),
    },
    DROPOUT: {
        "citation": "playbook-dropout#Diagnosis steps",
        "symptoms": (
            "snr_db collapses while packet_loss_pct rises and downlink_mbps "
            "falls, a link-budget dropout rather than scheduled occultation."
        ),
        "remediation": (
            "Above the SEV-2 packet-loss threshold, plan failover_downlink to "
            "the secondary path. Marginal loss is held at monitor."
        ),
    },
    THERMAL_RUNAWAY: {
        "citation": "playbook-thermal-runaway#Diagnosis steps",
        "symptoms": (
            "temp_battery_c rises and accelerates, temp_payload_c follows, and "
            "bus_current_a climbs: a self-reinforcing heat-and-current loop."
        ),
        "remediation": (
            "An accelerating trend above the SEV-1 critical threshold is "
            "vehicle-threatening; plan enter_safe_mode to load-shed and point "
            "power-safe. Early mild warming is held at monitor."
        ),
    },
}


def citation(label: str) -> str:
    return PLAYBOOKS[label]["citation"]


def symptoms(label: str) -> str:
    return PLAYBOOKS[label]["symptoms"]


def remediation_text(label: str) -> str:
    return PLAYBOOKS[label]["remediation"]
