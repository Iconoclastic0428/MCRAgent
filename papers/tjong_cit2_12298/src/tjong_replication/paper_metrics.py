"""Paper-reported constants used by replication checks."""

from __future__ import annotations

PAPER_NAME = "Tjong CIT2.12298"

PAPER_REPORTED_SUPERVISED_METRICS = {
    "action_accuracy": 0.9463,
    "claim_accuracy": 0.9855,
    "discard_accuracy": 0.8151,
}

# Half of one unit in the last printed decimal place for fraction metrics.
DEFAULT_PAPER_METRIC_TOLERANCE = 0.00005
