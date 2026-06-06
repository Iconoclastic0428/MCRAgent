"""Plot supervised training progress from JSON epoch logs or metrics files."""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Iterable


def _json_objects_from_text(text: str) -> Iterable[dict]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def load_epoch_metrics(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw.count(b"\x00") > max(4, len(raw) // 20):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and isinstance(value.get("epochs"), list):
            return [epoch for epoch in value["epochs"] if isinstance(epoch, dict) and "epoch" in epoch]
    epochs = [obj for obj in _json_objects_from_text(text) if "epoch" in obj]
    return sorted(epochs, key=lambda item: int(item["epoch"]))


def _example_count(epoch: dict) -> int:
    counts = [
        int(epoch.get(key) or 0)
        for key in ("examples", "example_count", "action_count", "claim_count", "discard_count")
    ]
    return max(counts)


def _x_values(epochs: list[dict], *, x_axis: str, batch_size: int) -> list[int]:
    if x_axis == "epoch":
        return [int(epoch["epoch"]) for epoch in epochs]
    if batch_size <= 0:
        raise ValueError("batch size must be positive for batch x-axis")
    return [int(epoch["epoch"]) * math.ceil(_example_count(epoch) / batch_size) for epoch in epochs]


def summarize_epochs(epochs: list[dict], *, x_axis: str = "epoch", batch_size: int = 1024) -> dict:
    if not epochs:
        return {"epochs": 0}
    latest = epochs[-1]
    summary = {
        "epochs": len(epochs),
        "latest_epoch": int(latest["epoch"]),
        "latest_action_accuracy": latest.get("action_accuracy"),
        "latest_claim_accuracy": latest.get("claim_accuracy"),
        "latest_discard_accuracy": latest.get("discard_accuracy"),
        "latest_decision_loss": latest.get("decision_loss"),
        "latest_optimization_loss": latest.get("optimization_loss"),
        "x_axis": x_axis,
    }
    if x_axis == "batches":
        summary["batch_size"] = batch_size
        summary["latest_cumulative_batches"] = _x_values(epochs, x_axis=x_axis, batch_size=batch_size)[-1]
    return summary


def plot_epochs(epochs: list[dict], out_path: Path, *, title: str, x_axis: str, batch_size: int) -> None:
    if not epochs:
        raise ValueError("no epoch metrics found")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = _x_values(epochs, x_axis=x_axis, batch_size=batch_size)
    x_label = "epoch" if x_axis == "epoch" else f"cumulative optimizer batches (batch size {batch_size})"
    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for key, label in (
        ("action_accuracy", "action"),
        ("claim_accuracy", "claim"),
        ("discard_accuracy", "discard"),
    ):
        values = [epoch.get(key) for epoch in epochs]
        if any(value is not None for value in values):
            ax_acc.plot(xs, values, marker="o", linewidth=1.8, label=label)
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(loc="best")
    ax_acc.set_title(title)

    for key, label in (
        ("optimization_loss", "optimization"),
        ("decision_loss", "decision"),
    ):
        values = [epoch.get(key) for epoch in epochs]
        if any(value is not None for value in values):
            ax_loss.plot(xs, values, marker="o", linewidth=1.8, label=label)
    ax_loss.set_xlabel(x_label)
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--title", default="Tjong supervised training progress")
    parser.add_argument("--x-axis", choices=("epoch", "batches"), default="epoch")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    epochs = load_epoch_metrics(args.in_path)
    plot_epochs(epochs, args.out, title=args.title, x_axis=args.x_axis, batch_size=args.batch_size)
    summary = summarize_epochs(epochs, x_axis=args.x_axis, batch_size=args.batch_size)
    summary.update({"input": str(args.in_path), "output": str(args.out)})
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
