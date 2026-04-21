from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_local_training_report(
    *,
    metrics: dict[str, Any],
    report_dir: Path,
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    report_files: dict[str, str] = {}
    history_path = report_dir / "history.csv"
    history_df = pd.DataFrame(metrics.get("history", []))
    history_df.to_csv(history_path, index=False)
    report_files["history_csv"] = str(history_path)

    dataset_counts_path = report_dir / "dataset_counts.csv"
    _build_dataset_counts_frame(metrics).to_csv(dataset_counts_path, index=False)
    report_files["dataset_counts_csv"] = str(dataset_counts_path)

    test_dataset_metrics_path = report_dir / "test_by_dataset.csv"
    _build_dataset_metric_frame(metrics.get("test_by_dataset", {})).to_csv(test_dataset_metrics_path, index=False)
    report_files["test_by_dataset_csv"] = str(test_dataset_metrics_path)

    val_dataset_metrics_path = report_dir / "validation_by_dataset.csv"
    _build_dataset_metric_frame(metrics.get("validation_by_dataset", {})).to_csv(val_dataset_metrics_path, index=False)
    report_files["validation_by_dataset_csv"] = str(val_dataset_metrics_path)

    summary_json_path = report_dir / "summary.json"
    summary_json_path.write_text(json.dumps(_build_summary_payload(metrics), indent=2), encoding="utf-8")
    report_files["summary_json"] = str(summary_json_path)

    pngs = {
        "training_curves_png": report_dir / "training_curves.png",
        "dataset_distribution_png": report_dir / "dataset_distribution.png",
        "overall_test_metrics_png": report_dir / "overall_test_metrics.png",
        "test_dataset_heatmap_png": report_dir / "test_dataset_heatmap.png",
        "validation_dataset_heatmap_png": report_dir / "validation_dataset_heatmap.png",
    }
    _plot_training_curves(history_df, pngs["training_curves_png"])
    _plot_dataset_distribution(_build_dataset_counts_frame(metrics), pngs["dataset_distribution_png"])
    _plot_overall_test_metrics(metrics, pngs["overall_test_metrics_png"])
    _plot_dataset_heatmap(metrics.get("test_by_dataset", {}), "test", pngs["test_dataset_heatmap_png"])
    _plot_dataset_heatmap(metrics.get("validation_by_dataset", {}), "validation", pngs["validation_dataset_heatmap_png"])
    report_files.update({name: str(path) for name, path in pngs.items()})

    html_path = report_dir / "index.html"
    html_path.write_text(_render_html(metrics, report_files), encoding="utf-8")
    report_files["index_html"] = str(html_path)
    return report_files


def _build_summary_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_duration_seconds": metrics.get("training_duration_seconds"),
        "size_profile": metrics.get("size_profile"),
        "thresholds": metrics.get("thresholds"),
        "severity_accuracy": metrics.get("severity_accuracy"),
        "severity_macro_f1": metrics.get("severity_macro_f1"),
        "dataset": metrics.get("dataset"),
        "category_metrics": metrics.get("category_metrics"),
        "auxiliary_metrics": metrics.get("auxiliary_metrics"),
        "severity_metrics": metrics.get("severity_metrics"),
    }


def _build_dataset_counts_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split = metrics.get("dataset", {}).get(split_name, {})
        for dataset_name, count in split.get("dataset_counts", {}).items():
            rows.append({"split": split_name, "dataset": dataset_name, "count": count})
    return pd.DataFrame(rows)


def _build_dataset_metric_frame(metrics_by_dataset: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, entry in metrics_by_dataset.items():
        evaluation = entry.get("evaluation", {})
        scalar_metrics = evaluation.get("scalar_metrics", {})
        row = {
            "dataset": dataset_name,
            "sample_count": entry.get("sample_count", 0),
            "loss_total": scalar_metrics.get("loss_total"),
            "severity_accuracy": scalar_metrics.get("severity/accuracy"),
            "severity_macro_f1": scalar_metrics.get("severity/macro_f1"),
        }
        for head_name in ("approval", "destination", "simulation", "address_malicious", "failure_aux"):
            row[f"{head_name}_f1"] = scalar_metrics.get(f"{head_name}/f1")
            row[f"{head_name}_accuracy"] = scalar_metrics.get(f"{head_name}/accuracy")
            row[f"{head_name}_precision"] = scalar_metrics.get(f"{head_name}/precision")
            row[f"{head_name}_recall"] = scalar_metrics.get(f"{head_name}/recall")
            row[f"{head_name}_support"] = scalar_metrics.get(f"{head_name}/support")
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_training_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if history_df.empty:
        for axis in axes.ravel():
            axis.text(0.5, 0.5, "No history available", ha="center", va="center")
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    epochs = history_df["epoch"]
    axes[0, 0].plot(epochs, history_df.get("train/loss_total"), label="train_loss")
    axes[0, 0].plot(epochs, history_df.get("val/loss_total"), label="val_loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    for column in ("val/approval/f1", "val/destination/f1", "val/simulation/f1"):
        if column in history_df:
            axes[0, 1].plot(epochs, history_df[column], label=column.split("/")[-2])
    axes[0, 1].set_title("Validation Main-Head F1")
    axes[0, 1].legend()

    for column in ("train/loss_approval", "train/loss_destination", "train/loss_simulation", "train/loss_severity"):
        if column in history_df:
            axes[1, 0].plot(epochs, history_df[column], label=column.replace("train/loss_", ""))
    axes[1, 0].set_title("Per-Head Training Loss")
    axes[1, 0].legend()

    for column in ("val/severity/accuracy", "val/severity/macro_f1", "train/grad_norm"):
        if column in history_df:
            axes[1, 1].plot(epochs, history_df[column], label=column)
    axes[1, 1].set_title("Severity / Optimization")
    axes[1, 1].legend()

    for axis in axes.ravel():
        axis.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_dataset_distribution(dataset_counts_df: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    if dataset_counts_df.empty:
        axis.text(0.5, 0.5, "No dataset counts available", ha="center", va="center")
        axis.axis("off")
    else:
        sns.barplot(data=dataset_counts_df, x="dataset", y="count", hue="split", ax=axis)
        axis.set_title("Dataset Distribution by Split")
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_overall_test_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    test_main = metrics.get("test_metrics", {}).get("binary_metrics", {}).get("main", {})
    for head_name, payload in test_main.items():
        for metric_name in ("accuracy", "precision", "recall", "f1", "balanced_accuracy"):
            value = payload.get(metric_name)
            if value is None:
                continue
            rows.append({"head": head_name, "metric": metric_name, "value": value})

    fig, axis = plt.subplots(figsize=(12, 6))
    if not rows:
        axis.text(0.5, 0.5, "No test metrics available", ha="center", va="center")
        axis.axis("off")
    else:
        frame = pd.DataFrame(rows)
        sns.barplot(data=frame, x="head", y="value", hue="metric", ax=axis)
        axis.set_ylim(0, 1.05)
        axis.set_title("Overall Test Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_dataset_heatmap(metrics_by_dataset: dict[str, Any], split_name: str, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for dataset_name, entry in metrics_by_dataset.items():
        scalar_metrics = entry.get("evaluation", {}).get("scalar_metrics", {})
        row = {
            "dataset": dataset_name,
            "approval_f1": scalar_metrics.get("approval/f1"),
            "destination_f1": scalar_metrics.get("destination/f1"),
            "simulation_f1": scalar_metrics.get("simulation/f1"),
            "address_malicious_f1": scalar_metrics.get("address_malicious/f1"),
            "failure_aux_f1": scalar_metrics.get("failure_aux/f1"),
            "severity_accuracy": scalar_metrics.get("severity/accuracy"),
        }
        rows.append(row)

    fig, axis = plt.subplots(figsize=(10, max(3, 1.2 * max(len(rows), 1))))
    if not rows:
        axis.text(0.5, 0.5, f"No {split_name} dataset metrics available", ha="center", va="center")
        axis.axis("off")
    else:
        frame = pd.DataFrame(rows).set_index("dataset").fillna(0.0)
        sns.heatmap(frame, annot=True, fmt=".3f", cmap="Blues", vmin=0.0, vmax=1.0, ax=axis)
        axis.set_title(f"{split_name.title()} Metrics by Dataset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_html(metrics: dict[str, Any], report_files: dict[str, str]) -> str:
    duration = metrics.get("training_duration_seconds")
    summary_items = [
        ("Size profile", metrics.get("size_profile")),
        ("Training duration (s)", duration),
        ("Severity accuracy", metrics.get("severity_accuracy")),
        ("Severity macro F1", metrics.get("severity_macro_f1")),
        ("Log path", metrics.get("log_path")),
        ("JSONL log path", metrics.get("jsonl_log_path")),
    ]
    images = [
        ("Training curves", "training_curves_png"),
        ("Dataset distribution", "dataset_distribution_png"),
        ("Overall test metrics", "overall_test_metrics_png"),
        ("Test dataset heatmap", "test_dataset_heatmap_png"),
        ("Validation dataset heatmap", "validation_dataset_heatmap_png"),
    ]
    links = [
        ("History CSV", "history_csv"),
        ("Dataset counts CSV", "dataset_counts_csv"),
        ("Validation-by-dataset CSV", "validation_by_dataset_csv"),
        ("Test-by-dataset CSV", "test_by_dataset_csv"),
        ("Summary JSON", "summary_json"),
    ]

    summary_html = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary_items
    )
    image_html = "".join(
        f"<section><h2>{html.escape(title)}</h2><img src='{Path(report_files[key]).name}' alt='{html.escape(title)}'></section>"
        for title, key in images
    )
    link_html = "".join(
        f"<li><a href='{Path(report_files[key]).name}'>{html.escape(title)}</a></li>"
        for title, key in links
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ChainSentry Training Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; margin-bottom: 24px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>ChainSentry Local Training Report</h1>
  <section>
    <h2>Summary</h2>
    <table>{summary_html}</table>
  </section>
  <section>
    <h2>Artifacts</h2>
    <ul>{link_html}</ul>
  </section>
  {image_html}
</body>
</html>
"""
