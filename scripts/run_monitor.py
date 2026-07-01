from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from configs.global_config import get_monitor_config
from src.emailurgency.pipelines.monitoring import ModelMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monitoring for email urgency predictions")
    parser.add_argument("--input", type=Path, default=Path("data/eval.parquet"), help="Path to evaluation parquet file")
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/monitor_snapshot.json"), help="Where to write the monitoring snapshot")
    parser.add_argument("--review-output", type=Path, default=Path("artifacts/monitoring/review_queue.csv"), help="Where to write the review queue")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline parquet file to seed the monitor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_monitor_config()

    if not args.input.exists():
        raise FileNotFoundError(f"Monitoring input file not found: {args.input}")

    df = pd.read_parquet(args.input, engine="pyarrow")
    if df.empty:
        raise ValueError("Monitoring input file is empty")

    if "proba" not in df.columns or "pred" not in df.columns:
        raise ValueError("Input file must contain 'proba' and 'pred' columns")

    monitor = ModelMonitor(config=config)

    baseline_df = None
    if args.baseline and args.baseline.exists():
        baseline_df = pd.read_parquet(args.baseline, engine="pyarrow")
    elif "baseline" in df.columns:
        baseline_df = df

    if baseline_df is not None and "proba" in baseline_df.columns:
        monitor.set_baseline(
            predictions=baseline_df["pred"].fillna(0).astype(int).to_numpy(),
            probabilities=baseline_df["proba"].fillna(0.0).astype(float).to_numpy(),
            topics=baseline_df["topic"].fillna(-1).astype(int).to_numpy() if "topic" in baseline_df.columns else None,
            topic_probabilities=baseline_df["topic_probability"].fillna(0.0).astype(float).to_numpy() if "topic_probability" in baseline_df.columns else None,
        )
    else:
        monitor.set_baseline(
            predictions=df["pred"].fillna(0).astype(int).to_numpy()[: max(100, config.min_samples)],
            probabilities=df["proba"].fillna(0.0).astype(float).to_numpy()[: max(100, config.min_samples)],
            topics=df["topic"].fillna(-1).astype(int).to_numpy()[: max(100, config.min_samples)] if "topic" in df.columns else None,
            topic_probabilities=df["topic_probability"].fillna(0.0).astype(float).to_numpy()[: max(100, config.min_samples)] if "topic_probability" in df.columns else None,
        )

    snapshot = monitor.monitor(
        predictions=df["pred"].fillna(0).astype(int).to_numpy(),
        probabilities=df["proba"].fillna(0.0).astype(float).to_numpy(),
        topics=df["topic"].fillna(-1).astype(int).to_numpy() if "topic" in df.columns else None,
        topic_probabilities=df["topic_probability"].fillna(0.0).astype(float).to_numpy() if "topic_probability" in df.columns else None,
    )

    review_items = monitor.build_review_queue(
        df=df,
        predictions=df["pred"].fillna(0).astype(int).to_numpy(),
        probabilities=df["proba"].fillna(0.0).astype(float).to_numpy(),
        topics=df["topic"].fillna(-1).astype(int).to_numpy() if "topic" in df.columns else None,
        topic_probabilities=df["topic_probability"].fillna(0.0).astype(float).to_numpy() if "topic_probability" in df.columns else None,
        text_column="body",
        row_id_column="message_id",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        fh.write(snapshot.model_dump_json(indent=2))

    if review_items:
        review_df = pd.DataFrame([item.model_dump() for item in review_items])
        review_df.to_csv(args.review_output, index=False)
    else:
        pd.DataFrame(columns=["row_id", "text", "predicted_label", "predicted_probability", "topic", "topic_probability", "reason", "drift_score"]).to_csv(args.review_output, index=False)

    print(f"Monitoring snapshot written to {args.output}")
    print(f"Review queue written to {args.review_output}")


if __name__ == "__main__":
    main()
