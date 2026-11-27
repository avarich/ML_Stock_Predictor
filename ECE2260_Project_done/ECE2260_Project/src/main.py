"""Run the full stock-movement experiment (README step 8).

Usage (from the project root):
    python -m src.main                          # NVDA, data/raw/NVDA.csv
    python -m src.main --ticker NVDA --csv data/raw/NVDA.csv --train-fraction 0.8
"""

import argparse
from pathlib import Path

from src.data import load_stock_data, validate_stock_data
from src.evaluate import (
    accuracy_by_period,
    evaluate_predictions,
    format_metrics,
    misclassified_examples,
    save_confusion_matrix_plot,
    save_period_accuracy_plot,
)
from src.features import FEATURE_COLUMNS, build_dataset
from src.train import (
    build_models,
    chronological_split,
    cross_validate_train,
    get_xy,
    train_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment(
    ticker="NVDA", csv_path=None, train_fraction=0.8, output_dir=None, save_outputs=True
) -> dict:
    """Load data, build features, split, train, evaluate. Returns the results."""
    csv_path = (
        Path(csv_path) if csv_path else PROJECT_ROOT / "data" / "raw" / f"{ticker}.csv"
    )
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs"
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    # --- data (steps 1-3) ---
    raw = load_stock_data(csv_path)
    report = validate_stock_data(raw)
    log(f"=== {ticker} next-day direction experiment ===")
    log(f"Data file     : {csv_path.name}")
    log(
        f"Raw data range: {report['start'].date()} to {report['end'].date()} "
        f"({report['rows']} trading days)"
    )
    log(f"Duplicates dropped: {report['duplicates_dropped']}")
    if report["long_gaps"]:
        log(f"Gaps > 4 calendar days before: {report['long_gaps']}")
    else:
        log("No unusual gaps in dates (only normal weekends/holidays).")

    # --- target and features (step 4) ---
    data = build_dataset(raw)
    log(
        f"Usable rows after dropping incomplete ones: {len(data)} "
        f"({data['Date'].iloc[0].date()} to {data['Date'].iloc[-1].date()})"
    )
    log(f"Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")

    # --- chronological split (step 5) ---
    train_df, test_df = chronological_split(data, train_fraction)
    log(
        f"Train: {train_df['Date'].iloc[0].date()} to "
        f"{train_df['Date'].iloc[-1].date()} "
        f"({len(train_df)} rows)"
    )
    log(
        f"Test : {test_df['Date'].iloc[0].date()} to {test_df['Date'].iloc[-1].date()} "
        f"({len(test_df)} rows)"
    )
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)
    log(f"Share of 'up' days: train {y_train.mean():.3f}, test {y_test.mean():.3f}")

    # --- models (step 6) and evaluation (step 7) ---
    log()
    log("Time-series cross-validation on the TRAINING period only (balanced accuracy):")
    results, period_tables = {}, {}
    for name, model in build_models().items():
        cv_score = cross_validate_train(model, X_train, y_train)
        log(f"  {name:26s} {cv_score:.3f}")
        fitted = train_model(model, X_train, y_train)
        y_pred = fitted.predict(X_test)
        y_proba = (
            fitted.predict_proba(X_test)[:, 1]
            if hasattr(fitted, "predict_proba")
            else None
        )
        results[name] = {
            "cv_balanced_accuracy": cv_score,
            "metrics": evaluate_predictions(y_test, y_pred),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        period_tables[name] = accuracy_by_period(test_df, y_pred)

    log()
    log("Held-out TEST results (test set was not used to choose anything):")
    for name, res in results.items():
        log(format_metrics(name, res["metrics"]))
        log()

    baseline_acc = results["baseline (most frequent)"]["metrics"]["accuracy"]
    log(
        "Baseline (always predict the most common training class) "
        f"accuracy: {baseline_acc:.3f}"
    )
    for name, res in results.items():
        if name.startswith("baseline"):
            continue
        diff = res["metrics"]["accuracy"] - baseline_acc
        log(f"  {name}: {diff:+.3f} accuracy vs baseline")

    log()
    log("Accuracy by quarter of the test period:")
    for name, table in period_tables.items():
        log(f"  {name}")
        for _, row in table.iterrows():
            log(f"    {row['Period']}: {row['accuracy']:.3f} ({int(row['days'])} days)")

    best = max(
        (n for n in results if not n.startswith("baseline")),
        key=lambda n: results[n]["cv_balanced_accuracy"],
    )
    log()
    log(f"Most confident mistakes of {best} (picked by training CV, not test score):")
    wrong = misclassified_examples(
        test_df.reset_index(drop=True),
        results[best]["y_pred"],
        results[best]["y_proba"],
        n=8,
    )
    log(wrong.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- save charts and report (step 8) ---
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, res in results.items():
            slug = name.split(" (")[0].replace(" ", "_")
            save_confusion_matrix_plot(
                res["metrics"]["confusion_matrix"],
                name,
                output_dir / f"{ticker}_confusion_{slug}.png",
            )
        save_period_accuracy_plot(
            period_tables, output_dir / f"{ticker}_accuracy_by_quarter.png"
        )
        (output_dir / f"{ticker}_report.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        log()
        log(f"Saved charts and report to {output_dir}")

    return {"results": results, "train": train_df, "test": test_df, "best_model": best}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Next-day stock direction classifier")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--csv", default=None, help="path to the price CSV")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    run_experiment(args.ticker, args.csv, args.train_fraction, args.output_dir)


if __name__ == "__main__":
    main()
