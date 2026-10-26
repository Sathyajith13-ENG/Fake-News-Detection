"""Train and evaluate the fake news classifier.

    python -m scripts.train_model --data data/news.csv
"""
from __future__ import annotations

import argparse
import sys

from fake_news_detection.config import get_settings
from fake_news_detection.dataset import DatasetError, load_dataset
from fake_news_detection.model import save_model, train_and_evaluate


def format_report(ev: dict) -> str:
    lab = ev["labels"]
    lines = [
        f"Rows: train={ev['n_train']}  test={ev['n_test']}  (test_size={ev['test_size']})",
        f"Accuracy:        {ev['accuracy']:.3f}",
        f"Macro precision: {ev['macro_precision']:.3f}",
        f"Macro recall:    {ev['macro_recall']:.3f}",
        f"Macro F1:        {ev['macro_f1']:.3f}",
        "",
        f"{'class':<8}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for c in lab:
        m = ev["per_class"][c]
        lines.append(f"{c:<8}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10}")
    cm = ev["confusion_matrix"]
    lines += ["", "Confusion matrix (rows = true, columns = predicted)", " " * 8 + "".join(f"{c:>8}" for c in lab)]
    lines += [f"{c:<8}" + "".join(f"{v:>8}" for v in row) for c, row in zip(lab, cm)]
    lines += ["", "Metrics come from this dataset and split only; they are not a general accuracy claim."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(s.data_path))
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--model-out", default=str(s.model_path))
    a = ap.parse_args(argv)
    try:
        df = load_dataset(a.data, a.text_column, a.label_column)
        pipe, ev = train_and_evaluate(df, a.test_size)
    except DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(format_report(ev))
    print(f"\nSaved model to {save_model(pipe, ev, a.model_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
