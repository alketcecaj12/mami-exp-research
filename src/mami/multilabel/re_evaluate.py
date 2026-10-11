"""Re-evaluate saved Task B responses without loading a model."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .prompts import LABELS
from .parsing import parse_response
from .evaluation import evaluate


def re_evaluate(input_csv, output_json=None):
    source = Path(input_csv)
    frame = pd.read_csv(source, keep_default_na=False)
    required = {"response"} | {"true_" + name for name in LABELS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing saved prediction columns: {sorted(missing)}")
    truth = frame[["true_" + name for name in LABELS]].copy()
    truth.columns = list(LABELS)
    for name in LABELS:
        truth[name] = pd.to_numeric(truth[name], errors="raise")
    predictions = [parse_response(response) for response in frame["response"]]
    result = evaluate(truth, predictions)
    destination = Path(output_json) if output_json else source.with_name(source.stem + "_reparsed_metrics.json")
    if destination.resolve() == source.resolve():
        raise ValueError("Output cannot overwrite source CSV")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing metrics: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result, destination


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate saved MAMI Task B raw CSV responses")
    parser.add_argument("input_csv")
    parser.add_argument("--output", help="New JSON metrics path; must not exist")
    args = parser.parse_args()
    result, destination = re_evaluate(args.input_csv, args.output)
    print(json.dumps(result, indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
