import argparse
import json
from pathlib import Path
import pandas as pd
from .config import load_config
from .data import load_splits
from .metrics import evaluate


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description="Simple MAMI research commands")
    parser.add_argument("command", choices=["inspect", "download", "baseline", "predict", "vlm", "train"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, help="Smoke test only; omit for a complete run")
    parser.add_argument("--image")
    parser.add_argument("--text", default="")
    parser.add_argument("--mode", choices=["multimodal", "image", "text"], default="multimodal")
    parser.add_argument("--adapter", help="Path to trained LoRA adapter")
    parser.add_argument("--resume", help="Training checkpoint .pt file created by this version")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["experiment"]["output"])
    out.mkdir(parents=True, exist_ok=True)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "download":
        from huggingface_hub import snapshot_download, HfApi
        info = HfApi().model_info(cfg["model"]["id"], revision=cfg["model"]["revision"])
        snapshot_download(repo_id=cfg["model"]["id"], revision=info.sha, local_dir=cfg["model"]["local_dir"],
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE*"])
        write_json(Path(cfg["model"]["local_dir"]) / "model_info.json", {"id": info.id, "commit": info.sha})
        return
    if args.command == "predict":
        if not args.image:
            parser.error("predict requires --image")
        from .vlm import Classifier
        print(json.dumps(Classifier(cfg, args.adapter).predict(args.image, args.text, args.mode), indent=2))
        return
    splits, report = load_splits(cfg)
    write_json(out / "data_report.json", report)
    if args.command == "inspect":
        print(json.dumps(report, indent=2))
        return
    if args.command == "train":
        from .train import train_lora
        train = splits["train"]
        if args.limit:
            train = train.head(args.limit)
        train_lora(cfg, train, resume=args.resume)
        write_json(out / "lora_adapter" / "run_config.json", cfg)
        return
    frame = splits[args.split]
    if args.limit:
        frame = frame.head(args.limit)
    if args.command == "baseline":
        from .baseline import fit_baselines
        systems = fit_baselines(splits["train"], cfg["experiment"]["seed"])
        results = {name: model.predict(frame.text).tolist() for name, model in systems.items()}
    else:
        from .vlm import Classifier
        model = Classifier(cfg, args.adapter)
        rows = []
        for i, row in enumerate(frame.itertuples()):
            prediction = model.predict(row.image_path, row.text, args.mode)
            rows.append({"file_name": row.file_name, "label": row.label, **prediction})
            # Save every completed prediction so a failure does not erase the run.
            name = ("lora" if args.adapter else "zero_shot") + "_" + args.mode
            pd.DataFrame(rows).to_csv(out / f"{name}_{args.split}_raw.csv", index=False)
            print(f"{i + 1}/{len(frame)} {row.file_name}: {prediction['response']}", flush=True)
        results = {name: [row["prediction"] for row in rows]}
    summary = {}
    for name, predictions in results.items():
        table = frame[["file_name", "label"]].copy()
        table["prediction"] = predictions
        table.to_csv(out / f"{name}_{args.split}.csv", index=False)
        summary[name] = evaluate(frame.label, predictions)
    write_json(out / f"{args.command}_{args.split}_metrics.json", summary)
    write_json(out / f"{args.command}_{args.split}_config.json", {"config": cfg, "mode": args.mode, "adapter": args.adapter, "limit": args.limit})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
