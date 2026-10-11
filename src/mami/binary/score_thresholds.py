"""Run from the mami_research root using its existing virtual environment."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relative_score(log0, log1):
    margin = log1 - log0
    if not math.isfinite(margin):
        raise ValueError("Non-finite label score; stopping without classifying it.")
    if margin >= 0:
        return 1 / (1 + math.exp(-margin))
    e = math.exp(margin)
    return e / (1 + e)


def threshold_metrics(rows, threshold):
    tn = fp = fn = tp = 0
    for row in rows:
        y = int(row['label'])
        pred = float(row['score_1']) >= threshold
        if y == 1:
            tp += int(pred)
            fn += int(not pred)
        else:
            fp += int(pred)
            tn += int(not pred)
    def divide(a, b):
        return a / b if b else None
    f1pos = divide(2 * tp, 2 * tp + fp + fn) or 0.0
    f1neg = divide(2 * tn, 2 * tn + fp + fn) or 0.0
    return dict(threshold=threshold, n=len(rows), true_negatives=tn,
                false_positives=fp, false_negatives=fn, true_positives=tp,
                precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                false_positive_rate=divide(fp, fp + tn),
                accuracy=divide(tp + tn, len(rows)), macro_f1=(f1pos + f1neg) / 2)


def label_loglikelihood(classifier, messages, prefix, label):
    """Sum conditional log probabilities for label tokens plus <|im_end|>."""
    torch = classifier.torch
    full = classifier.encode(messages + [{"role": "assistant", "content": label}], generation=False)
    ids = full['input_ids']
    start = prefix.shape[1]
    if not torch.equal(ids[:, :start], prefix):
        raise ValueError("Chat-template prefix mismatch: no scores were accepted.")
    end_id = classifier.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    if end_id is None or end_id == classifier.processor.tokenizer.unk_token_id:
        raise ValueError("Expected Qwen end-of-answer token is missing.")
    tail = ids[0, start:].tolist()
    if end_id not in tail:
        raise ValueError("Answer end token not found.")
    count = tail.index(end_id) + 1
    if count < 2:
        raise ValueError("Empty candidate label.")
    stop = start + count
    with torch.inference_mode():
        output = classifier.model(**full, use_cache=False, return_dict=True)
        # Position t-1 predicts token t. Convert only answer logits to FP32.
        logits = output.logits[:, start - 1:stop - 1, :].float()
        target = ids[:, start:stop]
        token_logs = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        if not torch.isfinite(token_logs).all().item():
            raise ValueError("Non-finite token log probabilities.")
        total = token_logs.sum().item()
    return total, count


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True, help='New directory; existing directories are refused.')
    parser.add_argument('--limit', type=int, help='Smoke test only; omit for full validation.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    # Works even if the local package was not installed in editable mode.
    root = Path.cwd()
    if not (root / 'src/mami/vlm.py').is_file():
        parser.error('Run this script from your mami_research project directory.')
    sys.path.insert(0, str(root / 'src'))
    from mami.config import load_config
    from mami.data import load_splits
    from mami.vlm import Classifier, LABELS, PROMPT
    from PIL import Image

    config = load_config(args.config)
    # Use one explicitly recorded precision for the new scoring experiment.
    config['model']['dtype'] = 'float32'
    if config['model']['id'] != 'Qwen/Qwen2.5-VL-3B-Instruct':
        parser.error('This addon is intended for your Qwen2.5-VL-3B adapters.')
    adapter = Path(args.adapter)
    if not (adapter / 'adapter_config.json').is_file():
        parser.error('Adapter directory is missing adapter_config.json')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='started', config=config, adapter=str(adapter.resolve()),
                    split='validation', limit=args.limit, prompt=PROMPT,
                    method='sum of log probabilities of complete label plus im_end; no length normalization',
                    score='exp(logp1) / (exp(logp0) + exp(logp1)); not calibrated',
                    python=platform.python_version(), platform=platform.platform(),
                    script_sha256=file_hash(__file__),
                    adapter_hashes={p.name: file_hash(p) for p in sorted(adapter.glob('*'))
                                    if p.suffix in ('.safetensors', '.json')},
                    source_hashes={name: file_hash(root / 'src/mami' / name)
                                   for name in ('vlm.py', 'data.py', 'config.py')})
    metadata['versions'] = {name: importlib.metadata.version(name)
                            for name in ('torch', 'transformers', 'peft', 'Pillow')}
    info = Path(config['model']['local_dir']) / 'model_info.json'
    metadata['base_model_info'] = json.loads(info.read_text()) if info.exists() else None
    write_json(out / 'run.json', metadata)
    try:
        splits, report = load_splits(config)
        write_json(out / 'data_report.json', report)
        frame = splits['validation']
        if args.limit:
            frame = frame.head(args.limit)
        classifier = Classifier(config, str(adapter))
        metadata['device'] = str(classifier.device)
        metadata['actual_dtype'] = str(next(classifier.model.parameters()).dtype)
        metadata['image_processor'] = classifier.processor.image_processor.to_dict()
        write_json(out / 'run.json', metadata)
        rows = []
        fields = ['file_name', 'label', 'image_sha256', 'text_sha256', 'logp_0', 'logp_1',
                  'tokens_0', 'tokens_1', 'log_odds_1', 'score_1', 'prediction_at_0_5']
        with (out / 'scores.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, row in enumerate(frame.itertuples(), 1):
                with Image.open(row.image_path) as source:
                    image = source.convert('RGB')
                messages = classifier.messages(image, row.text, 'multimodal')
                prefix_inputs = classifier.encode(messages)
                prefix = prefix_inputs['input_ids'].clone()
                del prefix_inputs
                log0, n0 = label_loglikelihood(classifier, messages, prefix, LABELS[0])
                log1, n1 = label_loglikelihood(classifier, messages, prefix, LABELS[1])
                score = relative_score(log0, log1)
                result = dict(file_name=row.file_name, label=int(row.label),
                              image_sha256=file_hash(row.image_path),
                              text_sha256=hashlib.sha256(row.text.encode()).hexdigest(),
                              logp_0=log0, logp_1=log1, tokens_0=n0, tokens_1=n1,
                              log_odds_1=log1-log0, score_1=score,
                              prediction_at_0_5=int(score >= 0.5))
                rows.append(result)
                writer.writerow(result)
                f.flush()
                print(f'{i}/{len(frame)} {row.file_name}: score_1={score:.6f}', flush=True)
        # Unique scores cover every attainable decision set, including all-negative.
        thresholds = sorted({0.0, 0.5, 1.0, math.nextafter(1.0, math.inf),
                             *(i / 100 for i in range(1, 100)),
                             *(r['score_1'] for r in rows)})
        table = [threshold_metrics(rows, t) for t in thresholds]
        with (out / 'thresholds.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        summary = threshold_metrics(rows, 0.5)
        summary['note'] = 'Candidate scoring is different from greedy label generation. No threshold has been selected.'
        summary['smoke_test'] = args.limit is not None
        write_json(out / 'summary.json', summary)
        metadata.update(status='completed', n=len(rows))
        write_json(out / 'run.json', metadata)
        print(json.dumps(summary, indent=2))
        print(f'Saved scores and threshold table to {out}')
    except BaseException as error:
        metadata.update(status='failed', error=str(error))
        write_json(out / 'run.json', metadata)
        raise


if __name__ == '__main__':
    main()
