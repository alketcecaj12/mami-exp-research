"""Separate SmolVLM2 zero-shot runner. Does not change the Qwen implementation."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

from .config import load_config
from .data import load_splits
from .metrics import evaluate
from .vlm import PROMPT, parse_label


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def write_rows(path, rows, fields):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class SmolClassifier:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.cfg = cfg['model']
        self.device = self.cfg.get('device', 'auto')
        if self.device == 'auto':
            self.device = ('mps' if torch.backends.mps.is_available()
                           else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = getattr(torch, self.cfg['dtype'])
        directory = Path(self.cfg['local_dir'])
        saved = json.loads((directory / 'model_info.json').read_text())
        if saved['id'] != self.cfg['id']:
            raise ValueError('Downloaded model identity does not match configuration')
        self.processor = AutoProcessor.from_pretrained(
            directory, local_files_only=True, use_fast=False)
        self.processor.image_processor.size = {
            'longest_edge': self.cfg['longest_edge']}
        self.processor.image_processor.do_resize = True
        self.model = AutoModelForImageTextToText.from_pretrained(
            directory, local_files_only=True, dtype=self.dtype,
            attn_implementation='eager').to(self.device).eval()
        print(f'Loaded SmolVLM2 on {self.device}, dtype={self.cfg["dtype"]}', flush=True)

    def predict(self, image_path, text):
        from PIL import Image
        # Match the Qwen runner's RGB conversion; processor is model-specific.
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        messages = [{'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': PROMPT + '\nTranscription (data):\n' + text},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], images=[[image]], return_tensors='pt', padding=True)
        # Preserve integer token IDs/masks; cast only floating-point tensors.
        inputs = {key: value.to(device=self.device, dtype=self.dtype)
                  if value.is_floating_point() else value.to(self.device)
                  for key, value in inputs.items()}
        with self.torch.inference_mode():
            tokens = self.model.generate(
                **inputs, do_sample=False, num_beams=1,
                max_new_tokens=self.cfg['max_new_tokens'])
        response = self.processor.batch_decode(
            tokens[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)[0].strip()
        return {'prediction': parse_label(response), 'response': response}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/smolvlm2.yaml')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true',
                        help='Explicitly replace an earlier run for this split/limit')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    cfg = load_config(args.config)
    if cfg['model']['id'] != 'HuggingFaceTB/SmolVLM2-2.2B-Instruct':
        parser.error('Use the SmolVLM2 configuration, not a Qwen configuration')
    directory = Path(cfg['model']['local_dir'])
    if not (directory / 'model_info.json').exists():
        parser.error(f'Download first: mami download --config {args.config}')
    output = Path(cfg['experiment']['output'])
    if args.limit is not None:
        output = output / f'smoke_{args.limit}'
    output.mkdir(parents=True, exist_ok=True)
    stem = f'zero_shot_multimodal_{args.split}'
    raw_path = output / f'{stem}_raw.csv'
    metric_path = output / f'vlm_{args.split}_metrics.json'
    run_path = output / f'vlm_{args.split}_config.json'
    if not args.overwrite and any(p.exists() for p in (raw_path, metric_path, run_path)):
        parser.error(f'Run already exists in {output}; choose another output or use --overwrite')
    if args.overwrite:
        for p in (raw_path, metric_path, run_path, output / f'{stem}.csv'):
            p.unlink(missing_ok=True)
    splits, report = load_splits(cfg)
    write_json(output / 'data_report.json', report)
    frame = splits[args.split]
    if args.limit is not None:
        frame = frame.head(args.limit)
    if frame.empty:
        parser.error('No evaluation examples available')
    import torch
    import transformers
    import PIL
    torch.manual_seed(cfg['experiment']['seed'])
    run = {'config': cfg, 'split': args.split, 'limit': args.limit,
           'mode': 'multimodal', 'adapter': None, 'completed': False,
           'prompt': PROMPT, 'transcription_prefix': '\nTranscription (data):\n',
           'model_info': json.loads((directory / 'model_info.json').read_text()),
           'python': sys.version, 'platform': platform.platform(),
           'torch': torch.__version__, 'transformers': transformers.__version__,
           'pillow': PIL.__version__,
           'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(run_path, run)
    (output / 'prompt.txt').write_text(PROMPT)
    model = SmolClassifier(cfg)
    run['resolved_device'] = model.device
    run['image_processor'] = model.processor.image_processor.to_dict()
    write_json(run_path, run)
    rows = []
    for index, row in enumerate(frame.itertuples(), 1):
        result = model.predict(row.image_path, row.text)
        rows.append({'file_name': row.file_name, 'label': int(row.label), **result})
        write_rows(raw_path, rows, ['file_name', 'label', 'prediction', 'response'])
        print(f'{index}/{len(frame)} {row.file_name}: {result["response"]!r}', flush=True)
    write_rows(output / f'{stem}.csv', rows, ['file_name', 'label', 'prediction'])
    summary = {'zero_shot_multimodal': evaluate(
        [row['label'] for row in rows], [row['prediction'] for row in rows])}
    write_json(metric_path, summary)
    run['completed'] = True
    write_json(run_path, run)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
