"""Dataset manifest builder for label-blind clustering experiments."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from ..config import load_config
from ..data.loader import load_splits


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_corpus(config_path, splits, limit, seed):
    config = load_config(config_path)
    frames, report = load_splits(config)
    rows = []
    for split in splits:
        for r in frames[split].to_dict("records"):
            rows.append(dict(uid=f"{split}:{r['file_name']}", split=split,
                             file_name=str(r["file_name"]), image_path=str(Path(r["image_path"]).resolve()),
                             text=str(r["text"]), label=int(r["label"])))
    rows.sort(key=lambda r: r["uid"])
    if limit and len(rows) > limit:
        chosen = sorted(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
        rows = [rows[i] for i in chosen]
    for r in rows:
        with Image.open(r["image_path"]) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
            r["pixels_sha256"] = digest(str(im.size).encode() + im.tobytes())
        r["text_sha256"] = digest(r["text"].encode())
    loader_source = Path(__file__).resolve().parents[1] / "data" / "loader.py"
    return rows, report, digest(loader_source.read_bytes())
