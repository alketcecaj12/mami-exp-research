"""Exploratory, label-blind MAMI clustering. Run from the existing project root."""
import argparse
import csv
import hashlib
import html
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image, ImageOps

MODEL = 'google/embeddinggemma-2'
PREFIX = 'task: clustering | query: '


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def unit_vector(value):
    x = np.asarray(value, dtype=np.float32).reshape(-1)
    if x.shape != (768,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-10:
        raise ValueError(f'Invalid embedding: shape={x.shape}; expected finite nonzero 768-vector')
    return x / np.linalg.norm(x)


def load_corpus(config_path, splits, limit, seed):
    import yaml
    source = Path('src/mami/data.py').resolve()
    if not source.is_file():
        raise FileNotFoundError('Run from /Users/alket/Downloads/mami_research')
    spec = importlib.util.spec_from_file_location('mami_cluster_data', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = yaml.safe_load(Path(config_path).read_text())
    frames, report = module.load_splits(cfg)
    rows = []
    for split in splits:
        for r in frames[split].to_dict('records'):
            rows.append(dict(uid=f"{split}:{r['file_name']}", split=split,
                             file_name=str(r['file_name']), image_path=str(Path(r['image_path']).resolve()),
                             text=str(r['text']), label=int(r['label'])))
    rows.sort(key=lambda r: r['uid'])
    if limit and len(rows) > limit:
        chosen = sorted(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
        rows = [rows[i] for i in chosen]
    # Hash decoded pixels plus text to ensure cache cannot silently use changed data.
    for r in rows:
        with Image.open(r['image_path']) as source_image:
            im = ImageOps.exif_transpose(source_image).convert('RGB')
            r['pixels_sha256'] = digest(str(im.size).encode() + im.tobytes())
        r['text_sha256'] = digest(r['text'].encode())
    return rows, report, digest(source.read_bytes())


def extract(rows, out, modes, device_arg):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from sentence_transformers import SentenceTransformer
    versions = {p: importlib.metadata.version(p) for p in
                ['torch', 'transformers', 'sentence-transformers', 'Pillow']}
    device = device_arg if device_arg != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu')
    metadata_path = out / 'embedding_run.json'
    settings = dict(model=MODEL, dtype='float32', device=device, versions=versions,
                    prefix=PREFIX, dimensions=768, audio_disabled=True,
                    image_orientation='EXIF transpose, RGB', batch_size=1)
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta['settings'] != settings:
            raise ValueError('Embedding settings/environment changed. Use a new output directory.')
        revision = meta['revision']
    else:
        revision = HfApi().model_info(MODEL).sha
        meta = dict(settings=settings, revision=revision, status='started', python=platform.python_version())
        save_json(metadata_path, meta)
    directory = snapshot_download(MODEL, revision=revision,
                                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    print(f'Loading {MODEL} revision={revision} on {device}, float32', flush=True)
    model = SentenceTransformer(directory, device=device, trust_remote_code=False,
                                config_kwargs={'audio_config': None},
                                model_kwargs={'torch_dtype': torch.float32})
    model.eval()
    matrices = {}
    for mode in modes:
        cache = out / 'cache' / mode
        cache.mkdir(parents=True, exist_ok=True)
        vectors = []
        for i, r in enumerate(rows, 1):
            key = digest((r['pixels_sha256'] + r['text_sha256'] + revision + mode).encode())
            target = cache / f'{key}.npy'
            if target.exists():
                x = unit_vector(np.load(target, allow_pickle=False))
            else:
                # Normalize orientation once, then use the documented path-based media input.
                with Image.open(r['image_path']) as im:
                    image = ImageOps.exif_transpose(im).convert('RGB')
                    temporary_image = out / 'current_image.png'
                    image.save(temporary_image)
                if '<|image|>' in r['text']:
                    raise ValueError(f"Reserved image placeholder in transcription: {r['uid']}")
                content = '<|image|>' if mode == 'image' else PREFIX + '<|image|>\n' + r['text']
                payload = {'text': content, 'image': [str(temporary_image.resolve())]}
                with torch.inference_mode():
                    encoded = model.encode(payload, prompt='', normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False, batch_size=1)
                x = unit_vector(encoded)
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as f:
                    np.save(f, x)
                temporary.replace(target)
                temporary_image.unlink(missing_ok=True)
            vectors.append(x)
            if i % 20 == 0 or i in (1, len(rows)):
                print(f'{mode}: {i}/{len(rows)} embedded or cached', flush=True)
        matrices[mode] = np.stack(vectors)
        np.save(out / f'embeddings_{mode}.npy', matrices[mode])
    meta['status'] = 'completed'
    save_json(metadata_path, meta)
    return matrices


STYLE = '''<style>body{font:16px system-ui;margin:32px auto;max-width:1250px;padding:0 20px;color:#1b2632;background:#f5f7fa}a{color:#185b99}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:18px}article,section{background:white;padding:18px;border:1px solid #ddd;border-radius:10px;margin-bottom:18px}img{width:100%;height:250px;object-fit:contain}p{line-height:1.5;white-space:pre-wrap}small{color:#52616d}input{padding:10px;width:90%;margin-bottom:20px}</style>'''


def page(title, body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title>' + STYLE + '<body><h1>' + html.escape(title) + '</h1>' + body + '</body></html>'


def atlas(rows, x, labels, centers, out, mode, k, seed):
    folder = out / f'atlas_{mode}_k{k}'
    folder.mkdir(exist_ok=True)
    thumbs = out / 'thumbnails'
    thumbs.mkdir(exist_ok=True)
    for r in rows:
        target = thumbs / (digest(r['uid'].encode()) + '.jpg')
        if not target.exists():
            with Image.open(r['image_path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((640, 640))
                im.save(target, quality=85)
    annotations, links = [], []
    for c in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels == c)
        distance = np.linalg.norm(x[ids] - centers[c], axis=1)
        ordered = ids[np.argsort(distance)]
        rank = {int(v): i+1 for i, v in enumerate(ordered)}
        typical = ordered[:5].tolist()
        peripheral = [i for i in ordered[-3:].tolist() if i not in typical]
        remaining = [i for i in ids.tolist() if i not in typical + peripheral]
        sampled = np.random.default_rng(seed+c).choice(remaining, min(3,len(remaining)), replace=False).tolist()
        roles = {i:'Typical (nearest centroid)' for i in typical}
        roles.update({i:'Peripheral (farthest centroid)' for i in peripheral})
        roles.update({i:'Random review sample' for i in sampled})
        display = typical + peripheral + sampled + [int(i) for i in ordered if int(i) not in roles]
        cards = []
        for i in display:
            r = rows[i]
            cards.append('<article><small>' + html.escape(roles.get(i,'Other member')) + '</small><img loading="lazy" src="../thumbnails/' + digest(r['uid'].encode()) + '.jpg" alt="Meme ' + html.escape(r['uid'],quote=True) + '"><h3>' + html.escape(r['uid']) + '</h3><p>' + html.escape(r['text']) + '</p><details><summary>Reveal dataset label</summary>' + str(r['label']) + '</details></article>')
        positives = sum(rows[i]['label'] for i in ids)
        title = f'{mode} — cluster {c}, {len(ids)} memes'
        body = '<p><a href="index.html">All clusters</a> · Numbered clusters need human interpretation.</p><details><summary>Reveal label composition</summary>' + f'{positives}/{len(ids)} dataset-positive; descriptive only.' + '</details><input id="search" placeholder="Filter by filename or transcription"><div class="grid">' + ''.join(cards) + '</div><script>document.getElementById("search").addEventListener("input",e=>{let q=e.target.value.toLowerCase();document.querySelectorAll("article").forEach(a=>a.hidden=!a.textContent.toLowerCase().includes(q));});</script>'
        (folder / f'cluster_{c}.html').write_text(page(title, body))
        links.append(f'<section><a href="cluster_{c}.html">Cluster {c}</a> — {len(ids)} memes</section>')
        annotations.append(dict(cluster=c,n=len(ids),theme='',visual_motifs='',template_or_watermark_effect='',
                                interpretation='',counterexamples='',confidence='',reviewer='',
                                typical_ids=';'.join(rows[i]['uid'] for i in typical),
                                peripheral_ids=';'.join(rows[i]['uid'] for i in peripheral)))
    (folder / 'index.html').write_text(page(f'MAMI atlas: {mode}, k={k}', '<p>Explore visual motifs and themes. Cluster numbers are not thematic labels. Labels were excluded from clustering.</p>' + ''.join(links)))
    annotation_path = folder / 'human_review.csv'
    if not annotation_path.exists():
        csv_write(annotation_path, annotations)


def cluster(rows, matrices, out, ks, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    metrics, links, assignments = [], [], {}
    for mode, x in matrices.items():
        if x.shape != (len(rows),768) or not np.isfinite(x).all():
            raise ValueError('Embedding matrix does not match corpus')
        for k in ks:
            if not 2 <= k < len(rows):
                raise ValueError('Each k must be at least 2 and smaller than corpus size')
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
            other = KMeans(n_clusters=k, n_init=10, random_state=seed+1).fit_predict(x)
            y = model.labels_
            assignments[(mode,k)] = y
            unique = len(np.unique(y))
            score = silhouette_score(x,y,metric='cosine',sample_size=min(2000,len(rows)),random_state=seed) if 1 < unique < len(rows) else None
            metrics.append(dict(mode=mode,k=k,actual_clusters=unique,n=len(rows),
                                cosine_silhouette=score,initialization_ARI=adjusted_rand_score(y,other)))
            csv_write(out / f'assignments_{mode}_k{k}.csv',
                      [dict(**r,cluster=int(c)) for r,c in zip(rows,y)])
            atlas(rows,x,y,model.cluster_centers_,out,mode,k,seed)
            links.append(f'<section><a href="atlas_{mode}_k{k}/index.html">{mode}, k={k}</a></section>')
    csv_write(out/'cluster_metrics.csv',metrics)
    comparisons = [dict(k=k,image_vs_multimodal_ARI=adjusted_rand_score(assignments[('image',k)],assignments[('multimodal',k)]))
                   for k in ks if ('image',k) in assignments and ('multimodal',k) in assignments]
    if comparisons:
        csv_write(out/'modality_comparison.csv',comparisons)
    (out/'index.html').write_text(page('MAMI visual corpus atlas',
        '<p>Exploratory clustering of frozen embeddings. Review typical, peripheral and random examples before naming clusters. This corpus has no audience comments or platform context.</p>'+''.join(links)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/full_corrected.yaml')
    p.add_argument('--output',required=True)
    p.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation','test'])
    p.add_argument('--limit',type=int)
    p.add_argument('--k',type=int,nargs='+',default=[10,20,30])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    p.add_argument('--cluster-only',action='store_true',help='Reuse saved embeddings; no model load')
    a=p.parse_args()
    if a.limit is not None and a.limit<3:p.error('--limit must be at least 3')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows,report,loader_hash=load_corpus(a.config,list(dict.fromkeys(a.splits)),a.limit,a.seed)
    signature=dict(rows=rows,loader_sha256=loader_hash,limit=a.limit,seed=a.seed,splits=a.splits)
    manifest=out/'corpus.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:
        raise ValueError('Corpus changed: use a new output directory to preserve the previous run.')
    save_json(manifest,signature);save_json(out/'data_report.json',report)
    modes=['image','multimodal']
    if any(k<2 or k>=len(rows) for k in a.k):p.error('Require 2 <= k < number of selected examples')
    print(f'Corpus: {len(rows)} examples; label values excluded from embeddings and clustering.',flush=True)
    if a.cluster_only:
        matrices={m:np.load(out/f'embeddings_{m}.npy',allow_pickle=False) for m in modes}
    else:
        matrices=extract(rows,out,modes,a.device)
    cluster(rows,matrices,out,sorted(set(a.k)),a.seed)
    save_json(out/'clustering_run.json',dict(status='completed',n=len(rows),k=a.k,seed=a.seed,
              sklearn=importlib.metadata.version('scikit-learn'),script_sha256=digest(Path(__file__).read_bytes()),
              method='KMeans on unit-normalized 768-dimensional embeddings; not on 2D projection',
              smoke_test=a.limit is not None))
    print(f'Finished. Open {out / "index.html"}',flush=True)


if __name__=='__main__':
    main()
