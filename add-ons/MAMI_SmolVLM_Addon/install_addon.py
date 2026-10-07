"""Run from the existing mami_research project root."""
from copy import deepcopy
from pathlib import Path
import ast
import yaml


def main():
    root = Path.cwd()
    original = root / 'configs/full_corrected.yaml'
    if not original.exists() or not (root / 'src/mami/vlm.py').exists():
        raise SystemExit('Run this installer from your mami_research project directory.')
    cfg = deepcopy(yaml.safe_load(original.read_text()))
    if not cfg['data'].get('image_dirs'):
        raise SystemExit('Expected the corrected configuration with data.image_dirs.')
    cfg['model'] = {
        'id': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
        'revision': 'main', 'local_dir': 'models/smolvlm2-2.2b',
        'device': 'auto', 'dtype': 'float32',
        'longest_edge': 1024, 'max_new_tokens': 16,
    }
    cfg['experiment']['output'] = 'outputs_smolvlm2'
    cfg.pop('training', None)
    here = Path(__file__).resolve().parent
    module = (here / 'smolvlm.py').read_text()
    ast.parse(module)
    files = {
        root / 'src/mami/smolvlm.py': module,
        root / 'configs/smolvlm2.yaml': yaml.safe_dump(cfg, sort_keys=False),
        root / 'SMOLVLM_README.md': (here / 'README.md').read_text(),
    }
    # Preflight every target before writing any file. Re-running is idempotent.
    for path, content in files.items():
        if path.exists() and path.read_text() != content:
            raise SystemExit(f'Refusing to overwrite a different existing file: {path}')
    for path, content in files.items():
        if not path.exists():
            with path.open('x') as stream:
                stream.write(content)
        print(f'Ready: {path.relative_to(root)}')
    print('Add-on installed. Existing project code, data, models and results were not modified.')


if __name__ == '__main__':
    main()
