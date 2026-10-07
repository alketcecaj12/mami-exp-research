"""Run: python /path/to/MAMI_Training_Fix/install_fix.py /path/to/mami_research"""
from pathlib import Path
from datetime import datetime
import shutil
import sys

source = Path(__file__).resolve().parent
target = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
if not (target / 'src/mami/train.py').exists():
    raise SystemExit('Target must be your mami_research directory.')
backup = target / ('code_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
for name in ['src/mami/train.py', 'src/mami/vlm.py', 'src/mami/cli.py', 'configs/mac_stable.yaml']:
    original = target / name
    if original.exists():
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, saved)
    original.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / name, original)
print('Fix installed. Previous code backed up to:', backup)
print('Dataset, downloaded model, default config and previous outputs were not modified.')
