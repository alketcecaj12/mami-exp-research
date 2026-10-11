"""Audit tracked root folders before archival. Read-only; makes no changes."""
from collections import Counter
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    paths = [p for p in paths if p]
    counts = Counter(p.split('/')[0] if '/' in p else '(root files)' for p in paths)
    print('Tracked files by top-level location:')
    for folder, count in sorted(counts.items()):
        print(f'{folder:45} {count:5}')
    print('\nCandidates for review (no files changed):')
    for folder in sorted(counts):
        if folder.startswith(('outputs_', 'code_backup')) or folder in ('add-ons', 'addons'):
            matches = [p for p in paths if p.startswith(folder + '/')]
            print(f'\n{folder}: {len(matches)} tracked files')
            for path in matches[:12]:
                print('  ' + path)
            if len(matches) > 12:
                print(f'  ... and {len(matches)-12} more')
    print('\nDo not remove tracked research outputs before archiving them.')


if __name__ == '__main__':
    main()
