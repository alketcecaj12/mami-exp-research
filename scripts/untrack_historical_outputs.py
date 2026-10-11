"""Untrack historical experiment outputs while preserving local files.

Run from the repository root on cleanup/research-repository-root.
This script NEVER deletes local files and NEVER creates a commit or pushes.
"""
from pathlib import Path
import subprocess
import sys

BRANCH = "cleanup/research-repository-root"
ARCHIVE = "origin/archive/pre-root-cleanup-20261011"


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise SystemExit("Run from repository root.")
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise SystemExit(f"Expected {BRANCH}, found {branch!r}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Tracked working-tree changes detected. Commit/stash first.")
    subprocess.run(["git", "fetch", "origin", "archive/pre-root-cleanup-20261011"], check=True)
    if not git("rev-parse", "--verify", ARCHIVE):
        raise SystemExit("Archive branch not available; refusing to proceed.")
    paths = git("ls-files", "-z").split("\0")
    tracked = [p for p in paths if p.startswith("outputs_") and "/" in p]
    folders = sorted({p.split("/", 1)[0] for p in tracked})
    if not folders:
        print("No tracked outputs_* folders remain.")
        return
    print(f"Will untrack {len(tracked)} files in {len(folders)} historical experiment directories:")
    for folder in folders:
        print("  " + folder)
    print("Local files remain on disk; archived commits remain on the archive branch.")
    if "--apply" not in sys.argv:
        print("\nDRY RUN ONLY. To stage untracking, rerun with --apply.")
        return
    if "--yes" not in sys.argv:
        raise SystemExit("Add --yes to acknowledge staging the removals.")
    subprocess.run(["git", "rm", "-r", "--cached", "--", *folders], check=True)
    print("\nStaged untracking only. Review git diff --cached --stat before committing.")
    print("Do NOT use git clean -fd or git reset --hard to remove the local outputs.")


if __name__ == "__main__":
    main()
