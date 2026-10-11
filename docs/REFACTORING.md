# Refactoring plan

This branch isolates binary Task A logic without changing existing commands.

## Implemented
- `src/mami/binary/prompts.py`: original binary prompt and parser
- `src/mami/binary/evaluation.py`: original binary metrics
- `src/mami/binary/baseline.py`: original text baselines
- `src/mami/binary/inference.py`: binary classifier selection
- Existing import paths are retained as compatibility wrappers.

## Next
- Run unit tests and binary regression checks locally.
- Introduce independent multilabel Task B modules and CLI.
- Only then consider moving model loading, training, and configs.

Keep the previous test outputs unchanged. Do not move historical backups or add-ons before an audit.
