"""Execute notebook code cells in order and assert that training was skipped."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from IPython.display import display

os.environ.setdefault('MPLBACKEND', 'Agg')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
VALIDATION_DEPS = Path(__file__).resolve().parent / '.validation_deps'
if VALIDATION_DEPS.is_dir():
    sys.path.insert(0, str(VALIDATION_DEPS))


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: validate_notebook_load_only.py NOTEBOOK.ipynb')
        return 2
    path = Path(sys.argv[1]).resolve()
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {
        '__name__': '__main__',
        '__file__': str(path),
        '__builtins__': __builtins__,
        'display': display,
    }
    os.chdir(path.parent)
    for index, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        if not source.strip():
            continue
        print(f'[{path.name}] cell {index}', flush=True)
        try:
            exec(compile(source, f'{path.name}:cell-{index}', 'exec'), namespace)
        except Exception:
            print(f'FAILED at cell {index}', flush=True)
            traceback.print_exc()
            return 1
        if '# LOAD-FIRST CHECKPOINT GUARD' in source:
            if namespace.get('checkpoint_loaded') is not True:
                print('FAILED: checkpoint guard did not load an existing checkpoint.')
                return 1
            if namespace.get('execution_status') != 'LOADED_EXISTING_BEST_CHECKPOINT_NO_TRAINING':
                print('FAILED: notebook did not report the load-only execution status.')
                return 1
    if namespace.get('checkpoint_loaded') is not True:
        print('FAILED: no checkpoint was loaded.')
        return 1
    print(f'PASS: {path.name} completed from its saved checkpoint without training.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
