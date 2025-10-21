# Code Workspace

A tiny workspace with a Jupyter notebook.

## Quick start

1) Create/select Python 3.11 env
- Recommended: use the existing venv at `.venv311`.
- Or create a new one:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Register the kernel (if needed)

```bash
python -m ipykernel install --user --name code-venv311 --display-name "Python 3.11 (code/.venv311)"
```

3) Open `ttt.ipynb` in VS Code and pick the kernel:
- Select kernel: "Python 3.11 (code/.venv311)".

## Project layout
- `ttt.ipynb` — main notebook
- `test.py` — sample script
- `requirements.txt` — base dependencies

## Notes
- This repo ignores common Python/Jupyter/VS Code artifacts via `.gitignore`.
- If you need additional packages, add them to `requirements.txt` and re-run `pip install -r requirements.txt`.
