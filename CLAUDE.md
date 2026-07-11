# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Course materials for **T81-558: Applications of Deep Neural Networks** (Washington University in St. Louis, instructor Jeff Heaton). It is not an application — there is no build, lint, or test suite. The deliverables are **Jupyter notebooks** that teach applied deep learning with **PyTorch**, and every notebook must remain runnable end-to-end on **Google Colab**.

`README.md` is the live syllabus (module schedule, due dates, notebook links). When notebook filenames or module contents change, update the syllabus table in `README.md` to match. `README-fall.md`, `README_old.md`, `intro.md`, and `copyright.md` are book/alternate-semester text.

## Layout and naming

- **Lesson notebooks**: `t81_558_class_<MODULE>_<PART>_<topic>.ipynb` (e.g. `t81_558_class_02_2_pytorch_neural.ipynb` = Module 2, Part 2.2). Module and part numbers are load-bearing — they drive the syllabus and the cross-links inside each notebook.
- `assignments/` — student assignment templates, `assignment_yourname_t81_558_class<N>.ipynb`.
- `install/` — conda environment files (`torch.yml`, `torch-conda.yml`, `torch-cuda.yml`) and an install-walkthrough notebook.
- `prompts/` — plain-text LLM prompt specs (named `<module>_<part>_<topic>.txt`) used to generate notebook code cells. They follow a strict "Specs for Cell 1 / Cell 2 …" format; each generated cell is delimited by a `# Cell N` comment and shares variable names so cells run in sequence.
- Data is not committed — notebooks download it at runtime from `https://data.heatonresearch.com/data/t81-558/...`.

## Notebook conventions (follow these when editing or adding notebooks)

Each lesson notebook opens with the same fixed preamble, in order:
1. A Google Colab badge cell linking to `.../blob/main/<this-notebook>.ipynb`.
2. A course-header markdown cell (title, "Module N Material", instructor links).
3. A "Module N Material" list linking all five parts of the module.
4. A Colab-detection + device-selection boilerplate code cell. **Device selection is standardized** — prefer MPS, then CUDA, then CPU:
   ```python
   try:
       import google.colab
       COLAB = True
   except:
       COLAB = False
   import torch
   has_mps = torch.backends.mps.is_built()
   device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
   ```

Downstream code assumes `device` already exists — pass `device` explicitly rather than re-detecting it. Keep imports non-duplicated across cells (later cells rely on earlier imports), and keep all code Colab-compatible (`pip install` any package Colab lacks inside the notebook).

## Editing notebooks

Edit `.ipynb` files with the notebook tools (NotebookEdit), not by hand-editing JSON. Several notebooks embed large base64 image/media outputs and are multiple MB — when inspecting them programmatically, read `source` cells and avoid dumping cell `outputs`. To run/preview locally, use Jupyter with an environment built from `install/torch.yml` (or the CUDA variant on an NVIDIA machine).
