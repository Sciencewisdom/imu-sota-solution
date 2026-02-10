How To Run (Organizer Interface)

1) Put hidden test files into: ./test_data/*.txt
2) Run:
   python3 run.py

Outputs:
- ./submission.xlsx

Executable (PyInstaller, optional):
- Build:
   ./build_pyinstaller.sh
- Run (in the built folder):
   ./submit_runner --test-data-dir ./test_data --out-xlsx ./submission.xlsx

Optional:
- Use GPU explicitly:
   python3 run.py --device cuda
- Use CPU:
   python3 run.py --device cpu

Notes:
- best_cfg.json + mapping.txt must stay next to run.py
- models/*.model can contain 1 or multiple models; when multiple exist, run.py will ensemble them by averaging logits.
