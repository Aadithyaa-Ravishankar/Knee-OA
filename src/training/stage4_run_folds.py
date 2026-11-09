"""
stage4_run_folds.py
====================
Run several folds of one configuration, spreading them over the available GPUs
(e.g. Kaggle T4 x2 → two folds at a time). Each fold is resume-safe, so simply
re-run the same command after a disconnect.

Every argument not listed below is passed through to stage4_cv_train.py,
including --aux-csv and --aux-weight.

Example:
  python src/training/stage4_run_folds.py --folds 0 1 2 3 4 --run-name r50_dual_cascade_sqrt \
      --backbone resnet50 --head cascade --sampler sqrt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN = PROJECT_ROOT / "src" / "training" / "stage4_cv_train.py"
PAUSED_EXIT = 75  # must match stage4_cv_train.PAUSED_EXIT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpus", type=int, nargs="*", default=None, help="default: all visible GPUs")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv" / "runs")
    args, passthrough = ap.parse_known_args()

    gpus = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
    if not gpus:
        gpus = [None]
    queues = {g: args.folds[i::len(gpus)] for i, g in enumerate(gpus)}
    failures, paused = [], []
    deadline = float(os.environ.get("PIPELINE_DEADLINE", "inf"))

    def worker(gpu, folds):
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for fold in folds:
            log = args.out_dir / args.run_name / f"fold{fold}" / "train.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            if not (log.parent / "DONE").exists() and time.time() > deadline - 3600:
                print(f"[gpu {gpu}] fold {fold} not started: less than 1 h left in this session", flush=True)
                paused.append(fold)
                continue
            cmd = [sys.executable, str(TRAIN), "--fold", str(fold), "--run-name", args.run_name,
                   "--out-dir", str(args.out_dir), *passthrough]
            print(f"[gpu {gpu}] fold {fold} → {log}", flush=True)
            with log.open("a", encoding="utf-8") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
            if rc == 0:
                status = "ok"
            elif rc == PAUSED_EXIT:
                status = "paused (session time budget) — re-run to resume"
                paused.append(fold)
            else:
                status = f"FAILED (exit {rc}) — see {log}"
                failures.append(fold)
            tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            print(f"[gpu {gpu}] fold {fold} {status}\n    " + "\n    ".join(tail), flush=True)

    threads = [threading.Thread(target=worker, args=(g, q)) for g, q in queues.items() if q]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        sys.exit(f"folds failed: {sorted(failures)} — re-run the same command to resume")
    if paused:
        print(f"paused folds (unfinished, will resume next session): {sorted(paused)}")
        return
    print("all folds finished")


if __name__ == "__main__":
    main()
