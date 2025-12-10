#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# Base configuration
CACHE_DIR = os.path.expanduser("~/scratch/relbench")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Command template pieces
BASE_CMD = [
    "pixi",
    "run",
    "python",
    "gnn_recommendation.py",
]

# (dataset, task) tuples
jobs = [
    ("rel-amazon", "user-item-purchase"),
    ("rel-amazon", "user-item-rate"),
    ("rel-amazon", "user-item-review"),
    ("rel-avito", "user-ad-visit"),
    ("rel-f1", "driver-race-compete"),
    ("rel-hm", "user-item-purchase"),
    ("rel-stack", "user-post-comment"),
    ("rel-stack", "post-post-related"),
    ("rel-trial", "condition-sponsor-run"),
    ("rel-trial", "site-sponsor-run"),
]

processes = []

for idx, (dataset, task) in enumerate(jobs):
    cuda_device = idx % 8  # cycle from 0 to 7

    # Build output file path
    out_file = OUTPUT_DIR / f"{dataset}_{task}.out"

    # Build full command
    cmd = BASE_CMD + [
        "--dataset", dataset,
        "--task", task,
        "--epochs", "20",
    ]

    # Environment: inherit and override CUDA_VISIBLE_DEVICES and RELBENCH_CACHE_DIR
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env["RELBENCH_CACHE_DIR"] = CACHE_DIR

    # Open output file and start process in "background" (non-blocking)
    f = open(out_file, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    processes.append((p, f))

    print(f"Started job {idx}: dataset={dataset}, task={task}, "
          f"CUDA_VISIBLE_DEVICES={cuda_device}, pid={p.pid}, output={out_file}")

# If you truly want the script to exit immediately and leave processes running,
# comment out the block below. As written, it waits for all jobs to finish.

for p, f in processes:
    p.wait()
    f.close()
