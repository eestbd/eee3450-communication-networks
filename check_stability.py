"""Additional long C experiment because its probability continues adapting at 100k."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from aimd_sim import COL, Config, measure, simulate
from run_experiments import aggregate, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results_stability")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    rows, blocks, checks = [], [], []
    for seed in args.seeds:
        cfg = Config(steps=args.steps, warmup=args.warmup)
        d, check = simulate("C", seed, cfg)
        rows.append(measure(d, cfg, "C", seed, "C_extended"))
        checks.append(check)
        for left in range(0, args.steps, 10_000):
            right = min(left + 10_000, args.steps)
            r = {"seed": seed, "first_slot": left + 1, "last_slot": right}
            for i in (1, 2, 3):
                r[f"throughput{i}"] = float(d[left:right, COL[f"received{i}"]].mean())
                r[f"mean_p{i}"] = float(d[left:right, COL[f"p{i}"]].mean())
            blocks.append(r)
        print(f"C stability seed={seed}: " + ", ".join(f"{rows[-1][f'throughput{i}']:.5f}" for i in (1,2,3)), flush=True)
    write_csv(out / "per_seed.csv", rows)
    write_csv(out / "summary.csv", aggregate(rows))
    write_csv(out / "stability_blocks.csv", blocks)
    (out / "verification.json").write_text(json.dumps({"runs": checks, "elapsed_seconds": time.perf_counter()-start}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
