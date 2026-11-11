"""Run the requested experiments and write raw CSV + checked diagnostics."""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np

from aimd_sim import BASE_P, COL, FIELDS, PF, PRICES, ROUTE_PRICES, Config, fair_service, measure, simulate


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_validation(seeds, steps=300_000, warmup=20_000):
    rows = []
    n = steps - warmup
    for seed in seeds:
        rng, w = random.Random(seed), [1.0] * 3
        sums, squares = [0.0] * 3, [0.0] * 3
        for k in range(steps):
            for i in range(3):
                if k >= warmup:
                    sums[i] += w[i]
                    squares[i] += w[i] ** 2
                w[i] = w[i] / 2 if rng.random() < BASE_P[i] else w[i] + 1
        for i in range(3):
            p = BASE_P[i]
            theory = 2 * (1 - p) / p
            variance = 4 * (1 - p) / (3 * p ** 2)
            a = 1 - p / 2  # E[W_next | W] has slope 1-p/2
            se = np.sqrt(variance / n * (1 + a) / (1 - a))
            observed = sums[i] / n
            rows.append({"seed": seed, "flow": i + 1, "steps": steps, "warmup": warmup,
                         "p": p, "theory_mean": theory, "sample_mean": observed,
                         "relative_error_pct": 100 * abs(observed / theory - 1),
                         "theory_variance": variance, "sample_variance": squares[i] / n - observed**2,
                         "autocorrelation_adjusted_se": se, "z": (observed - theory) / se})
    # Autocorrelation is not ignored when checking this stochastic process.
    assert max(abs(r["z"]) for r in rows) < 6
    return rows


def unit_checks():
    for a, b, c, expected in [(100, 100, 90, (45, 45)), (10, 100, 90, (10, 80)),
                             (100, 10, 90, (80, 10)), (0, 3, 90, (0, 3))]:
        assert fair_service(a, b, c) == expected
    # A deliberately overloaded serial path must lose data at R2, not teleport it.
    cfg = Config(steps=50, warmup=10, initial_cwnd=(250.0, 1.0, 250.0), queue_factor=1)
    d, _ = simulate("A", 0, cfg)
    assert d[:, COL["drop_r2_f1"]].sum() > 0
    assert not d[:, [COL[f"mark{i}"] for i in (1, 2, 3)]].any()
    for i in (1, 2, 3):
        if i == 1:
            loss = d[:, COL["drop_r1_f1"]] + d[:, COL["drop_r2_f1"]]
        else:
            loss = d[:, COL["drop_r1_f2" if i == 2 else "drop_r2_f3"]]
        assert np.array_equal(d[1:, COL[f"reaction{i}"]], loss[:-1] > 1e-10)
    # Exact repeatability includes every trace sample, not just aggregate means.
    for mode in ("A", "B", "C"):
        a, _ = simulate(mode, 42, cfg)
        b, _ = simulate(mode, 42, cfg)
        assert np.array_equal(a, b)
    return {"scheduler_examples": "passed", "serial_route_and_r2_loss": "passed",
            "A_loss_only_feedback": "passed", "exact_repeatability_A_B_C": "passed"}


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["mode"]), []).append(row)
    result = []
    metrics = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.number))
               and k not in ("seed", "steps", "warmup", "queue_factor")]
    for (scenario, mode), group in groups.items():
        r = {"scenario": scenario, "mode": mode, "n_runs": len(group)}
        for metric in metrics:
            values = [x[metric] for x in group]
            r[f"{metric}_mean"] = float(np.mean(values))
            r[f"{metric}_seed_sd"] = float(np.std(values, ddof=1)) if len(group) > 1 else 0.0
        result.append(r)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--queue-factor", type=float, default=2.0)
    parser.add_argument("--initial-cwnd", type=float, nargs=3, default=(1, 1, 1))
    parser.add_argument("--modes", nargs="+", choices=("A", "B", "C"), default=("A", "B", "C"))
    parser.add_argument("--main-only", action="store_true", help="Skip initial/queue/length sensitivity, retain unit and independent checks")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an existing experiment
    start = time.perf_counter()
    tests = unit_checks()
    independent = independent_validation(args.seeds)
    write_csv(out / "independent_aimd.csv", independent)
    config = Config(steps=args.steps, warmup=args.warmup, queue_factor=args.queue_factor,
                    initial_cwnd=tuple(args.initial_cwnd))
    jobs = [("main", config, args.seeds)]
    if not args.main_only:
        jobs += [("initial_high", replace(config, initial_cwnd=(150.0, 150.0, 150.0)), args.seeds),
                 ("queue_1x", replace(config, queue_factor=1.0), args.seeds[:3]),
                 ("queue_4x", replace(config, queue_factor=4.0), args.seeds[:3]),
                 ("long", replace(config, steps=max(100_000, args.steps), warmup=max(20_000, args.warmup)), args.seeds[:3])]
    rows, checks, stability = [], [], []
    for scenario, cfg, seeds in jobs:
        for mode in args.modes:
            # A is deterministic: one run, never ten duplicate pseudo-replicates.
            for seed in seeds[:1] if mode == "A" else seeds:
                data, check = simulate(mode, seed, cfg)
                rows.append(measure(data, cfg, mode, seed, scenario))
                check["scenario"] = scenario
                checks.append(check)
                for left in range(cfg.warmup, cfg.steps, 5000):
                    right = min(left + 5000, cfg.steps)
                    r = {"scenario": scenario, "mode": mode, "seed": seed,
                         "first_slot": left + 1, "last_slot": right}
                    for i in (1, 2, 3):
                        r[f"throughput{i}"] = float(data[left:right, COL[f"received{i}"]].mean())
                    stability.append(r)
                if scenario == "main" and seed == seeds[0]:
                    np.savetxt(out / f"trace_{mode}.csv", data, delimiter=",", header=",".join(FIELDS), comments="", fmt="%.12g")
                print(f"{scenario:13s} {mode} seed={seed}: x=" + ", ".join(f"{rows[-1][f'throughput{i}']:.4f}" for i in (1,2,3)), flush=True)
    write_csv(out / "per_seed.csv", rows)
    write_csv(out / "summary.csv", aggregate(rows))
    write_csv(out / "stability_blocks.csv", stability)
    hashes = {}
    for name in ("aimd_sim.py", "run_experiments.py"):
        hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    metadata = {"python": platform.python_version(), "numpy": np.__version__,
                "config": asdict(config), "seeds": args.seeds, "modes": args.modes,
                "pf": PF.tolist(), "prices": PRICES.tolist(), "route_prices": ROUTE_PRICES.tolist(),
                "base_probabilities": BASE_P.tolist(), "utility_at_pf": float(np.log(PF).sum()),
                "tests": tests, "runs": checks, "source_sha256": hashes,
                "elapsed_seconds": time.perf_counter() - start}
    (out / "verification.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Completed {len(rows)} network runs; {sum(x['slots_checked'] for x in checks):,} slots checked; elapsed {metadata['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
