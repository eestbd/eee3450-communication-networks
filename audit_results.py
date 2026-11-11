"""Independent checks of stored traces and an actually adapting C trajectory."""
import argparse
import json
from pathlib import Path

import numpy as np

from aimd_sim import BASE_P, COL, ROUTE_PRICES, Config, simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_long")
    args = parser.parse_args()
    checks = {}
    for mode in ("A", "B", "C"):
        d = np.genfromtxt(Path(args.results) / f"trace_{mode}.csv", delimiter=",", names=True)
        # Verify OR aggregation from stored outputs, independent of simulator asserts.
        for i in (1, 2, 3):
            loss = d["drop_r1_f1"] + d["drop_r2_f1"] if i == 1 else d["drop_r1_f2" if i == 2 else "drop_r2_f3"]
            assert np.array_equal(d[f"reaction{i}"][1:], (loss[:-1] > 1e-10) | (d[f"mark{i}"][:-1] > 0))
            w = d[f"cwnd{i}"]
            expected = np.where(d[f"reaction{i}"][1:] > 0, w[:-1]/2, w[:-1]+1)
            assert np.allclose(w[1:], expected, rtol=2e-11, atol=1e-9)
        assert np.max(d["r1_tx_f1"] + d["r1_tx_f2"]) <= 90 + 1e-8
        assert np.max(d["r2_tx_f1"] + d["r2_tx_f3"]) <= 45 + 1e-8
        checks[f"stored_trace_{mode}_capacity_OR_AIMD"] = "passed"
    cfg = Config(steps=1600, warmup=200, adaptation_block=200)
    a, _ = simulate("C", 42, cfg)
    b, _ = simulate("C", 42, cfg)
    assert np.array_equal(a, b)
    assert not np.allclose(a[-1, [COL[f"p{i}"] for i in (1,2,3)]], BASE_P)
    max_error = 0.0
    for boundary in range(cfg.adaptation_block, cfg.steps, cfg.adaptation_block):
        for i in (1, 2, 3):
            xbar = a[boundary-cfg.adaptation_block:boundary, COL[f"received{i}"]].mean()
            expected = np.clip(a[boundary-1, COL[f"p{i}"]] * np.exp(cfg.adaptation_gain*(ROUTE_PRICES[i-1]*xbar-1)), *cfg.p_bounds)
            max_error = max(max_error, abs(expected-a[boundary, COL[f"p{i}"]]))
    assert max_error < 1e-12
    checks["C_adapting_path_exact_reproducibility"] = "passed"
    checks["C_probability_update_max_abs_error"] = max_error
    path = Path(args.results) / "independent_audit.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
