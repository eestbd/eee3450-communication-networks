"""Plot C's long-run block rates and marking probabilities across seeds.

Usage: python plot_stability.py --results results_stability
Uses saved block measurements and does not rerun or change the experiment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from aimd_sim import PF
from plot_results import COLORS, save, setup_style
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results_stability"))
    args = parser.parse_args()
    with (args.results / "stability_blocks.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    meta = json.loads((args.results / "verification.json").read_text(encoding="utf-8"))
    config = meta["runs"][0]["config"]
    groups = {}
    for row in rows:
        key = (int(row["first_slot"]), int(row["last_slot"]))
        groups.setdefault(key, []).append(row)
    intervals = sorted(groups)
    nseeds = len({int(row["seed"]) for row in rows})
    assert all(len(groups[key]) == nseeds for key in intervals)
    assert all(len({r["seed"] for r in groups[key]}) == nseeds for key in intervals)
    widths = {end - begin + 1 for begin, end in intervals}
    assert len(widths) == 1
    width = next(iter(widths))
    x = np.array([(begin+end)/2 for begin, end in intervals]) / 1000
    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.1), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.4, 1)})
    for i, color in enumerate(COLORS, 1):
        for j, column in enumerate((f"throughput{i}", f"mean_p{i}")):
            values = np.array([[float(r[column]) for r in groups[key]] for key in intervals])
            mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
            ax = axes[i-1,j]
            ax.axvspan(0, config["warmup"]/1000, color="#D8DBDF", alpha=.50, zorder=0)
            ax.axvline(config["warmup"]/1000, ls=":", color="#555555", lw=1)
            ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.20,
                            label="seed 간 평균 ± 표준편차")
            ax.plot(x, mean, color=color, lw=1.3, marker="o", markersize=2.5,
                    label=f"{width:,} RTT 블록 평균")
            ax.set_xlim(0, config["steps"]/1000)
            if j == 0:
                ax.axhline(PF[i-1], color="#444444", ls="--", lw=1,
                           label=f"PF 목표 {PF[i-1]:.3f}")
                ax.set_ylabel(f"TCP{i} 수신률\n(packets/RTT)")
                ax.legend(loc="lower right", fontsize=8)
            else:
                ax.set_ylabel(f"TCP{i} 알림 확률 p{i}")
                ax.set_ylim(bottom=0)
            if i == 1:
                ax.set_title(("실제 수신률: 초기 변화와 후반 변동", "보완 C가 조정한 알림 확률")[j], fontsize=11)
                if j == 1:
                    ax.legend(loc="lower right", fontsize=8)
            if i == 3:
                ax.set_xlabel("시간 (×1,000 공통 기본 RTT)")
    fig.suptitle(f"별도 보완 C의 장기 안정성 확인 · {nseeds} seeds, 총 {config['steps']:,} RTT\n"
                 f"각 점은 {width:,} RTT 블록 통계 · 회색 {config['warmup']:,} RTT는 최종 평균에서 제외", fontsize=13)
    directory = args.results / "figures"
    directory.mkdir(exist_ok=True)
    save(fig, directory, "stabilization_C")
    print(f"Wrote {directory / 'stabilization_C.png'} and SVG")


if __name__ == "__main__":
    main()
