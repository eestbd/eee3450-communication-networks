"""Create publication-readable PNG/SVG plots from saved simulator CSV results.

Usage: python plot_results.py --results results
No simulator or raw result file is modified.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from aimd_sim import PF


COLORS = ("#C03E3E", "#2465A5", "#23845D")
MODE_NAMES = {"A": "A: 손실 기반 AIMD", "B": "B: 원래 고정 확률 방식", "C": "C: 수신률 피드백 보완"}


def setup_style():
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "grid.linewidth": .5,
                         "savefig.dpi": 200, "svg.fonttype": "path",
                         "legend.frameon": False})


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", facecolor="white")
    fig.savefig(directory / f"{name}.svg", facecolor="white")
    plt.close(fig)


def block_stats(x, y, width=100):
    pieces = [(x[k:k+width], y[k:k+width]) for k in range(0, len(y), width)]
    return tuple(np.array([f(a, b) for a, b in pieces]) for f in (
        lambda a,b: np.mean(a), lambda a,b: np.min(b),
        lambda a,b: np.max(b), lambda a,b: np.mean(b)))


def cwnd_plot(data, mode, warmup, directory, seed):
    t, n = data["slot"], len(data)
    width = max(100, int(np.ceil(n / 200)))
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.65, 1)})
    for i, color in enumerate(COLORS, 1):
        y = data[f"cwnd{i}"]
        bt, low, high, avg = block_stats(t, y, width)
        ax = axes[i-1, 0]
        ax.axvspan(0, warmup, color="#D8DBDF", alpha=.5, zorder=0)
        ax.fill_between(bt, low, high, color=color, alpha=.20,
                        label=f"{width:,} RTT 블록 최소–최대")
        ax.plot(bt, avg, color=color, linewidth=1.2, label=f"{width:,} RTT 블록 평균")
        ax.set(xlim=(0, n), ylim=(0, max(y)*1.08), ylabel=f"TCP{i} cwnd\n(packets)")
        ax.axvline(warmup, color="#555555", ls=":", linewidth=1)
        detail = axes[i-1, 1]
        left = max(0, n - 1000)
        detail.plot(t[left:], y[left:], color=color, linewidth=.85)
        detail.set(xlim=(t[left], n), ylim=(0, max(y[left:])*1.08))
        if i == 1:
            ax.set_title("전체 실행: 블록 통계로 변동 범위 표시", fontsize=11)
            detail.set_title("마지막 1,000 RTT: 원래 cwnd", fontsize=11)
            ax.legend(loc="upper right", fontsize=8)
        if i == 3:
            ax.set_xlabel("시간 (공통 기본 RTT)")
            detail.set_xlabel("시간 (공통 기본 RTT)")
    fig.suptitle(f"{MODE_NAMES[mode]} · 대표 seed {seed}\n회색: 초기 {warmup:,} RTT 제외, 측정: {warmup+1:,}–{n:,} RTT", fontsize=13)
    save(fig, directory, f"cwnd_{mode}")


def throughput_plot(data, mode, warmup, directory, seed):
    t = data["slot"][warmup:]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), constrained_layout=True, sharex=True)
    for i, color in enumerate(COLORS, 1):
        received = data[f"received{i}"][warmup:]
        cummean = np.cumsum(received) / np.arange(1, len(received)+1)
        ax = axes[i-1]
        ax.plot(t, cummean, color=color, lw=1.35, label="측정 시작 이후 누적평균")
        ax.axhline(PF[i-1], color="#444444", ls="--", lw=1,
                   label=f"PF 목표 {PF[i-1]:.3f}")
        ax.set(ylabel=f"TCP{i} 수신률\n(packets/RTT)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.text(.985, .08, f"최종 {cummean[-1]:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=color,
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":.85})
    axes[-1].set(xlabel="시간 (공통 기본 RTT)", xlim=(warmup+1, len(data)))
    fig.suptitle(f"{MODE_NAMES[mode]} · 대표 seed {seed}\n누적평균 범위: {warmup+1:,}번째 RTT부터 현재 RTT까지 (초기 표본은 변동이 큼)", fontsize=13)
    save(fig, directory, f"throughput_{mode}")


def comparison_plot(results, directory, config):
    with (results / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {r["mode"]: r for r in csv.DictReader(f) if r["scenario"] == "main"}
    modes = [m for m in ("A", "B", "C") if m in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.45, 1)})
    x, width = np.arange(3), .22
    for k, mode in enumerate(modes):
        row = rows[mode]
        means = [float(row[f"throughput{i}_mean"]) for i in (1,2,3)]
        sd = [float(row[f"throughput{i}_seed_sd"]) for i in (1,2,3)]
        axes[0].bar(x+(k-1)*width, means, width=width, yerr=sd, capsize=3,
                    color=("#6B7785", "#3287B7", "#DC8C32")[k],
                    label=f"{mode} (n={row['n_runs']})")
    axes[0].scatter(x, PF, marker="D", color="#111111", s=24, zorder=5, label="PF 목표")
    axes[0].set(xticks=x, xticklabels=["TCP1", "TCP2", "TCP3"],
                ylabel="평균 수신률 (packets/RTT)", ylim=(0, 85),
                title="수신률: 오차막대는 seed 간 표준편차")
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    for j, (router, color) in enumerate(((1,"#366D99"),(2,"#23845D"))):
        values = [100*float(rows[m][f"utilization_r{router}_mean"]) for m in modes]
        axes[1].bar(np.arange(len(modes))+(j-.5)*.32, values, width=.32,
                    color=color, label=f"R{router}")
        for k, value in enumerate(values):
            axes[1].text(k+(j-.5)*.32, value+1.2, f"{value:.1f}", ha="center", fontsize=8)
    axes[1].set(xticks=np.arange(len(modes)), xticklabels=modes, ylabel="출력 링크 활용률 (%)",
                ylim=(0, 112), title="실제 라우터 출력량으로 계산")
    axes[1].axhline(100, ls=":", color="#555555", lw=1)
    axes[1].legend(fontsize=8, loc="upper center", ncol=2)
    samples = ", ".join(f"{m}: {rows[m]['n_runs']}회" for m in modes)
    fig.suptitle(f"공통 네트워크에서 A / 원래 B / 별도 보완 C 비교\n측정: {config['warmup']+1:,}–{config['steps']:,} RTT · {samples} (A는 결정론적)", fontsize=12)
    save(fig, directory, "comparison_main")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    setup_style()
    metadata = json.loads((args.results / "verification.json").read_text(encoding="utf-8"))
    warmup = metadata["config"]["warmup"]
    seed = metadata["seeds"][0]
    directory = args.results / "figures"
    directory.mkdir(exist_ok=True)
    for mode in ("A", "B", "C"):
        path = args.results / f"trace_{mode}.csv"
        if path.exists():
            data = np.genfromtxt(path, delimiter=",", names=True)
            cwnd_plot(data, mode, warmup, directory, seed)
            throughput_plot(data, mode, warmup, directory, seed)
    comparison_plot(args.results, directory, metadata["config"])
    print(f"Wrote PNG/SVG plots to {directory}")


if __name__ == "__main__":
    main()
