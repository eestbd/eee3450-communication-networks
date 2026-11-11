"""Two-router fluid AIMD simulator. Time unit: one common base RTT.

Only numpy is needed for storage/measurement; the network itself uses scalar math.
See README.md for the exact event order and abstraction limits.
"""
from dataclasses import asdict, dataclass
import math
import random

import numpy as np


PF = np.array([45 - 15 * math.sqrt(3), 45 + 15 * math.sqrt(3), 15 * math.sqrt(3)])
PRICES = np.array([1 / PF[1], 1 / PF[2]])
ROUTE_PRICES = np.array([sum(PRICES), PRICES[0], PRICES[1]])
BASE_P = 2 * ROUTE_PRICES / (1 + 2 * ROUTE_PRICES)
FIELDS = (
    ["slot"] + [f"cwnd{i}" for i in (1, 2, 3)]
    + [f"sent{i}" for i in (1, 2, 3)]
    + [f"received{i}" for i in (1, 2, 3)]
    + [f"flight{i}" for i in (1, 2, 3)]
    + ["q_r1_f1", "q_r1_f2", "q_r2_f1", "q_r2_f3"]
    + ["r1_tx_f1", "r1_tx_f2", "r2_tx_f1", "r2_tx_f3"]
    + ["drop_r1_f1", "drop_r1_f2", "drop_r2_f1", "drop_r2_f3"]
    + [f"mark{i}" for i in (1, 2, 3)]
    + [f"reaction{i}" for i in (1, 2, 3)]
    + [f"p{i}" for i in (1, 2, 3)]
)
COL = {name: i for i, name in enumerate(FIELDS)}


@dataclass(frozen=True)
class Config:
    steps: int = 20_000
    warmup: int = 5_000
    capacities: tuple = (90.0, 45.0)
    queue_factor: float = 2.0  # each independent ingress queue = factor * link capacity
    initial_cwnd: tuple = (1.0, 1.0, 1.0)
    adaptation_block: int = 500
    adaptation_gain: float = 0.5
    p_bounds: tuple = (1e-5, 0.5)

    def validate(self):
        assert 0 <= self.warmup < self.steps
        assert min(self.capacities) > 0 and self.queue_factor > 0
        assert len(self.initial_cwnd) == 3 and min(self.initial_cwnd) > 0
        assert self.adaptation_block > 0 and self.adaptation_gain > 0
        assert 0 < self.p_bounds[0] < self.p_bounds[1] < 1


def fair_service(a, b, capacity):
    """Unweighted max-min service for two queues; all spare capacity is reused."""
    sa = min(a, max(capacity / 2, capacity - b))
    sb = min(b, capacity - sa)
    return sa, sb


def enqueue(old, incoming, limit):
    total = old + incoming
    dropped = max(0.0, total - limit)
    return min(total, limit), dropped


def simulate(mode="B", seed=0, config=Config()):
    """A=loss only; B=fixed original p; C=separate receiver-rate feedback extension.

    Returned trace has one row per base RTT, measured at end of slot except cwnd
    (used to send in that slot), reaction (applied at its start), p (marking p).
    Every slot checks conservation, capacity, queue bounds and the serial route.
    """
    config.validate()
    assert mode in ("A", "B", "C")
    rng = random.Random(seed)
    c1, c2 = config.capacities
    limits = [config.queue_factor * c1] * 2 + [config.queue_factor * c2] * 2
    w, flight = list(config.initial_cwnd), [0.0] * 3
    q, terminal, last_congestion = [0.0] * 4, [0.0] * 3, [False] * 3
    sent_total, recv_total, drop_total, freed_total = ([0.0] * 3 for _ in range(4))
    p = [0.0] * 3 if mode == "A" else BASE_P.tolist()
    block_received = [0.0] * 3
    data = np.empty((config.steps, len(FIELDS)), dtype=np.float64)
    max_residual = 0.0
    for k in range(config.steps):
        reaction = [0, 0, 0]
        if k:
            for i in range(3):
                flight[i] = max(0.0, flight[i] - terminal[i])
                freed_total[i] += terminal[i]
                reaction[i] = int(last_congestion[i])
                w[i] = w[i] / 2 if last_congestion[i] else w[i] + 1
        sent = [max(w[i] - flight[i], 0.0) for i in range(3)]
        for i in range(3):
            flight[i] += sent[i]
            sent_total[i] += sent[i]
        old_q = q.copy()
        q[0], d10 = enqueue(q[0], sent[0], limits[0])
        q[1], d11 = enqueue(q[1], sent[1], limits[1])
        s10, s11 = fair_service(q[0], q[1], c1)
        q[0] -= s10
        q[1] -= s11
        # Flow 1 can enter R2 only through R1's actual departure s10.
        q[2], d20 = enqueue(q[2], s10, limits[2])
        q[3], d21 = enqueue(q[3], sent[2], limits[3])
        s20, s21 = fair_service(q[2], q[3], c2)
        q[2] -= s20
        q[3] -= s21
        received, dropped = [s20, s11, s21], [d10 + d20, d11, d21]
        marks = [int(rng.random() < p[i]) if mode != "A" else 0 for i in range(3)]
        terminal = [received[i] + dropped[i] for i in range(3)]
        last_congestion = [bool(marks[i] or dropped[i] > 1e-10) for i in range(3)]
        for i in range(3):
            recv_total[i] += received[i]
            drop_total[i] += dropped[i]
            block_received[i] += received[i]
        queued = [q[0] + q[2], q[1], q[3]]
        # Independent identities: attempted data = terminal data + live queue data;
        # flight includes terminal data awaiting next-slot notification.
        residuals = [
            abs(sent_total[i] - recv_total[i] - drop_total[i] - queued[i])
            for i in range(3)
        ] + [abs(flight[i] - queued[i] - terminal[i]) for i in range(3)]
        residuals += [abs(sent_total[i] - freed_total[i] - flight[i]) for i in range(3)]
        # Local queue equations verify the route and all four drop locations.
        residuals += [abs(old_q[0] + sent[0] - d10 - s10 - q[0]),
                      abs(old_q[1] + sent[1] - d11 - s11 - q[1]),
                      abs(old_q[2] + s10 - d20 - s20 - q[2]),
                      abs(old_q[3] + sent[2] - d21 - s21 - q[3])]
        max_residual = max(max_residual, *residuals)
        assert max(residuals) < 1e-6, (mode, seed, k, residuals)
        assert s10 + s11 <= c1 + 1e-9 and s20 + s21 <= c2 + 1e-9
        assert all(-1e-9 <= q[j] <= limits[j] + 1e-9 for j in range(4))
        assert all(d >= 0 for d in dropped)
        assert abs((s10 + s11) - min(c1, old_q[0] + sent[0] - d10 + old_q[1] + sent[1] - d11)) < 1e-9
        assert abs((s20 + s21) - min(c2, old_q[2] + s10 - d20 + old_q[3] + sent[2] - d21)) < 1e-9
        data[k] = [k + 1, *w, *sent, *received, *flight, *q, s10, s11, s20, s21,
                   d10, d11, d20, d21, *marks, *reaction, *p]
        # C changes only the sender's marking probabilities, never router service.
        # xbar is the receiver's last complete block rate (equiv. delayed ACK rate,
        # apart from one boundary sample); the new p first applies next slot.
        if mode == "C" and (k + 1) % config.adaptation_block == 0:
            for i in range(3):
                xbar = block_received[i] / config.adaptation_block
                logp = math.log(p[i]) + config.adaptation_gain * (ROUTE_PRICES[i] * xbar - 1)
                p[i] = math.exp(min(math.log(config.p_bounds[1]), max(math.log(config.p_bounds[0]), logp)))
            block_received = [0.0] * 3
    return data, {"slots_checked": config.steps, "max_conservation_residual": max_residual,
                  "mode": mode, "seed": seed, "config": asdict(config)}


def measure(data, config, mode, seed, scenario="main"):
    d = data[config.warmup:]
    mean = lambda name: float(np.mean(d[:, COL[name]]))
    result = {"scenario": scenario, "mode": mode, "seed": seed,
              "steps": config.steps, "warmup": config.warmup,
              "queue_factor": config.queue_factor, "initial_cwnd": ":".join(map(str, config.initial_cwnd))}
    rates = np.array([mean(f"received{i}") for i in (1, 2, 3)])
    for i in (1, 2, 3):
        result.update({f"throughput{i}": rates[i-1], f"pf_error_pct{i}": abs(rates[i-1] / PF[i-1] - 1) * 100,
                       f"mean_cwnd{i}": mean(f"cwnd{i}"), f"mean_flight{i}": mean(f"flight{i}"),
                       f"mean_sent{i}": mean(f"sent{i}"), f"mark_rate{i}": mean(f"mark{i}"),
                       f"reaction_rate{i}": mean(f"reaction{i}"), f"mean_p{i}": mean(f"p{i}"),
                       f"blocked_slot_fraction{i}": float(np.mean(d[:, COL[f"sent{i}"]] < 1e-10))})
    for j, cols in enumerate((("r1_tx_f1", "r1_tx_f2"), ("r2_tx_f1", "r2_tx_f3")), 1):
        load = d[:, [COL[c] for c in cols]].sum(axis=1)
        result[f"utilization_r{j}"] = float(load.mean() / config.capacities[j-1])
        result[f"not_full_fraction_r{j}"] = float(np.mean(load < config.capacities[j-1] - 1e-8))
    for col in FIELDS:
        if col.startswith("drop_") or col.startswith("q_"):
            result[f"mean_{col}"] = mean(col)
    result["drop_per_rtt"] = sum(mean(c) for c in FIELDS if c.startswith("drop_"))
    result["utility"] = float(np.log(rates).sum())
    result["utility_gap"] = float(np.log(PF).sum() - result["utility"])
    result["jain"] = float(rates.sum() ** 2 / (3 * np.sum(rates ** 2)))
    return result
