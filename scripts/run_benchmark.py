"""Прогон методов на бенчмарке Solomon с одинаковым бюджетом времени на запуск.

    uv run python scripts/run_benchmark.py --time 10 --seeds 3 --jobs 4

Результаты дописываются в results/runs.csv, уже посчитанные запуски пропускаются,
так что прогон можно прервать и продолжить. Решения с историей сходимости
лежат в results/solutions/<метод>/<инстанс>_s<seed>.json.

Число процессов не больше числа производительных ядер: на эффективных ядрах
запуск получил бы меньше вычислений за то же время, и сравнение стало бы нечестным.
"""

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from vrptw.alns import ALNS
from vrptw.greedy import nearest_neighbor
from vrptw.instance import all_instance_names, instance_path, load_solomon
from vrptw.ortools_solver import solve_ortools
from vrptw.solution import evaluate

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIELDS = ["instance", "family", "method", "seed", "time_limit", "vehicles", "distance",
          "objective", "feasible", "iterations", "wall_time"]

# Детерминированные методы запускаются один раз, стохастические — по числу seed.
METHODS = {
    "greedy": ("deterministic", lambda inst, seed, t: (nearest_neighbor(inst), {})),
    "ortools": ("deterministic", lambda inst, seed, t: (solve_ortools(inst, t), {})),
    "alns": ("stochastic", lambda inst, seed, t: _alns(inst, seed, t)),
    "alns-uniform-q": ("stochastic", lambda inst, seed, t: _alns(inst, seed, t, q_dist="uniform")),
    "alns-no-adapt": ("stochastic", lambda inst, seed, t: _alns(inst, seed, t, adaptive=False)),
}


def _alns(inst, seed, time_limit, **kwargs):
    result = ALNS(inst, seed=seed, **kwargs).solve(time_limit)
    extra = {
        "iterations": result.iterations,
        "history": result.history,
        "operators": {name: asdict(s) for name, s in result.operators.items()},
    }
    return result.routes, extra


def run_one(instance: str, method: str, seed: int, time_limit: float) -> dict:
    inst = load_solomon(instance_path(instance))
    t0 = time.perf_counter()
    routes, extra = METHODS[method][1](inst, seed, time_limit)
    wall = time.perf_counter() - t0
    ev = evaluate(inst, routes)

    out = RESULTS / "solutions" / method / f"{instance}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"routes": routes, **extra}))

    return {
        "instance": instance, "family": inst.family, "method": method, "seed": seed,
        "time_limit": time_limit, "vehicles": ev.vehicles, "distance": round(ev.distance, 4),
        "objective": round(ev.objective, 4), "feasible": ev.feasible,
        "iterations": extra.get("iterations", ""), "wall_time": round(wall, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--instances", default="all", help="all или список через запятую")
    parser.add_argument("--time", type=float, default=10.0, help="секунд на запуск")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=RESULTS / "runs.csv")
    args = parser.parse_args()

    instances = all_instance_names() if args.instances == "all" else args.instances.split(",")
    methods = args.methods.split(",")
    done = set()
    if args.out.exists():
        with args.out.open() as f:
            done = {(r["instance"], r["method"], int(r["seed"])) for r in csv.DictReader(f)}

    tasks = []
    for method in methods:
        seeds = [0] if METHODS[method][0] == "deterministic" else range(args.seeds)
        tasks += [(i, method, s) for i in instances for s in seeds if (i, method, s) not in done]
    print(f"запусков: {len(tasks)}, уже есть: {len(done)}, бюджет {args.time} с, процессов {args.jobs}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with args.out.open("a", newline="") as f, ProcessPoolExecutor(args.jobs) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        futures = [pool.submit(run_one, *task, args.time) for task in tasks]
        for k, future in enumerate(as_completed(futures), 1):
            row = future.result()
            writer.writerow(row)
            f.flush()
            print(f"[{k}/{len(tasks)}] {row['instance']:6s} {row['method']:15s} s{row['seed']} "
                  f"машин {row['vehicles']:2d} расстояние {row['distance']:9.2f} ok={row['feasible']}", flush=True)


if __name__ == "__main__":
    main()
