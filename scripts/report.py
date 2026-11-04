"""Сводка по results/runs.csv: таблицы для README, карта маршрутов и доли операторов ALNS.

    uv run python scripts/report.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from vrptw.instance import instance_path, load_bks, load_solomon
from vrptw.solution import VEHICLE_COST, evaluate

RESULTS = Path(__file__).resolve().parents[1] / "results"
FAMILIES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
LABELS = {
    "greedy": "Жадный",
    "ortools": "OR-tools (GLS)",
    "alns": "ALNS",
    "alns-uniform-q": "ALNS, q равномерно",
    "alns-no-adapt": "ALNS без адаптации",
}
TIE = 0.001  # различие цели меньше 0,1 % считаем ничьей
OP_NAMES = {"random": "случайные", "worst": "самые дорогие", "shaw": "похожие (Shaw)", "route": "маршрут целиком"}

# Цвета и «чернила» — эталонная палитра скилла dataviz, светлая тема.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    # itertuples, а не iterrows: iterrows приводит строку к общему типу и превращает целые в float.
    for idx, *row in df.itertuples():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def per_instance(runs: pd.DataFrame) -> pd.DataFrame:
    """Медиана по seed для каждого (инстанс, метод)."""
    agg = runs.groupby(["instance", "family", "method"], as_index=False).agg(
        vehicles=("vehicles", "median"), distance=("distance", "median"), objective=("objective", "median")
    )
    return agg


def family_table(med: pd.DataFrame, bks: dict) -> str:
    methods = [m for m in LABELS if m in set(med.method)]
    header = "| Класс | Лучшее известное | " + " | ".join(LABELS[m] for m in methods) + " |"
    lines = [header, "|" + "---|" * (len(methods) + 2)]
    rows = FAMILIES + ["Всего (CNV / CTD)"]
    for fam in rows:
        if fam in FAMILIES:
            names = sorted(med[med.family == fam].instance.unique())
            b = np.array([bks[n] for n in names])
            cells = [f"{b[:, 0].mean():.2f} / {b[:, 1].mean():.1f}"]
            for m in methods:
                sub = med[(med.family == fam) & (med.method == m)]
                cells.append(f"{sub.vehicles.mean():.2f} / {sub.distance.mean():.1f}")
        else:
            b = np.array(list(bks.values()))
            cells = [f"{b[:, 0].sum():.0f} / {b[:, 1].sum():.0f}"]
            for m in methods:
                sub = med[med.method == m]
                cells.append(f"{sub.vehicles.sum():.0f} / {sub.distance.sum():.0f}")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def head_to_head(med: pd.DataFrame, a: str, b: str) -> tuple[str, dict]:
    """Победы/ничьи/поражения a против b по цели (машины × 1000 + расстояние) и тест Вилкоксона."""
    pivot = med.pivot(index="instance", columns="method", values="objective")
    fam = med.drop_duplicates("instance").set_index("instance").family
    rel = (pivot[a] - pivot[b]) / pivot[b]
    lines = [f"| Класс | {LABELS[a]} лучше | ничья | {LABELS[b]} лучше | медиана разницы цели |",
             "|---|---|---|---|---|"]
    totals = [0, 0, 0]
    for f in FAMILIES + ["Всего"]:
        r = rel if f == "Всего" else rel[fam == f]
        w, t, l = int((r < -TIE).sum()), int((r.abs() <= TIE).sum()), int((r > TIE).sum())
        if f != "Всего":
            totals = [totals[0] + w, totals[1] + t, totals[2] + l]
        lines.append(f"| {f} | {w} | {t} | {l} | {r.median() * 100:+.2f} % |")
    p = wilcoxon(pivot[a], pivot[b]).pvalue if (pivot[a] != pivot[b]).any() else 1.0
    return "\n".join(lines), {"wins": totals[0], "ties": totals[1], "losses": totals[2], "p": p}


def operator_shares(runs: pd.DataFrame, method: str = "alns") -> pd.DataFrame:
    rows = []
    for r in runs[runs.method == method].itertuples():
        data = json.loads((RESULTS / "solutions" / method / f"{r.instance}_s{r.seed}.json").read_text())
        for name, s in data["operators"].items():
            if name.startswith("destroy:"):
                rows.append({"family": r.family, "op": name.split(":")[1], "uses": s["uses"], "reward": s["reward"]})
    t = pd.DataFrame(rows).groupby(["family", "op"], as_index=False)[["uses", "reward"]].sum()
    t["share"] = t.uses / t.groupby("family").uses.transform("sum")
    t["success"] = t.reward / t.uses
    return t


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_operator_shares(shares: pd.DataFrame, out: Path) -> None:
    ops = list(OP_NAMES)
    names = OP_NAMES
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(FAMILIES))[::-1]
    left = np.zeros(len(FAMILIES))
    for op, color in zip(ops, SERIES):
        vals = np.array([shares[(shares.family == f) & (shares.op == op)].share.sum() for f in FAMILIES])
        # Белый зазор между сегментами — разделитель, а не часть данных.
        ax.barh(y, vals, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2, label=names[op])
        for yi, l, v in zip(y, left, vals):
            if v >= 0.06:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color=INK if color in ("#eda100", "#1baf7a") else "#ffffff")
        left += vals
    ax.set_yticks(y, FAMILIES)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Какие операторы разрушения выбирает адаптивный слой ALNS (доля вызовов)",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9,
              labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_routes(name: str, out: Path, low_util: float = 0.5) -> list[dict]:
    """Карта маршрутов OR-tools и ALNS; маршруты с загрузкой ниже low_util выделены."""
    inst = load_solomon(instance_path(name))
    panels, analysis = [], []
    for method, seed in (("ortools", 0), ("alns", 0)):
        routes = json.loads((RESULTS / "solutions" / method / f"{name}_s{seed}.json").read_text())["routes"]
        ev = evaluate(inst, routes)
        panels.append((method, ev))
        for i, s in enumerate(ev.routes):
            analysis.append({"method": method, "route": i + 1, "customers": len(s.customers),
                             "utilization": s.utilization, "distance": s.distance,
                             "waiting": s.waiting, "duration": s.duration})

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=SURFACE)
    for ax, (method, ev) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        weak = 0
        for s in ev.routes:
            path = [0, *s.customers, 0]
            is_weak = s.utilization < low_util
            weak += is_weak
            ax.plot(inst.x[path], inst.y[path], color=SERIES[1] if is_weak else SERIES[0],
                    linewidth=1.6 if is_weak else 1.1, alpha=0.95, zorder=2 if is_weak else 1)
        ax.scatter(inst.x[1:], inst.y[1:], s=9, color=MUTED, zorder=3, linewidths=0)
        ax.scatter([inst.x[0]], [inst.y[0]], s=70, marker="s", color=INK, zorder=4)
        ax.set_title(f"{LABELS[method]}: машин {ev.vehicles}, пробег {ev.distance:.1f}\n"
                     f"маршрутов с загрузкой ниже {low_util:.0%}: {weak}",
                     loc="left", fontsize=10, color=INK)
    handles = [
        plt.Line2D([], [], color=SERIES[0], lw=1.4, label="маршрут"),
        plt.Line2D([], [], color=SERIES[1], lw=1.8, label=f"маршрут с загрузкой ниже {low_util:.0%}"),
        plt.Line2D([], [], color=INK, marker="s", lw=0, label="склад"),
        plt.Line2D([], [], color=MUTED, marker="o", markersize=4, lw=0, label="клиент"),
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(f"Инстанс {name.upper()}, бюджет 10 с", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RESULTS / "runs.csv")
    parser.add_argument("--map", default=None, help="инстанс для карты маршрутов (по умолчанию — где ALNS и OR-tools разошлись сильнее всего)")
    args = parser.parse_args()

    runs = pd.read_csv(args.runs)
    infeasible = runs[~runs.feasible]
    bks = load_bks()
    med = per_instance(runs[runs.feasible])

    parts = [f"# Результаты\n\nБюджет {runs.time_limit.max():g} с на запуск, стохастические методы — "
             f"медиана по {runs[runs.method == 'alns'].seed.nunique()} seed. "
             f"Цель — {VEHICLE_COST:g} × машины + расстояние. Недопустимых решений: {len(infeasible)}.\n"]

    parts.append("## Среднее число машин / среднее расстояние по классам\n\n" + family_table(med, bks))

    h2h, stats = head_to_head(med, "alns", "ortools")
    parts.append(f"## ALNS против OR-tools\n\nТест Вилкоксона по 56 инстансам: p = {stats['p']:.3g}.\n\n{h2h}")

    ablation_notes = []
    for variant in ("alns-uniform-q", "alns-no-adapt"):
        if variant in set(med.method):
            table, s = head_to_head(med, "alns", variant)
            parts.append(f"## Абляция: ALNS против «{LABELS[variant]}»\n\nТест Вилкоксона: p = {s['p']:.3g}.\n\n{table}")
            ablation_notes.append((variant, s))

    shares = operator_shares(runs)
    plot_operator_shares(shares, RESULTS / "operators.png")
    pivot = shares.pivot(index="family", columns="op", values="success").reindex(FAMILIES)
    pivot = pivot[["random", "worst", "shaw", "route"]].rename(columns=OP_NAMES)
    pivot.index.name = "Класс"
    parts.append("## Операторы разрушения\n\n![доли операторов](operators.png)\n\n"
                 "Средняя награда за вызов (1 — новый рекорд, 0,5 — улучшение, 0,1 — принято хуже, 0 — отклонено):\n\n"
                 + md_table(pivot))

    if args.map is None:
        p = med.pivot(index="instance", columns="method", values="vehicles")
        args.map = (p["ortools"] - p["alns"]).abs().sort_values(kind="stable").index[-1]
    analysis = pd.DataFrame(plot_routes(args.map, RESULTS / f"routes_{args.map}.png"))
    summary = analysis.groupby("method").agg(
        routes=("route", "count"), mean_util=("utilization", "mean"),
        min_util=("utilization", "min"), waiting=("waiting", "sum"),
    ).rename(index=LABELS)
    summary.index.name = "Метод"
    summary.columns = ["Маршрутов", "Средняя загрузка", "Минимальная загрузка", "Суммарное ожидание"]
    summary["Маршрутов"] = summary["Маршрутов"].astype(int)
    parts.append(f"## Разбор маршрутов на {args.map.upper()}\n\n![маршруты](routes_{args.map}.png)\n\n"
                 + md_table(summary, fmt="{:.2f}"))

    (RESULTS / "summary.md").write_text("\n\n".join(parts) + "\n")
    print((RESULTS / "summary.md").read_text())


if __name__ == "__main__":
    main()
