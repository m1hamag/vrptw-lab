"""Инстансы Solomon для VRPTW: склад 0 и клиенты 1..n."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Instance:
    name: str
    capacity: int
    max_vehicles: int
    x: np.ndarray
    y: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray
    # Евклидовы расстояния без округления; время в пути равно расстоянию.
    dist: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.demand)

    @property
    def customers(self) -> range:
        return range(1, self.n_nodes)

    @property
    def family(self) -> str:
        """C1, C2, R1, R2, RC1 или RC2."""
        letters = self.name.rstrip("0123456789")
        return letters.upper() + self.name[len(letters)]


def load_solomon(path: str | Path) -> Instance:
    lines = Path(path).read_text().splitlines()
    name = lines[0].strip().lower()

    vehicle_header = next(i for i, line in enumerate(lines) if "CAPACITY" in line)
    max_vehicles, capacity = map(int, lines[vehicle_header + 1].split())

    rows = []
    for line in lines[vehicle_header + 2:]:
        parts = line.split()
        if len(parts) == 7 and all(p.lstrip("-").isdigit() for p in parts):
            rows.append([int(p) for p in parts])
    table = np.array(rows, dtype=float)
    if not np.array_equal(table[:, 0], np.arange(len(table))):
        raise ValueError(f"{path}: узлы должны идти по порядку с 0")

    x, y = table[:, 1], table[:, 2]
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    return Instance(
        name=name,
        capacity=capacity,
        max_vehicles=max_vehicles,
        x=x,
        y=y,
        demand=table[:, 3].astype(int),
        ready=table[:, 4],
        due=table[:, 5],
        service=table[:, 6],
        dist=dist,
    )


def instance_path(name: str) -> Path:
    return DATA_DIR / "solomon" / f"{name.lower()}.txt"


def all_instance_names() -> list[str]:
    return sorted(p.stem for p in (DATA_DIR / "solomon").glob("*.txt"))


def load_bks() -> dict[str, tuple[int, float]]:
    """Лучшие известные решения (сначала минимум машин, потом расстояние), источник — SINTEF."""
    import pandas as pd

    table = pd.read_csv(DATA_DIR / "bks.csv")
    return {row.instance: (int(row.vehicles), float(row.distance)) for row in table.itertuples()}
