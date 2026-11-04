"""Проверка решения и метрики маршрутов.

Решение — список маршрутов, маршрут — список клиентов без склада:
каждая машина выезжает со склада 0 и возвращается на него.
"""

from dataclasses import dataclass, field

from .instance import Instance

EPS = 1e-6

# Цель обоих решателей: VEHICLE_COST * машины + расстояние. Машина дороже любого
# выигрыша в километраже, поэтому это почти лексикографический порядок (машины, расстояние),
# как в лучших известных решениях Solomon, и та же утилизация рейсов, что нужна в доставке.
VEHICLE_COST = 1000.0


@dataclass
class RouteStats:
    customers: list[int]
    distance: float
    load: int
    utilization: float  # загрузка / вместимость
    duration: float  # от выезда со склада до возвращения
    waiting: float  # суммарное ожидание открытия окон
    late: float  # суммарное опоздание (у допустимого решения 0)


@dataclass
class Evaluation:
    distance: float
    vehicles: int
    feasible: bool
    violations: list[str] = field(default_factory=list)
    routes: list[RouteStats] = field(default_factory=list)

    @property
    def objective(self) -> float:
        return VEHICLE_COST * self.vehicles + self.distance

    @property
    def mean_utilization(self) -> float:
        return sum(r.utilization for r in self.routes) / max(len(self.routes), 1)


def route_stats(inst: Instance, route: list[int]) -> RouteStats:
    d, ready, due, service = inst.dist, inst.ready, inst.due, inst.service
    t = ready[0]  # начало обслуживания в текущем узле
    prev = 0
    distance = waiting = late = 0.0
    for c in [*route, 0]:
        distance += d[prev, c]
        arrival = t + service[prev] + d[prev, c]
        if arrival < ready[c]:
            waiting += ready[c] - arrival
            arrival = ready[c]
        if arrival > due[c] + EPS:
            late += arrival - due[c]
        t, prev = arrival, c
    load = int(sum(inst.demand[c] for c in route))
    return RouteStats(
        customers=list(route),
        distance=distance,
        load=load,
        utilization=load / inst.capacity,
        duration=t - ready[0],
        waiting=waiting,
        late=late,
    )


def evaluate(inst: Instance, routes: list[list[int]]) -> Evaluation:
    routes = [r for r in routes if r]
    stats = [route_stats(inst, r) for r in routes]
    violations = []

    visited = [c for r in routes for c in r]
    missing = set(inst.customers) - set(visited)
    if missing:
        violations.append(f"не посещены клиенты: {sorted(missing)[:10]}")
    if len(visited) != len(set(visited)):
        violations.append("клиент посещён больше одного раза")
    if any(c == 0 or c >= inst.n_nodes for c in visited):
        violations.append("в маршруте есть склад или несуществующий узел")
    if len(routes) > inst.max_vehicles:
        violations.append(f"машин {len(routes)} > {inst.max_vehicles}")
    for i, s in enumerate(stats):
        if s.load > inst.capacity:
            violations.append(f"маршрут {i}: загрузка {s.load} > {inst.capacity}")
        if s.late > 0:
            violations.append(f"маршрут {i}: опоздание {s.late:.2f}")

    return Evaluation(
        distance=sum(s.distance for s in stats),
        vehicles=len(routes),
        feasible=not violations,
        violations=violations,
        routes=stats,
    )
