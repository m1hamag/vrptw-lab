"""ALNS для VRPTW: разрушить часть решения, починить, принять или откатить.

Два отличия от классического ALNS (Ropke & Pisinger, 2006):

* Выбор операторов — probability matching по доле успешных применений, как в
  гиперэвристике HyperTree: оператор, который чаще улучшает решение, вызывается чаще,
  но у каждого остаётся минимальная вероятность.
* Число удаляемых клиентов берётся из распределения Леви, как длина шага в Cuckoo Search:
  обычно правка маленькая и дешёвая, изредка — крупная, чтобы выбраться из локального минимума.
"""

import math
import random
import time
from dataclasses import dataclass, field

from .instance import Instance
from .solution import VEHICLE_COST

INF = float("inf")


class _Route:
    """Маршрут со складом на концах и кэшем расписания для проверки вставки за O(1)."""

    __slots__ = ("nodes", "start", "latest", "load", "cost")

    def copy(self) -> "_Route":
        r = _Route()
        r.nodes, r.start, r.latest = self.nodes[:], self.start[:], self.latest[:]
        r.load, r.cost = self.load, self.cost
        return r


@dataclass
class OperatorStats:
    uses: int = 0
    reward: float = 0.0
    probability: float = 0.0


@dataclass
class ALNSResult:
    routes: list[list[int]]
    vehicles: int
    distance: float
    iterations: int
    history: list[tuple[float, int, float]]  # (секунда, машины, расстояние) при каждом новом рекорде
    operators: dict[str, OperatorStats] = field(default_factory=dict)


class ALNS:
    def __init__(
        self,
        inst: Instance,
        seed: int = 0,
        q_dist: str = "levy",
        q_min: int = 4,
        q_max: int | None = None,
        levy_beta: float = 1.5,
        levy_scale: float = 4.0,
        adaptive: bool = True,
        pm_alpha: float = 0.1,
        pm_min: float = 0.05,
        start_worse: float = 0.05,
        vehicle_cost: float = VEHICLE_COST,
    ):
        if q_dist not in ("levy", "uniform"):
            raise ValueError("q_dist: levy или uniform")
        self.inst = inst
        self.rng = random.Random(seed)
        self.q_dist, self.levy_beta, self.levy_scale = q_dist, levy_beta, levy_scale
        n = len(inst.customers)
        self.q_min = q_min
        self.q_max = q_max or min(40, int(0.4 * n))
        self.adaptive, self.pm_alpha, self.pm_min = adaptive, pm_alpha, pm_min
        self.start_worse = start_worse
        self.vehicle_cost = vehicle_cost

        # Списки Python заметно быстрее поэлементного доступа к numpy.
        self.d = inst.dist.tolist()
        self.ready, self.due = inst.ready.tolist(), inst.due.tolist()
        self.svc, self.dem = inst.service.tolist(), inst.demand.tolist()
        self.cap, self.max_vehicles = inst.capacity, inst.max_vehicles
        self.d_max = float(inst.dist.max())
        self.horizon = float(inst.due[0])
        self.dem_max = float(max(inst.demand.max(), 1))
        self.new_route_cost = [self._new_route_cost(u) for u in range(inst.n_nodes)]

        b = levy_beta
        self.levy_sigma = (
            math.gamma(1 + b) * math.sin(math.pi * b / 2)
            / (math.gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))
        ) ** (1 / b)

        self.destroy_ops = {
            "random": self._random_removal,
            "worst": self._worst_removal,
            "shaw": self._shaw_removal,
            "route": self._route_removal,
        }
        self.repair_ops = {"greedy": 1, "regret2": 2}

    # ------------------------------------------------------------------ маршруты

    def _make_route(self, customers: list[int]) -> _Route:
        r = _Route()
        r.nodes = [0, *customers, 0]
        self._refresh(r)
        return r

    def _refresh(self, r: _Route) -> None:
        """Пересчитывает начало обслуживания и самое позднее допустимое начало в каждом узле."""
        d, ready, due, svc, dem = self.d, self.ready, self.due, self.svc, self.dem
        nodes = r.nodes
        m = len(nodes)
        start = [0.0] * m
        start[0] = ready[0]
        cost, load = 0.0, 0
        for k in range(1, m):
            a, b = nodes[k - 1], nodes[k]
            cost += d[a][b]
            load += dem[b]
            s = start[k - 1] + svc[a] + d[a][b]
            start[k] = s if s > ready[b] else ready[b]
        latest = [0.0] * m
        latest[-1] = due[0]
        for k in range(m - 2, -1, -1):
            a, b = nodes[k], nodes[k + 1]
            l = latest[k + 1] - d[a][b] - svc[a]
            latest[k] = l if l < due[a] else due[a]
        r.start, r.latest, r.load, r.cost = start, latest, load, cost

    def _best_insertion(self, u: int, r: _Route) -> tuple[float, int]:
        """Самая дешёвая допустимая позиция для клиента u в маршруте r: (прирост, позиция)."""
        if r.load + self.dem[u] > self.cap:
            return INF, -1
        d, svc = self.d, self.svc
        du = d[u]
        ready_u, due_u, svc_u = self.ready[u], self.due[u], svc[u]
        nodes, start, latest = r.nodes, r.start, r.latest
        best, pos = INF, -1
        for k in range(len(nodes) - 1):
            a = nodes[k]
            departure = start[k] + svc[a]
            if departure > due_u:
                break  # выезды дальше по маршруту только позже
            s_u = departure + du[a]
            if s_u > due_u:
                continue
            if s_u < ready_u:
                s_u = ready_u
            b = nodes[k + 1]
            if s_u + svc_u + du[b] > latest[k + 1]:
                continue
            delta = du[a] + du[b] - d[a][b]
            if delta < best:
                best, pos = delta, k + 1
        return best, pos

    def _new_route_cost(self, u: int) -> float:
        if u == 0:
            return INF
        s = max(self.ready[u], self.ready[0] + self.d[0][u])
        if s > self.due[u] or s + self.svc[u] + self.d[u][0] > self.due[0]:
            return INF
        return 2 * self.d[0][u] + self.vehicle_cost

    def _objective(self, routes: list[_Route]) -> float:
        return sum(r.cost for r in routes) + self.vehicle_cost * len(routes)

    def _remove(self, routes: list[_Route], removed: set[int]) -> None:
        for r in routes:
            if any(c in removed for c in r.nodes):
                r.nodes = [c for c in r.nodes if c not in removed]
                self._refresh(r)
        routes[:] = [r for r in routes if len(r.nodes) > 2]

    # ------------------------------------------------------------- разрушение

    def _sample_q(self, n_assigned: int) -> int:
        hi = min(self.q_max, n_assigned)
        if self.q_dist == "uniform":
            return self.rng.randint(min(self.q_min, hi), hi)
        u = self.rng.gauss(0.0, self.levy_sigma)
        v = self.rng.gauss(0.0, 1.0)
        step = abs(u) / max(abs(v), 1e-12) ** (1 / self.levy_beta)
        return min(hi, self.q_min + int(self.levy_scale * step))

    def _random_removal(self, routes: list[_Route], q: int) -> list[int]:
        assigned = [c for r in routes for c in r.nodes[1:-1]]
        removed = self.rng.sample(assigned, min(q, len(assigned)))
        self._remove(routes, set(removed))
        return removed

    def _worst_removal(self, routes: list[_Route], q: int, power: float = 3.0) -> list[int]:
        """Удаляет клиентов, без которых маршрут сильнее всего сокращается (с рандомизацией)."""
        d = self.d
        removed, touched = [], []
        for _ in range(q):
            gains = []
            for r in routes:
                nodes = r.nodes
                for k in range(1, len(nodes) - 1):
                    a, c, b = nodes[k - 1], nodes[k], nodes[k + 1]
                    gains.append((d[a][c] + d[c][b] - d[a][b], c, r))
            if not gains:
                break
            gains.sort(key=lambda g: -g[0])
            _, c, r = gains[int(self.rng.random() ** power * len(gains))]
            r.nodes.remove(c)
            removed.append(c)
            if r not in touched:
                touched.append(r)
        # Узлы уже убраны из маршрутов, поэтому расписание пересчитываем здесь, а не в _remove.
        for r in touched:
            self._refresh(r)
        routes[:] = [r for r in routes if len(r.nodes) > 2]
        return removed

    def _shaw_removal(self, routes: list[_Route], q: int, power: float = 6.0) -> list[int]:
        """Удаляет «похожих» клиентов: близких по месту, времени обслуживания и спросу."""
        d, dem = self.d, self.dem
        start_of = {c: r.start[k] for r in routes for k, c in enumerate(r.nodes) if c}
        remaining = set(start_of)
        seed = self.rng.choice(sorted(remaining))
        removed = [seed]
        remaining.discard(seed)

        def relatedness(i, j):
            return (
                9 * d[i][j] / self.d_max
                + 3 * abs(start_of[i] - start_of[j]) / self.horizon
                + 2 * abs(dem[i] - dem[j]) / self.dem_max
            )

        while len(removed) < q and remaining:
            ref = self.rng.choice(removed)
            candidates = sorted(remaining, key=lambda j: relatedness(ref, j))
            j = candidates[int(self.rng.random() ** power * len(candidates))]
            removed.append(j)
            remaining.discard(j)
        self._remove(routes, set(removed))
        return removed

    def _route_removal(self, routes: list[_Route], q: int) -> list[int]:
        """Расформировывает маршрут целиком: так ALNS пробует обойтись меньшим числом машин."""
        if len(routes) <= 1:
            return self._random_removal(routes, q)
        if self.rng.random() < 0.5:
            victim = min(routes, key=lambda r: len(r.nodes))
        else:
            victim = self.rng.choice(routes)
        routes.remove(victim)
        return victim.nodes[1:-1]

    # ---------------------------------------------------------------- починка

    def _repair(self, routes: list[_Route], removed: list[int], regret: int) -> list[_Route] | None:
        """Вставляет клиентов по одному: regret=1 — самый дешёвый, regret=2 — с наибольшим сожалением.

        Сожаление — насколько дороже станет вставка, если упустить лучший маршрут.
        Кэш лучших вставок пересчитывается только для изменённого маршрута.
        """
        pending = list(removed)
        self.rng.shuffle(pending)
        cache = {u: [self._best_insertion(u, r) for r in routes] for u in pending}
        while pending:
            chosen, chosen_key, chosen_target = None, None, None
            can_open = len(routes) < self.max_vehicles
            for u in pending:
                best = second = INF
                best_route = -1
                for ri, (cost, _) in enumerate(cache[u]):
                    if cost < best:
                        best, second, best_route = cost, best, ri
                    elif cost < second:
                        second = cost
                new_cost = self.new_route_cost[u] if can_open else INF
                if new_cost < best:
                    best, second, best_route = new_cost, best, -1
                elif new_cost < second:
                    second = new_cost
                if best == INF:
                    return None  # клиента некуда поставить
                # Больший ключ выбирается раньше; при равном сожалении — более дешёвая вставка.
                key = (-best,) if regret == 1 else (second - best, -best)
                if chosen_key is None or key > chosen_key:
                    chosen, chosen_key, chosen_target = u, key, best_route
            pending.remove(chosen)
            del cache[chosen]
            if chosen_target == -1:
                route = self._make_route([chosen])
                routes.append(route)
                for u in pending:
                    cache[u].append(self._best_insertion(u, route))
            else:
                route = routes[chosen_target]
                _, pos = self._best_insertion(chosen, route)
                route.nodes.insert(pos, chosen)
                self._refresh(route)
                for u in pending:
                    cache[u][chosen_target] = self._best_insertion(u, route)
        return routes

    # ------------------------------------------------------- адаптивный выбор

    def _pick(self, probs: list[float]) -> int:
        x, acc = self.rng.random(), 0.0
        for i, p in enumerate(probs):
            acc += p
            if x < acc:
                return i
        return len(probs) - 1

    def _probabilities(self, quality: list[float]) -> list[float]:
        k = len(quality)
        total = sum(quality)
        if not self.adaptive or total <= 0:
            return [1 / k] * k
        return [self.pm_min + (1 - k * self.pm_min) * q / total for q in quality]

    # ------------------------------------------------------------- основной цикл

    def solve(self, time_limit_s: float) -> ALNSResult:
        t0 = time.perf_counter()
        customers = list(self.inst.customers)
        current = self._repair([], customers, regret=2)
        if current is None:
            raise RuntimeError(f"{self.inst.name}: не удалось построить стартовое решение")
        cur_cost = self._objective(current)
        best, best_cost = [r.copy() for r in current], cur_cost
        history = [(time.perf_counter() - t0, len(best), sum(r.cost for r in best))]

        # Начальная температура: маршрут на start_worse длиннее принимается с вероятностью 1/2.
        # Считаем от расстояния, а не от цели: лишняя машина почти никогда не принимается.
        t_start = -self.start_worse * sum(r.cost for r in current) / math.log(0.5)
        t_end = t_start * 1e-3

        d_names, r_names = list(self.destroy_ops), list(self.repair_ops)
        d_quality, r_quality = [1.0] * len(d_names), [1.0] * len(r_names)
        stats = {f"destroy:{n}": OperatorStats() for n in d_names}
        stats.update({f"repair:{n}": OperatorStats() for n in r_names})

        iterations = 0
        while (elapsed := time.perf_counter() - t0) < time_limit_s:
            iterations += 1
            temperature = t_start * (t_end / t_start) ** (elapsed / time_limit_s)
            di = self._pick(self._probabilities(d_quality))
            ri = self._pick(self._probabilities(r_quality))

            candidate = [r.copy() for r in current]
            q = self._sample_q(len(customers))
            removed = self.destroy_ops[d_names[di]](candidate, q)
            candidate = self._repair(candidate, removed, self.repair_ops[r_names[ri]])

            reward = 0.0
            if candidate is not None:
                cand_cost = self._objective(candidate)
                delta = cand_cost - cur_cost
                if cand_cost < best_cost - 1e-9:
                    reward = 1.0
                    best, best_cost = [r.copy() for r in candidate], cand_cost
                    history.append((time.perf_counter() - t0, len(best), sum(r.cost for r in best)))
                elif delta < -1e-9:
                    reward = 0.5
                elif delta > 1e-9 and self.rng.random() < math.exp(-delta / temperature):
                    reward = 0.1
                if reward > 0 or abs(delta) <= 1e-9:
                    current, cur_cost = candidate, cand_cost

            d_quality[di] += self.pm_alpha * (reward - d_quality[di])
            r_quality[ri] += self.pm_alpha * (reward - r_quality[ri])
            for name in (f"destroy:{d_names[di]}", f"repair:{r_names[ri]}"):
                stats[name].uses += 1
                stats[name].reward += reward

        for names, quality, prefix in ((d_names, d_quality, "destroy"), (r_names, r_quality, "repair")):
            for name, p in zip(names, self._probabilities(quality)):
                stats[f"{prefix}:{name}"].probability = p

        return ALNSResult(
            routes=[r.nodes[1:-1] for r in best],
            vehicles=len(best),
            distance=sum(r.cost for r in best),
            iterations=iterations,
            history=history,
            operators=stats,
        )
