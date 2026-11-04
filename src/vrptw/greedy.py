"""Жадный бейзлайн: ближайший сосед с учётом окон (Solomon, 1987)."""

from .instance import Instance


def nearest_neighbor(inst: Instance, w_dist=0.4, w_time=0.4, w_urgency=0.2) -> list[list[int]]:
    """Строит маршруты по одному: из текущей точки едем к ближайшему допустимому клиенту.

    Близость — смесь расстояния, времени до начала обслуживания и запаса до закрытия окна.
    Если допустимых клиентов нет, машина возвращается на склад и выезжает следующая.
    """
    d, ready, due, service, demand = inst.dist, inst.ready, inst.due, inst.service, inst.demand
    unvisited = set(inst.customers)
    routes = []
    while unvisited:
        route, prev, t, load = [], 0, ready[0], 0
        while True:
            best, best_score, best_start = None, float("inf"), 0.0
            for c in unvisited:
                if load + demand[c] > inst.capacity:
                    continue
                start = max(ready[c], t + service[prev] + d[prev, c])
                if start > due[c] or start + service[c] + d[c, 0] > due[0]:
                    continue
                score = (
                    w_dist * d[prev, c]
                    + w_time * (start - (t + service[prev]))
                    + w_urgency * (due[c] - (t + service[prev] + d[prev, c]))
                )
                if score < best_score:
                    best, best_score, best_start = c, score, start
            if best is None:
                break
            route.append(best)
            unvisited.remove(best)
            prev, t, load = best, best_start, load + demand[best]
        if not route:
            raise ValueError(f"{inst.name}: клиента {min(unvisited)} нельзя обслужить даже отдельной машиной")
        routes.append(route)
    return routes
