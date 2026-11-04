"""VRPTW через Routing Solver из OR-tools — та же постановка, что в Ozon fresh."""

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .instance import Instance
from .solution import VEHICLE_COST

# OR-tools работает с целыми числами: расстояния и время умножаем на SCALE.
SCALE = 100


def solve_ortools(
    inst: Instance,
    time_limit_s: float = 10.0,
    metaheuristic: str = "GUIDED_LOCAL_SEARCH",
    # PATH_CHEAPEST_ARC не находит допустимого старта на узких окнах (R101), вставка находит.
    first_solution: str = "PARALLEL_CHEAPEST_INSERTION",
    vehicle_cost: float = VEHICLE_COST,
) -> list[list[int]]:
    n, n_vehicles = inst.n_nodes, inst.max_vehicles
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    cost = np.rint(inst.dist * SCALE).astype(int).tolist()
    # Время округляем вверх: решение, допустимое для солвера, допустимо и в точной арифметике.
    travel = np.ceil((inst.dist + inst.service[:, None]) * SCALE).astype(int).tolist()
    demand = inst.demand.tolist()

    def cost_cb(i, j):
        return cost[manager.IndexToNode(i)][manager.IndexToNode(j)]

    def time_cb(i, j):
        return travel[manager.IndexToNode(i)][manager.IndexToNode(j)]

    def demand_cb(i):
        return demand[manager.IndexToNode(i)]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(cost_cb))
    routing.SetFixedCostOfAllVehicles(int(vehicle_cost * SCALE))

    horizon = int(inst.due[0] * SCALE)
    routing.AddDimension(routing.RegisterTransitCallback(time_cb), horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in inst.customers:
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(
            int(np.ceil(inst.ready[node] * SCALE)), int(inst.due[node] * SCALE)
        )
    for v in range(n_vehicles):
        for index in (routing.Start(v), routing.End(v)):
            time_dim.CumulVar(index).SetRange(int(inst.ready[0] * SCALE), horizon)

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_cb), 0, [inst.capacity] * n_vehicles, True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, first_solution)
    params.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic)
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError(f"{inst.name}: OR-tools не нашёл допустимого решения за {time_limit_s} с")

    routes = []
    for v in range(n_vehicles):
        index, route = solution.Value(routing.NextVar(routing.Start(v))), []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        if route:
            routes.append(route)
    return routes
