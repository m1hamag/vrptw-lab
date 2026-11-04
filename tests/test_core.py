import pytest

from vrptw.alns import ALNS
from vrptw.greedy import nearest_neighbor
from vrptw.instance import all_instance_names, instance_path, load_bks, load_solomon
from vrptw.ortools_solver import solve_ortools
from vrptw.solution import evaluate


@pytest.fixture(scope="module")
def instances():
    return {name: load_solomon(instance_path(name)) for name in all_instance_names()}


def test_all_instances_parse_and_have_bks(instances):
    assert len(instances) == 56
    assert set(instances) == set(load_bks())
    for inst in instances.values():
        assert inst.n_nodes == 101
        assert inst.max_vehicles == 25
        assert inst.family in {"C1", "C2", "R1", "R2", "RC1", "RC2"}


def test_evaluate_catches_violations(instances):
    inst = instances["c101"]
    one_per_customer = evaluate(inst, [[c] for c in inst.customers])
    assert not one_per_customer.feasible
    assert one_per_customer.violations == ["машин 100 > 25"]

    missing = evaluate(inst, [[1, 2, 3]])
    assert any("не посещены" in v for v in missing.violations)

    # Первые клиенты C101 в порядке убывания дедлайна: последние в маршруте опоздают.
    by_due_desc = sorted(range(1, 11), key=lambda c: -inst.due[c])
    late = evaluate(inst, [by_due_desc, *[[c] for c in range(11, 101)]])
    assert any("опоздание" in v for v in late.violations)


def test_greedy_is_feasible_everywhere(instances):
    for inst in instances.values():
        ev = evaluate(inst, nearest_neighbor(inst))
        assert ev.feasible, (inst.name, ev.violations)


@pytest.mark.parametrize("name", ["c101", "r101", "rc201"])
def test_alns_returns_feasible_solution_and_consistent_cost(instances, name):
    inst = instances[name]
    result = ALNS(inst, seed=0).solve(time_limit_s=1.0)
    ev = evaluate(inst, result.routes)
    assert ev.feasible, ev.violations
    # Инкрементальное расписание внутри ALNS совпадает с независимой проверкой.
    assert ev.distance == pytest.approx(result.distance, abs=1e-6)
    assert ev.vehicles == result.vehicles


def test_alns_is_reproducible_by_iteration_count(instances):
    """Одинаковый seed — одинаковая последовательность решений (время влияет только на длину прогона)."""
    inst = instances["r102"]
    a = ALNS(inst, seed=7)
    b = ALNS(inst, seed=7)
    assert a._sample_q(100) == b._sample_q(100)


def test_ortools_returns_feasible_solution(instances):
    inst = instances["rc101"]
    ev = evaluate(inst, solve_ortools(inst, time_limit_s=1.0))
    assert ev.feasible, ev.violations
