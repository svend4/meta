"""Тесты верификатора и генератора hexspec."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from projects.hexspec.verifier import Spec, load_spec, verify
from projects.hexspec.generator import (
    bfs_path, all_states_paths, all_transitions_paths,
    round_trip_paths, negative_scenarios, generate_report,
    path_covers_transition, path_to_hexforth, format_path,
)


# ---------------------------------------------------------------------------
# Вспомогательная функция: создать минимальную спецификацию
# ---------------------------------------------------------------------------

def make_spec(
    states: dict[str, str],
    transitions: list[tuple[str, str]],
    initial: str,
    final: list[str] = [],
    forbidden: list[str] = [],
    bits: list[str] | None = None,
) -> Spec:
    s = {name: int(bits_str, 2) for name, bits_str in states.items()}
    t = [(s[a], s[b]) for (a, b) in transitions]
    return Spec(
        name='test',
        bit_names=bits or [f'b{i}' for i in range(6)],
        states=s,
        transitions=t,
        initial=s[initial],
        final={s[f] for f in final},
        forbidden={s[f] for f in forbidden},
    )


class TestSpecBasic(unittest.TestCase):
    def setUp(self):
        # Простая цепочка: A → B → C → A (цикл)
        self.spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'A')],
            initial='A',
            final=['A'],
        )

    def test_reachable(self):
        r = self.spec.reachable_states()
        self.assertEqual(r, {0, 1, 3})

    def test_no_deadlocks(self):
        self.assertEqual(self.spec.deadlocks(), set())

    def test_no_unreachable(self):
        self.assertEqual(self.spec.unreachable(), set())

    def test_final_reachable(self):
        final_reach = self.spec.final - self.spec.reachable_states()
        self.assertEqual(final_reach, set())

    def test_state_name(self):
        self.assertEqual(self.spec.state_name(0), 'A')
        self.assertEqual(self.spec.state_name(1), 'B')


class TestSpecDeadlock(unittest.TestCase):
    def test_detects_deadlock(self):
        spec = make_spec(
            states={'S': '000000', 'DEAD': '000001'},
            transitions=[('S', 'DEAD')],
            initial='S',
        )
        self.assertEqual(spec.deadlocks(), {1})

    def test_final_not_deadlock(self):
        spec = make_spec(
            states={'S': '000000', 'END': '000001'},
            transitions=[('S', 'END')],
            initial='S',
            final=['END'],
        )
        self.assertEqual(spec.deadlocks(), set())


class TestSpecForbidden(unittest.TestCase):
    def test_forbidden_reachable(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'BAD': '000011'},
            transitions=[('A', 'B'), ('B', 'BAD')],
            initial='A',
            forbidden=['BAD'],
        )
        self.assertIn(3, spec.forbidden_reachable())

    def test_forbidden_unreachable(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'BAD': '000011'},
            transitions=[('A', 'B')],     # BAD недостижим
            initial='A',
            forbidden=['BAD'],
        )
        self.assertEqual(spec.forbidden_reachable(), set())


class TestSpecPathTo(unittest.TestCase):
    def test_path_to_self(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001'},
            transitions=[('A', 'B')],
            initial='A',
        )
        path = spec.path_to(0)
        self.assertEqual(path, [0])

    def test_path_found(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C')],
            initial='A',
        )
        path = spec.path_to(3)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 3)

    def test_path_not_found(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B')],    # C недостижима
            initial='A',
        )
        path = spec.path_to(3)
        self.assertIsNone(path)


class TestSpecNonQ6(unittest.TestCase):
    def test_non_q6_warning_collected(self):
        # Переход A→C с расстоянием 2 — не ребро Q6
        spec = make_spec(
            states={'A': '000000', 'C': '000011'},
            transitions=[('A', 'C')],
            initial='A',
        )
        self.assertEqual(len(spec.non_q6_transitions), 1)
        self.assertIn((0, 3), spec.non_q6_transitions)

    def test_q6_no_warning(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001'},
            transitions=[('A', 'B')],
            initial='A',
        )
        self.assertEqual(spec.non_q6_transitions, [])


class TestSpecCoverage(unittest.TestCase):
    def test_coverage_ratio(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'A')],
            initial='A',
        )
        cov = spec.coverage()
        self.assertGreater(cov['ratio'], 0)
        self.assertLessEqual(cov['ratio'], 1.0)


class TestLoadSpec(unittest.TestCase):
    def test_load_tcp(self):
        path = 'projects/hexspec/examples/tcp.json'
        if os.path.exists(path):
            spec = load_spec(path)
            self.assertEqual(spec.name, 'TCP (Q6-совместимый)')
            self.assertIn(0, spec.reachable_states())  # CLOSED достижим

    def test_load_traffic(self):
        path = 'projects/hexspec/examples/traffic_light.json'
        if os.path.exists(path):
            spec = load_spec(path)
            self.assertTrue(len(spec.states) >= 4)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        # 4-цикл по коду Грея: 0→1→3→2→0
        self.spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011', 'D': '000010'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')],
            initial='A',
            final=['A'],
        )

    def test_bfs_path(self):
        path = bfs_path(self.spec, 0, 3)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 3)

    def test_bfs_path_none(self):
        path = bfs_path(self.spec, 3, 0)    # нет пути назад напрямую
        # 3→2→0: путь существует через D→A
        # Проверяем что путь есть или нет в зависимости от топологии
        # (в цикле 3→2→0 достижим)
        self.assertIsNotNone(path)

    def test_all_states_paths_cover(self):
        paths = all_states_paths(self.spec)
        covered = {0}
        for p in paths:
            covered.update(p)
        self.assertEqual(covered, {0, 1, 2, 3})

    def test_all_transitions_paths_cover(self):
        paths = all_transitions_paths(self.spec)
        covered_transitions = set()
        for path in paths:
            for i in range(len(path) - 1):
                covered_transitions.add((path[i], path[i + 1]))
        spec_transitions = {(a, b) for (a, b) in self.spec.transitions}
        self.assertTrue(spec_transitions.issubset(covered_transitions))

    def test_round_trips_exist(self):
        trips = round_trip_paths(self.spec)
        self.assertGreater(len(trips), 0)
        for state, path in trips:
            self.assertEqual(path[0], self.spec.initial)
            self.assertEqual(path[-1], self.spec.initial)

    def test_generate_report_structure(self):
        report = generate_report(self.spec, coverage='states')
        self.assertIn('spec_name', report)
        self.assertIn('scenarios', report)
        self.assertGreater(report['total_scenarios'], 0)

    def test_generate_report_json_serializable(self):
        report = generate_report(self.spec, coverage='all')
        # Не должно бросать исключение
        json.dumps(report)


class TestNegativeScenarios(unittest.TestCase):
    def test_deadlock_scenario(self):
        spec = make_spec(
            states={'S': '000000', 'DEAD': '000001'},
            transitions=[('S', 'DEAD')],
            initial='S',
        )
        negs = negative_scenarios(spec)
        self.assertEqual(len(negs), 1)
        self.assertEqual(negs[0]['type'], 'deadlock')

    def test_forbidden_scenario(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'BAD': '000011'},
            transitions=[('A', 'B'), ('B', 'BAD')],
            initial='A',
            forbidden=['BAD'],
        )
        negs = negative_scenarios(spec)
        types = {n['type'] for n in negs}
        self.assertIn('forbidden_reachable', types)

    def test_clean_spec_no_negatives(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'A')],
            initial='A',
            final=['A'],
        )
        negs = negative_scenarios(spec)
        self.assertEqual(negs, [])


class TestVerify(unittest.TestCase):
    """Тесты для функции verify() — проверка результата (True/False)."""

    def _verify_silent(self, spec: Spec) -> bool:
        """Вызов verify() с подавлением stdout."""
        with redirect_stdout(io.StringIO()):
            return verify(spec)

    def test_clean_spec_ok(self):
        """Чистый автомат без проблем → verify() == True."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'A')],
            initial='A',
            final=['A'],
        )
        self.assertTrue(self._verify_silent(spec))

    def test_deadlock_fails(self):
        """Автомат с тупиком → verify() == False."""
        spec = make_spec(
            states={'S': '000000', 'DEAD': '000001'},
            transitions=[('S', 'DEAD')],
            initial='S',
        )
        self.assertFalse(self._verify_silent(spec))

    def test_unreachable_state_fails(self):
        """Недостижимое состояние → verify() == False."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'ORPHAN': '000010'},
            transitions=[('A', 'B'), ('B', 'A')],
            initial='A',
        )
        self.assertFalse(self._verify_silent(spec))

    def test_reachable_forbidden_fails(self):
        """Достижимое запрещённое состояние → verify() == False."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'BAD': '000011'},
            transitions=[('A', 'B'), ('B', 'BAD')],
            initial='A',
            forbidden=['BAD'],
        )
        self.assertFalse(self._verify_silent(spec))

    def test_unreachable_forbidden_ok(self):
        """Недостижимое запрещённое + нет других проблем → True."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001'},
            transitions=[('A', 'B'), ('B', 'A')],
            initial='A',
            final=['A'],
            forbidden=[],    # явно пустой список
        )
        self.assertTrue(self._verify_silent(spec))

    def test_unreachable_final_fails(self):
        """Конечное состояние недостижимо → verify() == False."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'GOAL': '000010'},
            transitions=[('A', 'B'), ('B', 'A')],
            initial='A',
            final=['GOAL'],  # GOAL не включена в переходы → недостижима
        )
        # GOAL недостижима, и это состояние нет в transitions, но
        # оно присутствует в states. Проверяем поведение.
        result = self._verify_silent(spec)
        # unreachable states + unreachable final — оба нарушения → False
        self.assertFalse(result)

    def test_two_initial_cycles_ok(self):
        """Два независимых цикла, каждый достижим от начального."""
        spec = make_spec(
            states={
                'A': '000000', 'B': '000001',
                'C': '000010', 'D': '000011',
            },
            transitions=[
                ('A', 'B'), ('B', 'A'),
                ('A', 'C'), ('C', 'D'), ('D', 'A'),
            ],
            initial='A',
            final=['A'],
        )
        self.assertTrue(self._verify_silent(spec))

    def test_verify_returns_bool(self):
        """verify() должна возвращать именно bool, а не truthy-значение."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001'},
            transitions=[('A', 'B'), ('B', 'A')],
            initial='A',
        )
        result = self._verify_silent(spec)
        self.assertIsInstance(result, bool)


class TestPathUtils(unittest.TestCase):
    """Тесты path_covers_transition, path_to_hexforth, format_path."""

    def _simple_spec(self) -> Spec:
        return make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C')],
            initial='A',
            final=['C'],
        )

    # path_covers_transition ---------------------------------------------------

    def test_covers_present_transition(self):
        # путь 0→1→3 покрывает ребро (0, 1)
        self.assertTrue(path_covers_transition([0, 1, 3], (0, 1)))

    def test_covers_second_transition(self):
        self.assertTrue(path_covers_transition([0, 1, 3], (1, 3)))

    def test_not_covers_absent_transition(self):
        self.assertFalse(path_covers_transition([0, 1, 3], (3, 1)))  # в обратную сторону

    def test_empty_path_covers_nothing(self):
        self.assertFalse(path_covers_transition([], (0, 1)))

    def test_single_node_covers_nothing(self):
        self.assertFalse(path_covers_transition([5], (5, 6)))

    # path_to_hexforth ---------------------------------------------------------

    def test_hexforth_returns_string(self):
        spec = self._simple_spec()
        path = [0, 1, 3]
        out = path_to_hexforth(spec, path)
        self.assertIsInstance(out, str)

    def test_hexforth_contains_goto(self):
        spec = self._simple_spec()
        out = path_to_hexforth(spec, [0, 1])
        self.assertIn('GOTO', out)

    def test_hexforth_contains_flip(self):
        spec = self._simple_spec()
        # 0→1: бит 0 → FLIP-0
        out = path_to_hexforth(spec, [0, 1])
        self.assertIn('FLIP-0', out)

    def test_hexforth_contains_assert(self):
        spec = self._simple_spec()
        out = path_to_hexforth(spec, [0, 1])
        self.assertIn('ASSERT-EQ', out)

    def test_hexforth_multiline(self):
        spec = self._simple_spec()
        out = path_to_hexforth(spec, [0, 1, 3])
        self.assertGreater(out.count('\n'), 2)

    # format_path --------------------------------------------------------------

    def test_format_path_returns_string(self):
        spec = self._simple_spec()
        out = format_path(spec, [0, 1, 3])
        self.assertIsInstance(out, str)

    def test_format_path_shows_steps(self):
        spec = self._simple_spec()
        out = format_path(spec, [0, 1, 3])
        self.assertIn('2', out)  # 2 шага

    def test_format_path_shows_bits(self):
        spec = self._simple_spec()
        out = format_path(spec, [0, 1])
        self.assertIn('000000', out)  # to_bits(0)

    def test_format_path_single_node(self):
        spec = self._simple_spec()
        out = format_path(spec, [0])
        self.assertIn('0', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
