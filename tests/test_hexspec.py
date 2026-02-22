"""Тесты верификатора и генератора hexspec."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import json
import tempfile
import os
import unittest
from projects.hexspec.verifier import Spec, load_spec, verify
from projects.hexspec.verifier import verify
from projects.hexspec.generator import (
    bfs_path, all_states_paths, all_transitions_paths,
    round_trip_paths, negative_scenarios, generate_report,
    path_to_hexforth, format_path,
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


class TestSpecInvariant(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'A')],
            initial='A',
            final=['A'],
        )

    def test_invariant_all_pass(self):
        """Инвариант, всегда True → пустой список нарушителей."""
        violations = self.spec.check_invariant(lambda s: True, 'always_true')
        self.assertEqual(violations, [])

    def test_invariant_some_fail(self):
        """Инвариант, ложный для состояния 1 → [1] в нарушителях."""
        violations = self.spec.check_invariant(lambda s: s != 1, 'not_B')
        self.assertIn(1, violations)


class TestSpecBackwardReachable(unittest.TestCase):
    def test_backward_from_final(self):
        """Обратная достижимость из финального состояния включает начальное."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C')],
            initial='A',
            final=['C'],
        )
        back = spec.backward_reachable({3})   # 3 = C
        self.assertIn(0, back)  # A достижима назад

    def test_reverse_transitions(self):
        """reverse_transitions содержит (B,A) для каждого (A,B)."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001'},
            transitions=[('A', 'B')],
            initial='A',
        )
        rev = spec.reverse_transitions()
        self.assertIn((1, 0), rev)
        self.assertNotIn((0, 1), rev)


class TestVerifyFunction(unittest.TestCase):
    def test_clean_spec_returns_true(self):
        """verify() для чистой спецификации (без deadlock/forbidden) → True."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'A')],
            initial='A',
            final=['A'],
        )
        self.assertTrue(verify(spec))

    def test_forbidden_spec_returns_false(self):
        """verify() для спецификации с достижимым запрещённым → False."""
        spec = make_spec(
            states={'A': '000000', 'B': '000001', 'BAD': '000011'},
            transitions=[('A', 'B'), ('B', 'BAD')],
            initial='A',
            forbidden=['BAD'],
        )
        self.assertFalse(verify(spec))


class TestPathFormatting(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'A')],
            initial='A',
        )

    def test_path_to_hexforth_contains_goto(self):
        s = path_to_hexforth(self.spec, [0, 1, 3])
        self.assertIn('GOTO', s)
        self.assertIn('ASSERT-EQ', s)

    def test_format_path_contains_names(self):
        s = format_path(self.spec, [0, 1])
        self.assertIn('A', s)
        self.assertIn('B', s)

    def test_format_path_contains_steps(self):
        s = format_path(self.spec, [0, 1, 3])
        self.assertIn('Шагов: 2', s)


if __name__ == '__main__':
    unittest.main(verbosity=2)
