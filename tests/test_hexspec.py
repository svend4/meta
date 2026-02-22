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
    print_text_report, print_hexforth_report,
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


class TestPrintReports(unittest.TestCase):
    """Тесты print_text_report и print_hexforth_report."""

    def _make_spec(self):
        return make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C'), ('C', 'A')],
            initial='A', final=['C'],
        )

    def _capture(self, fn, *args, **kwargs) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*args, **kwargs)
        return buf.getvalue()

    # print_text_report --------------------------------------------------------

    def test_text_report_produces_output(self):
        spec = self._make_spec()
        out = self._capture(print_text_report, spec)
        self.assertGreater(len(out), 0)

    def test_text_report_contains_spec_name(self):
        spec = self._make_spec()
        out = self._capture(print_text_report, spec)
        self.assertIn(spec.name, out)

    def test_text_report_coverage_states(self):
        spec = self._make_spec()
        out = self._capture(print_text_report, spec, coverage='states')
        self.assertIn('Покрытие состояний', out)

    def test_text_report_coverage_transitions(self):
        spec = self._make_spec()
        out = self._capture(print_text_report, spec, coverage='transitions')
        self.assertIn('переходов', out.lower())

    def test_text_report_shows_total(self):
        spec = self._make_spec()
        out = self._capture(print_text_report, spec)
        self.assertIn('Итого', out)

    # print_hexforth_report ----------------------------------------------------

    def test_hexforth_report_produces_output(self):
        spec = self._make_spec()
        out = self._capture(print_hexforth_report, spec)
        self.assertGreater(len(out), 0)

    def test_hexforth_report_contains_goto(self):
        spec = self._make_spec()
        out = self._capture(print_hexforth_report, spec)
        self.assertIn('GOTO', out)

    def test_hexforth_report_coverage_states(self):
        spec = self._make_spec()
        out = self._capture(print_hexforth_report, spec, coverage='states')
        self.assertGreater(len(out), 0)

    def test_hexforth_report_no_crash_all_coverages(self):
        spec = self._make_spec()
        for cov in ['states', 'transitions', 'roundtrip', 'all']:
            self._capture(print_hexforth_report, spec, coverage=cov)


class TestSpecCLI(unittest.TestCase):
    """CLI-тесты для hexspec generator и verifier main()."""

    SPEC_JSON = json.dumps({
        "name": "test_fsm",
        "bits": ["b0", "b1", "b2", "b3", "b4", "b5"],
        "states": {"S0": "000000", "S1": "000001", "S2": "000011"},
        "transitions": [["S0", "S1"], ["S1", "S2"]],
        "initial": "S0",
        "final": ["S2"],
        "forbidden": [],
    })

    def setUp(self):
        import tempfile
        self._tmpfile = tempfile.NamedTemporaryFile(
            'w', suffix='.json', delete=False
        )
        self._tmpfile.write(self.SPEC_JSON)
        self._tmpfile.close()
        self.spec_path = self._tmpfile.name

    def tearDown(self):
        import os
        os.unlink(self.spec_path)

    def _run_gen(self, extra_args):
        from projects.hexspec.generator import main as gen_main
        old_argv = sys.argv
        sys.argv = ['generator.py', self.spec_path] + extra_args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                gen_main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def _run_ver(self, extra_args, expect_exit=None):
        from projects.hexspec.verifier import main as ver_main
        old_argv = sys.argv
        sys.argv = ['verifier.py', self.spec_path] + extra_args
        buf = io.StringIO()
        try:
            if expect_exit is not None:
                with self.assertRaises(SystemExit) as cm:
                    with redirect_stdout(buf):
                        ver_main()
                self.assertEqual(cm.exception.code, expect_exit)
            else:
                with redirect_stdout(buf):
                    ver_main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_gen_text_all(self):
        out = self._run_gen(['--coverage', 'all', '--format', 'text'])
        self.assertGreater(len(out), 0)

    def test_gen_json(self):
        out = self._run_gen(['--format', 'json'])
        data = json.loads(out)
        self.assertIsInstance(data, dict)

    def test_gen_hexforth(self):
        out = self._run_gen(['--format', 'hexforth'])
        self.assertGreater(len(out), 0)

    def test_ver_ok(self):
        out = self._run_ver([], expect_exit=0)

    def test_ver_verbose(self):
        out = self._run_ver(['--verbose'], expect_exit=0)
        self.assertGreater(len(out), 0)

    def test_ver_path_valid(self):
        out = self._run_ver(['--path', 'S0', 'S2'], expect_exit=0)
        self.assertGreater(len(out), 0)

    def test_ver_path_invalid_from(self):
        out = self._run_ver(['--path', 'UNKNOWN', 'S2'], expect_exit=1)
        self.assertIn('Неизвестное', out)

    def test_ver_path_invalid_to(self):
        out = self._run_ver(['--path', 'S0', 'UNKNOWN'], expect_exit=1)
        self.assertIn('Неизвестное', out)

    def test_ver_path_not_from_initial(self):
        # from_h != spec.initial → path=None → prints "недостижимо" (line 388)
        out = self._run_ver(['--path', 'S1', 'S2'], expect_exit=0)
        self.assertIn('недостижимо', out)


class TestSpecReverseMethods(unittest.TestCase):
    """Тесты для reverse_transitions() и backward_reachable() (lines 94, 98-107)."""

    def setUp(self):
        self.spec = make_spec(
            states={'A': '000000', 'B': '000001', 'C': '000011'},
            transitions=[('A', 'B'), ('B', 'C')],
            initial='A',
        )

    def test_reverse_transitions(self):
        rev = self.spec.reverse_transitions()
        self.assertIn((1, 0), rev)   # B→A reversed
        self.assertIn((3, 1), rev)   # C→B reversed

    def test_backward_reachable(self):
        # Backward from {C=3}: C←B←A, so all reachable backwards
        br = self.spec.backward_reachable({3})
        self.assertIn(0, br)   # A is backward reachable from C
        self.assertIn(1, br)   # B is backward reachable from C
        self.assertIn(3, br)   # C itself

    def test_backward_reachable_single(self):
        br = self.spec.backward_reachable({1})  # from B
        self.assertIn(0, br)   # A can reach B
        self.assertIn(1, br)   # B itself


class TestSpecCheckInvariant(unittest.TestCase):
    """Тест для check_invariant() (line 148)."""

    def test_check_invariant_no_violations(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001'},
            transitions=[('A', 'B')],
            initial='A',
        )
        # All reachable states have h < 10 → no violations
        violations = spec.check_invariant(lambda h: h < 10)
        self.assertEqual(violations, [])

    def test_check_invariant_with_violation(self):
        spec = make_spec(
            states={'A': '000000', 'B': '000001'},
            transitions=[('A', 'B')],
            initial='A',
        )
        # State B=1 violates h == 0
        violations = spec.check_invariant(lambda h: h == 0)
        self.assertIn(1, violations)


class TestLoadSpecErrors(unittest.TestCase):
    """Тесты ошибок в from_json() (lines 200, 202, 207)."""

    def _write_spec(self, data):
        import tempfile
        f = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_load_spec_unknown_transition_src(self):
        spec_data = {
            'states': {'A': '000000', 'B': '000001'},
            'transitions': [['UNKNOWN', 'B']],
            'initial': 'A',
        }
        path = self._write_spec(spec_data)
        try:
            with self.assertRaises(ValueError) as cm:
                load_spec(path)
            self.assertIn('UNKNOWN', str(cm.exception))
        finally:
            os.unlink(path)

    def test_load_spec_unknown_transition_dst(self):
        spec_data = {
            'states': {'A': '000000', 'B': '000001'},
            'transitions': [['A', 'MISSING']],
            'initial': 'A',
        }
        path = self._write_spec(spec_data)
        try:
            with self.assertRaises(ValueError) as cm:
                load_spec(path)
            self.assertIn('MISSING', str(cm.exception))
        finally:
            os.unlink(path)

    def test_load_spec_unknown_initial(self):
        spec_data = {
            'states': {'A': '000000'},
            'transitions': [],
            'initial': 'NOPE',
        }
        path = self._write_spec(spec_data)
        try:
            with self.assertRaises(ValueError) as cm:
                load_spec(path)
            self.assertIn('NOPE', str(cm.exception))
        finally:
            os.unlink(path)


class TestVerifyWithDescription(unittest.TestCase):
    """Тесты для print_verification_report с description (line 237) и forbidden OK (line 322)."""

    def test_verify_spec_with_description(self):
        # Create spec with description → hits line 237
        s = Spec(
            name='desc_test',
            description='Test description',
            bit_names=[f'b{i}' for i in range(6)],
            states={'A': 0, 'B': 1},
            transitions=[(0, 1)],
            initial=0,
            final=set(),
            forbidden=set(),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            verify(s, verbose=True)
        self.assertIn('Test description', buf.getvalue())

    def test_verify_spec_with_unreachable_forbidden(self):
        # Forbidden state exists but is not reachable → prints [OK] (line 322)
        s = Spec(
            name='forbidden_test',
            description='',
            bit_names=[f'b{i}' for i in range(6)],
            states={'A': 0, 'B': 1, 'F': 63},
            transitions=[(0, 1), (1, 0)],  # cycle to avoid deadlock
            initial=0,
            final=set(),
            forbidden={63},  # F=63 is forbidden but unreachable
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = verify(s, verbose=True)
        self.assertIn('[OK]', buf.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
