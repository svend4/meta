"""Тесты интерпретатора, компилятора и верификатора HexForth."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexforth.interpreter import HexForth, HexForthError
import json as _json
from projects.hexforth.compiler import compile_to_ir, to_python, to_json_bytecode, to_dot, path_stats
from projects.hexforth.verifier import (
    build_transition_graph, reachable_from, shortest_path_in_graph,
    analyze_source, all_paths_bounded, fmt_path, ProgramAnalysis,
)
from libs.hexcore.hexcore import flip, antipode, hamming


class TestInterpreterFlip(unittest.TestCase):
    def test_flip_changes_state(self):
        h = HexForth(start=0)
        h.run('FLIP-0')
        self.assertEqual(h.state, 1)

    def test_flip_involution(self):
        h = HexForth(start=42)
        h.run('FLIP-3 FLIP-3')
        self.assertEqual(h.state, 42)

    def test_flip_all_bits(self):
        h = HexForth(start=0)
        h.run('FLIP-0 FLIP-1 FLIP-2 FLIP-3 FLIP-4 FLIP-5')
        self.assertEqual(h.state, 63)

    def test_set_clr(self):
        h = HexForth(start=0)
        h.run('SET-3')
        self.assertEqual(h.state, 8)
        h.run('CLR-3')
        self.assertEqual(h.state, 0)

    def test_set_idempotent(self):
        h = HexForth(start=63)
        h.run('SET-0')
        self.assertEqual(h.state, 63)  # уже установлен

    def test_clr_idempotent(self):
        h = HexForth(start=0)
        h.run('CLR-0')
        self.assertEqual(h.state, 0)   # уже сброшен


class TestInterpreterGoto(unittest.TestCase):
    def test_goto_self(self):
        h = HexForth(start=42)
        h.run('GOTO 42')
        self.assertEqual(h.state, 42)

    def test_goto_path_valid(self):
        h = HexForth(start=0)
        h.run('GOTO 63')
        self.assertEqual(h.state, 63)
        # Каждый шаг — ребро Q6
        for i in range(len(h.trace) - 1):
            self.assertEqual(hamming(h.trace[i], h.trace[i + 1]), 1)

    def test_goto_length(self):
        h = HexForth(start=0)
        h.run('GOTO 42')
        # Длина пути = hamming(0, 42) = 3
        self.assertEqual(len(h.trace) - 1, hamming(0, 42))


class TestInterpreterAntipode(unittest.TestCase):
    def test_antipode_zero(self):
        h = HexForth(start=0)
        h.run('ANTIPODE')
        self.assertEqual(h.state, 63)

    def test_antipode_42(self):
        h = HexForth(start=42)
        h.run('ANTIPODE')
        self.assertEqual(h.state, antipode(42))

    def test_double_antipode(self):
        h = HexForth(start=17)
        h.run('ANTIPODE ANTIPODE')
        self.assertEqual(h.state, 17)


class TestInterpreterAssert(unittest.TestCase):
    def test_assert_pass(self):
        h = HexForth(start=42)
        h.run('ASSERT-EQ 42')  # не должен бросить исключение

    def test_assert_fail(self):
        h = HexForth(start=42)
        with self.assertRaises(HexForthError):
            h.run('ASSERT-EQ 0')


class TestInterpreterDefine(unittest.TestCase):
    def test_define_and_call(self):
        h = HexForth(start=0)
        h.run('DEFINE GO42 : GOTO 42 ; GO42')
        self.assertEqual(h.state, 42)

    def test_define_multi_word(self):
        h = HexForth(start=0)
        h.run('DEFINE STEP : FLIP-0 FLIP-1 ; STEP')
        self.assertEqual(h.state, 3)


class TestInterpreterNop(unittest.TestCase):
    def test_nop_no_change(self):
        h = HexForth(start=42)
        h.run('NOP NOP NOP')
        self.assertEqual(h.state, 42)
        self.assertEqual(len(h.trace), 1)


class TestInterpreterTrace(unittest.TestCase):
    def test_trace_starts_with_initial(self):
        h = HexForth(start=7)
        self.assertEqual(h.trace[0], 7)

    def test_trace_grows(self):
        h = HexForth(start=0)
        h.run('FLIP-0 FLIP-1 FLIP-2')
        self.assertEqual(len(h.trace), 4)

    def test_nop_no_trace(self):
        h = HexForth(start=0)
        h.run('NOP')
        self.assertEqual(len(h.trace), 1)


class TestCompilerIR(unittest.TestCase):
    def test_flip_produces_flip_op(self):
        ir = compile_to_ir('FLIP-0', start=0)
        flips = [i for i in ir if i['op'] == 'FLIP']
        self.assertEqual(len(flips), 1)
        self.assertEqual(flips[0]['bit'], 0)
        self.assertEqual(flips[0]['from'], 0)
        self.assertEqual(flips[0]['to'], 1)

    def test_goto_expands(self):
        ir = compile_to_ir('GOTO 7', start=0)
        flips = [i for i in ir if i['op'] == 'FLIP']
        self.assertEqual(len(flips), 3)  # hamming(0, 7) = 3

    def test_final_state(self):
        ir = compile_to_ir('FLIP-0 FLIP-1', start=0)
        final = [i for i in ir if i['op'] == 'FINAL']
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]['state'], 3)


class TestCompilerPython(unittest.TestCase):
    def test_generated_code_runnable(self):
        ir = compile_to_ir('GOTO 42', start=0)
        code = to_python(ir, func_name='test_fn')
        ns = {}
        exec(code, ns)
        result = ns['test_fn'](0)
        self.assertEqual(result, 42)

    def test_nop_program(self):
        ir = compile_to_ir('NOP', start=10)
        code = to_python(ir)
        ns = {}
        exec(code, ns)
        result = ns['hexforth_program'](10)
        self.assertEqual(result, 10)


class TestCompilerStats(unittest.TestCase):
    def test_total_steps(self):
        ir = compile_to_ir('GOTO 63', start=0)
        s = path_stats(ir)
        self.assertEqual(s['total_steps'], 6)

    def test_bit_usage(self):
        ir = compile_to_ir('FLIP-2 FLIP-2 FLIP-3', start=0)
        s = path_stats(ir)
        self.assertEqual(s['bit_usage'].get(2, 0), 2)
        self.assertEqual(s['bit_usage'].get(3, 0), 1)


class TestVerifierGraph(unittest.TestCase):
    def test_all_flips_full_reachability(self):
        graph = build_transition_graph([f'FLIP-{i}' for i in range(6)])
        reach = reachable_from(0, graph)
        self.assertEqual(len(reach), 64)

    def test_one_flip_limited(self):
        graph = build_transition_graph(['FLIP-0'])
        reach = reachable_from(0, graph)
        self.assertEqual(reach, {0, 1})

    def test_two_flips_reachability(self):
        graph = build_transition_graph(['FLIP-0', 'FLIP-1'])
        reach = reachable_from(0, graph)
        self.assertEqual(reach, {0, 1, 2, 3})

    def test_path_found(self):
        graph = build_transition_graph([f'FLIP-{i}' for i in range(6)])
        path = shortest_path_in_graph(0, 63, graph)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 63)

    def test_path_not_found(self):
        graph = build_transition_graph(['FLIP-0'])  # только бит 0
        path = shortest_path_in_graph(0, 42, graph)  # 42 требует биты 1,3,5
        self.assertIsNone(path)

    def test_set_word_sets_bit(self):
        """SET-0 переводит 0→1, но не меняет состояния где бит уже установлен."""
        graph = build_transition_graph(['SET-0'])
        # 0 (000000) → SET-0 → 1 (000001)
        self.assertIn(1, graph[0])
        # 1 (000001) → SET-0 → остаётся 1 (бит уже установлен)
        self.assertIn(1, graph[1])

    def test_clr_word_clears_bit(self):
        """CLR-0 переводит 1→0, но не меняет состояния где бит уже сброшен."""
        graph = build_transition_graph(['CLR-0'])
        # 1 (000001) → CLR-0 → 0 (000000)
        self.assertIn(0, graph[1])
        # 0 (000000) → CLR-0 → остаётся 0
        self.assertIn(0, graph[0])

    def test_nop_word_self_loop(self):
        """NOP создаёт самопетли во всех состояниях."""
        graph = build_transition_graph(['NOP'])
        for state in range(64):
            self.assertIn(state, graph[state])

    def test_antipode_word_reachability(self):
        """ANTIPODE позволяет достичь антипода из любой вершины."""
        from libs.hexcore.hexcore import antipode
        graph = build_transition_graph(['ANTIPODE'])
        reach = reachable_from(0, graph)
        self.assertIn(antipode(0), reach)  # 63

    def test_goto_word_jump(self):
        """GOTO 42 позволяет достичь вершины 42 из любой вершины."""
        graph = build_transition_graph(['GOTO 42'])
        reach = reachable_from(0, graph)
        self.assertIn(42, reach)


class TestStaticAnalysis(unittest.TestCase):
    def test_clean_program(self):
        a = analyze_source('FLIP-0 GOTO 42 ASSERT-EQ 42')
        self.assertTrue(a.is_ok())
        self.assertIn(42, a.gotos)
        self.assertIn(42, a.asserts)

    def test_out_of_range_goto(self):
        a = analyze_source('GOTO 100')
        self.assertFalse(a.is_ok())

    def test_flip_tracking(self):
        a = analyze_source('FLIP-0 FLIP-0 FLIP-3')
        self.assertEqual(a.flips.count(0), 2)
        self.assertEqual(a.flips.count(3), 1)

    def test_define_tracked(self):
        a = analyze_source('DEFINE MOVE : FLIP-0 FLIP-1 ;')
        self.assertIn('MOVE', a.defines)


class TestCompilerJSON(unittest.TestCase):
    """Тесты to_json_bytecode: валидный JSON, корректные поля."""

    def _ir(self, src: str, start: int = 0):
        return compile_to_ir(src, start=start)

    def test_valid_json(self):
        """to_json_bytecode должна возвращать валидный JSON."""
        ir = self._ir('FLIP-0 FLIP-1', start=0)
        out = to_json_bytecode(ir)
        parsed = _json.loads(out)
        self.assertIsInstance(parsed, dict)

    def test_version_field(self):
        ir = self._ir('FLIP-0', start=0)
        parsed = _json.loads(to_json_bytecode(ir))
        self.assertEqual(parsed['version'], 1)

    def test_start_field(self):
        ir = self._ir('FLIP-0', start=7)
        parsed = _json.loads(to_json_bytecode(ir))
        self.assertEqual(parsed['start'], 7)

    def test_instructions_list(self):
        ir = self._ir('FLIP-0 FLIP-1', start=0)
        parsed = _json.loads(to_json_bytecode(ir))
        self.assertIn('instructions', parsed)
        self.assertIsInstance(parsed['instructions'], list)

    def test_nop_empty_instructions(self):
        """NOP не порождает FLIP-инструкции в JSON."""
        ir = self._ir('NOP', start=0)
        parsed = _json.loads(to_json_bytecode(ir))
        flip_instrs = [i for i in parsed['instructions'] if i['op'] == 'FLIP']
        self.assertEqual(len(flip_instrs), 0)

    def test_goto_expands_flips(self):
        """GOTO 63 от 0 порождает 6 FLIP-инструкций."""
        ir = self._ir('GOTO 63', start=0)
        parsed = _json.loads(to_json_bytecode(ir))
        flip_count = sum(1 for i in parsed['instructions'] if i['op'] == 'FLIP')
        self.assertEqual(flip_count, 6)


class TestCompilerDOT(unittest.TestCase):
    """Тесты to_dot: структура DOT-файла для Graphviz."""

    def _ir(self, src: str, start: int = 0):
        return compile_to_ir(src, start=start)

    def test_returns_string(self):
        ir = self._ir('FLIP-0', start=0)
        out = to_dot(ir)
        self.assertIsInstance(out, str)

    def test_digraph_keyword(self):
        ir = self._ir('FLIP-0', start=0)
        out = to_dot(ir)
        self.assertIn('digraph', out)

    def test_closing_brace(self):
        ir = self._ir('FLIP-0', start=0)
        out = to_dot(ir)
        self.assertIn('}', out)

    def test_node_labels(self):
        """Узлы должны содержать метки в формате h<n>."""
        ir = self._ir('FLIP-0', start=0)
        out = to_dot(ir)
        self.assertIn('h0', out)
        self.assertIn('h1', out)

    def test_edge_label(self):
        """Рёбра подписаны 'bit<n>'."""
        ir = self._ir('FLIP-0', start=0)
        out = to_dot(ir)
        self.assertIn('bit0', out)

    def test_custom_title(self):
        ir = self._ir('FLIP-0', start=0)
        out = to_dot(ir, title='MyPath')
        self.assertIn('MyPath', out)

    def test_nop_empty_graph(self):
        """NOP → граф без рёбер (только завершающий FINAL, без FLIP)."""
        ir = self._ir('NOP', start=0)
        out = to_dot(ir)
        self.assertIn('digraph', out)
        # Нет ни одного ребра bit-N
        self.assertNotIn('bit0', out)


class TestAllPathsBounded(unittest.TestCase):
    """Тесты all_paths_bounded — DFS с ограничением длины."""

    def _simple_graph(self) -> dict[int, set[int]]:
        """Граф: 0→1→2→3, 0→2."""
        return {0: {1, 2}, 1: {2}, 2: {3}}

    def test_direct_path_found(self):
        g = self._simple_graph()
        paths = all_paths_bounded(0, 3, g, max_len=10)
        self.assertGreater(len(paths), 0)

    def test_all_paths_end_at_target(self):
        g = self._simple_graph()
        for path in all_paths_bounded(0, 3, g, max_len=10):
            self.assertEqual(path[-1], 3)
            self.assertEqual(path[0], 0)

    def test_max_len_limits_paths(self):
        g = self._simple_graph()
        paths_short = all_paths_bounded(0, 3, g, max_len=2)
        paths_long = all_paths_bounded(0, 3, g, max_len=10)
        self.assertLessEqual(len(paths_short), len(paths_long))

    def test_no_path_returns_empty(self):
        g = {0: {1}}  # 1 не ведёт никуда
        paths = all_paths_bounded(0, 3, g, max_len=5)
        self.assertEqual(paths, [])

    def test_same_start_and_target_returns_empty(self):
        """start == target, но путь длины 1 (только start) не считается."""
        g = {0: {1}, 1: {0}}
        paths = all_paths_bounded(0, 0, g, max_len=5)
        # Требуется len(path) > 1, поэтому путь [0, 1, 0] может быть найден
        for p in paths:
            self.assertGreater(len(p), 1)


class TestFmtPath(unittest.TestCase):
    """Тесты fmt_path — форматирование пути как строки."""

    def test_returns_string(self):
        self.assertIsInstance(fmt_path([0, 1]), str)

    def test_contains_arrow(self):
        out = fmt_path([0, 1, 3])
        self.assertIn('→', out)

    def test_shows_bits(self):
        out = fmt_path([0])
        self.assertIn('000000', out)

    def test_single_node(self):
        out = fmt_path([42])
        self.assertIn('42', out)

    def test_path_length_reflected(self):
        """Путь из N узлов содержит N-1 стрелок."""
        out = fmt_path([0, 1, 3, 7])
        self.assertEqual(out.count('→'), 3)


class TestProgramAnalysis(unittest.TestCase):
    """Тесты класса ProgramAnalysis и его метода is_ok."""

    def test_empty_analysis_is_ok(self):
        pa = ProgramAnalysis()
        self.assertTrue(pa.is_ok())

    def test_error_makes_not_ok(self):
        pa = ProgramAnalysis()
        pa.errors.append('some error')
        self.assertFalse(pa.is_ok())

    def test_warning_is_still_ok(self):
        pa = ProgramAnalysis()
        pa.warnings.append('some warning')
        self.assertTrue(pa.is_ok())

    def test_analyze_source_clean(self):
        """Корректный код → is_ok()."""
        src = 'FLIP-0\nFLIP-1'
        pa = analyze_source(src)
        self.assertTrue(pa.is_ok())

    def test_analyze_source_define(self):
        """DEFINE с непустым телом не вызывает ошибку."""
        src = 'DEFINE myword : FLIP-0 ;'
        pa = analyze_source(src)
        self.assertTrue(pa.is_ok())

    def test_analyze_source_empty_define_warning(self):
        """DEFINE с пустым телом → предупреждение (warnings)."""
        src = 'DEFINE myword : ;'
        pa = analyze_source(src)
        self.assertGreater(len(pa.warnings), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
