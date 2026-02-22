"""Тесты интерпретатора, компилятора и верификатора HexForth."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from projects.hexforth.interpreter import HexForth, HexForthError, repl
import json as _json
from projects.hexforth.compiler import compile_to_ir, to_python, to_json_bytecode, to_dot, path_stats
from projects.hexforth.verifier import (
    build_transition_graph, reachable_from, shortest_path_in_graph,
    analyze_source, all_paths_bounded, fmt_path, ProgramAnalysis,
    _parse_allowed_word,
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


class TestVerifierExtended(unittest.TestCase):
    """Дополнительные тесты для покрытия непокрытых ветвей verifier.py."""

    def test_build_graph_with_explicit_states(self):
        """build_transition_graph с явным параметром states."""
        states = {0, 1, 2, 4}
        graph = build_transition_graph([f'FLIP-{i}' for i in range(6)], states=states)
        # Только указанные состояния входят в граф
        for s in graph:
            self.assertIn(s, states)

    def test_shortest_path_start_equals_target(self):
        """shortest_path_in_graph когда start == target → [start]."""
        graph = build_transition_graph([f'FLIP-{i}' for i in range(6)])
        path = shortest_path_in_graph(42, 42, graph)
        self.assertEqual(path, [42])

    def test_analyze_source_parse_error(self):
        """Синтаксическая ошибка → errors содержит сообщение."""
        # DEFINE без ':' вызывает HexForthError в parse()
        pa = analyze_source('DEFINE bad')
        self.assertFalse(pa.is_ok())
        self.assertGreater(len(pa.errors), 0)

    def test_analyze_recursive_define(self):
        """DEFINE с прямой рекурсией → ошибка."""
        pa = analyze_source('DEFINE LOOP : LOOP ;')
        self.assertFalse(pa.is_ok())
        self.assertTrue(any('рекурсия' in e for e in pa.errors))

    def test_analyze_indirect_recursion(self):
        """Взаимная рекурсия двух слов → ошибка."""
        src = 'DEFINE A : B ;\nDEFINE B : A ;'
        pa = analyze_source(src)
        self.assertFalse(pa.is_ok())

    def test_analyze_goto_no_arg(self):
        """GOTO без аргумента → ошибка."""
        pa = analyze_source('GOTO')
        self.assertFalse(pa.is_ok())
        self.assertTrue(any('требует аргумент' in e or 'GOTO' in e for e in pa.errors))

    def test_analyze_goto_non_number(self):
        """GOTO с нечисловым аргументом → ошибка."""
        pa = analyze_source('GOTO foo')
        self.assertFalse(pa.is_ok())

    def test_analyze_assert_eq_out_of_range(self):
        """ASSERT-EQ с аргументом вне диапазона → ошибка."""
        pa = analyze_source('ASSERT-EQ 100')
        self.assertFalse(pa.is_ok())

    def test_analyze_assert_eq_non_number(self):
        """ASSERT-EQ с нечисловым аргументом → ошибка."""
        pa = analyze_source('ASSERT-EQ bar')
        self.assertFalse(pa.is_ok())

    def test_parse_allowed_word_unknown_returns_none(self):
        """_parse_allowed_word с неизвестным словом → None."""
        self.assertIsNone(_parse_allowed_word('UNKNOWN_WORD', 0))


class TestInterpreterExtended(unittest.TestCase):
    """Тесты для покрытия непокрытых ветвей interpreter.py."""

    def test_start_out_of_range_raises(self):
        """Стартовая гексаграмма вне диапазона → HexForthError."""
        with self.assertRaises(HexForthError):
            HexForth(start=100)

    def test_debug_word_emits(self):
        """DEBUG записывает информацию в output."""
        h = HexForth(start=42)
        h.run('DEBUG')
        self.assertGreater(len(h.output), 0)
        self.assertTrue(any('DEBUG' in s or '42' in s for s in h.output))

    def test_print_word_emits_state(self):
        """PRINT записывает текущее состояние в output."""
        h = HexForth(start=7)
        h.run('PRINT')
        self.assertGreater(len(h.output), 0)
        self.assertTrue(any('7' in s for s in h.output))

    def test_render_word_emits(self):
        """RENDER записывает несколько строк в output."""
        h = HexForth(start=0)
        h.run('RENDER')
        self.assertGreater(len(h.output), 0)

    def test_verbose_mode_prints(self):
        """verbose=True выводит сообщения при DEBUG."""
        import io
        import sys
        h = HexForth(start=0, verbose=True)
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            h.run('DEBUG')
        finally:
            sys.stdout = old_stdout
        self.assertGreater(len(captured.getvalue()), 0)

    def test_goto_out_of_range_raises(self):
        """GOTO с целью вне диапазона → HexForthError."""
        h = HexForth(start=0)
        with self.assertRaises(HexForthError):
            h.run('GOTO 100')

    def test_goto_no_arg_raises(self):
        """GOTO без аргумента в run() → HexForthError."""
        h = HexForth(start=0)
        with self.assertRaises(HexForthError):
            h.run('GOTO')

    def test_unknown_word_raises(self):
        """Неизвестное слово → HexForthError."""
        h = HexForth(start=0)
        with self.assertRaises(HexForthError):
            h.run('NONEXISTENT_WORD')

    def test_user_word_with_goto_inside(self):
        """Пользовательское слово с GOTO внутри тела."""
        h = HexForth(start=0)
        h.run('DEFINE JUMP : GOTO 7 ; JUMP')
        self.assertEqual(h.state, 7)

    def test_user_word_goto_without_arg_raises(self):
        """GOTO внутри пользовательского слова без аргумента → HexForthError."""
        h = HexForth(start=0)
        with self.assertRaises(HexForthError):
            h.run('DEFINE BAD : GOTO ; BAD')

    def test_assert_eq_success_emits_ok(self):
        """ASSERT-EQ с правильным значением записывает [OK] в output."""
        h = HexForth(start=42)
        h.run('ASSERT-EQ 42')
        self.assertTrue(any('OK' in s for s in h.output))


# ── compiler.py — дополнительные тесты ───────────────────────────────────────

class TestCompilerEmit(unittest.TestCase):
    """Тесты TracingHexForth._emit и to_python с PRINT-инструкциями."""

    def test_compile_debug_produces_print_op(self):
        """DEBUG во время компиляции добавляет {'op': 'PRINT'} в IR."""
        ir = compile_to_ir('DEBUG', start=42)
        ops = [i['op'] for i in ir]
        self.assertIn('PRINT', ops)

    def test_to_python_with_print_instr(self):
        """to_python с PRINT-инструкцией генерирует print(...)."""
        ir = compile_to_ir('DEBUG', start=42)
        code = to_python(ir)
        self.assertIn('print(', code)

    def test_to_python_no_final_in_ir(self):
        """to_python без FINAL-инструкции добавляет 'return state'."""
        ir_no_final = [{'op': 'FLIP', 'bit': 0, 'from': 0, 'to': 1}]
        code = to_python(ir_no_final)
        self.assertIn('return state', code)


class TestCompilerCLI(unittest.TestCase):
    """Тесты main() компилятора через sys.argv."""

    def _run(self, args):
        import io as _io
        from projects.hexforth.compiler import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ['compiler.py'] + args
        sys.stdout = _io.StringIO()
        try:
            main()
            return sys.stdout.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

    def test_target_python(self):
        out = self._run(['--program', 'FLIP-0', '--target', 'python'])
        self.assertIn('def hexforth_program', out)

    def test_target_json(self):
        out = self._run(['--program', 'FLIP-0', '--target', 'json'])
        self.assertIn('"op"', out)

    def test_target_dot(self):
        out = self._run(['--program', 'FLIP-0', '--target', 'dot'])
        self.assertIn('digraph', out)

    def test_target_stats(self):
        out = self._run(['--program', 'FLIP-0', '--target', 'stats'])
        self.assertIn('Шагов', out)

    def test_compile_error_exits(self):
        import io as _io
        from projects.hexforth.compiler import main
        sys.argv = ['compiler.py', '--program', 'NONEXISTENT_WORD_XYZ']
        sys.stdout = _io.StringIO()
        try:
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)
        finally:
            sys.argv = ['compiler.py']
            sys.stdout = sys.__stdout__

    def test_from_file(self):
        import tempfile, os, io as _io
        from projects.hexforth.compiler import main as compiler_main
        src = 'FLIP-0 FLIP-1'
        with tempfile.NamedTemporaryFile('w', suffix='.hf', delete=False) as f:
            f.write(src)
            fname = f.name
        try:
            old_argv = sys.argv
            old_stdout = sys.stdout
            sys.argv = ['compiler.py', fname, '--target', 'json']
            sys.stdout = _io.StringIO()
            try:
                compiler_main()
                out = sys.stdout.getvalue()
            finally:
                sys.argv = old_argv
                sys.stdout = old_stdout
            self.assertIn('"op"', out)
        finally:
            os.unlink(fname)

    def test_out_to_file(self):
        import tempfile, os, io as _io
        from projects.hexforth.compiler import main as compiler_main
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            outfile = f.name
        try:
            old_argv = sys.argv
            old_stdout = sys.stdout
            sys.argv = ['compiler.py', '--program', 'FLIP-0',
                        '--target', 'python', '--out', outfile]
            sys.stdout = _io.StringIO()
            try:
                compiler_main()
            finally:
                sys.argv = old_argv
                sys.stdout = old_stdout
            with open(outfile, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('def hexforth_program', content)
        finally:
            os.unlink(outfile)


# ── дополнительные тесты interpreter.py ─────────────────────────────────────

class TestInterpreterExtra(unittest.TestCase):
    """Тесты непокрытых веток interpreter.py."""

    def test_transition_non_neighbor_raises(self):
        """_transition к не-соседу → HexForthError."""
        h = HexForth(start=0)
        with self.assertRaises(HexForthError):
            h._transition(63)  # 63 не сосед 0 (расстояние = 6)

    def test_parse_comment_stripping(self):
        """Комментарий после # удаляется перед разбором."""
        h = HexForth(start=0)
        h.run('FLIP-0 # это комментарий')
        self.assertEqual(h.state, 1)

    def test_parse_empty_line_skipped(self):
        """Пустые строки пропускаются при разборе."""
        h = HexForth(start=0)
        h.run('\n\n# комментарий\n\nFLIP-0')
        self.assertEqual(h.state, 1)

    def test_define_semicolon_before_colon_raises(self):
        """DEFINE с ';' перед ':' → HexForthError."""
        h = HexForth(start=0)
        with self.assertRaises(HexForthError):
            h.run('DEFINE FOO ; : FLIP-0')

    def test_run_file(self):
        """run_file читает программу из файла и выполняет её."""
        import tempfile, os
        src = 'FLIP-0\nFLIP-1\n'
        with tempfile.NamedTemporaryFile('w', suffix='.hf', delete=False) as f:
            f.write(src)
            fname = f.name
        try:
            h = HexForth(start=0)
            h.run_file(fname)
            self.assertEqual(h.state, 3)  # bits 0 and 1 flipped
        finally:
            os.unlink(fname)

    def test_summary_returns_string(self):
        """summary() возвращает строку с информацией о пути."""
        h = HexForth(start=0)
        h.run('FLIP-0 FLIP-1')
        s = h.summary()
        self.assertIn('Конечная гексаграмма', s)
        self.assertIn('Путь', s)


class TestRepl(unittest.TestCase):
    """Тесты repl() с мок-вводом."""

    def _run_repl(self, inputs, start=0):
        buf = io.StringIO()
        with patch('builtins.input', side_effect=inputs):
            with redirect_stdout(buf):
                repl(start=start)
        return buf.getvalue()

    def test_quit_with_q(self):
        """'q' завершает REPL."""
        out = self._run_repl(['q'])
        self.assertIn('HexForth', out)

    def test_quit_with_quit(self):
        """'quit' завершает REPL."""
        out = self._run_repl(['quit'])
        self.assertIn('HexForth', out)

    def test_empty_line_continues(self):
        """Пустая строка → продолжить (не завершить)."""
        out = self._run_repl(['', 'q'])
        self.assertIn('HexForth', out)

    def test_valid_command_executes(self):
        """Корректная команда выполняется и показывает output."""
        out = self._run_repl(['FLIP-0', 'q'])
        self.assertIsInstance(out, str)

    def test_error_in_command(self):
        """Ошибка в команде → печатает сообщение, не падает."""
        out = self._run_repl(['NONEXISTENT_WORD', 'q'])
        self.assertIn('Ошибка', out)

    def test_eof_exits_gracefully(self):
        """EOFError → graceful exit."""
        buf = io.StringIO()
        with patch('builtins.input', side_effect=EOFError()):
            with redirect_stdout(buf):
                repl(start=0)
        self.assertIn('HexForth', buf.getvalue())


class TestInterpreterMainCLI(unittest.TestCase):
    """Тесты main() интерпретатора."""

    def _run_main(self, args):
        from projects.hexforth.interpreter import main
        old_argv = sys.argv
        sys.argv = ['hexforth.py'] + args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_repl_flag(self):
        """--repl запускает REPL (сразу q)."""
        with patch('builtins.input', return_value='q'):
            out = self._run_main(['--repl'])
        self.assertIn('HexForth', out)

    def test_no_file_starts_repl(self):
        """Без файла запускается REPL."""
        with patch('builtins.input', return_value='q'):
            out = self._run_main([])
        self.assertIn('HexForth', out)

    def test_run_file_verbose(self):
        """С файлом и --verbose выполняет программу."""
        import tempfile, os
        src = 'DEBUG\n'
        with tempfile.NamedTemporaryFile('w', suffix='.hf', delete=False) as f:
            f.write(src)
            fname = f.name
        try:
            out = self._run_main([fname, '--verbose'])
            self.assertIsInstance(out, str)
        finally:
            os.unlink(fname)

    def test_run_file_error_exits(self):
        """Файл с ошибкой → stderr + sys.exit(1)."""
        import tempfile, os
        src = 'NONEXISTENT_WORD\n'
        with tempfile.NamedTemporaryFile('w', suffix='.hf', delete=False) as f:
            f.write(src)
            fname = f.name
        try:
            from projects.hexforth.interpreter import main
            old_argv = sys.argv
            sys.argv = ['hexforth.py', fname]
            try:
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
            finally:
                sys.argv = old_argv
        finally:
            os.unlink(fname)


class TestVerifierCLI(unittest.TestCase):
    """CLI-тесты для hexforth/verifier.py main()."""

    def _run(self, args, expect_exit=None):
        from projects.hexforth.verifier import main as verifier_main
        old_argv = sys.argv
        sys.argv = ['verifier.py'] + args
        buf = io.StringIO()
        try:
            if expect_exit is not None:
                with self.assertRaises(SystemExit) as cm:
                    with redirect_stdout(buf):
                        verifier_main()
                self.assertEqual(cm.exception.code, expect_exit)
            else:
                with redirect_stdout(buf):
                    verifier_main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_no_args_shows_reachable(self):
        out = self._run([])
        self.assertIn('Достижимо', out)

    def test_target_reachable(self):
        out = self._run(['--target', '1'])
        self.assertIn('OK', out)

    def test_target_unreachable(self):
        # FLIP-1 flips bit 1 (gives 2), so target 1 unreachable with only FLIP-1
        out = self._run(['--target', '1', '--allowed', 'FLIP-1'], expect_exit=1)
        self.assertIn('FAIL', out)

    def test_target_with_all_paths(self):
        out = self._run(['--target', '1', '--all-paths', '--max-path', '3'])
        self.assertIn('пути', out)

    def test_reachable_flag(self):
        out = self._run(['--reachable'], expect_exit=0)
        self.assertIn('Достижимо', out)

    def test_file_check_syntax_ok(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile('w', suffix='.hf', delete=False) as f:
            f.write('FLIP-0 FLIP-1\n')
            fname = f.name
        try:
            out = self._run([fname, '--check-syntax'], expect_exit=0)
            self.assertIn('OK', out)
        finally:
            os.unlink(fname)

    def test_file_check_syntax_error(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile('w', suffix='.hf', delete=False) as f:
            f.write('GOTO\n')  # GOTO without arg → error
            fname = f.name
        try:
            out = self._run([fname, '--check-syntax'], expect_exit=1)
            self.assertIn('FAIL', out)
        finally:
            os.unlink(fname)

    def test_target_out_of_range(self):
        out = self._run(['--target', '100'], expect_exit=1)
        self.assertIn('Ошибка', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
