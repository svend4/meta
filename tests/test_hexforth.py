"""Тесты интерпретатора, компилятора и верификатора HexForth."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexforth.interpreter import HexForth, HexForthError
from projects.hexforth.compiler import compile_to_ir, to_python, to_json_bytecode, path_stats
from projects.hexforth.verifier import (
    build_transition_graph, reachable_from, shortest_path_in_graph,
    analyze_source,
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
