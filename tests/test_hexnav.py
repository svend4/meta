"""Тесты для hexnav — CLI навигатор по графу Q6."""
import sys
import os
import io
import json
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from projects.hexnav.hexnav import fmt_hex, print_current, print_path, TRIGRAM_NAMES, run


class TestTrigramNames(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(TRIGRAM_NAMES), 8)
    def test_all_keys_valid(self):
        for k in TRIGRAM_NAMES: self.assertGreaterEqual(k, 0); self.assertLessEqual(k, 7)
    def test_keys_complete(self):
        self.assertEqual(set(TRIGRAM_NAMES.keys()), set(range(8)))
    def test_values_are_strings(self):
        for v in TRIGRAM_NAMES.values():
            self.assertIsInstance(v, str); self.assertGreater(len(v), 0)
    def test_kun_zero(self):
        self.assertIn('☷', TRIGRAM_NAMES[0b000])
    def test_qian_seven(self):
        self.assertIn('☰', TRIGRAM_NAMES[0b111])
    def test_kan_water(self):
        self.assertIn('☵', TRIGRAM_NAMES[0b010])
    def test_li_fire(self):
        self.assertIn('☲', TRIGRAM_NAMES[0b101])


class TestFmtHex(unittest.TestCase):
    def test_returns_string(self):
        self.assertIsInstance(fmt_hex(0), str)
    def test_contains_number(self):
        self.assertIn('42', fmt_hex(42))
    def test_contains_bits(self):
        self.assertIn('000000', fmt_hex(0))
    def test_fmt_hex_zero(self):
        result = fmt_hex(0)
        self.assertIn('0', result)
    def test_fmt_hex_max(self):
        result = fmt_hex(63)
        self.assertIn('63', result); self.assertIn('111111', result)
    def test_fmt_hex_one(self):
        result = fmt_hex(1)
        self.assertIn('1', result); self.assertIn('000001', result)
    def test_fmt_hex_format(self):
        result = fmt_hex(5)
        self.assertIn('(', result); self.assertIn(')', result)


class TestPrintPath(unittest.TestCase):
    def _capture(self, func, *args):
        captured = io.StringIO()
        with patch('sys.stdout', captured):
            func(*args)
        return captured.getvalue()

    def test_empty_path(self):
        out = self._capture(print_path, [])
        self.assertTrue(len(out) >= 0)  # does not crash
    def test_single_node_path(self):
        out = self._capture(print_path, [0])
        self.assertIn('0', out)
    def test_path_shows_steps(self):
        out = self._capture(print_path, [0, 1, 3, 7])
        self.assertIn('3', out)
    def test_path_shows_arrow(self):
        out = self._capture(print_path, [0, 1])
        self.assertIn('→', out)
    def test_path_two_nodes_one_step(self):
        out = self._capture(print_path, [0, 1])
        self.assertIn('1', out)
    def test_path_contains_nodes(self):
        out = self._capture(print_path, [5, 7, 15])
        self.assertIn('5', out); self.assertIn('7', out); self.assertIn('15', out)


class TestPrintCurrent(unittest.TestCase):
    def _capture(self, node):
        captured = io.StringIO()
        with patch('sys.stdout', captured):
            print_current(node)
        return captured.getvalue()

    def test_output_contains_hexagram_number(self):
        self.assertIn('0', self._capture(0))
    def test_output_contains_neighbors_header(self):
        out = self._capture(0)
        self.assertTrue(any(x in out for x in ['0','1','2','3','4','5']))
    def test_output_contains_antipod(self):
        out = self._capture(0)
        self.assertTrue('нтипод' in out or 'antipod' in out.lower())
    def test_output_nonempty(self):
        self.assertGreater(len(self._capture(42)), 0)
    def test_output_for_max(self):
        self.assertIn('63', self._capture(63))
    def test_six_transitions_shown(self):
        out = self._capture(0)
        for i in range(6): self.assertIn(f'[{i}]', out)


class TestRun(unittest.TestCase):
    def _run_with_inputs(self, inputs, start=0):
        input_iter = iter(inputs + ['q'])
        with patch('builtins.input', side_effect=input_iter):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                try:
                    run(start)
                except StopIteration:
                    pass
            return captured.getvalue()

    def test_run_starts_and_quits(self):
        self.assertGreater(len(self._run_with_inputs([])), 0)
    def test_run_shows_initial_hexagram(self):
        self.assertIn('0', self._run_with_inputs([], start=0))
    def test_run_flip_bit_0(self):
        self.assertIn('1', self._run_with_inputs(['0']))
    def test_run_flip_bit_changes_state(self):
        self.assertIn('2', self._run_with_inputs(['1']))
    def test_run_help_command(self):
        out = self._run_with_inputs(['h'])
        self.assertTrue('g' in out or 'G' in out or 'r' in out)
    def test_run_info_command(self):
        self.assertGreater(len(self._run_with_inputs(['i'])), 100)
    def test_run_hist_command(self):
        out = self._run_with_inputs(['hist'])
        self.assertTrue('стори' in out or 'путь' in out.lower() or 'Путь' in out)
    def test_run_antipode_command(self):
        self.assertIn('63', self._run_with_inputs(['a']))
    def test_run_goto_command(self):
        self.assertIn('63', self._run_with_inputs(['g 63']))
    def test_run_goto_invalid(self):
        out = self._run_with_inputs(['g 999'])
        self.assertTrue('ошибк' in out.lower() or '0-63' in out)
    def test_run_goto_no_target(self):
        out = self._run_with_inputs(['g'])
        self.assertTrue('использование' in out.lower() or 'g' in out)
    def test_run_reset_command(self):
        out = self._run_with_inputs(['0', 'r'])
        self.assertTrue('брос' in out.lower() or 'Сброс' in out)
    def test_run_unknown_command(self):
        out = self._run_with_inputs(['xyz'])
        self.assertTrue('xyz' in out or 'неизвестн' in out.lower())
    def test_run_empty_command_ignored(self):
        self.assertGreater(len(self._run_with_inputs(['', ''])), 0)
    def test_run_multiple_flips(self):
        self.assertIn('7', self._run_with_inputs(['0', '1', '2']))
    def test_run_export_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'out.json')
            input_seq = ['0', f'e {filepath}', 'q']
            with patch('builtins.input', side_effect=iter(input_seq)):
                captured = io.StringIO()
                with patch('sys.stdout', captured):
                    try: run(0)
                    except StopIteration: pass
            self.assertTrue(os.path.exists(filepath))
            with open(filepath) as f: data = json.load(f)
            self.assertIn('path', data); self.assertIn('start', data)
            self.assertEqual(data['start'], 0)
    def test_run_from_nonzero_start(self):
        self.assertIn('42', self._run_with_inputs([], start=42))
    def test_run_path_in_final_summary(self):
        out = self._run_with_inputs(['0'])
        self.assertTrue('путь' in out.lower() or 'Путь' in out)
    def test_run_eofstop_graceful(self):
        with patch('builtins.input', side_effect=EOFError):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                run(0)
            self.assertGreater(len(captured.getvalue()), 0)


class TestNavMain(unittest.TestCase):
    """Тесты для hexnav.main() (lines 202-214)."""

    def _run_main(self, argv_args):
        from projects.hexnav.hexnav import main
        captured = io.StringIO()
        with patch('sys.argv', ['hexnav.py'] + argv_args):
            with patch('projects.hexnav.hexnav.run') as mock_run:
                with patch('sys.stdout', captured):
                    main()
        return captured.getvalue(), mock_run

    def test_main_default_start(self):
        """main() без аргументов → run(0)."""
        _, mock_run = self._run_main([])
        mock_run.assert_called_once_with(0)

    def test_main_with_start_arg(self):
        """main() с аргументом 42 → run(42)."""
        _, mock_run = self._run_main(['42'])
        mock_run.assert_called_once_with(42)

    def test_main_out_of_range_exits(self):
        """main() с аргументом 100 (out of range) → parser.error."""
        from projects.hexnav.hexnav import main
        with patch('sys.argv', ['hexnav.py', '100']):
            with self.assertRaises(SystemExit):
                main()

    def test_main_out_of_range_high(self):
        """main() с аргументом 64 (=SIZE, out of range) → parser.error."""
        from projects.hexnav.hexnav import main
        with patch('sys.argv', ['hexnav.py', '64']):
            with self.assertRaises(SystemExit):
                main()


if __name__ == '__main__':
    unittest.main()
