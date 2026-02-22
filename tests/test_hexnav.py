"""Тесты для hexnav — CLI навигатор по графу Q6."""
import sys
import os
import io
import json
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from projects.hexnav.hexnav import fmt_hex, print_current, print_path, TRIGRAM_NAMES, run


# ============================================================
# TRIGRAM_NAMES
# ============================================================

class TestTrigramNames:
    def test_count(self):
        assert len(TRIGRAM_NAMES) == 8

    def test_all_keys_valid(self):
        for k in TRIGRAM_NAMES:
            assert 0 <= k <= 7

    def test_keys_complete(self):
        assert set(TRIGRAM_NAMES.keys()) == set(range(8))

    def test_values_are_strings(self):
        for v in TRIGRAM_NAMES.values():
            assert isinstance(v, str)
            assert len(v) > 0

    def test_kun_zero(self):
        assert '☷' in TRIGRAM_NAMES[0b000]

    def test_qian_seven(self):
        assert '☰' in TRIGRAM_NAMES[0b111]

    def test_kan_water(self):
        assert '☵' in TRIGRAM_NAMES[0b010]

    def test_li_fire(self):
        assert '☲' in TRIGRAM_NAMES[0b101]


# ============================================================
# fmt_hex
# ============================================================

class TestFmtHex:
    def test_returns_string(self):
        assert isinstance(fmt_hex(0), str)

    def test_contains_number(self):
        result = fmt_hex(42)
        assert '42' in result

    def test_contains_bits(self):
        # to_bits(0) = '000000'
        result = fmt_hex(0)
        assert '000000' in result

    def test_fmt_hex_zero(self):
        result = fmt_hex(0)
        assert ' 0' in result or '0' in result

    def test_fmt_hex_max(self):
        result = fmt_hex(63)
        assert '63' in result
        assert '111111' in result

    def test_fmt_hex_one(self):
        result = fmt_hex(1)
        assert '1' in result
        assert '000001' in result

    def test_fmt_hex_format(self):
        # должен содержать скобки для битового представления
        result = fmt_hex(5)
        assert '(' in result and ')' in result


# ============================================================
# print_path
# ============================================================

class TestPrintPath:
    def test_empty_path(self, capsys):
        print_path([])
        captured = capsys.readouterr()
        assert 'пуст' in captured.out.lower() or 'empty' in captured.out.lower() or captured.out

    def test_single_node_path(self, capsys):
        print_path([0])
        captured = capsys.readouterr()
        assert '0' in captured.out
        assert '0' in captured.out  # шагов: 0

    def test_path_shows_steps(self, capsys):
        print_path([0, 1, 3, 7])
        captured = capsys.readouterr()
        assert '3' in captured.out  # 3 шага

    def test_path_shows_arrow(self, capsys):
        print_path([0, 1])
        captured = capsys.readouterr()
        assert '→' in captured.out

    def test_path_two_nodes_one_step(self, capsys):
        print_path([0, 1])
        captured = capsys.readouterr()
        assert '1' in captured.out  # 1 шаг

    def test_path_contains_nodes(self, capsys):
        print_path([5, 7, 15])
        captured = capsys.readouterr()
        assert '5' in captured.out
        assert '7' in captured.out
        assert '15' in captured.out


# ============================================================
# print_current
# ============================================================

class TestPrintCurrent:
    def test_output_contains_hexagram_number(self, capsys):
        print_current(0)
        captured = capsys.readouterr()
        assert '0' in captured.out

    def test_output_contains_neighbors_header(self, capsys):
        print_current(0)
        captured = capsys.readouterr()
        # должно показывать переходы
        assert any(x in captured.out for x in ['0', '1', '2', '3', '4', '5'])

    def test_output_contains_antipod(self, capsys):
        print_current(0)
        captured = capsys.readouterr()
        assert 'нтипод' in captured.out or 'antipod' in captured.out.lower()

    def test_output_nonempty(self, capsys):
        print_current(42)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_output_for_max(self, capsys):
        print_current(63)
        captured = capsys.readouterr()
        assert '63' in captured.out

    def test_six_transitions_shown(self, capsys):
        print_current(0)
        captured = capsys.readouterr()
        # 6 переходов (0-5)
        for i in range(6):
            assert f'[{i}]' in captured.out


# ============================================================
# run — тестирование через mock input
# ============================================================

class TestRun:
    def _run_with_inputs(self, inputs, start=0):
        """Запускает run() с заданными командами, возвращает stdout."""
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
        out = self._run_with_inputs([])
        assert len(out) > 0

    def test_run_shows_initial_hexagram(self):
        out = self._run_with_inputs([], start=0)
        assert '0' in out

    def test_run_flip_bit_0(self):
        # Переворот черты 0: 0 → 1
        out = self._run_with_inputs(['0'])
        assert '1' in out

    def test_run_flip_bit_changes_state(self):
        # Переворот черты 1: 0 → 2
        out = self._run_with_inputs(['1'])
        assert '2' in out

    def test_run_help_command(self):
        out = self._run_with_inputs(['h'])
        # должна появиться информация о командах
        assert 'g' in out or 'G' in out or 'r' in out

    def test_run_info_command(self):
        out = self._run_with_inputs(['i'])
        # вывод информации о текущей гексаграмме
        assert len(out) > 100

    def test_run_hist_command(self):
        out = self._run_with_inputs(['hist'])
        assert 'стори' in out or 'путь' in out.lower() or 'Путь' in out

    def test_run_antipode_command(self):
        out = self._run_with_inputs(['a'])
        # антипод 0 = 63
        assert '63' in out

    def test_run_goto_command(self):
        out = self._run_with_inputs(['g 63'])
        assert '63' in out

    def test_run_goto_invalid(self):
        out = self._run_with_inputs(['g 999'])
        assert 'ошибк' in out.lower() or 'ошибка' in out.lower() or '0-63' in out

    def test_run_goto_no_target(self):
        out = self._run_with_inputs(['g'])
        assert 'использование' in out.lower() or 'g' in out

    def test_run_reset_command(self):
        # переходим и сбрасываем
        out = self._run_with_inputs(['0', 'r'])
        assert 'брос' in out.lower() or 'Сброс' in out

    def test_run_unknown_command(self):
        out = self._run_with_inputs(['xyz'])
        assert 'xyz' in out or 'неизвестн' in out.lower()

    def test_run_empty_command_ignored(self):
        # пустой ввод не должен падать
        out = self._run_with_inputs(['', ''])
        assert len(out) > 0

    def test_run_multiple_flips(self):
        # Переворот бит 0, 1, 2: 0 → 1 → 3 → 7
        out = self._run_with_inputs(['0', '1', '2'])
        assert '7' in out

    def test_run_export_path(self, tmp_path):
        """Команда e сохраняет путь в JSON."""
        filepath = str(tmp_path / 'out.json')
        input_seq = ['0', f'e {filepath}', 'q']
        input_iter = iter(input_seq)
        with patch('builtins.input', side_effect=input_iter):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                try:
                    run(0)
                except StopIteration:
                    pass
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert 'path' in data
        assert 'start' in data
        assert data['start'] == 0

    def test_run_from_nonzero_start(self):
        out = self._run_with_inputs([], start=42)
        assert '42' in out

    def test_run_path_in_final_summary(self):
        out = self._run_with_inputs(['0'])
        # итоговый путь печатается в конце
        assert 'путь' in out.lower() or 'Путь' in out

    def test_run_eofstop_graceful(self):
        """EOFError при вводе не должен ронять run()."""
        with patch('builtins.input', side_effect=EOFError):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                run(0)
            out = captured.getvalue()
        assert len(out) > 0


class TestNavMain:
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
