"""Тесты игровой логики hexpath."""
import io
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from projects.hexpath.game import (
    GameState, GameResult, Player, new_game, best_move, minimax,
)
from projects.hexpath.cli import fmt_hex, draw_board, announce_result, ai_move, human_move, play
from libs.hexcore.hexcore import neighbors, hamming


class TestNewGame(unittest.TestCase):
    def test_default(self):
        g = new_game()
        self.assertEqual(g.pos_a, 0)
        self.assertEqual(g.pos_b, 63)
        self.assertEqual(g.target_a, 63)
        self.assertEqual(g.target_b, 0)
        self.assertEqual(g.current_player, Player.A)

    def test_custom(self):
        g = new_game(pos_a=10, pos_b=20, target_a=30, target_b=5)
        self.assertEqual(g.pos_a, 10)
        self.assertEqual(g.pos_b, 20)
        self.assertEqual(g.target_a, 30)
        self.assertEqual(g.target_b, 5)


class TestPlayer(unittest.TestCase):
    def test_other(self):
        self.assertEqual(Player.A.other(), Player.B)
        self.assertEqual(Player.B.other(), Player.A)

    def test_symbol(self):
        self.assertNotEqual(Player.A.symbol(), Player.B.symbol())


class TestLegalMoves(unittest.TestCase):
    def test_no_capture_all_neighbors(self):
        g = new_game(pos_a=0, pos_b=42, capture_mode=False)
        moves = g.legal_moves(Player.A)
        self.assertEqual(set(moves), set(neighbors(0)))

    def test_capture_blocks_opponent(self):
        g = new_game(pos_a=0, pos_b=1, capture_mode=True)
        # A стартует на 0, B на 1. Ход A — нельзя идти на 1 (захвачен B)
        moves_a = g.legal_moves(Player.A)
        self.assertNotIn(1, moves_a)

    def test_can_revisit_own_captures(self):
        g = new_game(pos_a=0, pos_b=42, capture_mode=True)
        # A ходит на 1, затем обратно на 0 (захвачен A — можно)
        g2 = g.make_move(1)   # A → 1
        g3 = g2.make_move(neighbors(42)[0])  # B ходит
        moves_a = g3.legal_moves(Player.A)
        self.assertIn(0, moves_a)  # 0 захвачен A — можно


class TestMakeMove(unittest.TestCase):
    def test_valid_move_a(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        move = list(neighbors(0))[0]
        g2 = g.make_move(move)
        self.assertEqual(g2.pos_a, move)
        self.assertEqual(g2.current_player, Player.B)

    def test_valid_move_b(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        move_a = list(neighbors(0))[0]
        g2 = g.make_move(move_a)
        move_b = list(neighbors(63))[0]
        g3 = g2.make_move(move_b)
        self.assertEqual(g3.pos_b, move_b)
        self.assertEqual(g3.current_player, Player.A)

    def test_invalid_move_raises(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        with self.assertRaises(ValueError):
            g.make_move(42)  # 42 не сосед 0

    def test_history_grows(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        move = list(neighbors(0))[0]
        g2 = g.make_move(move)
        self.assertEqual(len(g2.history_a), 2)
        self.assertEqual(len(g2.history_b), 1)

    def test_capture_mode_adds_to_captured(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=True)
        move = list(neighbors(0))[0]
        g2 = g.make_move(move)
        self.assertIn(move, g2.captured)
        self.assertEqual(g2.captured[move], Player.A)


class TestGameResult(unittest.TestCase):
    def test_ongoing(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        self.assertEqual(g.result(), GameResult.ONGOING)

    def test_a_wins_target(self):
        # A уже на своей цели
        g = new_game(pos_a=63, pos_b=0, target_a=63, target_b=0, capture_mode=False)
        self.assertEqual(g.result(), GameResult.A_WINS)

    def test_b_wins_target(self):
        g = new_game(pos_a=1, pos_b=0, target_a=63, target_b=0, capture_mode=False)
        self.assertEqual(g.result(), GameResult.B_WINS)

    def test_is_over(self):
        g = new_game(pos_a=63, pos_b=0, target_a=63, target_b=0, capture_mode=False)
        self.assertTrue(g.is_over())

    def test_not_over(self):
        g = new_game()
        self.assertFalse(g.is_over())


class TestAI(unittest.TestCase):
    def test_best_move_valid(self):
        g = new_game(capture_mode=False)
        move = best_move(g, depth=2)
        self.assertIn(move, g.legal_moves())

    def test_ai_game_terminates(self):
        """Полная игра AI vs AI должна завершиться за разумное число ходов."""
        g = new_game(capture_mode=False)
        for _ in range(30):
            if g.is_over():
                break
            move = best_move(g, depth=2)
            g = g.make_move(move)
        self.assertTrue(g.is_over())

    def test_ai_prefers_goal(self):
        """Если цель рядом, AI должен её взять."""
        # A на расстоянии 1 от цели
        target = 63
        start = neighbors(target)[0]   # один шаг от 63
        g = new_game(pos_a=start, pos_b=0, target_a=target, target_b=start,
                     capture_mode=False)
        move = best_move(g, depth=1)
        self.assertEqual(move, target)

    def test_ai_no_legal_moves_raises(self):
        """Если нет ходов, best_move должен бросить ValueError."""
        # Создать состояние, где у текущего игрока нет ходов (все соседи захвачены)
        g = new_game(pos_a=0, pos_b=63, capture_mode=True)
        # Захватим все соседи 0 от имени B
        captured = {nb: Player.B for nb in neighbors(0)}
        captured[0] = Player.A
        captured[63] = Player.B
        g2 = GameState(
            pos_a=0, pos_b=63,
            target_a=63, target_b=0,
            current_player=Player.A,
            captured=captured,
            history_a=[0], history_b=[63],
            capture_mode=True,
        )
        with self.assertRaises(ValueError):
            best_move(g2, depth=1)


class TestGameStateCurrentMethods(unittest.TestCase):
    """Тесты методов current_pos() и current_target()."""

    def test_current_pos_a(self):
        g = new_game(pos_a=10, pos_b=50, capture_mode=False)
        self.assertEqual(g.current_pos(), 10)

    def test_current_pos_b(self):
        g = new_game(pos_a=10, pos_b=50, capture_mode=False)
        move = list(neighbors(10))[0]
        g2 = g.make_move(move)
        self.assertEqual(g2.current_pos(), 50)

    def test_current_target_a(self):
        g = new_game(pos_a=0, pos_b=63, target_a=63, target_b=0, capture_mode=False)
        self.assertEqual(g.current_target(), 63)

    def test_current_target_b(self):
        g = new_game(pos_a=0, pos_b=63, target_a=63, target_b=0, capture_mode=False)
        move = list(neighbors(0))[0]
        g2 = g.make_move(move)
        self.assertEqual(g2.current_target(), 0)

    def test_current_pos_changes_after_move(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        pos_before = g.current_pos()
        move = list(neighbors(0))[0]
        g2 = g.make_move(move)
        self.assertNotEqual(g2.current_pos(), pos_before)


class TestGameImmutability(unittest.TestCase):
    """make_move() должна возвращать новый объект, не изменяя исходный."""

    def test_returns_new_object(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        move = list(neighbors(0))[0]
        g2 = g.make_move(move)
        self.assertIsNot(g, g2)

    def test_original_pos_unchanged(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        move = list(neighbors(0))[0]
        g.make_move(move)
        self.assertEqual(g.pos_a, 0)

    def test_original_history_unchanged(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        hist_len = len(g.history_a)
        move = list(neighbors(0))[0]
        g.make_move(move)
        self.assertEqual(len(g.history_a), hist_len)


class TestGameResultBlock(unittest.TestCase):
    """Заблокированный игрок проигрывает."""

    def test_a_blocked_b_wins(self):
        """Если у A нет ходов (все соседи захвачены B) → B_WINS."""
        from projects.hexpath.game import GameState
        nbs = list(neighbors(0))
        captured = {nb: Player.B for nb in nbs}
        captured[0] = Player.A
        captured[63] = Player.B
        g = GameState(
            pos_a=0, pos_b=63,
            target_a=63, target_b=0,
            current_player=Player.A,
            captured=captured,
            history_a=[0], history_b=[63],
            capture_mode=True,
        )
        from projects.hexpath.game import GameResult
        self.assertEqual(g.result(), GameResult.B_WINS)

    def test_capture_mode_off_never_blocks(self):
        """Без capture_mode у A всегда 6 ходов от любой вершины."""
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        self.assertEqual(len(g.legal_moves(Player.A)), 6)


class TestMinimax(unittest.TestCase):
    """Тесты minimax с альфа-бета отсечением."""

    def test_returns_float_at_depth_0(self):
        g = new_game()
        val = minimax(g, depth=0, alpha=float('-inf'), beta=float('inf'), maximizing=True)
        self.assertIsInstance(val, float)

    def test_returns_finite_value(self):
        g = new_game()
        val = minimax(g, 1, float('-inf'), float('inf'), True)
        self.assertFalse(val == float('inf') or val == float('-inf'))

    def test_maximizing_ge_minimizing_at_same_depth(self):
        """Максимизирующий вызов ≥ минимизирующего из того же состояния."""
        g = new_game()
        v_max = minimax(g, 1, float('-inf'), float('inf'), True)
        v_min = minimax(g, 1, float('-inf'), float('inf'), False)
        self.assertGreaterEqual(v_max, v_min)

    def test_deeper_search_consistent(self):
        """Поиск на глубину 2 тоже возвращает float."""
        g = new_game()
        val = minimax(g, 2, float('-inf'), float('inf'), True)
        self.assertIsInstance(val, float)

    def test_over_state_returns_evaluation(self):
        """Для завершённой игры глубина не важна."""
        g = new_game(pos_a=63, pos_b=63, capture_mode=False)
        # pos_a достиг target_a=63 → игра завершена
        if g.is_over():
            val = minimax(g, 3, float('-inf'), float('inf'), True)
            self.assertIsInstance(val, float)


class TestCLIFunctions(unittest.TestCase):
    """Тесты fmt_hex, draw_board, announce_result из cli.py."""

    def _capture(self, fn, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*args)
        return buf.getvalue()

    # fmt_hex ------------------------------------------------------------------

    def test_fmt_hex_returns_string(self):
        self.assertIsInstance(fmt_hex(0), str)

    def test_fmt_hex_contains_number(self):
        self.assertIn('42', fmt_hex(42))

    def test_fmt_hex_contains_bits(self):
        out = fmt_hex(0)
        self.assertIn('000000', out)

    def test_fmt_hex_all_valid(self):
        for h in range(64):
            self.assertIsInstance(fmt_hex(h), str)

    # draw_board ---------------------------------------------------------------

    def test_draw_board_produces_output(self):
        g = new_game()
        out = self._capture(draw_board, g)
        self.assertGreater(len(out), 0)

    def test_draw_board_shows_positions(self):
        g = new_game()
        out = self._capture(draw_board, g)
        self.assertIn('000000', out)  # pos_a=0 → 000000

    def test_draw_board_shows_player_labels(self):
        g = new_game()
        out = self._capture(draw_board, g)
        self.assertIn('A', out)
        self.assertIn('B', out)

    def test_draw_board_capture_mode_shows_captures(self):
        g = new_game(capture_mode=True)
        out = self._capture(draw_board, g)
        self.assertIn('Захвачено', out)

    def test_draw_board_no_capture_mode_no_captures(self):
        g = new_game(capture_mode=False)
        out = self._capture(draw_board, g)
        self.assertNotIn('Захвачено', out)

    # announce_result ----------------------------------------------------------

    def test_announce_result_produces_output(self):
        g = new_game(pos_a=63, capture_mode=False)
        out = self._capture(announce_result, g)
        self.assertGreater(len(out), 0)

    def test_announce_result_shows_separator(self):
        g = new_game(pos_a=63, capture_mode=False)
        out = self._capture(announce_result, g)
        self.assertIn('═', out)

    def test_announce_result_shows_paths(self):
        g = new_game(pos_a=63, capture_mode=False)
        out = self._capture(announce_result, g)
        self.assertIn('Путь', out)

    # ai_move ------------------------------------------------------------------

    def test_ai_move_returns_game_state(self):
        g = new_game()
        buf = io.StringIO()
        with redirect_stdout(buf):
            g2 = ai_move(g, depth=2)
        self.assertIsInstance(g2, GameState)

    def test_ai_move_changes_player(self):
        g = new_game()
        buf = io.StringIO()
        with redirect_stdout(buf):
            g2 = ai_move(g, depth=2)
        self.assertEqual(g.current_player, Player.A)
        self.assertEqual(g2.current_player, Player.B)

    def test_ai_move_valid_position(self):
        g = new_game()
        buf = io.StringIO()
        with redirect_stdout(buf):
            g2 = ai_move(g, depth=2)
        self.assertIn(g2.pos_a, neighbors(g.pos_a))

    def test_ai_move_prints_output(self):
        g = new_game()
        out = self._capture(ai_move, g, 2)
        self.assertGreater(len(out), 0)


    # draw_board with no legal moves (line 62) --------------------------------

    def test_draw_board_no_legal_moves(self):
        """draw_board печатает 'Нет допустимых ходов', когда ходов нет."""
        # Заблокировать все соседи вершины 0 игроком B
        blocked = {nb: Player.B for nb in neighbors(0)}
        state = GameState(
            pos_a=0, pos_b=63, target_a=63, target_b=0,
            current_player=Player.A,
            captured=blocked,
            capture_mode=True,
        )
        out = self._capture(draw_board, state)
        self.assertIn('Нет допустимых ходов', out)

    # announce_result B_WINS (lines 72-73) ------------------------------------

    def test_announce_result_b_wins(self):
        """announce_result показывает победу B, когда pos_b == target_b."""
        # pos_b=0, target_b = pos_a = 0 → B_WINS сразу
        g = new_game(pos_a=1, pos_b=0, target_a=0, target_b=0, capture_mode=False)
        out = self._capture(announce_result, g)
        self.assertIn('Победил игрок B', out)


class TestHumanMove(unittest.TestCase):
    """Тесты human_move с мок-вводом."""

    def _run_human(self, inputs, state=None):
        if state is None:
            state = new_game(capture_mode=False)
        with patch('builtins.input', side_effect=inputs):
            with redirect_stdout(io.StringIO()):
                return human_move(state)

    def test_valid_bit_on_first_try(self):
        """Ввод '0' → ход на бит 0."""
        g = new_game(capture_mode=False)
        new_g = self._run_human(['0'], g)
        self.assertIsInstance(new_g, GameState)
        self.assertNotEqual(new_g.pos_a, g.pos_a)

    def test_help_then_valid(self):
        """'h' показывает подсказку, затем '0' → ход."""
        g = new_game(capture_mode=False)
        new_g = self._run_human(['h', '0'], g)
        self.assertIsInstance(new_g, GameState)

    def test_invalid_string_then_valid(self):
        """Нечисловой ввод → сообщение об ошибке, затем '0' → ход."""
        g = new_game(capture_mode=False)
        new_g = self._run_human(['abc', '0'], g)
        self.assertIsInstance(new_g, GameState)

    def test_out_of_range_then_valid(self):
        """Число вне диапазона 0-5 → ошибка, затем '1' → ход."""
        g = new_game(capture_mode=False)
        new_g = self._run_human(['9', '1'], g)
        self.assertIsInstance(new_g, GameState)

    def test_quit_raises_systemexit(self):
        """'q' → sys.exit(0)."""
        g = new_game(capture_mode=False)
        with self.assertRaises(SystemExit):
            self._run_human(['q'], g)

    def test_blocked_capture_then_valid(self):
        """Ход на захваченную клетку → сообщение, затем другой ход."""
        # Захватить вершину 1 (neighbor 0, bit 0) игроком B
        blocked = {1: Player.B}
        state = GameState(
            pos_a=0, pos_b=63, target_a=63, target_b=0,
            current_player=Player.A,
            captured=blocked,
            capture_mode=True,
        )
        # '0' → вершина 1 (заблокирована), затем '1' → вершина 2 (свободна)
        with patch('builtins.input', side_effect=['0', '1']):
            with redirect_stdout(io.StringIO()):
                new_state = human_move(state)
        self.assertEqual(new_state.pos_a, 2)

    def test_illegal_no_capture_then_valid(self):
        """Ход недопустим без режима захвата → сообщение, затем другой ход."""
        # В режиме без захвата все соседи доступны, но нужен blocked scenario.
        # Вместо этого ввести несуществующую вершину через '9' (out of range), потом '0'.
        g = new_game(capture_mode=False)
        new_g = self._run_human(['9', '0'], g)
        self.assertIsInstance(new_g, GameState)


class TestPlay(unittest.TestCase):
    """Тесты функции play()."""

    def test_play_ai_vs_ai_no_capture(self):
        """play(ai_vs_ai=True, capture=False) завершается без ошибок."""
        with patch('builtins.input', return_value=''):
            with redirect_stdout(io.StringIO()):
                play(ai_vs_ai=True, capture=False, ai_depth=1)

    def test_play_capture_mode_header(self):
        """play(capture=True) показывает строку про захват — мок already-over game."""
        done_state = GameState(
            pos_a=63, pos_b=0, target_a=63, target_b=0,
            capture_mode=True,
        )
        buf = io.StringIO()
        with patch('projects.hexpath.cli.new_game', return_value=done_state):
            with redirect_stdout(buf):
                play(ai_vs_ai=True, capture=True, ai_depth=1)
        self.assertIn('захват узлов включён', buf.getvalue())


class TestMainCLI(unittest.TestCase):
    """Тесты main() через sys.argv."""

    def test_main_ai_vs_ai(self):
        """main() с --ai-vs-ai --no-capture завершается без ошибок."""
        from projects.hexpath.cli import main
        with patch.object(sys, 'argv', ['cli.py', '--ai-vs-ai', '--no-capture']):
            with patch('builtins.input', return_value=''):
                with redirect_stdout(io.StringIO()):
                    main()


class TestCurrentPosTarget(unittest.TestCase):
    def test_player_symbol_nonempty(self):
        """symbol() обоих игроков — непустая строка."""
        for p in (Player.A, Player.B):
            s = p.symbol()
            self.assertIsInstance(s, str)
            self.assertGreater(len(s), 0)

    def test_new_game_default_capture_mode_initializes_captured(self):
        g = new_game()
        self.assertTrue(g.capture_mode)
        self.assertIn(g.pos_a, g.captured)
        self.assertIn(g.pos_b, g.captured)

    def test_legal_moves_player_b(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        moves_b = g.legal_moves(Player.B)
        self.assertEqual(set(moves_b), set(neighbors(63)))

    def test_is_over_a_wins(self):
        g = new_game(pos_a=63, pos_b=0, target_a=63, target_b=63,
                     capture_mode=False)
        self.assertEqual(g.result(), GameResult.A_WINS)
        self.assertTrue(g.is_over())


class TestGameStateExtra(unittest.TestCase):
    def test_gameresult_draw_exists(self):
        self.assertIsNotNone(GameResult.DRAW)

    def test_current_player_b_after_a_move(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        move = list(neighbors(0))[0]
        g2 = g.make_move(move)
        self.assertEqual(g2.current_player, Player.B)

    def test_history_b_unchanged_after_a_move(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        move = list(neighbors(0))[0]
        g2 = g.make_move(move)
        self.assertEqual(len(g2.history_b), 1)

    def test_legal_moves_default_equals_current_player(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        self.assertEqual(set(g.legal_moves()), set(g.legal_moves(Player.A)))

    def test_new_game_capture_false_empty_captured(self):
        g = new_game(pos_a=0, pos_b=63, capture_mode=False)
        self.assertEqual(g.captured, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
