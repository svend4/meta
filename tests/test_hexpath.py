"""Тесты игровой логики hexpath."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexpath.game import (
    GameState, GameResult, Player, new_game, best_move,
)
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


class TestCurrentPosTarget(unittest.TestCase):
    def test_current_pos_a(self):
        g = new_game(pos_a=10, pos_b=20)
        self.assertEqual(g.current_pos(), 10)

    def test_current_pos_b(self):
        g = new_game(pos_a=10, pos_b=20)
        move = list(neighbors(10))[0]
        g2 = g.make_move(move)
        self.assertEqual(g2.current_pos(), 20)

    def test_current_target_a(self):
        g = new_game(pos_a=0, pos_b=63, target_a=63, target_b=0)
        self.assertEqual(g.current_target(), 63)

    def test_blocked_player_loses(self):
        """Если у текущего игрока нет ходов в capture_mode → проигрывает."""
        captured = {nb: Player.B for nb in neighbors(0)}
        captured[0] = Player.A
        captured[63] = Player.B
        from projects.hexpath.game import GameState
        g = GameState(
            pos_a=0, pos_b=63,
            target_a=63, target_b=0,
            current_player=Player.A,
            captured=captured,
            history_a=[0], history_b=[63],
            capture_mode=True,
        )
        self.assertEqual(g.result(), GameResult.B_WINS)
        self.assertTrue(g.is_over())


if __name__ == '__main__':
    unittest.main(verbosity=2)
