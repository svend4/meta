"""Тесты однопользовательского режима hexpath (puzzle)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexpath.puzzle import (
    Puzzle, PuzzleState, solve, solve_all,
    generate_puzzle, get_builtin, BUILTIN_PUZZLES,
)
from libs.hexcore.hexcore import neighbors, hamming, SIZE


class TestPuzzleBasic(unittest.TestCase):
    def test_no_blocks_solvable(self):
        """Без блоков любая пара узлов разрешима."""
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        self.assertTrue(p.is_solvable())

    def test_direct_distance(self):
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        self.assertEqual(p.direct_distance, 6)

    def test_par_equals_hamming_no_blocks(self):
        """Без блоков par = расстояние Хэмминга."""
        p = Puzzle(start=0, goal=7, blocked=frozenset())
        self.assertEqual(p.par, hamming(0, 7))

    def test_blocked_start_raises(self):
        with self.assertRaises(ValueError):
            Puzzle(start=5, goal=10, blocked=frozenset({5}))

    def test_blocked_goal_raises(self):
        with self.assertRaises(ValueError):
            Puzzle(start=0, goal=10, blocked=frozenset({10}))

    def test_same_start_goal(self):
        p = Puzzle(start=42, goal=42, blocked=frozenset())
        self.assertTrue(p.is_solvable())
        self.assertEqual(p.par, 0)

    def test_summary_contains_title(self):
        p = Puzzle(start=0, goal=7, blocked=frozenset(), title='Тест')
        s = p.summary()
        self.assertIn('Тест', s)

    def test_summary_contains_start_goal(self):
        p = Puzzle(start=5, goal=20, blocked=frozenset())
        s = p.summary()
        self.assertIn('5', s)
        self.assertIn('20', s)

    def test_summary_unsolvable_shows_status(self):
        """Summary неразрешимой головоломки содержит 'НЕРАЗРЕШИМА'."""
        blocked = frozenset(neighbors(0))
        p = Puzzle(start=0, goal=63, blocked=blocked)
        s = p.summary()
        self.assertIn('НЕРАЗРЕШИМА', s)


class TestSolve(unittest.TestCase):
    def test_solve_trivial(self):
        p = Puzzle(start=0, goal=0, blocked=frozenset())
        path = solve(p)
        self.assertEqual(path, [0])

    def test_solve_one_step(self):
        nb = neighbors(0)[0]
        p = Puzzle(start=0, goal=nb, blocked=frozenset())
        path = solve(p)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], nb)

    def test_solve_path_valid(self):
        """Каждый шаг пути — ребро Q6."""
        p = Puzzle(start=0, goal=42, blocked=frozenset())
        path = solve(p)
        self.assertIsNotNone(path)
        for i in range(len(path) - 1):
            self.assertEqual(hamming(path[i], path[i + 1]), 1)

    def test_solve_blocked_route(self):
        """Блокировка всех соседей start → нет пути."""
        blocked = frozenset(neighbors(0))
        p = Puzzle(start=0, goal=42, blocked=blocked)
        self.assertIsNone(solve(p))

    def test_solve_detour(self):
        """Блокировка прямых путей, но обход существует."""
        # 0→1 заблокирован, но 0→2→3 существует (если 3=1+2)
        p = Puzzle(start=0, goal=3, blocked=frozenset({1}))
        path = solve(p)
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], 3)
        self.assertNotIn(1, path)

    def test_solve_length_optimal(self):
        """BFS даёт кратчайший путь."""
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        path = solve(p)
        self.assertEqual(len(path) - 1, 6)   # hamming(0,63)=6

    def test_solve_unsolvable(self):
        """Все соседи заблокированы — нет пути."""
        blocked = frozenset(neighbors(0))
        p = Puzzle(start=0, goal=63, blocked=blocked)
        self.assertIsNone(solve(p))


class TestSolveAll(unittest.TestCase):
    def test_returns_list(self):
        p = Puzzle(start=0, goal=3, blocked=frozenset())
        paths = solve_all(p)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

    def test_all_paths_correct_length(self):
        """Все пути в solve_all имеют одинаковую длину (оптимальную)."""
        p = Puzzle(start=0, goal=3, blocked=frozenset())
        opt_len = len(solve(p))
        for path in solve_all(p):
            self.assertEqual(len(path), opt_len)

    def test_all_paths_end_at_goal(self):
        p = Puzzle(start=0, goal=7, blocked=frozenset())
        for path in solve_all(p):
            self.assertEqual(path[-1], 7)

    def test_no_paths_when_unsolvable(self):
        blocked = frozenset(neighbors(0))
        p = Puzzle(start=0, goal=63, blocked=blocked)
        self.assertEqual(solve_all(p), [])


class TestPuzzleState(unittest.TestCase):
    def test_initial_position(self):
        p = Puzzle(start=5, goal=20, blocked=frozenset())
        st = PuzzleState(puzzle=p)
        self.assertEqual(st.current, 5)
        self.assertEqual(st.moves, 0)
        self.assertFalse(st.is_solved)

    def test_move_to_neighbor(self):
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        st = PuzzleState(puzzle=p)
        nb = neighbors(0)[0]
        st2 = st.move(nb)
        self.assertEqual(st2.current, nb)
        self.assertEqual(st2.moves, 1)

    def test_move_invalid_not_neighbor(self):
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        st = PuzzleState(puzzle=p)
        with self.assertRaises(ValueError):
            st.move(42)  # 42 не сосед 0

    def test_move_to_blocked_raises(self):
        nb = neighbors(0)[0]
        p = Puzzle(start=0, goal=63, blocked=frozenset({nb}))
        st = PuzzleState(puzzle=p)
        with self.assertRaises(ValueError):
            st.move(nb)

    def test_undo(self):
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        st = PuzzleState(puzzle=p)
        nb = neighbors(0)[0]
        st2 = st.move(nb)
        st3 = st2.undo()
        self.assertEqual(st3.current, 0)
        self.assertEqual(st3.moves, 1)   # откат не уменьшает счётчик

    def test_undo_at_start_no_change(self):
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        st = PuzzleState(puzzle=p)
        st2 = st.undo()
        self.assertEqual(st2.current, 0)

    def test_is_solved(self):
        nb = neighbors(0)[0]
        p = Puzzle(start=0, goal=nb, blocked=frozenset())
        st = PuzzleState(puzzle=p)
        self.assertFalse(st.is_solved)
        st2 = st.move(nb)
        self.assertTrue(st2.is_solved)

    def test_available_moves_no_revisit(self):
        """available_moves не включает уже посещённые узлы."""
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        st = PuzzleState(puzzle=p)
        nb = neighbors(0)[0]
        st2 = st.move(nb)
        self.assertNotIn(0, st2.available_moves())

    def test_rating_perfect(self):
        """Решение за par ходов = идеально."""
        p = Puzzle(start=0, goal=7, blocked=frozenset())
        par = p.par
        st = PuzzleState(puzzle=p)
        for step in solve(p)[1:]:
            st = st.move(step)
        self.assertIn('★★★', st.rating())

    def test_rating_unsolved(self):
        p = Puzzle(start=0, goal=63, blocked=frozenset())
        st = PuzzleState(puzzle=p)
        self.assertEqual(st.rating(), '?')


class TestGeneratePuzzle(unittest.TestCase):
    def test_easy_solvable(self):
        for seed in range(5):
            p = generate_puzzle(difficulty='easy', seed=seed)
            self.assertTrue(p.is_solvable())

    def test_medium_solvable(self):
        for seed in range(5):
            p = generate_puzzle(difficulty='medium', seed=seed)
            self.assertTrue(p.is_solvable())

    def test_hard_solvable(self):
        for seed in range(3):
            p = generate_puzzle(difficulty='hard', seed=seed)
            self.assertTrue(p.is_solvable())

    def test_start_not_blocked(self):
        p = generate_puzzle(difficulty='medium', seed=99)
        self.assertNotIn(p.start, p.blocked)

    def test_goal_not_blocked(self):
        p = generate_puzzle(difficulty='medium', seed=99)
        self.assertNotIn(p.goal, p.blocked)

    def test_invalid_difficulty(self):
        with self.assertRaises(ValueError):
            generate_puzzle(difficulty='impossible')

    def test_reproducible_with_seed(self):
        p1 = generate_puzzle(difficulty='easy', seed=7)
        p2 = generate_puzzle(difficulty='easy', seed=7)
        self.assertEqual(p1.start, p2.start)
        self.assertEqual(p1.goal, p2.goal)
        self.assertEqual(p1.blocked, p2.blocked)


class TestBuiltinPuzzles(unittest.TestCase):
    def test_all_solvable(self):
        for i, puz in enumerate(BUILTIN_PUZZLES):
            with self.subTest(i=i):
                self.assertTrue(puz.is_solvable(), f"Встроенная #{i} неразрешима")

    def test_get_builtin_valid(self):
        for i in range(len(BUILTIN_PUZZLES)):
            p = get_builtin(i)
            self.assertIsInstance(p, Puzzle)

    def test_get_builtin_out_of_range(self):
        with self.assertRaises(IndexError):
            get_builtin(len(BUILTIN_PUZZLES))

    def test_builtin_titles(self):
        for puz in BUILTIN_PUZZLES:
            self.assertIsInstance(puz.title, str)
            self.assertGreater(len(puz.title), 0)

    def test_builtin_par_positive(self):
        for i, puz in enumerate(BUILTIN_PUZZLES):
            with self.subTest(i=i):
                self.assertGreater(puz.par, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
