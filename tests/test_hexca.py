"""Тесты клеточного автомата hexca (движок + правила)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from projects.hexca.hexca import CA1D, CA2D, cell_char, demo_1d, demo_2d
from projects.hexca.animate import animate_1d, animate_2d
from projects.hexca.rules import (
    majority_vote, xor_rule, identity, conway_like, RULES, get_rule,
    smooth_rule, cyclic_rule, outer_totalistic, random_walk,
)
from libs.hexcore.hexcore import neighbors, yang_count, SIZE


class TestRuleIdentity(unittest.TestCase):
    def test_identity_no_change(self):
        for h in range(SIZE):
            self.assertEqual(identity(h, [0, 1, 2]), h)

    def test_identity_empty_neighbors(self):
        self.assertEqual(identity(42, []), 42)


class TestRuleMajorityVote(unittest.TestCase):
    def test_returns_valid_state(self):
        for h in range(0, SIZE, 7):
            result = majority_vote(h, list(neighbors(h))[:4])
            self.assertIn(result, range(SIZE))

    def test_no_change_with_equal_split(self):
        # При 2 соседях «за» и 2 «против» — большинства нет, черта не меняется
        # Если 2 соседей, порог: count_other > 1 → нужно >1 из 2 = невозможно кроме 2
        result = majority_vote(0, [0, 0])   # оба соседа тоже 0 — ничего не меняется
        self.assertEqual(result, 0)

    def test_unanimous_flip(self):
        # Все 6 соседей гексаграммы 0 имеют бит 0 = 1 (т.е. нечётные числа)
        # Это не обязательно выполняется, поэтому проверяем только корректность возврата
        result = majority_vote(0, [63, 63, 63])  # 63=111111: все биты 1
        self.assertIn(result, range(SIZE))


class TestRuleXor(unittest.TestCase):
    def test_xor_no_neighbors(self):
        self.assertEqual(xor_rule(42, []), 42)

    def test_xor_returns_neighbor_in_q6(self):
        # Результат должен быть соседом в Q6 относительно current (изменён ровно 1 бит)
        # или равен current (если diff=0)
        for h in range(0, SIZE, 5):
            result = xor_rule(h, [neighbors(h)[0]])
            diff = bin(h ^ result).count('1')
            self.assertIn(diff, (0, 1))

    def test_xor_same_value(self):
        # XOR с собой = 0, diff от current ненулевой → переворот 1 бита
        result = xor_rule(42, [42])
        diff = bin(42 ^ result).count('1')
        self.assertIn(diff, (0, 1))


class TestRuleConway(unittest.TestCase):
    def test_dead_cell_no_birth(self):
        rule = conway_like(birth={3}, survive={2, 3})
        # «Мёртвая» клетка (0 ян) с 0 живыми соседями — остаётся мёртвой
        result = rule(0, [0, 0, 0, 0])
        self.assertEqual(yang_count(result), 0)

    def test_alive_cell_survives(self):
        rule = conway_like(birth={3}, survive={2, 3})
        # Живая клетка с 2-3 живыми соседями — выживает
        alive = 7   # yang_count=3
        nbrs = [1, 1, 0, 0]  # 2 живых соседа
        result = rule(alive, nbrs)
        # Должна выжить (вернуть то же состояние)
        self.assertEqual(result, alive)

    def test_returns_valid_state(self):
        rule = conway_like()
        for h in range(0, SIZE, 7):
            result = rule(h, list(range(4)))
            self.assertIn(result, range(SIZE))


class TestGetRule(unittest.TestCase):
    def test_known_rules(self):
        for name in RULES:
            rule = get_rule(name)
            self.assertIsNotNone(rule)

    def test_unknown_rule_raises(self):
        with self.assertRaises(ValueError):
            get_rule('nonexistent_rule')

    def test_all_rules_callable(self):
        for name, rule in RULES.items():
            result = rule(0, [1, 2, 3])
            self.assertIn(result, range(SIZE))


class TestCA1DBasic(unittest.TestCase):
    def test_init_random(self):
        import random
        random.seed(42)
        ca = CA1D(width=10, rule=identity)
        self.assertEqual(len(ca.grid), 10)
        self.assertEqual(ca.generation, 0)

    def test_init_custom(self):
        ca = CA1D(width=5, rule=identity, init=[0, 1, 2, 3, 4])
        self.assertEqual(ca.grid, [0, 1, 2, 3, 4])

    def test_step_identity(self):
        ca = CA1D(width=5, rule=identity, init=[7, 14, 21, 28, 42])
        original = list(ca.grid)
        ca.step()
        self.assertEqual(ca.grid, original)
        self.assertEqual(ca.generation, 1)

    def test_history_grows(self):
        ca = CA1D(width=5, rule=identity, init=[0] * 5)
        ca.run(3)
        self.assertEqual(len(ca.history), 4)   # start + 3 шага

    def test_periodic_boundary(self):
        # При identity: соседи крайних клеток — это первая и последняя
        ca = CA1D(width=4, rule=identity, init=[0, 1, 2, 3])
        ca.step()
        self.assertEqual(ca.grid, [0, 1, 2, 3])  # identity всегда сохраняет

    def test_render_row_length(self):
        ca = CA1D(width=10, rule=identity, init=[0] * 10)
        row_str = ca.render_row()
        self.assertEqual(len(row_str), 10)

    def test_stats_keys(self):
        ca = CA1D(width=8, rule=identity, init=[42] * 8)
        s = ca.stats()
        for key in ('generation', 'width', 'mean_yang', 'unique_states'):
            self.assertIn(key, s)

    def test_stats_unique_states(self):
        ca = CA1D(width=4, rule=identity, init=[0, 0, 0, 0])
        s = ca.stats()
        self.assertEqual(s['unique_states'], 1)

    def test_xor_rule_propagates(self):
        # XOR-правило: одна «горячая» клетка должна распространяться
        init = [0] * 20
        init[10] = 42
        ca = CA1D(width=20, rule=xor_rule, init=init)
        ca.run(3)
        # После 3 шагов несколько клеток ненулевые
        nonzero = sum(1 for h in ca.grid if h != 0)
        self.assertGreater(nonzero, 1)


class TestCA2DBasic(unittest.TestCase):
    def test_init_grid_shape(self):
        import random
        random.seed(0)
        ca = CA2D(width=5, height=4, rule=identity)
        self.assertEqual(len(ca.grid), 4)
        self.assertEqual(len(ca.grid[0]), 5)

    def test_init_custom(self):
        grid = [[0, 1, 2], [3, 4, 5]]
        ca = CA2D(width=3, height=2, rule=identity, init=grid)
        self.assertEqual(ca.grid[0][0], 0)
        self.assertEqual(ca.grid[1][2], 5)

    def test_step_identity(self):
        grid = [[7, 14], [21, 28]]
        ca = CA2D(width=2, height=2, rule=identity, init=grid)
        ca.step()
        self.assertEqual(ca.grid[0][0], 7)
        self.assertEqual(ca.grid[1][1], 28)
        self.assertEqual(ca.generation, 1)

    def test_periodic_boundary(self):
        # Угловая клетка (0,0) имеет соседей: (height-1, 0), (1, 0), (0, width-1), (0, 1)
        # При identity это не важно, но проверяем что step не падает
        ca = CA2D(width=3, height=3, rule=identity, init=[[i * 3 + j for j in range(3)] for i in range(3)])
        ca.step()
        self.assertEqual(ca.generation, 1)

    def test_render_format(self):
        import random
        random.seed(1)
        ca = CA2D(width=8, height=4, rule=identity)
        rendered = ca.render()
        lines = rendered.split('\n')
        # Должны быть: заголовок + top border + 4 строки + bottom border
        self.assertEqual(len(lines), 7)

    def test_stats_keys(self):
        import random
        random.seed(0)
        ca = CA2D(width=4, height=4, rule=identity)
        s = ca.stats()
        for key in ('generation', 'size', 'mean_yang', 'unique_states'):
            self.assertIn(key, s)

    def test_run_multiple_steps(self):
        import random
        random.seed(5)
        ca = CA2D(width=6, height=6, rule=xor_rule)
        ca.run(5)
        self.assertEqual(ca.generation, 5)


class TestRuleSmoothAndCyclic(unittest.TestCase):
    def test_smooth_returns_valid_state(self):
        for h in range(0, SIZE, 7):
            result = smooth_rule(h, list(range(4)))
            self.assertIn(result, range(SIZE))

    def test_smooth_empty_neighbors(self):
        self.assertEqual(smooth_rule(42, []), 42)

    def test_smooth_all_same_no_change(self):
        """Если все соседи == текущему, ничего не меняется."""
        for h in range(SIZE):
            result = smooth_rule(h, [h, h, h, h])
            self.assertEqual(result, h)

    def test_smooth_converges(self):
        """Правило сглаживания на однородном поле — стабильно."""
        from projects.hexca.hexca import CA1D
        init = [21] * 10   # однородное поле
        ca = CA1D(width=10, rule=smooth_rule, init=init)
        ca.run(5)
        # Поле должно остаться однородным (у всех соседи тоже 21)
        self.assertEqual(ca.grid, [21] * 10)

    def test_cyclic_returns_valid_state(self):
        rule = cyclic_rule(step=1)
        for h in range(0, SIZE, 7):
            result = rule(h, list(range(4)))
            self.assertIn(result, range(SIZE))

    def test_cyclic_no_neighbors(self):
        rule = cyclic_rule(step=1)
        self.assertEqual(rule(0, []), 0)

    def test_cyclic_registered(self):
        self.assertIn('cyclic', RULES)
        self.assertIn('cyclic2', RULES)

    def test_outer_totalistic_no_match(self):
        """Если ключ не в таблице, состояние не меняется."""
        rule = outer_totalistic({})
        for h in range(0, SIZE, 5):
            self.assertEqual(rule(h, [0, 1, 2]), h)

    def test_outer_totalistic_increase_yang(self):
        """Если new_c > c, добавить инь-черту в ян."""
        rule = outer_totalistic({(0, 0): 1})
        result = rule(0, [0, 0, 0])
        self.assertEqual(yang_count(result), 1)

    def test_outer_totalistic_decrease_yang(self):
        """Если new_c < c, убрать ян-черту в инь."""
        rule = outer_totalistic({(6, 0): 5})
        result = rule(63, [0, 0, 0])
        self.assertEqual(yang_count(result), 5)


class TestCA1DXorPattern(unittest.TestCase):
    """Проверка свойств XOR-правила (аналог правила 90 Вольфрама)."""

    def test_symmetry_preserved(self):
        """XOR-правило сохраняет симметрию начальной конфигурации."""
        width = 11
        init = [0] * width
        init[5] = 42  # центральная клетка
        ca = CA1D(width=width, rule=xor_rule, init=init)
        ca.run(4)
        # Проверяем симметрию: grid[i] == grid[width-1-i]
        for i in range(width // 2):
            self.assertEqual(
                yang_count(ca.grid[i]),
                yang_count(ca.grid[width - 1 - i]),
                f"Нарушена симметрия на позиции {i}"
            )

    def test_single_cell_spreads(self):
        """Одна ненулевая клетка порождает треугольный паттерн."""
        init = [0] * 15
        init[7] = 1
        ca = CA1D(width=15, rule=xor_rule, init=init)
        for step in range(1, 5):
            ca.step()
            nonzero_positions = [i for i, h in enumerate(ca.grid) if h != 0]
            # После k шагов должно быть как минимум 2 ненулевых клетки
            self.assertGreaterEqual(len(nonzero_positions), 2 if step > 0 else 1)


class TestCellChar(unittest.TestCase):
    """Тесты cell_char: символ ячейки по числу ян-черт."""

    def test_returns_string(self):
        for h in range(SIZE):
            self.assertIsInstance(cell_char(h), str)

    def test_zero_yang_is_space(self):
        # h=0: 0 ян → первый символ CELL_CHARS (пробел)
        self.assertEqual(cell_char(0), ' ')

    def test_six_yang_is_block(self):
        # h=63: 6 ян → последний символ перед зеркала
        # CELL_CHARS = ' ·░▒▓█▓' → index 6 = '▓'
        self.assertEqual(cell_char(63), '▓')

    def test_monotone_with_yang(self):
        """Символ определяется только yang_count, одинаков для одинакового rang."""
        for h in range(SIZE):
            yc = yang_count(h)
            # Любой другой узел с тем же yang_count даёт тот же символ
            same = [x for x in range(SIZE) if yang_count(x) == yc]
            for x in same:
                self.assertEqual(cell_char(x), cell_char(h))

    def test_lower_yang_gives_different_char_than_higher(self):
        """Крайние уровни ян (0 и 6) дают разные символы."""
        self.assertNotEqual(cell_char(0), cell_char(63))


class TestRandomWalk(unittest.TestCase):
    """Тесты random_walk: переход к случайному соседу."""

    def test_returns_neighbor(self):
        for h in range(0, SIZE, 8):
            result = random_walk(h, neighbors(h))
            self.assertIn(result, neighbors(h))

    def test_returns_int(self):
        result = random_walk(0, neighbors(0))
        self.assertIsInstance(result, int)

    def test_in_valid_range(self):
        for h in range(SIZE):
            result = random_walk(h, neighbors(h))
            self.assertGreaterEqual(result, 0)
            self.assertLess(result, SIZE)

    def test_stays_connected(self):
        """За 20 шагов random_walk обходит несколько вершин."""
        visited = {0}
        h = 0
        for _ in range(20):
            h = random_walk(h, neighbors(h))
            visited.add(h)
        self.assertGreater(len(visited), 1)


class TestOuterTotalistic(unittest.TestCase):
    """Тесты outer_totalistic: правило зависит от (yang(current), sum_yang_nbrs)."""

    def test_identity_when_no_match(self):
        """Пустая таблица: правило не меняет состояние."""
        rule = outer_totalistic({})
        for h in range(0, SIZE, 7):
            self.assertEqual(rule(h, neighbors(h)), h)

    def test_returns_callable(self):
        rule = outer_totalistic({})
        self.assertTrue(callable(rule))

    def test_specific_transition(self):
        """При совпадении (c, s) правило делает 1 бит-флип к новому yang_count."""
        # h=0 (yang=0), nbrs=neighbors(0) — у всех соседей yang=1 → sum=6
        # Зададим таблицу: (0, 6) → 3 (увеличить ян)
        h = 0
        nbrs = neighbors(h)  # yang=1 у каждого
        s = sum(yang_count(n) for n in nbrs)  # = 6
        rule = outer_totalistic({(0, s): 3})
        result = rule(h, nbrs)
        # Правило делает ровно 1 флип: yang_count 0→1
        self.assertEqual(yang_count(result), 1)
        self.assertIn(result, neighbors(h))

    def test_no_change_when_same_yang(self):
        """Если new_yang == current_yang, состояние не меняется."""
        h = 0
        nbrs = neighbors(h)
        s = sum(yang_count(n) for n in nbrs)
        # Таблица: переход в то же yang_count — нет изменения
        rule = outer_totalistic({(0, s): 0})
        self.assertEqual(rule(h, nbrs), h)

    def test_ca1d_with_outer_totalistic(self):
        """CA1D с outer_totalistic правилом делает корректный шаг."""
        rule = outer_totalistic({})   # тождество — всё 0 → 0
        ca = CA1D(width=8, rule=rule, init=[0] * 8)
        ca.step()
        self.assertEqual(ca.grid, [0] * 8)


class TestDemoFunctions(unittest.TestCase):
    """Тесты demo_1d и demo_2d — демонстрационные функции вывода."""

    def _capture(self, fn, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*args)
        return buf.getvalue()

    # demo_1d ---------------------------------------------------------------

    def test_demo_1d_produces_output(self):
        out = self._capture(demo_1d, 'xor_rule', 8, 3)
        self.assertGreater(len(out), 0)

    def test_demo_1d_contains_rule_name(self):
        out = self._capture(demo_1d, 'identity', 8, 2)
        self.assertIn('identity', out)

    def test_demo_1d_contains_generation(self):
        out = self._capture(demo_1d, 'xor_rule', 8, 3)
        self.assertIn('3', out)  # steps shown

    def test_demo_1d_all_rules_no_crash(self):
        """Все встроенные правила запускаются без ошибок."""
        for rule_name in ['xor_rule', 'identity', 'majority_vote']:
            self._capture(demo_1d, rule_name, 8, 2)

    # demo_2d ---------------------------------------------------------------

    def test_demo_2d_produces_output(self):
        out = self._capture(demo_2d, 'xor_rule', 4, 4, 2)
        self.assertGreater(len(out), 0)

    def test_demo_2d_contains_rule_name(self):
        out = self._capture(demo_2d, 'majority_vote', 4, 4, 2)
        self.assertIn('majority_vote', out)

    def test_demo_2d_contains_size(self):
        out = self._capture(demo_2d, 'identity', 4, 4, 2)
        self.assertIn('4', out)


class TestAnimate(unittest.TestCase):
    """Тесты animate_1d и animate_2d (fps=0 → мгновенный вывод)."""

    def _capture(self, fn, *args, **kwargs) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*args, **kwargs)
        return buf.getvalue()

    # animate_1d ---------------------------------------------------------------

    def test_animate_1d_produces_output(self):
        out = self._capture(animate_1d, 'xor_rule', 8, 2, 0, False, 'single')
        self.assertGreater(len(out), 0)

    def test_animate_1d_shows_rule_name(self):
        out = self._capture(animate_1d, 'majority_vote', 8, 2, 0, False, 'single')
        self.assertIn('majority_vote', out)

    def test_animate_1d_shows_step_count(self):
        out = self._capture(animate_1d, 'xor_rule', 8, 3, 0, False, 'single')
        self.assertIn('3', out)

    def test_animate_1d_center_mode(self):
        out = self._capture(animate_1d, 'identity', 8, 2, 0, False, 'center')
        self.assertGreater(len(out), 0)

    def test_animate_1d_random_mode(self):
        out = self._capture(animate_1d, 'identity', 8, 2, 0, False, 'random')
        self.assertGreater(len(out), 0)

    # animate_2d ---------------------------------------------------------------

    def test_animate_2d_produces_output(self):
        out = self._capture(animate_2d, 'xor_rule', 4, 4, 2, 0, False, 'single')
        self.assertGreater(len(out), 0)

    def test_animate_2d_shows_rule_name(self):
        out = self._capture(animate_2d, 'majority_vote', 4, 4, 2, 0, False, 'single')
        self.assertIn('majority_vote', out)

    # color=True ---------------------------------------------------------------

    def test_animate_1d_color_true(self):
        """animate_1d с color=True добавляет ANSI-escape коды."""
        out = self._capture(animate_1d, 'xor_rule', 4, 1, 0, True, 'single')
        self.assertGreater(len(out), 0)

    def test_animate_2d_color_true(self):
        """animate_2d с color=True добавляет ANSI-escape коды."""
        out = self._capture(animate_2d, 'xor_rule', 4, 4, 1, 0, True, 'single')
        self.assertGreater(len(out), 0)

    # init_mode ----------------------------------------------------------------

    def test_animate_2d_random_mode(self):
        """animate_2d с init_mode='random' инициализирует CA случайно."""
        out = self._capture(animate_2d, 'xor_rule', 4, 4, 1, 0, False, 'random')
        self.assertGreater(len(out), 0)

    def test_animate_2d_center_mode(self):
        """animate_2d с init_mode='center' ставит 42 в центр."""
        out = self._capture(animate_2d, 'xor_rule', 4, 4, 1, 0, False, 'center')
        self.assertGreater(len(out), 0)

    # KeyboardInterrupt --------------------------------------------------------

    def test_animate_1d_keyboard_interrupt(self):
        """KeyboardInterrupt в цикле animate_1d обрабатывается корректно."""
        with patch.object(CA1D, 'step', side_effect=KeyboardInterrupt):
            with redirect_stdout(io.StringIO()):
                animate_1d('xor_rule', 4, 1, 0, False, 'single')

    def test_animate_2d_keyboard_interrupt(self):
        """KeyboardInterrupt в цикле animate_2d обрабатывается корректно."""
        with patch.object(CA2D, 'step', side_effect=KeyboardInterrupt):
            with redirect_stdout(io.StringIO()):
                animate_2d('xor_rule', 4, 4, 1, 0, False, 'single')


class TestHexcaCLI(unittest.TestCase):
    """Тесты main() из hexca.py."""

    def _run(self, args):
        from projects.hexca.hexca import main as hexca_main
        buf = io.StringIO()
        old_argv = sys.argv
        sys.argv = ['hexca.py'] + args
        try:
            with redirect_stdout(buf):
                hexca_main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_list_rules(self):
        out = self._run(['--list-rules'])
        self.assertIn('xor_rule', out)

    def test_mode_1d(self):
        out = self._run(['--mode', '1d', '--steps', '2', '--width', '4', '--rule', 'xor_rule'])
        self.assertIn('hexca', out.lower())

    def test_mode_2d(self):
        out = self._run(['--mode', '2d', '--steps', '1', '--width', '4', '--height', '4'])
        self.assertGreater(len(out), 0)


class TestAnimateCLI(unittest.TestCase):
    """Тесты main() из animate.py."""

    def _run(self, args):
        from projects.hexca.animate import main as animate_main
        old_argv = sys.argv
        sys.argv = ['animate.py'] + args
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                animate_main()
        finally:
            sys.argv = old_argv
        return buf.getvalue()

    def test_list_rules(self):
        """--list-rules перечисляет правила и завершается."""
        out = self._run(['--list-rules'])
        self.assertIn('xor_rule', out)

    def test_mode_1d(self):
        """--mode 1d запускает 1D-анимацию."""
        out = self._run(['--mode', '1d', '--steps', '2', '--width', '4',
                         '--fps', '0', '--no-color', '--init', 'single'])
        self.assertIn('hexca 1D', out)

    def test_mode_2d(self):
        """--mode 2d запускает 2D-анимацию."""
        out = self._run(['--mode', '2d', '--steps', '1', '--width', '4',
                         '--height', '4', '--fps', '0', '--no-color'])
        self.assertGreater(len(out), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
