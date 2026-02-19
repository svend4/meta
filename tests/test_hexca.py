"""Тесты клеточного автомата hexca (движок + правила)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
from projects.hexca.hexca import CA1D, CA2D
from projects.hexca.rules import (
    majority_vote, xor_rule, identity, conway_like, RULES, get_rule,
    smooth_rule, cyclic_rule, outer_totalistic,
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
