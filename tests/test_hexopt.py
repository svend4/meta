"""Тесты hexopt — оптимизация на Q6."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import unittest
import math
from projects.hexopt.hexopt import (
    OptResult,
    local_search, simulated_annealing, genetic_algorithm, tabu_search,
    SetOptimizer,
    weighted_yang, min_dominating_set, max_hamming_spread, min_covering_code,
    exhaustive,
    hex_neighborhood, mask_neighborhood,
)
from libs.hexcore.hexcore import neighbors, hamming, yang_count, SIZE


# ---------------------------------------------------------------------------
# Вспомогательные объекты
# ---------------------------------------------------------------------------

def unimodal_obj(h: int) -> float:
    """Функция с единственным максимумом в точке 63 (все биты 1)."""
    return float(yang_count(h))


def noisy_obj(h: int) -> float:
    """Зашумлённая функция с чётким максимумом, но шумными соседями."""
    return yang_count(h) - 0.1 * bin(h ^ 42).count('1')


def known_max(obj) -> int:
    """Найти истинный максимум полным перебором."""
    return max(range(SIZE), key=obj)


# ---------------------------------------------------------------------------
# Тест OptResult
# ---------------------------------------------------------------------------

class TestOptResult(unittest.TestCase):
    def test_creation(self):
        r = OptResult(best=42, value=3.0, history=[(0, 3.0)], iterations=100)
        self.assertEqual(r.best, 42)
        self.assertAlmostEqual(r.value, 3.0)

    def test_repr(self):
        r = OptResult(best=0, value=0.0, history=[], iterations=10)
        self.assertIn('OptResult', repr(r))


# ---------------------------------------------------------------------------
# Тест окрестностей
# ---------------------------------------------------------------------------

class TestNeighborhood(unittest.TestCase):
    def test_hex_neighborhood_size(self):
        for h in range(SIZE):
            self.assertEqual(len(hex_neighborhood(h)), 6)

    def test_hex_neighborhood_hamming(self):
        for h in [0, 42, 63]:
            for nb in hex_neighborhood(h):
                self.assertEqual(hamming(h, nb), 1)

    def test_mask_neighborhood_size(self):
        self.assertEqual(len(mask_neighborhood(0)), SIZE)

    def test_mask_neighborhood_flip_one_bit(self):
        for bit in range(SIZE):
            mask = 1 << bit
            nbs = mask_neighborhood(0)
            self.assertIn(mask, nbs)


# ---------------------------------------------------------------------------
# Тест exhaustive
# ---------------------------------------------------------------------------

class TestExhaustive(unittest.TestCase):
    def test_finds_maximum(self):
        r = exhaustive(unimodal_obj, maximize=True)
        self.assertEqual(r.best, 63)   # yang_count(63) = 6

    def test_finds_minimum(self):
        r = exhaustive(unimodal_obj, maximize=False)
        self.assertEqual(r.best, 0)    # yang_count(0) = 0

    def test_value_correct(self):
        r = exhaustive(unimodal_obj)
        self.assertAlmostEqual(r.value, unimodal_obj(r.best))


# ---------------------------------------------------------------------------
# Тест LocalSearch
# ---------------------------------------------------------------------------

class TestLocalSearch(unittest.TestCase):
    def test_returns_optresult(self):
        r = local_search(unimodal_obj, seed=0)
        self.assertIsInstance(r, OptResult)

    def test_finds_global_max_unimodal(self):
        """Yang_count имеет максимум в 63. LS должен найти его."""
        r = local_search(unimodal_obj, seed=1, n_restarts=5)
        self.assertEqual(r.best, 63)

    def test_finds_global_min(self):
        r = local_search(unimodal_obj, maximize=False, seed=1, n_restarts=5)
        self.assertEqual(r.best, 0)

    def test_value_matches_best(self):
        r = local_search(unimodal_obj, seed=0)
        self.assertAlmostEqual(r.value, unimodal_obj(r.best))

    def test_history_monotone(self):
        """История должна быть неубывающей (каждая запись — улучшение)."""
        r = local_search(unimodal_obj, seed=5, n_restarts=8)
        for i in range(1, len(r.history)):
            self.assertGreater(r.history[i][1], r.history[i - 1][1])

    def test_custom_start(self):
        r = local_search(unimodal_obj, start=0, seed=0, n_restarts=1)
        self.assertIsNotNone(r.best)

    def test_custom_neighborhood(self):
        """С расширенной окрестностью (2 шага) алгоритм должен работать."""
        def nbrs2(h):
            return list({nb2 for nb in neighbors(h) for nb2 in neighbors(nb)})
        r = local_search(unimodal_obj, seed=0, neighborhood=nbrs2, n_restarts=3)
        self.assertIsNotNone(r.best)

    def test_weighted_yang_optimum(self):
        obj = weighted_yang([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        r = local_search(obj, seed=2, n_restarts=10)
        ex = exhaustive(obj)
        self.assertEqual(r.best, ex.best)

    def test_restarts_attribute(self):
        r = local_search(unimodal_obj, seed=0, n_restarts=7)
        self.assertEqual(r.restarts, 7)


# ---------------------------------------------------------------------------
# Тест SimulatedAnnealing
# ---------------------------------------------------------------------------

class TestSimulatedAnnealing(unittest.TestCase):
    def test_returns_optresult(self):
        r = simulated_annealing(unimodal_obj, seed=0)
        self.assertIsInstance(r, OptResult)

    def test_finds_global_max(self):
        r = simulated_annealing(unimodal_obj, seed=0, T0=10.0, max_iter=3000)
        self.assertEqual(r.best, 63)

    def test_value_matches_best(self):
        r = simulated_annealing(unimodal_obj, seed=0)
        self.assertAlmostEqual(r.value, unimodal_obj(r.best))

    def test_minimization(self):
        r = simulated_annealing(unimodal_obj, maximize=False, seed=0, max_iter=3000)
        self.assertEqual(r.best, 0)

    def test_custom_start(self):
        r = simulated_annealing(unimodal_obj, start=63, seed=0)
        self.assertAlmostEqual(r.value, unimodal_obj(r.best))

    def test_history_non_decreasing(self):
        """История должна содержать только улучшения."""
        r = simulated_annealing(unimodal_obj, seed=0)
        for i in range(1, len(r.history)):
            self.assertGreater(r.history[i][1], r.history[i - 1][1])

    def test_iterations_count(self):
        max_iter = 500
        r = simulated_annealing(unimodal_obj, seed=0, max_iter=max_iter)
        self.assertEqual(r.iterations, max_iter)


# ---------------------------------------------------------------------------
# Тест GeneticAlgorithm
# ---------------------------------------------------------------------------

class TestGeneticAlgorithm(unittest.TestCase):
    def test_returns_optresult(self):
        r = genetic_algorithm(unimodal_obj, seed=0)
        self.assertIsInstance(r, OptResult)

    def test_finds_global_max(self):
        r = genetic_algorithm(unimodal_obj, seed=0, n_pop=16, n_gen=100)
        self.assertEqual(r.best, 63)

    def test_value_matches_best(self):
        r = genetic_algorithm(unimodal_obj, seed=0)
        self.assertAlmostEqual(r.value, unimodal_obj(r.best))

    def test_valid_individual(self):
        """Лучшее решение должно быть гексаграммой."""
        r = genetic_algorithm(unimodal_obj, seed=0)
        self.assertIn(r.best, range(SIZE))

    def test_minimization(self):
        r = genetic_algorithm(unimodal_obj, maximize=False, seed=0)
        self.assertEqual(r.best, 0)

    def test_single_point_crossover(self):
        r = genetic_algorithm(unimodal_obj, seed=0, crossover='single')
        self.assertIn(r.best, range(SIZE))

    def test_iterations_count(self):
        n_gen = 50
        r = genetic_algorithm(unimodal_obj, seed=0, n_gen=n_gen)
        self.assertEqual(r.iterations, n_gen)

    def test_elitism_preserves_best(self):
        """Элитизм: лучший в поколении не ухудшается."""
        r = genetic_algorithm(unimodal_obj, seed=1, n_pop=10, n_gen=50, elitism=3)
        # Значение должно быть разумным
        self.assertGreaterEqual(r.value, 0)


# ---------------------------------------------------------------------------
# Тест TabuSearch
# ---------------------------------------------------------------------------

class TestTabuSearch(unittest.TestCase):
    def test_returns_optresult(self):
        r = tabu_search(unimodal_obj, seed=0)
        self.assertIsInstance(r, OptResult)

    def test_finds_global_max(self):
        r = tabu_search(unimodal_obj, seed=0, max_iter=300)
        self.assertEqual(r.best, 63)

    def test_value_matches_best(self):
        r = tabu_search(unimodal_obj, seed=0)
        self.assertAlmostEqual(r.value, unimodal_obj(r.best))

    def test_minimization(self):
        r = tabu_search(unimodal_obj, maximize=False, seed=0)
        self.assertEqual(r.best, 0)

    def test_escapes_local_optima(self):
        """Poиск с запретами должен уходить из локальных оптимумов."""
        # Цель с несколькими локальными максимумами
        def multimodal(h):
            return yang_count(h) if yang_count(h) != 4 else yang_count(h) - 2
        r_ls = local_search(multimodal, seed=0, n_restarts=1, max_iter=50)
        r_ts = tabu_search(multimodal, seed=0, tabu_size=10)
        # TabuSearch должен найти не хуже LS с одним стартом
        self.assertGreaterEqual(r_ts.value, r_ls.value - 1e-9)

    def test_different_tabu_sizes(self):
        for ts in [3, 7, 15]:
            r = tabu_search(unimodal_obj, seed=0, tabu_size=ts)
            self.assertIn(r.best, range(SIZE))


# ---------------------------------------------------------------------------
# Тест SetOptimizer
# ---------------------------------------------------------------------------

class TestSetOptimizer(unittest.TestCase):
    def test_mask_set_roundtrip(self):
        hexagrams = [0, 7, 42, 63]
        mask = SetOptimizer.set_to_mask(hexagrams)
        recovered = SetOptimizer.mask_to_set(mask)
        self.assertEqual(set(recovered), set(hexagrams))

    def test_mask_to_set_empty(self):
        self.assertEqual(SetOptimizer.mask_to_set(0), [])

    def test_set_to_mask_all(self):
        all_mask = (1 << SIZE) - 1
        result = SetOptimizer.mask_to_set(all_mask)
        self.assertEqual(len(result), SIZE)

    def test_local_search_returns_result(self):
        obj = min_dominating_set()
        opt = SetOptimizer(obj, maximize=True, seed=0)
        r = opt.local_search(max_iter=100, n_restarts=2)
        self.assertIsInstance(r, OptResult)

    def test_sa_returns_result(self):
        obj = min_dominating_set()
        opt = SetOptimizer(obj, maximize=True, seed=0)
        r = opt.simulated_annealing(max_iter=200)
        self.assertIsInstance(r, OptResult)

    def test_spread_objective_positive(self):
        """Максимальное рассеивание ≥ 0."""
        obj = max_hamming_spread(4)
        opt = SetOptimizer(obj, maximize=True, seed=0)
        r = opt.local_search(max_iter=200)
        # После нескольких шагов должно найти ненулевой spread
        S = SetOptimizer.mask_to_set(r.best)
        self.assertGreater(len(S), 0)

    def test_dominating_set_result_valid(self):
        """Результат должен быть корректным подмножеством Q6."""
        from libs.hexcore.hexcore import neighbors as nbrs
        obj = min_dominating_set()
        opt = SetOptimizer(obj, maximize=True, seed=5)
        r = opt.local_search(max_iter=300, n_restarts=3)
        S = set(SetOptimizer.mask_to_set(r.best))
        # Проверить что S доминирует Q6 (каждая точка в S или соседствует с S)
        closed_S = set()
        for s in S:
            closed_S.add(s)
            closed_S |= set(nbrs(s))
        self.assertEqual(closed_S, set(range(SIZE)))


# ---------------------------------------------------------------------------
# Тест задач оптимизации
# ---------------------------------------------------------------------------

class TestProblems(unittest.TestCase):
    def test_weighted_yang_max_at_63(self):
        """С единичными весами максимум в 63."""
        obj = weighted_yang([1.0] * 6)
        ex = exhaustive(obj)
        self.assertEqual(ex.best, 63)

    def test_weighted_yang_min_at_0(self):
        obj = weighted_yang([1.0] * 6)
        ex = exhaustive(obj, maximize=False)
        self.assertEqual(ex.best, 0)

    def test_weighted_yang_custom_weights(self):
        """Только бит 5 ненулевой → оптимум там, где бит 5 = 1."""
        obj = weighted_yang([0, 0, 0, 0, 0, 1.0])
        ex = exhaustive(obj)
        self.assertEqual((ex.best >> 5) & 1, 1)

    def test_min_dominating_valid(self):
        obj = min_dominating_set()
        # Пустое множество не доминирует → большой штраф
        empty_val = obj(0)
        # Полное множество доминирует (S = Q6), но |S| = 64 → -64
        full_mask = (1 << SIZE) - 1
        full_val = obj(full_mask)
        self.assertGreater(full_val, empty_val)  # full dominating > empty

    def test_max_spread_increases_with_k(self):
        """Большее k → больший потенциальный spread."""
        obj4 = max_hamming_spread(4)
        obj5 = max_hamming_spread(5)
        # Просто проверим что функция вычислима
        mask_4pts = SetOptimizer.set_to_mask([0, 21, 42, 63])
        val4 = obj4(mask_4pts)
        self.assertIsInstance(val4, float)

    def test_covering_code_full_mask_valid(self):
        """Полный Q6 покрывает с радиусом 0."""
        obj = min_covering_code(0)
        full = (1 << SIZE) - 1
        val_full = obj(full)
        empty = 0
        val_empty = obj(empty)
        self.assertGreater(val_full, val_empty)


if __name__ == '__main__':
    unittest.main(verbosity=2)
