"""
hexopt — оптимизация на пространстве Q6

Q6 = (Z₂)⁶ — дискретное метрическое пространство 64 гексаграмм.
Два вида задач оптимизации:

  [Одна точка]  h* = argmax_{h ∈ Q6} f(h)
    Окрестность h: 6 соседей по рёбрам Q6 (Хэмминг-расстояние 1).

  [Подмножество] S* = argmin/argmax_{S ⊆ Q6} g(S)
    Представление: 64-битная маска (бит i = 1, если гексаграмма i ∈ S).
    Окрестность S: добавить или убрать одну гексаграмму (flip одного бита).

Алгоритмы:
  local_search        — подъём по холму + случайные перезапуски
  simulated_annealing — метод имитации отжига (критерий Метрополиса)
  genetic_algorithm   — ГА с однородным кроссовером и мутацией
  tabu_search         — поиск с запретами (запрет недавних состояний)

Задачи на Q6:
  weighted_yang(w)      — h: максимизировать взвешенный счёт ян
  min_dominating_set()  — S: минимальное доминирующее множество
  max_hamming_spread(k) — S: k гексаграмм с максимальным попарным расстоянием
  min_covering_code(t)  — S: минимальный код с покрывающим радиусом ≤ t
"""

from __future__ import annotations
import sys
import random
import math
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import neighbors, hamming, yang_count, ball, SIZE


# ---------------------------------------------------------------------------
# Результат оптимизации
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    """Результат запуска алгоритма оптимизации."""
    best: object                        # лучшее найденное решение
    value: float                        # значение цели в лучшей точке
    history: list[tuple[int, float]]    # [(iteration, best_value), ...]
    iterations: int                     # итераций всего
    restarts: int = 0                   # перезапусков (для LS)

    def __repr__(self) -> str:
        return (f"OptResult(best={self.best}, value={self.value:.4f}, "
                f"iters={self.iterations})")


# ---------------------------------------------------------------------------
# Окрестности
# ---------------------------------------------------------------------------

def hex_neighborhood(h: int) -> list[int]:
    """6 соседей гексаграммы h в Q6."""
    return neighbors(h)


def mask_neighborhood(mask: int) -> list[int]:
    """
    Окрестность подмножества (задаётся 64-битной маской):
    все маски, отличающиеся от mask ровно в 1 бите.
    """
    return [mask ^ (1 << i) for i in range(SIZE)]


# ---------------------------------------------------------------------------
# Подъём по холму с перезапусками
# ---------------------------------------------------------------------------

def local_search(
    objective: Callable,
    start: object | None = None,
    maximize: bool = True,
    max_iter: int = 300,
    n_restarts: int = 10,
    neighborhood: Callable | None = None,
    seed: int | None = None,
) -> OptResult:
    """
    Подъём по холму (steepest ascent) с n_restarts случайными перезапусками.

    objective   : функция оценки решения → float
    start       : начальная точка (None → случайная)
    maximize    : True — максимизация, False — минимизация
    max_iter    : итераций на один запуск
    n_restarts  : число перезапусков
    neighborhood: функция → список соседей (None → hex_neighborhood)
    """
    rng = random.Random(seed)
    nbrs = neighborhood or hex_neighborhood
    sign = 1 if maximize else -1

    # Определить тип точки автоматически
    def random_start():
        return rng.randrange(SIZE)

    best_global = None
    best_val_global = -math.inf if maximize else math.inf
    history: list[tuple[int, float]] = []
    total_iters = 0

    for restart in range(n_restarts):
        current = (start if restart == 0 and start is not None
                   else random_start())
        current_val = objective(current)

        for it in range(max_iter):
            total_iters += 1
            nb_list = nbrs(current)
            if maximize:
                best_nb = max(nb_list, key=objective)
                best_nb_val = objective(best_nb)
            else:
                best_nb = min(nb_list, key=objective)
                best_nb_val = objective(best_nb)

            if sign * best_nb_val > sign * current_val:
                current = best_nb
                current_val = best_nb_val
            else:
                break  # локальный оптимум

        if sign * current_val > sign * best_val_global:
            best_val_global = current_val
            best_global = current
            history.append((total_iters, best_val_global))

    return OptResult(
        best=best_global,
        value=best_val_global,
        history=history,
        iterations=total_iters,
        restarts=n_restarts,
    )


# ---------------------------------------------------------------------------
# Имитация отжига
# ---------------------------------------------------------------------------

def simulated_annealing(
    objective: Callable,
    start: object | None = None,
    maximize: bool = True,
    T0: float = 8.0,
    alpha: float = 0.95,
    T_min: float = 0.01,
    max_iter: int = 2000,
    neighborhood: Callable | None = None,
    seed: int | None = None,
) -> OptResult:
    """
    Метод имитации отжига.

    T0    : начальная температура
    alpha : коэффициент охлаждения (T ← T * alpha каждые SIZE шагов)
    T_min : минимальная температура (условие остановки)
    """
    rng = random.Random(seed)
    nbrs = neighborhood or hex_neighborhood
    sign = 1 if maximize else -1

    current = start if start is not None else rng.randrange(SIZE)
    current_val = objective(current)
    best = current
    best_val = current_val
    history: list[tuple[int, float]] = [(0, best_val)]

    T = T0
    for it in range(1, max_iter + 1):
        nb = rng.choice(nbrs(current))
        nb_val = objective(nb)
        delta = sign * (nb_val - current_val)

        if delta > 0:
            # Принять улучшение
            current = nb
            current_val = nb_val
        else:
            # Принять ухудшение с вероятностью exp(delta / T)
            if T > T_min and rng.random() < math.exp(delta / T):
                current = nb
                current_val = nb_val

        if sign * current_val > sign * best_val:
            best = current
            best_val = current_val
            history.append((it, best_val))

        # Охлаждение каждые SIZE шагов
        if it % SIZE == 0:
            T = max(T * alpha, T_min)

    return OptResult(
        best=best,
        value=best_val,
        history=history,
        iterations=max_iter,
    )


# ---------------------------------------------------------------------------
# Генетический алгоритм
# ---------------------------------------------------------------------------

def genetic_algorithm(
    fitness: Callable,
    n_pop: int = 20,
    n_gen: int = 150,
    mutation_rate: float = 1.0 / 6,
    elitism: int = 2,
    maximize: bool = True,
    crossover: str = 'uniform',
    seed: int | None = None,
) -> OptResult:
    """
    Генетический алгоритм для оптимизации на Q6 (одна гексаграмма).

    Кодирование: гексаграмма как 6-битный геном.
    Мутация    : переворот каждого бита с вероятностью mutation_rate.
    Кроссовер  : 'uniform' — равномерный, 'single' — одноточечный.
    Выборка    : турнирная (размер турнира 3).
    Элитизм    : top-elitism особей переходят без изменений.
    """
    rng = random.Random(seed)
    sign = 1 if maximize else -1

    # Начальная популяция
    population = [rng.randrange(SIZE) for _ in range(n_pop)]

    def tournament(pop: list[int], k: int = 3) -> int:
        candidates = rng.sample(pop, min(k, len(pop)))
        return max(candidates, key=lambda h: sign * fitness(h))

    def crossover_op(p1: int, p2: int) -> int:
        if crossover == 'uniform':
            child = 0
            for bit in range(6):
                if rng.random() < 0.5:
                    child |= (p1 & (1 << bit))
                else:
                    child |= (p2 & (1 << bit))
            return child
        else:  # single-point
            point = rng.randint(1, 5)
            lo_mask = (1 << point) - 1
            return (p1 & lo_mask) | (p2 & ~lo_mask & 63)

    def mutate(h: int) -> int:
        for bit in range(6):
            if rng.random() < mutation_rate:
                h ^= (1 << bit)
        return h

    best = max(population, key=lambda h: sign * fitness(h))
    best_val = fitness(best)
    history: list[tuple[int, float]] = [(0, best_val)]

    for gen in range(1, n_gen + 1):
        scored = sorted(population, key=lambda h: sign * fitness(h), reverse=True)

        # Следующее поколение: сохранить элитных
        next_pop = scored[:elitism]

        # Порождение остальных
        while len(next_pop) < n_pop:
            p1 = tournament(population)
            p2 = tournament(population)
            child = crossover_op(p1, p2)
            child = mutate(child)
            next_pop.append(child)

        population = next_pop

        gen_best = max(population, key=lambda h: sign * fitness(h))
        gen_val = fitness(gen_best)
        if sign * gen_val > sign * best_val:
            best = gen_best
            best_val = gen_val
            history.append((gen, best_val))

    return OptResult(
        best=best,
        value=best_val,
        history=history,
        iterations=n_gen,
    )


# ---------------------------------------------------------------------------
# Поиск с запретами
# ---------------------------------------------------------------------------

def tabu_search(
    objective: Callable,
    start: object | None = None,
    maximize: bool = True,
    tabu_size: int = 8,
    max_iter: int = 300,
    neighborhood: Callable | None = None,
    seed: int | None = None,
) -> OptResult:
    """
    Поиск с запретами.

    tabu_size : длина очереди запрещённых решений
    Запрет    : посещённое решение нельзя повторить в течение tabu_size шагов.
    Аспирация : запрет снимается, если решение лучше глобального оптимума.
    """
    rng = random.Random(seed)
    nbrs = neighborhood or hex_neighborhood
    sign = 1 if maximize else -1

    current = start if start is not None else rng.randrange(SIZE)
    best = current
    best_val = objective(current)
    tabu_queue: list[object] = [current]
    history: list[tuple[int, float]] = [(0, best_val)]

    for it in range(1, max_iter + 1):
        nb_list = nbrs(current)
        # Найти лучший незапрещённый сосед (или разрешённый аспирацией)
        best_nb = None
        best_nb_val = -math.inf * sign

        for nb in nb_list:
            nb_val = objective(nb)
            # Проверить аспирацию: разрешить, если лучше глобального
            if nb in tabu_queue and sign * nb_val <= sign * best_val:
                continue
            if sign * nb_val > sign * best_nb_val:
                best_nb = nb
                best_nb_val = nb_val

        if best_nb is None:
            # Все соседи запрещены — взять лучшего из запрещённых
            best_nb = max(nb_list, key=lambda h: sign * objective(h))
            best_nb_val = objective(best_nb)

        current = best_nb
        tabu_queue.append(current)
        if len(tabu_queue) > tabu_size:
            tabu_queue.pop(0)

        if sign * best_nb_val > sign * best_val:
            best = current
            best_val = best_nb_val
            history.append((it, best_val))

    return OptResult(
        best=best,
        value=best_val,
        history=history,
        iterations=max_iter,
    )


# ---------------------------------------------------------------------------
# Оптимизация подмножеств Q6 (маска 64 бита)
# ---------------------------------------------------------------------------

class SetOptimizer:
    """
    Оптимизация над подмножествами Q6 (0/1-вектор как 64-битная маска).

    Каждый бит i маски = индикатор i-й гексаграммы в наборе S.
    Окрестность: добавить/убрать одну гексаграмму.

    Поддерживает SA и локальный поиск.
    """

    def __init__(
        self,
        objective: Callable[[int], float],
        maximize: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        objective(mask) → float
        mask: 64-битное целое (бит i = 1 iff i ∈ S)
        """
        self._obj = objective
        self._sign = 1 if maximize else -1
        self._rng = random.Random(seed)

    def _neighbors(self, mask: int) -> list[int]:
        return [mask ^ (1 << i) for i in range(SIZE)]

    def _random_mask(self, density: float = 0.1) -> int:
        """Случайная маска с примерно SIZE*density единицами."""
        mask = 0
        for i in range(SIZE):
            if self._rng.random() < density:
                mask |= (1 << i)
        return mask

    def local_search(
        self,
        start: int | None = None,
        max_iter: int = 500,
        n_restarts: int = 5,
    ) -> OptResult:
        sign = self._sign

        def random_start():
            return self._random_mask()

        best_global = None
        best_val_global = -math.inf if self._sign > 0 else math.inf
        history: list[tuple[int, float]] = []
        total_iters = 0

        for restart in range(n_restarts):
            current = start if restart == 0 and start is not None else random_start()
            current_val = self._obj(current)

            for it in range(max_iter):
                total_iters += 1
                nbrs = self._neighbors(current)
                if sign > 0:
                    best_nb = max(nbrs, key=self._obj)
                else:
                    best_nb = min(nbrs, key=self._obj)
                best_nb_val = self._obj(best_nb)

                if sign * best_nb_val > sign * current_val:
                    current = best_nb
                    current_val = best_nb_val
                else:
                    break

            if sign * current_val > sign * best_val_global:
                best_val_global = current_val
                best_global = current
                history.append((total_iters, best_val_global))

        return OptResult(
            best=best_global,
            value=best_val_global,
            history=history,
            iterations=total_iters,
            restarts=n_restarts,
        )

    def simulated_annealing(
        self,
        start: int | None = None,
        T0: float = 5.0,
        alpha: float = 0.97,
        T_min: float = 0.01,
        max_iter: int = 3000,
    ) -> OptResult:
        sign = self._sign
        current = start if start is not None else self._random_mask()
        current_val = self._obj(current)
        best = current
        best_val = current_val
        history: list[tuple[int, float]] = [(0, best_val)]

        T = T0
        for it in range(1, max_iter + 1):
            # Выбрать случайного соседа (flip один бит)
            bit = self._rng.randrange(SIZE)
            nb = current ^ (1 << bit)
            nb_val = self._obj(nb)
            delta = sign * (nb_val - current_val)

            if delta > 0 or (T > T_min and self._rng.random() < math.exp(delta / T)):
                current = nb
                current_val = nb_val

            if sign * current_val > sign * best_val:
                best = current
                best_val = current_val
                history.append((it, best_val))

            if it % 100 == 0:
                T = max(T * alpha, T_min)

        return OptResult(
            best=best,
            value=best_val,
            history=history,
            iterations=max_iter,
        )

    @staticmethod
    def mask_to_set(mask: int) -> list[int]:
        """Маска → список гексаграмм."""
        return [i for i in range(SIZE) if (mask >> i) & 1]

    @staticmethod
    def set_to_mask(hexagrams: list[int]) -> int:
        """Список гексаграмм → маска."""
        mask = 0
        for h in hexagrams:
            mask |= (1 << h)
        return mask


# ---------------------------------------------------------------------------
# Стандартные задачи на Q6
# ---------------------------------------------------------------------------

def weighted_yang(weights: list[float] | None = None) -> Callable[[int], float]:
    """
    Задача: найти h с максимальным взвешенным числом ян-линий.
    weights[i] = вес i-й линии (бита i), i = 0..5.
    По умолчанию — единичные веса (= yang_count).
    """
    w = weights or [1.0] * 6

    def objective(h: int) -> float:
        return sum(w[i] * ((h >> i) & 1) for i in range(6))

    return objective


def min_dominating_set() -> Callable[[int], float]:
    """
    Задача: найти наименьшее доминирующее множество Q6.
    Доминирующее множество S: ∀h ∈ Q6: h ∈ S или ∃s ∈ S: d(h,s) = 1.

    Для подмножеств (маска): минимизировать штраф = |S| + λ × недоминированных.
    λ = SIZE + 1 — большой штраф за недоминированные вершины.
    """
    PENALTY = SIZE + 1

    # Предвычислить закрытые окрестности
    closed_nbrs = [frozenset([h] + neighbors(h)) for h in range(SIZE)]

    def objective(mask: int) -> float:
        S = [i for i in range(SIZE) if (mask >> i) & 1]
        size = len(S)
        # Покрытые вершины
        covered = set()
        for s in S:
            covered |= closed_nbrs[s]
        dominated = len(covered)
        not_dominated = SIZE - dominated
        return -(size + PENALTY * not_dominated)

    return objective


def max_hamming_spread(k: int) -> Callable[[int], float]:
    """
    Задача: найти k гексаграмм с максимальным суммарным попарным расстоянием.
    Для подмножеств (маска): максимизировать Σ d(u,v) для u,v ∈ S, |S|=k.
    Штраф за |S| ≠ k.
    """
    PENALTY = SIZE * 10

    def objective(mask: int) -> float:
        S = [i for i in range(SIZE) if (mask >> i) & 1]
        size_penalty = PENALTY * abs(len(S) - k)
        if len(S) < 2:
            return -size_penalty
        spread = sum(hamming(S[i], S[j])
                     for i in range(len(S)) for j in range(i + 1, len(S)))
        return float(spread) - size_penalty

    return objective


def min_covering_code(t: int) -> Callable[[int], float]:
    """
    Задача: найти наименьший код с покрывающим радиусом ≤ t.
    Для подмножеств (маска): минимизировать |S| при условии полного покрытия.
    Штраф за непокрытые вершины.
    """
    PENALTY = SIZE + 1

    # Предвычислить шары радиуса t для каждой точки
    balls_t = [frozenset(ball(h, t)) for h in range(SIZE)]

    def objective(mask: int) -> float:
        S = [i for i in range(SIZE) if (mask >> i) & 1]
        size = len(S)
        covered = set()
        for s in S:
            covered |= balls_t[s]
        not_covered = SIZE - len(covered)
        return -(size + PENALTY * not_covered)

    return objective


# ---------------------------------------------------------------------------
# Проверка достижимости глобального оптимума (для тестов)
# ---------------------------------------------------------------------------

def exhaustive(objective: Callable[[int], float], maximize: bool = True) -> OptResult:
    """Полный перебор Q6 (только для функций от одной гексаграммы)."""
    if maximize:
        best = max(range(SIZE), key=objective)
    else:
        best = min(range(SIZE), key=objective)
    return OptResult(
        best=best,
        value=objective(best),
        history=[(0, objective(best))],
        iterations=SIZE,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='hexopt — оптимизация на Q6')
    sub = parser.add_subparsers(dest='cmd')

    p_hex = sub.add_parser('hexagram', help='Оптимизация одной гексаграммы')
    p_hex.add_argument('algo', choices=['ls', 'sa', 'ga', 'tabu', 'all'])
    p_hex.add_argument('--weights', nargs=6, type=float,
                        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        help='Веса битов 0..5')
    p_hex.add_argument('--seed', type=int, default=42)

    p_set = sub.add_parser('subset', help='Оптимизация подмножества Q6')
    p_set.add_argument('problem', choices=['dominating', 'spread', 'covering'])
    p_set.add_argument('algo', choices=['ls', 'sa'])
    p_set.add_argument('--k', type=int, default=4,
                        help='Параметр задачи (k для spread, t для covering)')
    p_set.add_argument('--seed', type=int, default=42)

    p_cmp = sub.add_parser('compare', help='Сравнить алгоритмы')
    p_cmp.add_argument('--trials', type=int, default=10)
    p_cmp.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    if args.cmd == 'hexagram':
        obj = weighted_yang(args.weights)
        ex = exhaustive(obj)
        print(f"Оптимум (полный перебор): h={ex.best}, f={ex.value:.2f}")
        algos = {
            'ls': lambda: local_search(obj, seed=args.seed, n_restarts=8),
            'sa': lambda: simulated_annealing(obj, seed=args.seed),
            'ga': lambda: genetic_algorithm(obj, seed=args.seed),
            'tabu': lambda: tabu_search(obj, seed=args.seed),
        }
        run = list(algos.items()) if args.algo == 'all' else [(args.algo, algos[args.algo])]
        for name, fn in run:
            r = fn()
            status = '✓' if r.best == ex.best else '✗'
            print(f"  [{name:4s}] best={r.best:2d}  f={r.value:.2f}  "
                  f"iters={r.iterations:5d}  улучшений={len(r.history)}  {status}")

    elif args.cmd == 'subset':
        if args.problem == 'dominating':
            obj = min_dominating_set()
            label = 'Минимальное доминирующее множество'
        elif args.problem == 'spread':
            obj = max_hamming_spread(args.k)
            label = f'Максимальное рассеивание ({args.k} точек)'
        else:
            obj = min_covering_code(args.k)
            label = f'Минимальный код с покрывающим радиусом ≤ {args.k}'

        opt = SetOptimizer(obj, maximize=(args.problem != 'covering'), seed=args.seed)
        fn = opt.local_search if args.algo == 'ls' else opt.simulated_annealing
        result = fn()

        S = SetOptimizer.mask_to_set(result.best)
        print(f"{label}:")
        print(f"  Алгоритм: {'local_search' if args.algo == 'ls' else 'SA'}")
        print(f"  |S| = {len(S)}")
        print(f"  S   = {S}")
        print(f"  f   = {result.value:.2f}")
        print(f"  Итераций: {result.iterations}")

    elif args.cmd == 'compare':
        print(f"Сравнение алгоритмов (trials={args.trials}):")
        w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        obj = weighted_yang(w)
        ex = exhaustive(obj)
        print(f"  Оптимум: {ex.best} (f={ex.value:.2f})\n")

        algos = {
            'LocalSearch   ': lambda s: local_search(obj, seed=s, n_restarts=5, max_iter=100),
            'Sim.Annealing ': lambda s: simulated_annealing(obj, seed=s, max_iter=500),
            'GeneticAlg.   ': lambda s: genetic_algorithm(obj, seed=s, n_pop=16, n_gen=80),
            'TabuSearch    ': lambda s: tabu_search(obj, seed=s, max_iter=200),
        }
        for name, fn in algos.items():
            successes = sum(
                1 for t in range(args.trials) if fn(args.seed + t).best == ex.best
            )
            print(f"  {name}: {successes}/{args.trials} оптимумов")

    else:
        parser.print_help()
