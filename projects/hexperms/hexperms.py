"""hexperms.py — Алгоритм нахождения перестановок (Герман).

Источник: PDF «Алгоритм нахождения перестановок» Франца Германа.

Возможности:
  - Генерация всех перестановок в лексикографическом порядке (без рекурсии)
  - Следующая/предыдущая перестановка (алгоритм Германа)
  - Unranking: получить k-ю перестановку по номеру (факторадическая система)
  - Ranking: номер данной перестановки
  - Перестановки с ограничениями: фиксированная точка, беспорядки (derangements)
  - Применение к Q6: 720 = 6! перестановок позиций гексаграммы
"""
import math
import sys
import argparse
from typing import Iterator

# ── факторадическая система ───────────────────────────────────────────────────

def _factoradic(k: int, n: int) -> list[int]:
    """Разложить k в факторадической системе счисления длиной n.

    Возвращает список цифр [d_{n-1}, ..., d_1, d_0], где d_i ∈ [0, i].
    """
    digits = []
    for i in range(1, n + 1):
        digits.append(k % i)
        k //= i
    return list(reversed(digits))


def _factoradic_to_int(digits: list[int]) -> int:
    """Обратное преобразование: список факторадических цифр → целое."""
    result = 0
    for i, d in enumerate(reversed(digits)):
        result += d * math.factorial(i)
    return result


# ── основной класс ────────────────────────────────────────────────────────────

class PermutationEngine:
    """Генератор и анализатор перестановок {1, 2, …, n}."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError(f"n должно быть ≥ 1, получено {n}")
        self.n = n
        self._factorial = math.factorial(n)

    def __repr__(self):
        return f"PermutationEngine(n={self.n})"

    # ── базовые операции с перестановками ────────────────────────────────────

    def next_perm(self, perm: list[int]) -> list[int] | None:
        """Следующая перестановка в лексикографическом порядке.

        Возвращает None, если perm — максимальная перестановка.
        """
        p = list(perm)
        n = len(p)
        # 1. Найти наибольший i, где p[i] < p[i+1]
        i = n - 2
        while i >= 0 and p[i] >= p[i + 1]:
            i -= 1
        if i < 0:
            return None   # последняя перестановка
        # 2. Найти наибольший j, где p[i] < p[j]
        j = n - 1
        while p[j] <= p[i]:
            j -= 1
        # 3. Поменять p[i] и p[j]
        p[i], p[j] = p[j], p[i]
        # 4. Обратить суффикс p[i+1:]
        p[i + 1:] = reversed(p[i + 1:])
        return p

    def prev_perm(self, perm: list[int]) -> list[int] | None:
        """Предыдущая перестановка в лексикографическом порядке.

        Возвращает None, если perm — минимальная перестановка.
        """
        p = list(perm)
        n = len(p)
        i = n - 2
        while i >= 0 and p[i] <= p[i + 1]:
            i -= 1
        if i < 0:
            return None
        j = n - 1
        while p[j] >= p[i]:
            j -= 1
        p[i], p[j] = p[j], p[i]
        p[i + 1:] = reversed(p[i + 1:])
        return p

    # ── генерация ─────────────────────────────────────────────────────────────

    def generate_all(self) -> Iterator[list[int]]:
        """Генератор всех n! перестановок в лексикографическом порядке."""
        p = list(range(1, self.n + 1))
        while p is not None:
            yield list(p)
            p = self.next_perm(p)

    # ── unranking / ranking ───────────────────────────────────────────────────

    def unrank(self, k: int) -> list[int]:
        """Получить k-ю перестановку (0-индексированную) по факторадической системе.

        unrank(0)         → [1, 2, …, n]  (минимальная)
        unrank(n!-1)      → [n, n-1, …, 1]  (максимальная)
        """
        if not 0 <= k < self._factorial:
            raise ValueError(f"k должен быть в [0, {self._factorial - 1}]")
        digits = _factoradic(k, self.n)
        pool = list(range(1, self.n + 1))
        result = []
        for d in digits:
            result.append(pool.pop(d))
        return result

    def rank(self, perm: list[int]) -> int:
        """Номер перестановки perm в лексикографическом порядке (0-индексированный)."""
        if sorted(perm) != list(range(1, self.n + 1)):
            raise ValueError("perm должна быть перестановкой {1, …, n}")
        p = list(perm)
        result = 0
        available = list(range(1, self.n + 1))
        for i, elem in enumerate(p):
            idx = available.index(elem)
            result += idx * math.factorial(self.n - 1 - i)
            available.pop(idx)
        return result

    # ── специальные подмножества ──────────────────────────────────────────────

    def with_fixed_point(self, point: int) -> list[list[int]]:
        """Все перестановки, где p[point-1] = point (фиксируем элемент point).

        Это аналог группы-винта Bₙ при point=1.
        """
        if not 1 <= point <= self.n:
            raise ValueError(f"point должен быть в [1, {self.n}]")
        result = []
        for p in self.generate_all():
            if p[point - 1] == point:
                result.append(p)
        return result

    def derangements(self) -> list[list[int]]:
        """Все беспорядки (деранжементы): перестановки без неподвижных точек."""
        result = []
        for p in self.generate_all():
            if all(p[i] != i + 1 for i in range(self.n)):
                result.append(p)
        return result

    def derangement_count(self) -> int:
        """Число беспорядков: D(n) = n! * Σ(-1)^k/k!, k=0..n."""
        total = 0
        sign = 1
        fact_k = 1
        for k in range(self.n + 1):
            if k > 0:
                fact_k *= k
            total += sign * self._factorial // fact_k
            sign = -sign
        return total

    # ── применение к Q6 ───────────────────────────────────────────────────────

    def generate_aut_q6(self) -> list[list[int]]:
        """Все 720 перестановок позиций {1..6} = Aut(Q6) как перестановочная группа."""
        if self.n != 6:
            raise ValueError("generate_aut_q6 только для n=6")
        return list(self.generate_all())

    # ── бенчмарк ─────────────────────────────────────────────────────────────

    def benchmark(self, n: int | None = None) -> str:
        """Сравнить скорость генерации с itertools.permutations."""
        import time
        import itertools

        target_n = n or self.n
        pe = PermutationEngine(target_n)

        t0 = time.perf_counter()
        cnt_own = sum(1 for _ in pe.generate_all())
        t1 = time.perf_counter()
        cnt_iter = sum(1 for _ in itertools.permutations(range(1, target_n + 1)))
        t2 = time.perf_counter()

        own_ms = (t1 - t0) * 1000
        iter_ms = (t2 - t1) * 1000
        return (f"n={target_n}, {cnt_own}={cnt_iter} перестановок\n"
                f"  hexperms:  {own_ms:.2f} мс\n"
                f"  itertools: {iter_ms:.2f} мс")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexperms — перестановки (алгоритм Германа)")
    parser.add_argument("--all", type=int, metavar="N",
                        help="Вывести все перестановки {1..N}")
    parser.add_argument("--unrank", type=int, metavar="K",
                        help="Получить K-ю перестановку (0-индексация)")
    parser.add_argument("--n", type=int, metavar="N", default=5,
                        help="Размер перестановки (по умолчанию 5)")
    parser.add_argument("--rank", type=str, metavar="PERM",
                        help="Номер перестановки (через запятую, напр. 1,3,2,5,4)")
    parser.add_argument("--next", type=str, metavar="PERM",
                        help="Следующая перестановка")
    parser.add_argument("--prev", type=str, metavar="PERM",
                        help="Предыдущая перестановка")
    parser.add_argument("--derangements", type=int, metavar="N",
                        help="Все беспорядки для {1..N}")
    parser.add_argument("--fixed", nargs=2, type=int, metavar=("N", "POINT"),
                        help="Перестановки {1..N} с фиксированной точкой POINT")
    parser.add_argument("--benchmark", type=int, metavar="N",
                        help="Бенчмарк для n=N")
    args = parser.parse_args()

    n = args.n

    if args.all is not None:
        pe = PermutationEngine(args.all)
        for i, p in enumerate(pe.generate_all()):
            print(f"  {i:6d}: {p}")

    if args.unrank is not None:
        pe = PermutationEngine(n)
        p = pe.unrank(args.unrank)
        print(f"perm[{args.unrank}] = {p}")

    if args.rank is not None:
        perm = list(map(int, args.rank.split(",")))
        pe = PermutationEngine(len(perm))
        r = pe.rank(perm)
        print(f"rank({perm}) = {r}")

    if args.next is not None:
        perm = list(map(int, args.next.split(",")))
        pe = PermutationEngine(len(perm))
        nxt = pe.next_perm(perm)
        print(f"next({perm}) = {nxt}")

    if args.prev is not None:
        perm = list(map(int, args.prev.split(",")))
        pe = PermutationEngine(len(perm))
        prv = pe.prev_perm(perm)
        print(f"prev({perm}) = {prv}")

    if args.derangements is not None:
        pe = PermutationEngine(args.derangements)
        ders = pe.derangements()
        expected = pe.derangement_count()
        print(f"Беспорядки n={args.derangements}: {len(ders)} (ожидается {expected})")
        for d in ders:
            print(f"  {d}")

    if args.fixed is not None:
        n_val, point = args.fixed
        pe = PermutationEngine(n_val)
        fixed = pe.with_fixed_point(point)
        print(f"Перестановки {n_val}! с фиксированной точкой {point}: {len(fixed)}")
        for p in fixed[:10]:
            print(f"  {p}")
        if len(fixed) > 10:
            print(f"  … ещё {len(fixed) - 10}")

    if args.benchmark is not None:
        pe = PermutationEngine(args.benchmark)
        print(pe.benchmark(args.benchmark))

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
