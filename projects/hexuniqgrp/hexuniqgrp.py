"""hexuniqgrp.py — Уникальные группы (Герман).

Источник: PDF «Уникальные группы» Франца Германа.

Группа называется «уникальной», если её порядок n — составное число,
и существует ровно одна группа порядка n (с точностью до изоморфизма).

UVW-классификация нечётных чисел (n = 2k+1 или n нечётное):
  U = {n : n ≡ 1 (mod 6)} = {1, 7, 13, 19, 25, ...}   (u_k = 6k+1)
  V = {n : n ≡ 3 (mod 6)} = {3, 9, 15, 21, ...}          (кратные 3)
  W = {n : n ≡ 5 (mod 6)} = {5, 11, 17, 23, 29, ...}    (w_k = 6k-1)

Таблица умножения:
  U·U = U,  W·W = U,  U·W = W
  U·V = V,  W·V = V,  V·V = V
  Порядок из V → никогда не уникальный (если он составной из элементов V)

Условие уникальности (необходимое): n — нечётное, без квадратных делителей,
и для каждой пары простых делителей p, q | n: q ≢ 1 (mod p).

Список 35 уникальных порядков ≤ 300:
15, 33, 35, 51, 65, 69, 77, 85, 87, 91, 95, 115, 119, 123, 133,
141, 143, 145, 159, 177, 185, 187, 209, 213, 217, 235, 247, 249,
255, 259, 265, 267, 287, 295, 299
"""
import math
import sys
import argparse
from math import factorial

_UNIQUE_ORDERS_300 = [
    15, 33, 35, 51, 65, 69, 77, 85, 87, 91, 95, 115, 119, 123, 133,
    141, 143, 145, 159, 177, 185, 187, 209, 213, 217, 235, 247, 249,
    255, 259, 265, 267, 287, 295, 299
]


# ── вспомогательные ───────────────────────────────────────────────────────────

def _prime_factors(n: int) -> list[int]:
    """Разложить n на простые множители (с повторениями)."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _is_squarefree(n: int) -> bool:
    """Проверить отсутствие квадратных множителей."""
    factors = _prime_factors(n)
    return len(factors) == len(set(factors))


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


# ── UVW-классификация ─────────────────────────────────────────────────────────

def classify_uvw(n: int) -> str:
    """Классифицировать нечётное число по UVW.

    U: n ≡ 1 (mod 6)
    V: n ≡ 3 (mod 6)  (кратное 3)
    W: n ≡ 5 (mod 6)  (= 6k-1)
    Для чётных: не применяется.
    """
    if n % 2 == 0:
        return "E"   # чётное
    r = n % 6
    if r == 1:
        return "U"
    elif r == 3:
        return "V"
    elif r == 5:
        return "W"
    else:
        return "?"


def uvw_product(c1: str, c2: str) -> str:
    """Произведение классов UVW."""
    table = {
        ("U", "U"): "U",
        ("W", "W"): "U",
        ("U", "W"): "W",
        ("W", "U"): "W",
        ("U", "V"): "V",
        ("V", "U"): "V",
        ("W", "V"): "V",
        ("V", "W"): "V",
        ("V", "V"): "V",
    }
    return table.get((c1, c2), "?")


# ── основной класс ────────────────────────────────────────────────────────────

class UniqueGroups:
    """Уникальные группы по Францу Герману."""

    # ── UVW-классификация ─────────────────────────────────────────────────────

    def classify_odd(self, n: int) -> str:
        """Класс числа n в системе UVW."""
        return classify_uvw(n)

    # ── проверка уникальности ─────────────────────────────────────────────────

    def is_unique_order(self, n: int) -> bool:
        """Проверить: существует ли ровно одна группа порядка n?

        Условие (Hall, 1928): n уникален ⟺
          1. n нечётное
          2. n — свободное от квадратов (squarefree)
          3. Для каждой пары простых p | n и q | n: q ≢ 1 (mod p)
        """
        if n <= 1:
            return False
        if _is_prime(n):
            return False   # простые тривиальны (только одна группа, но не "интересная")
        if n % 2 == 0:
            return False
        if not _is_squarefree(n):
            return False

        primes = list(set(_prime_factors(n)))
        for p in primes:
            for q in primes:
                if p != q and (q - 1) % p == 0:
                    return False
        return True

    def unique_orders_up_to(self, limit: int) -> list[int]:
        """Все уникальные составные порядки ≤ limit."""
        return [n for n in range(3, limit + 1) if self.is_unique_order(n)]

    # ── построение группы G(n) ────────────────────────────────────────────────

    def build_group(self, n: int) -> "FiniteGroup":
        """Построить единственную группу порядка n (n — уникальный порядок)."""
        if not self.is_unique_order(n):
            raise ValueError(f"{n} — не уникальный порядок")
        return FiniteGroup(n)

    # ── формула хорд ─────────────────────────────────────────────────────────

    def chord_count(self, n: int, k: int) -> int:
        """Число расстановок k непересекающихся хорд на n точках окружности.

        chord(n, k) = (1/k!) · ∏_{i=1}^{k} C(n - 2(i-1), 2)
        """
        if k <= 0 or 2 * k > n:
            return 0
        result = 1
        for i in range(1, k + 1):
            m = n - 2 * (i - 1)
            result *= m * (m - 1) // 2
        return result // factorial(k)

    # ── известный список ──────────────────────────────────────────────────────

    def known_unique_orders_300(self) -> list[int]:
        """Авторский список 35 уникальных порядков ≤ 300."""
        return list(_UNIQUE_ORDERS_300)

    def verify_known_list(self) -> dict:
        """Сравнить вычисленный список с авторским (≤ 300)."""
        computed = set(self.unique_orders_up_to(300))
        known = set(_UNIQUE_ORDERS_300)
        return {
            "computed": sorted(computed),
            "known": sorted(known),
            "match": computed == known,
            "only_computed": sorted(computed - known),
            "only_known": sorted(known - computed),
        }


# ── конечная группа G(n) ─────────────────────────────────────────────────────

class FiniteGroup:
    """Единственная группа уникального порядка n (Z_p × Z_q × ...)."""

    def __init__(self, n: int):
        self.n = n
        self._primes = sorted(set(_prime_factors(n)))
        # G(n) ≅ Z_p1 × Z_p2 × ... (прямое произведение циклических групп)

    def __repr__(self):
        parts = " × ".join(f"Z{p}" for p in self._primes)
        return f"G({self.n}) ≅ {parts}"

    def order(self) -> int:
        return self.n

    def element_orders(self) -> dict[int, list]:
        """Порядки элементов (по теореме о прямом произведении).

        Порядок (a₁, ..., aₖ) = НОК(ord(aᵢ)).
        """
        from itertools import product
        component_orders = [list(range(p)) for p in self._primes]
        result: dict[int, list] = {}
        for elem in product(*component_orders):
            # Порядок элемента = НОК порядков компонент
            ord_elem = 1
            for i, v in enumerate(elem):
                p = self._primes[i]
                comp_ord = p if v != 0 else 1
                ord_elem = ord_elem * comp_ord // math.gcd(ord_elem, comp_ord)
            if ord_elem not in result:
                result[ord_elem] = []
            result[ord_elem].append(elem)
        return result

    def cayley_table(self) -> list[list[tuple]]:
        """Таблица Кэли (для малых групп). Элементы = кортежи остатков."""
        from itertools import product
        elems = list(product(*[range(p) for p in self._primes]))
        table = []
        for a in elems:
            row = []
            for b in elems:
                c = tuple((a[i] + b[i]) % self._primes[i] for i in range(len(self._primes)))
                row.append(c)
            table.append(row)
        return table

    def subgroups(self) -> list[str]:
        """Список подгрупп (по Теореме Гюссерова для прямых произведений)."""
        result = ["{e}"]
        for p in self._primes:
            result.append(f"Z{p}")
        result.append(str(self))
        return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexuniqgrp — уникальные группы (теория Германа)")
    parser.add_argument("--is-unique", type=int, metavar="N",
                        help="Проверить, является ли N уникальным порядком")
    parser.add_argument("--list-up-to", type=int, metavar="N",
                        help="Список уникальных порядков ≤ N")
    parser.add_argument("--build-group", type=int, metavar="N",
                        help="Построить группу G(N)")
    parser.add_argument("--cayley", type=int, metavar="N",
                        help="Таблица Кэли G(N)")
    parser.add_argument("--chord-count", nargs=2, type=int, metavar=("N", "K"),
                        help="Число расстановок K хорд на N точках")
    parser.add_argument("--classify-odd", type=int, metavar="N",
                        help="Классифицировать N по UVW")
    parser.add_argument("--verify-known", action="store_true",
                        help="Проверить авторский список уникальных групп ≤ 300")
    args = parser.parse_args()

    ug = UniqueGroups()

    if args.is_unique is not None:
        result = ug.is_unique_order(args.is_unique)
        print(f"G({args.is_unique}) уникальная: {result}")

    if args.list_up_to is not None:
        orders = ug.unique_orders_up_to(args.list_up_to)
        print(f"Уникальные порядки ≤ {args.list_up_to} ({len(orders)} штук):")
        print(orders)

    if args.build_group is not None:
        g = ug.build_group(args.build_group)
        print(f"G({args.build_group}) = {g}")
        print(f"Подгруппы: {g.subgroups()}")
        ords = g.element_orders()
        print("Порядки элементов:")
        for ord_val, elems in sorted(ords.items()):
            print(f"  порядок {ord_val}: {len(elems)} элементов")

    if args.cayley is not None:
        g = ug.build_group(args.cayley)
        table = g.cayley_table()
        print(f"Таблица Кэли G({args.cayley}) ({g}):")
        for row in table[:min(len(table), 10)]:
            print("  " + "  ".join(str(e) for e in row))

    if args.chord_count is not None:
        n, k = args.chord_count
        count = ug.chord_count(n, k)
        print(f"Хорд({n}, {k}) = {count}")

    if args.classify_odd is not None:
        cls = ug.classify_odd(args.classify_odd)
        print(f"Класс {args.classify_odd} по UVW: {cls}")

    if args.verify_known:
        res = ug.verify_known_list()
        print(f"Совпадение с авторским списком: {res['match']}")
        if res["only_computed"]:
            print(f"  Только в вычисленном: {res['only_computed']}")
        if res["only_known"]:
            print(f"  Только в авторском:   {res['only_known']}")
        print(f"  Всего вычислено: {len(res['computed'])}, известно: {len(res['known'])}")

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
