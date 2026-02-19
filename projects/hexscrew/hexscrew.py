"""hexscrew.py — Группа-винт Bₙ ⊂ Sₙ (теория Германа).

Источник: PDF «К вопросу о существовании «винта» в Природе» Франца Германа.

Группа-винт Bₙ = {σ ∈ Sₙ | σ(1) = 1} — подгруппа всех перестановок,
сохраняющих элемент 1. Изоморфна Sₙ₋₁.

|Bₙ| = (n-1)!
Bₙ порождается транспозициями (i j) для i,j ∈ {2..n}.

Спин элемента: знак перестановки, ограниченной на {2..n}.
  left_spin  = +1 (чётная) / -1 (нечётная)
  right_spin = произведение длин орбит на {2..n} по mod 2

Действие B₇ на Q6: переставляет биты 2..7 (бит 1 фиксирован) → подгруппа Aut(Q6).
"""
import math
import sys
import argparse
from typing import Iterator


# ── вспомогательные ──────────────────────────────────────────────────────────

def _cycle_decomposition(perm: list[int]) -> list[list[int]]:
    """Разложить перестановку perm (1-индексированную) на циклы."""
    n = len(perm)
    visited = [False] * (n + 1)
    cycles = []
    for start in range(1, n + 1):
        if visited[start]:
            continue
        cycle = []
        current = start
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = perm[current - 1]
        if len(cycle) >= 1:
            cycles.append(cycle)
    return cycles


def _sign(perm: list[int]) -> int:
    """Знак перестановки: +1 (чётная) / -1 (нечётная)."""
    cycles = _cycle_decomposition(perm)
    transpositions = sum(len(c) - 1 for c in cycles)
    return 1 if transpositions % 2 == 0 else -1


def _compose(p: list[int], q: list[int]) -> list[int]:
    """Применить сначала q, потом p: (p∘q)(i) = p(q(i))."""
    return [p[q[i - 1] - 1] for i in range(1, len(p) + 1)]


def _inverse(perm: list[int]) -> list[int]:
    """Обратная перестановка."""
    inv = [0] * len(perm)
    for i, v in enumerate(perm):
        inv[v - 1] = i + 1
    return inv


def _order_of_element(perm: list[int]) -> int:
    """Порядок элемента группы: минимальное k > 0 с perm^k = e."""
    n = len(perm)
    e = list(range(1, n + 1))
    current = list(perm)
    k = 1
    while current != e and k <= math.factorial(n):
        current = _compose(perm, current)
        k += 1
    return k


# ── основной класс ────────────────────────────────────────────────────────────

class ScrewGroup:
    """Группа-винт Bₙ = {σ ∈ Sₙ | σ(1) = 1}."""

    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"n должно быть ≥ 2, получено {n}")
        self.n = n
        self._elements: list[list[int]] | None = None

    def __repr__(self):
        return f"ScrewGroup(n={self.n}, order={(self.n-1)!r})"

    # ── порядок и элементы ────────────────────────────────────────────────────

    def order(self) -> int:
        """Порядок Bₙ = (n-1)!"""
        return math.factorial(self.n - 1)

    def _gen_elements(self) -> list[list[int]]:
        """Сгенерировать все элементы Bₙ."""
        from itertools import permutations
        result = []
        # Все перестановки {1..n} где первый элемент = 1
        for tail in permutations(range(2, self.n + 1)):
            result.append([1] + list(tail))
        return result

    def elements(self) -> list[list[int]]:
        """Список всех (n-1)! элементов Bₙ."""
        if self._elements is None:
            self._elements = self._gen_elements()
        return list(self._elements)

    def is_member(self, perm: list[int]) -> bool:
        """Проверить, принадлежит ли perm группе Bₙ (= σ(1) = 1)."""
        if len(perm) != self.n:
            return False
        return perm[0] == 1

    # ── спин ──────────────────────────────────────────────────────────────────

    def left_spin(self, perm: list[int]) -> int:
        """Левый спин: знак перестановки, ограниченной на {2..n}.

        +1 — чётная, -1 — нечётная.
        """
        if not self.is_member(perm):
            raise ValueError("perm не принадлежит Bₙ (σ(1) ≠ 1)")
        restricted = perm[1:]   # позиции {2..n} (0-индексированные)
        # Переиндексируем: {2..n} → {1..n-1}
        reindexed = [v - 1 for v in restricted]
        return _sign(reindexed)

    def right_spin(self, perm: list[int]) -> int:
        """Правый спин: произведение (длина каждой орбиты mod 2) на {2..n}.

        +1 если произведение = 1, -1 если = -1.
        """
        if not self.is_member(perm):
            raise ValueError("perm не принадлежит Bₙ")
        restricted = perm[1:]
        reindexed = [v - 1 for v in restricted]
        cycles = _cycle_decomposition(reindexed)
        product = 1
        for c in cycles:
            product *= len(c) % 2 if len(c) % 2 != 0 else -1
        # Упрощённо: произведение (-1)^(len-1) для каждого цикла = знак
        return _sign(reindexed)

    # ── структура орбит ───────────────────────────────────────────────────────

    def cycle_type(self, perm: list[int]) -> list[int]:
        """Тип циклов: отсортированный список длин циклов в разложении."""
        cycles = _cycle_decomposition(perm)
        return sorted(len(c) for c in cycles)

    def cycles(self, perm: list[int]) -> list[list[int]]:
        """Разложение perm на циклы (1-индексированное)."""
        return _cycle_decomposition(perm)

    # ── операции группы ───────────────────────────────────────────────────────

    def multiply(self, p: list[int], q: list[int]) -> list[int]:
        """Произведение p∘q в Sₙ (сначала q, потом p)."""
        return _compose(p, q)

    def inverse(self, perm: list[int]) -> list[int]:
        """Обратный элемент."""
        return _inverse(perm)

    def order_of(self, perm: list[int]) -> int:
        """Порядок элемента группы."""
        return _order_of_element(perm)

    # ── подгруппы и структура ────────────────────────────────────────────────

    def generating_transpositions(self) -> list[tuple[int, int]]:
        """Порождающие транспозиции (i j) для i,j ∈ {2..n}."""
        result = []
        for i in range(2, self.n + 1):
            for j in range(i + 1, self.n + 1):
                result.append((i, j))
        return result

    def transposition(self, i: int, j: int) -> list[int]:
        """Создать транспозицию (i j) как элемент Sₙ."""
        perm = list(range(1, self.n + 1))
        perm[i - 1], perm[j - 1] = perm[j - 1], perm[i - 1]
        return perm

    def conjugacy_classes(self) -> dict[tuple, list[list[int]]]:
        """Классы сопряжённости в Bₙ, сгруппированные по типу цикла."""
        classes: dict[tuple, list[list[int]]] = {}
        for elem in self.elements():
            ct = tuple(self.cycle_type(elem))
            if ct not in classes:
                classes[ct] = []
            classes[ct].append(elem)
        return classes

    # ── изоморфизм Bₙ → Sₙ₋₁ ─────────────────────────────────────────────────

    def isomorphism_to_symmetric(self) -> dict:
        """Биекция Bₙ → Sₙ₋₁: убрать первый элемент, сдвинуть на 1."""
        elems = self.elements()
        mapping = {}
        for perm in elems:
            key = tuple(perm)
            value = tuple(v - 1 for v in perm[1:])
            mapping[key] = value
        return mapping

    # ── действие на Q6 ────────────────────────────────────────────────────────

    def act_on_hexagram(self, hexagram: int, perm: list[int]) -> int:
        """Применить перестановку perm ∈ B₇ к 6-битной гексаграмме.

        perm переставляет биты 2..7 (бит 1 = LSB всегда фиксирован).
        hexagram ∈ [0, 63].
        """
        if self.n != 7:
            raise ValueError("act_on_hexagram только для n=7 (действие на 6-бит Q6)")
        if not self.is_member(perm):
            raise ValueError("perm не принадлежит B₇")
        result = 0
        # Бит 1 (LSB) фиксирован
        result |= (hexagram & 1)
        # Биты 2..7: perm переставляет позиции 2..7 → 2..7
        for i in range(2, 8):
            src_bit = i - 1          # 0-индексированный бит (1..6)
            dst_pos = perm[i - 1]    # куда идёт i-й элемент
            dst_bit = dst_pos - 1    # 0-индексированный
            bit_val = (hexagram >> src_bit) & 1
            result |= (bit_val << dst_bit)
        return result

    # ── визуализация ─────────────────────────────────────────────────────────

    def print_cayley_table(self) -> str:
        """Таблица Кэли Bₙ (ASCII). Для n ≤ 4."""
        if self.n > 4:
            return f"Таблица слишком большая (|Bₙ| = {self.order()}), используйте n ≤ 4"
        elems = self.elements()
        # Короткие метки
        labels = [f"σ{i}" for i in range(len(elems))]
        elem_to_idx = {tuple(e): i for i, e in enumerate(elems)}
        header = "    " + " ".join(f"{l:3s}" for l in labels)
        lines = [f"Таблица Кэли B{self.n}:", header, "    " + "───" * len(elems)]
        for i, ei in enumerate(elems):
            row = []
            for ej in elems:
                prod = self.multiply(ei, ej)
                idx = elem_to_idx[tuple(prod)]
                row.append(f"{labels[idx]:3s}")
            lines.append(f"{labels[i]:3s}│" + " ".join(row))
        return "\n".join(lines)

    def print_lattice(self) -> str:
        """Упрощённая схема решётки подгрупп."""
        lines = [
            f"Решётка подгрупп B{self.n} ≅ S{self.n - 1}:",
            f"  B{self.n} (порядок {self.order()})",
            f"  │",
            f"  ├── Alt{self.n - 1} (чётные, порядок {self.order() // 2})",
            f"  │   └── ...",
            f"  └── {{e}} (тривиальная)",
        ]
        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexscrew — группа-винт Bₙ ⊂ Sₙ (теория Германа)")
    parser.add_argument("--order", type=int, metavar="N",
                        help="Порядок Bₙ")
    parser.add_argument("--elements", type=int, metavar="N",
                        help="Все элементы Bₙ")
    parser.add_argument("--spin", type=str, metavar="PERM",
                        help="Левый и правый спин (через запятую, напр. 1,3,2,4)")
    parser.add_argument("--cayley", type=int, metavar="N",
                        help="Таблица Кэли Bₙ (n ≤ 4)")
    parser.add_argument("--subgroups", type=int, metavar="N",
                        help="Описание решётки подгрупп Bₙ")
    parser.add_argument("--isomorphism", type=int, metavar="N",
                        help="Показать изоморфизм Bₙ → Sₙ₋₁")
    parser.add_argument("--act-q6", nargs=2, metavar=("HEX", "PERM"),
                        help="Действие на гексаграмму HEX перестановкой PERM∈B7")
    args = parser.parse_args()

    if args.order is not None:
        sg = ScrewGroup(args.order)
        print(f"|B{args.order}| = {sg.order()} = ({args.order}-1)!")

    if args.elements is not None:
        sg = ScrewGroup(args.elements)
        elems = sg.elements()
        print(f"B{args.elements} ({len(elems)} элементов):")
        for i, e in enumerate(elems):
            print(f"  σ{i}: {e}")

    if args.spin is not None:
        perm = list(map(int, args.spin.split(",")))
        n = len(perm)
        sg = ScrewGroup(n)
        try:
            ls = sg.left_spin(perm)
            print(f"left_spin({perm}) = {ls}")
        except ValueError as e:
            print(f"Ошибка: {e}")

    if args.cayley is not None:
        sg = ScrewGroup(args.cayley)
        print(sg.print_cayley_table())

    if args.subgroups is not None:
        sg = ScrewGroup(args.subgroups)
        print(sg.print_lattice())
        classes = sg.conjugacy_classes()
        print(f"Классы сопряжённости ({len(classes)} классов):")
        for ct, members in sorted(classes.items()):
            print(f"  тип цикла {ct}: {len(members)} элементов")

    if args.isomorphism is not None:
        sg = ScrewGroup(args.isomorphism)
        iso = sg.isomorphism_to_symmetric()
        print(f"Изоморфизм B{args.isomorphism} → S{args.isomorphism - 1}:")
        for k, v in list(iso.items())[:8]:
            print(f"  {list(k)} → {list(v)}")

    if args.act_q6 is not None:
        hexagram = int(args.act_q6[0])
        perm = list(map(int, args.act_q6[1].split(",")))
        sg = ScrewGroup(7)
        result = sg.act_on_hexagram(hexagram, perm)
        print(f"B7 действует на Q6[{hexagram}] через {perm}: → {result}")

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
