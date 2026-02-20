"""hexpack.py — Теория упаковок замкнутых клеточных полей (Герман).

Источник: PDF «Теория упаковок замкнутых клеточных полей» Франца Германа.

Алгоритм упаковки (АУ):
  Число 1 ставится в произвольную клетку замкнутого поля.
  Число m+1 ставится через m-1 клеток после числа m.
  Позиция: pos(n) = n(n-1)/2   (сумма 0+1+2+...+(n-1))

Период поля: P = n(n-1)/2 + 1 (число клеток для упаковки n чисел).

Ключевые теоремы:
  ТЕОРЕМА: Полная упаковка без пробелов возможна ТОЛЬКО при P = 2^k.
  СЛЕДСТВИЕ 1: Всегда существует ровно одна позиция m, при которой
               ни одна клетка не содержит «собственный» номер.
  СЛЕДСТВИЕ 2: ring[k] + ring[k + P/2] = P + 1 для всех k.
               (Антиподальные пары суммируются в константу.)

Q6-связь: 64 = 2^6 — допустимый период по Герману.
  ring[n*(n-1)//2 % 64] = n для n=1..64 даёт полную упаковку Q6.
  Исключение (нет «собственных» номеров): начало m=32 (нумерация с 0).
"""
from __future__ import annotations
import math
from typing import Iterator


# ─── вспомогательные ──────────────────────────────────────────────────────────

def _triangular(n: int) -> int:
    """T(n) = n*(n-1)//2 = 0+1+2+...+(n-1)."""
    return n * (n - 1) // 2


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ─── основной класс ───────────────────────────────────────────────────────────

class PackedRing:
    """Замкнутое клеточное поле периода P, упакованное алгоритмом Германа.

    Для P = 2^k: ring заполнен числами 1..P без пробелов и коллизий.
    """

    def __init__(self, P: int) -> None:
        if P <= 0:
            raise ValueError("Period P must be positive")
        self.P = P
        self.packable = _is_power_of_two(P)
        self._ring: list[int] = self._build()

    # ── построение ──────────────────────────────────────────────────────────

    def _build(self) -> list[int]:
        """Заполнить кольцо: ring[pos(n)] = n, pos(n) = T(n-1) mod P."""
        P = self.P
        ring = [0] * P
        seen: set[int] = set()
        for n in range(1, P + 1):
            pos = _triangular(n) % P       # pos = n*(n-1)/2 mod P
            if pos in seen:
                self.packable = False
                break
            ring[pos] = n
            seen.add(pos)
        return ring

    # ── доступ ──────────────────────────────────────────────────────────────

    def __getitem__(self, pos: int) -> int:
        return self._ring[pos % self.P]

    def __len__(self) -> int:
        return self.P

    def as_list(self) -> list[int]:
        return list(self._ring)

    # ── следствие 1: фиксированные точки ────────────────────────────────────

    def fixed_points(self, start: int = 0) -> list[int]:
        """Номера n (1..P), которые стоят в «своей» позиции при нумерации с start.

        Число n «на своём месте», если ring[(start + n - 1) % P] == n.
        Следствие 1: для каждого start существует ровно одна такая n,
        кроме ровно одного start, при котором таких n нет.
        """
        P = self.P
        return [n for n in range(1, P + 1) if self._ring[(start + n - 1) % P] == n]

    def exceptional_start(self) -> int:
        """Единственная позиция старта, при которой нет фиксированных точек.

        По Следствию 1: для P=64 — это start=32 (0-indexed).
        """
        for m in range(self.P):
            if not self.fixed_points(m):
                return m
        return -1  # не должно быть

    # ── следствие 2: антиподальные пары ─────────────────────────────────────

    def antipodal_pairs(self) -> list[tuple[int, int, int, int, int]]:
        """Все пары (k, k+P/2): ring[k] + ring[k+P/2] = P+1.

        Возвращает список (pos1, pos2, val1, val2, sum).
        """
        P = self.P
        half = P // 2
        return [(k, k + half, self._ring[k], self._ring[k + half],
                 self._ring[k] + self._ring[k + half])
                for k in range(half)]

    def verify_antipodal(self) -> bool:
        """Проверить Следствие 2: ring[k]+ring[k+P/2] == P+1 для всех k."""
        target = self.P + 1
        return all(s == target for _, _, _, _, s in self.antipodal_pairs())

    # ── Q6 специфика ────────────────────────────────────────────────────────

    def q6_antipode(self, h: int) -> int:
        """Q6-антипод: h ↔ h XOR 63 (инвертировать все 6 бит).

        Совпадает с позиционным антиподом при P=64, P/2=32:
        если ring[h]=n, то ring[h XOR 32]=65-n (Следствие 2).
        Плюс: yang(h) + yang(h^63) = 6 всегда.
        """
        return h ^ 63

    def q6_antipodal_sum_constant(self) -> int:
        """Для P=64: ring[k]+ring[k+32]=65 для всех k."""
        return self.P + 1  # = 65


# ─── магические квадраты (из пакетов 2^(2k)) ─────────────────────────────────

class MagicSquare:
    """Магический квадрат side×side из упаковки поля P = side^2 = 2^(2k).

    Из Теоремы Германа: колонны квадрата имеют одинаковую сумму = (P+1)*side/2.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        P = 2 ** (2 * k)
        self.k = k
        self.P = P
        self.side = 2 ** k
        ring = PackedRing(P)
        if not ring.packable:
            raise ValueError(f"P={P} is not packable (expected 2^(2k))")
        flat = ring.as_list()
        self.matrix = [flat[i * self.side:(i + 1) * self.side]
                       for i in range(self.side)]
        self.magic_constant = (P + 1) * self.side // 2

    def column_sums(self) -> list[int]:
        return [sum(self.matrix[r][c] for r in range(self.side))
                for c in range(self.side)]

    def is_magic(self) -> bool:
        mc = self.magic_constant
        return all(s == mc for s in self.column_sums())

    def format(self) -> str:
        w = len(str(self.P))
        rows = [' '.join(f'{v:{w}}' for v in row) for row in self.matrix]
        return '\n'.join(rows)


# ─── утилиты ─────────────────────────────────────────────────────────────────

def period(n: int) -> int:
    """Период поля для упаковки n чисел: P = n(n-1)/2 + 1."""
    return _triangular(n) + 1


def valid_periods(max_n: int = 64) -> list[tuple[int, int]]:
    """Все пары (n, P) с P = 2^k для n = 1..max_n."""
    result = []
    for n in range(1, max_n + 1):
        P = period(n)
        if _is_power_of_two(P):
            k = P.bit_length() - 1
            result.append((n, P, k))
    return result


def prime_periods(max_n: int = 64) -> list[tuple[int, int]]:
    """Все пары (n, P) где P простое, для n = 1..max_n."""
    def is_prime(p):
        if p < 2: return False
        if p == 2: return True
        if p % 2 == 0: return False
        return all(p % i != 0 for i in range(3, int(p**0.5)+1, 2))
    return [(n, period(n)) for n in range(1, max_n + 1) if is_prime(period(n))]


# ─── главный объект Q6 ────────────────────────────────────────────────────────

Q6_RING = PackedRing(64)
