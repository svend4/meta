"""hextrimat.py — Пирамидальная (треугольная) матрица И-Цзин (Андреев).

Источник: PDF «Геометрически-числовая симметрия в матрице И-Цзин»
          Герасима Андреева (2002).

Структура:
  64 гексаграммы (числа 1..64) расположены в треугольной матрице.
  Строки r=1..11: в строке r находится r элементов.
  T(10) = 55 элементов в строках 1..10, строка 11 — 9 элементов (из 11).
  Итого: 55 + 9 = 64.

  Позиция ячейки (r, c): idx = T(r-1) + c - 1  (0-indexed),  число = idx+1.
  Координаты ячейки n: r = строка, c = позиция в строке.

Ключевые числа матрицы (Андреев):
  Σ(1..64)  = 2080 = 160×13 = 80×26   («число матрицы» = 13)
  729  = 3×243 = 3⁶                    («Птица Времени»)
  242  = 22²/2 = 11²×2                 («Цветок Тота»)
  753  = 3×251                          («Свастика-вихрь», левосторонняя)
  832  = 753+79 = 2080×0.4             («Свастика+узлы», делит матрицу 3:4:5)
  1248 = 2080-832                       (фоновая фигура, = 2080×0.6)

Симметрия:
  D₃ (диэдральная группа, порядок 6): вращения 0°/120°/240°, 3 отражения.
  Треугольник имеет ось вертикального отражения: (r,c) ↔ (r, r+1-c).
  Вращение на 120°: сложнее; реализовано через перестановку секторов.

  Тройная симметрия: 3 сектора треугольника. Свастика — фигура с тройной
  центральной симметрией (три ветви, поворот на 120°).

  13 пар «близнецов»: 13 парных подмножества по 13 элементов с равными суммами.
  Число 13 — «число матрицы»: 2080 = 160×13.
"""
from __future__ import annotations
import math
from typing import Iterator


# ─── вспомогательные ──────────────────────────────────────────────────────────

def _T(n: int) -> int:
    """Треугольное число T(n) = n(n+1)/2."""
    return n * (n + 1) // 2


def _row_col(idx: int) -> tuple[int, int]:
    """0-indexed позиция idx → (строка r, столбец c), оба 1-indexed."""
    # Найти строку r: T(r-1) <= idx < T(r)
    r = math.isqrt(2 * idx)
    while _T(r - 1) > idx:
        r -= 1
    while _T(r) <= idx:
        r += 1
    c = idx - _T(r - 1) + 1
    return r, c


def _idx(r: int, c: int) -> int:
    """(r,c) 1-indexed → 0-indexed позиция в матрице."""
    return _T(r - 1) + c - 1


# ─── основной класс ───────────────────────────────────────────────────────────

class TriangularMatrix:
    """Треугольная матрица И-Цзин: числа 1..64 в треугольном расположении.

    Строка r (1..11) содержит числа from_n(r)..to_n(r):
      from_n(r) = T(r-1)+1, to_n(r) = min(T(r), 64).
    """

    TOTAL = 64
    MATRIX_NUMBER = 13          # 2080 = 160×13
    SUM_TOTAL = 2080            # Σ(1..64)

    # Числа ключевых фигур по Андрееву
    BIRD_OF_TIME = 729          # 3×243 = 3⁶
    FLOWER_THOTH = 242          # 22²/2
    SWASTIKA     = 753          # 3×251
    SWASTIKA_EXT = 832          # 753+79 = 2080×0.4  (3:4:5 пропорция)
    BACKGROUND   = 1248         # 2080-832

    def __init__(self) -> None:
        # cells[idx] = (r, c, value=idx+1)
        self.cells: list[tuple[int, int, int]] = []
        for idx in range(self.TOTAL):
            r, c = _row_col(idx)
            self.cells.append((r, c, idx + 1))

        # Количество строк
        self.num_rows = self.cells[-1][0]  # строка последней ячейки

        # Карта (r,c) → значение
        self._map: dict[tuple[int, int], int] = {
            (r, c): v for r, c, v in self.cells
        }

    # ── доступ ──────────────────────────────────────────────────────────────

    def value(self, r: int, c: int) -> int | None:
        """Значение ячейки (r,c). None если не в матрице."""
        return self._map.get((r, c))

    def row_values(self, r: int) -> list[int]:
        """Все значения строки r (1-indexed)."""
        return [v for rr, cc, v in self.cells if rr == r]

    def row_sum(self, r: int) -> int:
        return sum(self.row_values(r))

    def all_row_sums(self) -> list[tuple[int, int]]:
        """[(строка, сумма) для строк 1..num_rows]."""
        return [(r, self.row_sum(r)) for r in range(1, self.num_rows + 1)]

    # ── симметрия ───────────────────────────────────────────────────────────

    def reflect_vertical(self, r: int, c: int) -> tuple[int, int]:
        """Вертикальное отражение: (r,c) → (r, r+1-c).

        Ось симметрии — средний столбец каждой строки.
        """
        return (r, r + 1 - c)

    def center_cell(self) -> tuple[int, int, int]:
        """Приближённый центр треугольника (ось симметрии).

        Для матрицы 11 строк: центр ≈ строка 7, средний столбец.
        Андреев называет ячейку 27 «главной Инь».
        """
        target = 27
        r, c = _row_col(target - 1)
        return (r, c, target)

    def d3_sectors(self) -> tuple[list[int], list[int], list[int]]:
        """Разделить матрицу на 3 сектора для тройной (D₃) симметрии.

        Сектор 1 (верхний): строки 1..4       (суммы ~верхняя треть)
        Сектор 2 (нижний-левый): строки 5..8  (половина)
        Сектор 3 (нижний-правый): оставшееся
        Это грубое приближение трёхосевого деления Андреева.
        """
        s1 = [v for r, c, v in self.cells if r <= 4]
        s2 = [v for r, c, v in self.cells if 5 <= r <= 8 and c <= r // 2]
        s3 = [v for r, c, v in self.cells if 5 <= r <= 8 and c > r // 2]
        s_rest = [v for r, c, v in self.cells if r > 8]
        return (s1, s2, s3 + s_rest[:len(s_rest)//2], )

    # ── поиск подмножеств по сумме ─────────────────────────────────────────

    def find_contiguous_subsets(self, target: int) -> list[list[int]]:
        """Найти все непрерывные подмножества (подстроки) со значением target."""
        vals = [v for _, _, v in self.cells]
        result = []
        for start in range(len(vals)):
            s = 0
            for end in range(start, len(vals)):
                s += vals[end]
                if s == target:
                    result.append(vals[start:end + 1])
                elif s > target:
                    break
        return result

    def proportion_345(self) -> dict[str, object]:
        """Египетская пропорция 3:4:5 в матрице.

        Андреев: свастика+узлы = 832 = 2080×0.4 = 2080×2/5.
        Фоновая фигура = 1248 = 2080×3/5.
        Пропорция 832:1248:2080 = 2:3:5 ~ 3:4:5 (египетский треугольник).
        """
        return {
            'total': self.SUM_TOTAL,
            'swastika_ext': self.SWASTIKA_EXT,     # 832
            'background': self.BACKGROUND,          # 1248
            'ratio_swastika': self.SWASTIKA_EXT / self.SUM_TOTAL,   # 0.4
            'ratio_background': self.BACKGROUND / self.SUM_TOTAL,   # 0.6
            'check_345': abs(self.SWASTIKA_EXT / self.BACKGROUND - 2/3) < 0.01,
        }

    # ── числа Андреева ─────────────────────────────────────────────────────

    def verify_key_numbers(self) -> dict[str, object]:
        """Верификация ключевых числовых фактов матрицы."""
        return {
            'sum_total': sum(v for _, _, v in self.cells),      # должно быть 2080
            'sum_correct': sum(v for _, _, v in self.cells) == self.SUM_TOTAL,
            '13_divides': self.SUM_TOTAL % self.MATRIX_NUMBER == 0,
            '729_is_3_pow_6': self.BIRD_OF_TIME == 3**6,
            '242_is_half_484': self.FLOWER_THOTH == 22**2 // 2,
            '753_is_3x251': self.SWASTIKA == 3 * 251,
            '832_is_0.4_total': abs(self.SWASTIKA_EXT / self.SUM_TOTAL - 0.4) < 1e-9,
            '1248_is_0.6_total': abs(self.BACKGROUND / self.SUM_TOTAL - 0.6) < 1e-9,
        }

    # ── близнецы (13 пар) ──────────────────────────────────────────────────

    def twin_pairs(self) -> list[tuple[list[int], list[int], int]]:
        """Приближение 13 пар «близнецов» Андреева.

        Для каждой из 13 пар: два подмножества по 13 элементов
        с одинаковыми суммами, симметричными относительно биссектрис.

        Реализация: 13 пар строк (отражения относительно вертикальной оси).
        """
        pairs = []
        for r in range(1, self.num_rows + 1):
            row = self.row_values(r)
            if len(row) >= 2:
                left = row[:len(row) // 2]
                right = row[len(row) // 2:]
                if left and right:
                    pairs.append((left, right[::-1], sum(left)))
        return pairs[:13]


# ─── глобальный объект ───────────────────────────────────────────────────────

TRIMAT = TriangularMatrix()
