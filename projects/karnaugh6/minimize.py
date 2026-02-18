#!/usr/bin/env python3
"""
karnaugh6 — минимизатор булевых функций от 6 переменных

Алгоритм Куайна–МакКласки для 6 переменных (64 минтерма).
Использует граф Q6 как основу: гиперкубическая структура
совпадает со структурой импликант.

Использование:
    python3 minimize.py <минтермы>
    python3 minimize.py 0 1 2 3 4 5 6 7           # верхняя половина
    python3 minimize.py --dc 3 5 --minterms 0 1 2  # с безразличными

Вывод:
    - Простые импликанты (prime implicants)
    - Существенные импликанты (essential PIs)
    - Минимальная ДНФ
"""

from __future__ import annotations
import sys
import argparse
from itertools import combinations

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import SIZE, to_bits

VARS = 6
VAR_NAMES = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']


# ---------------------------------------------------------------------------
# Импликанта — куб в гиперкубе Q6
# ---------------------------------------------------------------------------

class Implicant:
    """
    Импликанта представлена строкой из {0, 1, -}:
    '-' означает «переменная не входит в куб» (безразличная позиция).
    """

    def __init__(self, bits: str, covered: frozenset[int]) -> None:
        assert len(bits) == VARS
        self.bits = bits          # например '10-1--'
        self.covered = covered    # набор минтермов, покрытых этой импликантой

    @classmethod
    def from_minterm(cls, m: int) -> 'Implicant':
        return cls(to_bits(m), frozenset([m]))

    def combine(self, other: 'Implicant') -> 'Implicant | None':
        """
        Попытаться объединить две импликанты в одну (следующий уровень кубов).
        Возможно, если они отличаются ровно в одной позиции (не '-').
        """
        diffs = []
        for i, (a, b) in enumerate(zip(self.bits, other.bits)):
            if a != b:
                if a == '-' or b == '-':
                    return None  # разные размерности куба
                diffs.append(i)
        if len(diffs) != 1:
            return None
        new_bits = list(self.bits)
        new_bits[diffs[0]] = '-'
        return Implicant(''.join(new_bits), self.covered | other.covered)

    def covers(self, minterm: int) -> bool:
        """Проверить, покрывает ли импликанта данный минтерм."""
        bits = to_bits(minterm)
        return all(sb == '-' or sb == mb for sb, mb in zip(self.bits, bits))

    def size(self) -> int:
        """Размер куба: 2^(число '-' позиций)."""
        return 2 ** self.bits.count('-')

    def to_expr(self) -> str:
        """Преобразовать в выражение ДНФ."""
        terms = []
        for i, b in enumerate(reversed(self.bits)):  # reversed: бит 0 = x0
            var = VAR_NAMES[i]
            if b == '1':
                terms.append(var)
            elif b == '0':
                terms.append(f'~{var}')
        if not terms:
            return '1'  # тавтология
        return ' & '.join(terms)

    def __repr__(self) -> str:
        return f"Implicant({self.bits}, covers={sorted(self.covered)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Implicant):
            return NotImplemented
        return self.bits == other.bits

    def __hash__(self) -> int:
        return hash(self.bits)


# ---------------------------------------------------------------------------
# Алгоритм Куайна–МакКласки
# ---------------------------------------------------------------------------

def quine_mccluskey(minterms: list[int], dont_cares: list[int] = []) -> list[Implicant]:
    """
    Найти все простые импликанты для функции заданной списком минтермов.

    minterms:   минтермы, где функция = 1
    dont_cares: минтермы, где значение функции безразлично
    """
    all_ones = set(minterms) | set(dont_cares)
    if not all_ones:
        return []

    # Сгруппировать по числу единиц (для оптимизации объединения)
    groups: dict[int, list[Implicant]] = {}
    for m in all_ones:
        ones = bin(m).count('1')
        if ones not in groups:
            groups[ones] = []
        groups[ones].append(Implicant.from_minterm(m))

    prime_implicants: set[Implicant] = set()
    current_level = groups

    while True:
        next_level: dict[int, list[Implicant]] = {}
        used: set[str] = set()

        sorted_keys = sorted(current_level.keys())
        for i in range(len(sorted_keys) - 1):
            g1 = current_level[sorted_keys[i]]
            g2 = current_level[sorted_keys[i + 1]]
            for imp1 in g1:
                for imp2 in g2:
                    merged = imp1.combine(imp2)
                    if merged is not None:
                        key = sorted_keys[i]
                        if key not in next_level:
                            next_level[key] = []
                        if merged not in next_level[key]:
                            next_level[key].append(merged)
                        used.add(imp1.bits)
                        used.add(imp2.bits)

        # Всё, что не объединилось — простые импликанты
        for group in current_level.values():
            for imp in group:
                if imp.bits not in used:
                    prime_implicants.add(imp)

        if not next_level:
            break
        current_level = next_level

    return list(prime_implicants)


def essential_implicants(
    prime_imps: list[Implicant],
    minterms: list[int],
) -> list[Implicant]:
    """
    Выбрать существенные простые импликанты методом Петрика.
    Жадный алгоритм: выбирать импликанту, покрывающую наибольшее число непокрытых минтермов.
    """
    uncovered = set(minterms)
    selected: list[Implicant] = []

    # Сначала найти однозначно существенные (покрывают минтерм единственным образом)
    for m in list(minterms):
        covering = [p for p in prime_imps if p.covers(m)]
        if len(covering) == 1:
            imp = covering[0]
            if imp not in selected:
                selected.append(imp)
                uncovered -= imp.covered

    # Остальные — жадным методом
    while uncovered:
        best = max(
            prime_imps,
            key=lambda p: len(p.covered & uncovered)
        )
        if not (best.covered & uncovered):
            break  # не покрываем — что-то пошло не так
        selected.append(best)
        uncovered -= best.covered

    return selected


def minimize(
    minterms: list[int],
    dont_cares: list[int] = [],
) -> dict:
    """
    Минимизировать булеву функцию 6 переменных.

    Возвращает словарь:
      prime_implicants  — все простые импликанты
      essential         — существенные импликанты (минимальное покрытие)
      expression        — минимальная ДНФ (строка)
    """
    if not minterms:
        return {
            'prime_implicants': [],
            'essential': [],
            'expression': '0',
        }

    pis = quine_mccluskey(minterms, dont_cares)
    ess = essential_implicants(pis, minterms)

    if not ess:
        expr = '0'
    elif any(e.bits == '-' * VARS for e in ess):
        expr = '1'
    else:
        expr = ' | '.join(f'({e.to_expr()})' for e in ess)

    return {
        'prime_implicants': pis,
        'essential': ess,
        'expression': expr,
    }


# ---------------------------------------------------------------------------
# Визуализация в терминале
# ---------------------------------------------------------------------------

def print_truth_table(minterms: list[int], dont_cares: list[int] = []) -> None:
    """Вывести таблицу истинности (первые 16 строк или все)."""
    m_set = set(minterms)
    dc_set = set(dont_cares)
    print(f"\n  {'x5':>3} {'x4':>3} {'x3':>3} {'x2':>3} {'x1':>3} {'x0':>3}  │  f")
    print("  " + "─" * 31)
    for i in range(SIZE):
        bits = to_bits(i)
        val = '1' if i in m_set else ('-' if i in dc_set else '0')
        if val != '0':  # только единицы и безразличные
            b = list(bits)
            print(f"  {b[0]:>3} {b[1]:>3} {b[2]:>3} {b[3]:>3} {b[4]:>3} {b[5]:>3}  │  {val}  (#{i})")


# ---------------------------------------------------------------------------
# Визуальная карта Карно 8×8 (Q6 = 64 ячейки)
# ---------------------------------------------------------------------------

# Порядок строк и столбцов по коду Грея для 3 переменных
_GRAY3 = [0, 1, 3, 2, 6, 7, 5, 4]   # 000,001,011,010,110,111,101,100

# Метки для кода Грея 3-бит (x5x4x3 и x2x1x0)
_GRAY3_LABELS = ['000', '001', '011', '010', '110', '111', '101', '100']


def _cell_index(row: int, col: int) -> int:
    """Вычислить минтерм по позиции (row, col) в карте 8×8."""
    # row кодирует x5x4x3, col кодирует x2x1x0
    hi = _GRAY3[row]   # значение x5x4x3
    lo = _GRAY3[col]   # значение x2x1x0
    return (hi << 3) | lo


def print_karnaugh_map(minterms: list[int], dont_cares: list[int] = [],
                       essential: list | None = None) -> None:
    """
    Вывести карту Карно 8×8 для функции 6 переменных.

    Оси:
      Строки  = x5 x4 x3  (3 старших бита, код Грея)
      Столбцы = x2 x1 x0  (3 младших бита, код Грея)

    Символы ячеек:
      1  — минтерм
      -  — безразличный (don't care)
      .  — 0
      *  — минтерм, покрытый существенным импликантом (если передан)
    """
    m_set = set(minterms)
    dc_set = set(dont_cares)
    ess_covered: set[int] = set()
    if essential:
        for e in essential:
            ess_covered.update(e.covered & m_set)

    print()
    print("  Карта Карно Q6  (8×8)")
    print()

    # Заголовок столбцов (x2x1x0)
    col_header = '  x5x4x3╲x2x1x0 │ ' + '  '.join(_GRAY3_LABELS) + ' │'
    print(col_header)
    sep = '  ' + '─' * 10 + '┼─' + '────' * 8 + '──┤'
    print(sep)

    for row in range(8):
        label = _GRAY3_LABELS[row]
        cells = []
        for col in range(8):
            idx = _cell_index(row, col)
            if idx in ess_covered:
                cells.append('*')
            elif idx in m_set:
                cells.append('1')
            elif idx in dc_set:
                cells.append('-')
            else:
                cells.append('·')
        row_str = '  ' + ''.join(c.ljust(3) for c in cells)
        print(f"  {label:>10}  │ {row_str}│")

    print(sep.replace('┼', '┴'))
    print()
    print("  Обозначения: 1=минтерм  *=покрыт существенным  -=безразличный  ·=0")
    print()


def print_result(result: dict) -> None:
    pis = result['prime_implicants']
    ess = result['essential']
    expr = result['expression']

    print(f"\n  Простые импликанты ({len(pis)}):")
    for p in sorted(pis, key=lambda x: x.bits):
        mark = ' ← существенная' if p in ess else ''
        covered_str = str(sorted(p.covered))
        print(f"    {p.bits}  покрывает {covered_str:<30}{mark}")

    print(f"\n  Существенные импликанты ({len(ess)}):")
    for e in ess:
        print(f"    {e.bits}  ←  {e.to_expr()}")

    print(f"\n  Минимальная ДНФ:")
    print(f"    f = {expr}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='karnaugh6 — минимизатор булевых функций 6 переменных (Куайн–МакКласки)'
    )
    parser.add_argument(
        'minterms', nargs='*', type=int,
        help='Минтермы (номера строк таблицы истинности, где f=1)'
    )
    parser.add_argument(
        '--dc', nargs='*', type=int, default=[],
        help='Безразличные наборы (dont-care)'
    )
    parser.add_argument(
        '--table', action='store_true',
        help='Вывести таблицу истинности'
    )
    parser.add_argument(
        '--map', action='store_true',
        help='Вывести карту Карно 8×8'
    )
    args = parser.parse_args()

    minterms = args.minterms
    dont_cares = args.dc or []

    if not minterms:
        # Пример: f = x0 & x1 (минтермы 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
        print("  Пример: f(x0,...,x5) = x0 & x1")
        minterms = [m for m in range(SIZE) if (m & 0b11) == 0b11]
        print(f"  Минтермы: {minterms}")

    # Валидация
    for m in minterms + dont_cares:
        if not 0 <= m < SIZE:
            print(f"  Ошибка: {m} вне диапазона 0-63")
            sys.exit(1)

    if set(minterms) & set(dont_cares):
        print("  Ошибка: минтермы и безразличные наборы пересекаются")
        sys.exit(1)

    print(f"\n  karnaugh6 — минимизатор Q6")
    print(f"  Переменные: x5 x4 x3 x2 x1 x0  (x0 = бит 0)")
    print(f"  Минтермы ({len(minterms)}): {minterms}")
    if dont_cares:
        print(f"  Безразличные ({len(dont_cares)}): {dont_cares}")

    if args.table:
        print_truth_table(minterms, dont_cares)

    result = minimize(minterms, dont_cares)

    if args.map:
        print_karnaugh_map(minterms, dont_cares, essential=result['essential'])

    print_result(result)


if __name__ == '__main__':
    main()
