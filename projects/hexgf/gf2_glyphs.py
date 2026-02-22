"""gf2_glyphs — GF(2⁶) через систему глифов Q6.

Каждый глиф (0..63) — элемент поля Галуа GF(2⁶).
Сложение = XOR = симметрическая разность сегментов глифа.
Умножение — нетривиальная перестановка 63 ненулевых глифов.

Примитивный элемент g = 2 (= x), порядок 63.
Орбита g: g⁰=1, g¹=2, g²=4, g³=8, g⁴=16, g⁵=32, g⁶=3, ...

Визуализация:
  • Цикл примитивного элемента: 63 глифа по кольцу
  • Таблица умножения 8×8 (ненулевые элементы)
  • Подполя: GF(2) ⊂ GF(4) ⊂ GF(8) ⊂ GF(64) — глифы каждого подполя
  • Следы: Tr(h) = h⊕h²⊕h⁴⊕h⁸⊕h¹⁶⊕h³² ∈ {0,1}
  • Мультипликативные орбиты (циклотомические классы)
"""

from __future__ import annotations
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexgf.hexgf import (
    gf_mul, gf_pow, gf_inv, gf_log, gf_exp, gf_trace,
    ORDER, PRIMITIVE,
    all_cyclotomic_cosets as cyclotomic_classes,
    subfield_elements,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph, render_hasse_glyphs,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _glyph_colored(h: int, highlight: bool = False, color: bool = True) -> list[str]:
    """Три строки глифа с ANSI-цветом."""
    rows = render_glyph(h)
    if not color:
        return rows
    yc = yang_count(h)
    c = (_YANG_BG[yc] + _BOLD) if highlight else _YANG_ANSI[yc]
    return [c + row + _RESET for row in rows]


# ---------------------------------------------------------------------------
# 1. Цикл примитивного элемента g=2
# ---------------------------------------------------------------------------

def render_generator_cycle(color: bool = True, width: int = 16) -> str:
    """
    Цикл g⁰, g¹, ..., g⁶² — все 63 ненулевых элемента GF(2⁶) как глифы.
    Выводится по <width> глифов в строке.
    """
    lines: list[str] = []
    lines.append('═' * 60)
    lines.append(f'  Цикл примитивного элемента g=2 в GF(2⁶)')
    lines.append(f'  gᵏ = gf_exp[k],  k=0..62  (все 63 ненулевых элемента)')
    lines.append('═' * 60)

    # Разбиваем 63 элемента на строки по width
    for row_start in range(0, 63, width):
        chunk = list(range(row_start, min(row_start + width, 63)))
        elements = [gf_exp(k) for k in chunk]

        # Заголовок: степени k
        hdr = '  ' + '  '.join(f'g^{k:<2d}' for k in chunk)
        lines.append(hdr)

        # Три строки глифов
        for ri in range(3):
            parts: list[str] = []
            for k, h in zip(chunk, elements):
                cell = render_glyph(h)[ri]
                if color:
                    yc = yang_count(h)
                    cell = _YANG_ANSI[yc] + cell + _RESET
                parts.append(cell)
            lines.append('  ' + '  '.join(parts))

        # Значения (десятичные)
        val_row = '  ' + '  '.join(f'{h:>4d}' for h in elements)
        lines.append(val_row)
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Мультипликативные орбиты (циклотомические классы)
# ---------------------------------------------------------------------------

def render_cyclotomic(color: bool = True) -> str:
    """
    Циклотомические классы: орбиты {h, h², h⁴, h⁸, h¹⁶, h³²} под x↦x².
    Каждый класс соответствует минимальному многочлену над GF(2).
    """
    classes = cyclotomic_classes()
    lines: list[str] = []
    lines.append('═' * 60)
    lines.append('  Циклотомические классы GF(2⁶) (орбиты под x ↦ x²)')
    lines.append('  Каждый класс ↔ неприводимый многочлен над GF(2)')
    lines.append('═' * 60)

    for k, cls in enumerate(classes):
        cls_sorted = sorted(cls)
        n = len(cls_sorted)
        label = f'  C{k+1:2d}  (|C|={n})  '
        pad = ' ' * len(label)

        glyphs = [render_glyph(h) for h in cls_sorted]

        for ri in range(3):
            parts: list[str] = []
            for gi, h in enumerate(cls_sorted):
                cell = glyphs[gi][ri]
                if color:
                    yc = yang_count(h)
                    cell = _YANG_ANSI[yc] + cell + _RESET
                parts.append(cell)
            prefix = label if ri == 1 else pad
            lines.append(prefix + ' '.join(parts))

        # Значения
        vals = '  ' + '  '.join(f'{h:>2d}' for h in cls_sorted)
        lines.append(pad + vals)
        lines.append('')

    lines.append(f'  Итого: {len(classes)} классов')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Подполя GF(2) ⊂ GF(4) ⊂ GF(8) ⊂ GF(64)
# ---------------------------------------------------------------------------

_SUBFIELD_SIZES = [1, 3, 7, 63]  # размеры мультипликативных групп подполей
_SUBFIELD_DEGREES = [1, 2, 3, 6]  # степени расширений

def render_subfields(color: bool = True) -> str:
    """Глифы элементов каждого подполя GF(2ᵈ) ⊂ GF(2⁶)."""
    lines: list[str] = []
    lines.append('═' * 60)
    lines.append('  Подполя GF(2ᵈ) ⊂ GF(2⁶)  (d | 6,  d ∈ {1,2,3,6})')
    lines.append('═' * 60)

    for deg in _SUBFIELD_DEGREES:
        elements = sorted(subfield_elements(deg))
        n = len(elements)
        label = f'  GF(2^{deg})  ({n} эл.)  '
        pad = ' ' * len(label)
        glyphs = [render_glyph(h) for h in elements]

        for ri in range(3):
            parts: list[str] = []
            for gi, h in enumerate(elements):
                cell = glyphs[gi][ri]
                if color:
                    yc = yang_count(h)
                    cell = _YANG_ANSI[yc] + cell + _RESET
                parts.append(cell)
            prefix = label if ri == 1 else pad
            lines.append(prefix + ' '.join(parts))

        vals = '  ' + '  '.join(f'{h:>2d}' for h in elements)
        lines.append(pad + vals)
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Следы Tr(h) = h + h² + h⁴ + h⁸ + h¹⁶ + h³²
# ---------------------------------------------------------------------------

def render_trace_partition(color: bool = True) -> str:
    """
    Разбиение 64 глифов на два класса по абсолютному следу:
      Tr(h) = 0 (32 элемента) и Tr(h) = 1 (32 элемента).
    """
    tr0 = sorted(h for h in range(64) if gf_trace(h) == 0)
    tr1 = sorted(h for h in range(64) if gf_trace(h) == 1)

    lines: list[str] = []
    lines.append('═' * 60)
    lines.append('  Абсолютный след Tr: GF(2⁶) → GF(2)')
    lines.append('  Tr(h) = h ⊕ h² ⊕ h⁴ ⊕ h⁸ ⊕ h¹⁶ ⊕ h³²')
    lines.append('═' * 60)

    for label_text, elements in [('Tr=0', tr0), ('Tr=1', tr1)]:
        lines.append(f'\n  {label_text}  ({len(elements)} элементов):')
        # По 16 в строке
        for row_start in range(0, len(elements), 16):
            chunk = elements[row_start:row_start + 16]
            for ri in range(3):
                parts: list[str] = []
                for h in chunk:
                    cell = render_glyph(h)[ri]
                    if color:
                        yc = yang_count(h)
                        cell = _YANG_ANSI[yc] + cell + _RESET
                    parts.append(cell)
                lines.append('    ' + ' '.join(parts))
            lines.append('')

    # Диаграмма Хассе с подсветкой Tr=1
    lines.append('  Диаграмма Хассе B₆ (подсветка = Tr=1):')
    tr1_set = set(tr1)
    lines.append(render_hasse_glyphs(color=color, highlights=tr1_set))

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 5. Таблица умножения (8×8 фрагмент)
# ---------------------------------------------------------------------------

def render_mul_table_glyphs(
    rows: list[int] | None = None,
    cols: list[int] | None = None,
    color: bool = True,
) -> str:
    """
    Таблица умножения GF(2⁶): строки × столбцы → произведение как глиф.
    По умолчанию: строки и столбцы = {g⁰..g⁷} (первые 8 ненулевых).
    """
    if rows is None:
        rows = [gf_exp(k) for k in range(8)]
    if cols is None:
        cols = [gf_exp(k) for k in range(8)]

    lines: list[str] = []
    lines.append('  Таблица умножения GF(2⁶)  (первые 8 ненулевых элементов):')

    # Заголовок столбцов
    hdr = '        ' + '   '.join(f'{c:>2d}' for c in cols)
    lines.append(hdr)
    lines.append('        ' + '─' * (len(cols) * 5))

    for r in rows:
        glyphs_row = [render_glyph(gf_mul(r, c)) for c in cols]
        for ri in range(3):
            parts: list[str] = []
            for ci, c in enumerate(cols):
                prod = gf_mul(r, c)
                cell = glyphs_row[ci][ri]
                if color:
                    yc = yang_count(prod)
                    cell = _YANG_ANSI[yc] + cell + _RESET
                parts.append(cell)
            prefix = f'  {r:>2d} │  ' if ri == 1 else '       '
            lines.append(prefix + '   '.join(parts))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 6. Порядки элементов
# ---------------------------------------------------------------------------

def render_order_groups(color: bool = True) -> str:
    """
    Группировка ненулевых элементов по мультипликативному порядку.
    ord(h) делит 63 = 7 × 9 = 7 × 3².
    Делители 63: 1, 3, 7, 9, 21, 63.
    """
    from math import gcd

    def element_order(h: int) -> int:
        if h == 0:
            return 0
        if h == 1:
            return 1
        k = gf_log(h)  # h = g^k
        # ord(h) = 63 / gcd(63, k)
        return ORDER // gcd(ORDER, k)

    lines: list[str] = []
    lines.append('═' * 60)
    lines.append('  Порядки элементов мультипликативной группы GF(2⁶)*')
    lines.append('  ord(h) делит 63 = 7 × 9')
    lines.append('═' * 60)

    from collections import defaultdict
    by_order: dict[int, list[int]] = defaultdict(list)
    for h in range(1, 64):
        by_order[element_order(h)].append(h)

    for ord_val in sorted(by_order):
        elems = sorted(by_order[ord_val])
        n = len(elems)
        label = f'  ord={ord_val:2d}  (φ({ord_val})={n})  '
        pad = ' ' * len(label)
        glyphs = [render_glyph(h) for h in elems]

        for ri in range(3):
            parts: list[str] = []
            for gi, h in enumerate(elems):
                cell = glyphs[gi][ri]
                if color:
                    yc = yang_count(h)
                    cell = _YANG_ANSI[yc] + cell + _RESET
                parts.append(cell)
            prefix = label if ri == 1 else pad
            lines.append(prefix + ' '.join(parts))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='gf2_glyphs — GF(2⁶) визуализация через глифы Q6'
    )
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('cycle',  help='Цикл примитивного элемента g=2 (63 глифа)')
    sub.add_parser('cyclo',  help='Циклотомические классы')
    sub.add_parser('fields', help='Подполя GF(2ᵈ) ⊂ GF(2⁶)')
    sub.add_parser('trace',  help='Разбиение по абсолютному следу')
    sub.add_parser('mul',    help='Таблица умножения (8×8)')
    sub.add_parser('orders', help='Порядки элементов')

    for p in sub.choices.values():
        p.add_argument('--no-color', action='store_true')

    args = parser.parse_args()
    color = not getattr(args, 'no_color', False)

    if args.cmd == 'cycle' or args.cmd is None:
        print(render_generator_cycle(color=color))

    elif args.cmd == 'cyclo':
        print(render_cyclotomic(color=color))

    elif args.cmd == 'fields':
        print(render_subfields(color=color))

    elif args.cmd == 'trace':
        print(render_trace_partition(color=color))

    elif args.cmd == 'mul':
        print(render_mul_table_glyphs(color=color))

    elif args.cmd == 'orders':
        print(render_order_groups(color=color))
