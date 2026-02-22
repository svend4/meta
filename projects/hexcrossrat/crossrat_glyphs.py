"""hexcrossrat/crossrat_glyphs.py — Q6 глифы через группу перекрёстных отношений R6.

Группа R6 состоит из 6 преобразований перекрёстного отношения (Мёбиуса):
    r₀ = w           r₁ = 1/w
    r₂ = 1 − w       r₃ = 1/(1−w)
    r₄ = (w−1)/w     r₅ = w/(w−1)

Эта группа изоморфна S₃ (симметрическая группа на 3 элементах).

Инварианты (выполняются для любого w ≠ 0, 1):
    Σ rᵢ = 3   (сумма)
    Π rᵢ = 1   (произведение)

Связь с Q6: отображение yang_count(h) → rᵢ.
    ян=0 → r₀ = w        ян=1 → r₁ = 1/w
    ян=2 → r₂ = 1−w      ян=3 → r₃ = 1/(1−w)
    ян=4 → r₄ = (w−1)/w  ян=5 → r₅ = w/(w−1)
    ян=6 → r₀ (цикл)

Золотое сечение φ = (1+√5)/2 ≈ 1.618 особо красиво:
    r₀(φ) = φ      r₁(φ) = 1/φ = φ−1
    r₂(φ) = 1−φ = −1/φ   r₃(φ) = −φ
    r₄(φ) = (φ−1)/φ = 1/φ²   r₅(φ) = φ² = φ+1

Визуализация (8×8, Gray-код Q6):
  elements  [--w val]  — значение rᵢ = r_{yang}(w) для каждого глифа
  compose   [--w val]  — индекс rᵢ ∘ r_j для пар (h, antipode(h))
  identities [--w val] — верификация суммы и произведения по слоям
  s3map     [--w val]  — перестановка S₃ для каждого глифа

Команды CLI:
  elements  [--w val]
  compose   [--w val]
  identities [--w val]
  s3map     [--w val]
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexcrossrat.hexcrossrat import CrossRatioGroup, cross_ratio
from libs.hexcore.hexcore import yang_count, antipode
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_PHI = (1.0 + math.sqrt(5.0)) / 2.0   # φ ≈ 1.618

_GRAY3 = [i ^ (i >> 1) for i in range(8)]

# Имена элементов R6
_ELEM_NAMES = ['w', '1/w', '1−w', '1/(1−w)', '(w−1)/w', 'w/(w−1)']
_ELEM_LATEX = ['r₀', 'r₁', 'r₂', 'r₃', 'r₄', 'r₅']

# Цвета для 6 элементов R6 (ян 0..5, ян=6 → r₀ снова)
_R_COLOR = [
    '\033[38;5;226m',   # r₀ жёлтый
    '\033[38;5;39m',    # r₁ голубой
    '\033[38;5;82m',    # r₂ зелёный
    '\033[38;5;208m',   # r₃ оранжевый
    '\033[38;5;196m',   # r₄ красный
    '\033[38;5;200m',   # r₅ розовый
]

# Символ каждого элемента (1 буква)
_ELEM_SYM = ['0', '1', '2', '3', '4', '5']


def _yang_to_r(k: int) -> int:
    """yang 0..6 → индекс r (0..5, yang=6 → 0)."""
    return k % 6


def _make_group(w: float) -> CrossRatioGroup:
    return CrossRatioGroup(w)


def _header(title: str, subtitle: str = '') -> list[str]:
    lines = ['═' * 66, f'  {title}']
    if subtitle:
        lines.append(f'  {subtitle}')
    lines.append('═' * 66)
    col_hdr = '  '.join(format(g, '03b') for g in _GRAY3)
    lines.append(f'        {col_hdr}')
    lines.append('        ' + '─' * len(col_hdr))
    return lines


# ---------------------------------------------------------------------------
# 1. Значения элементов R6
# ---------------------------------------------------------------------------

def render_elements(w: float = _PHI, color: bool = True) -> str:
    """8×8 сетка: ярлык = индекс элемента R6 = yang_count(h) % 6.

    При w = φ элементы R6 суть степени золотого сечения:
      r₀=φ  r₁=1/φ  r₂=−1/φ  r₃=−φ  r₄=1/φ²  r₅=φ²
    """
    grp = _make_group(w)
    elems = grp.elements()   # [r₀, r₁, r₂, r₃, r₄, r₅]

    lines = _header(
        f'R6: элементы группы перекрёстного отношения  (w={w:.4f})',
        'Цифра = индекс rᵢ = yang%6  Цвет = элемент',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            ri = _yang_to_r(k)
            sym = _ELEM_SYM[ri]
            if color:
                c = _R_COLOR[ri]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  R6-группа для w = {w:.6f}:')
    for i, (name, lx, val) in enumerate(zip(_ELEM_NAMES, _ELEM_LATEX, elems)):
        c  = _R_COLOR[i] if color else ''
        r  = _RESET if color else ''
        cnt = sum(1 for h in range(64) if _yang_to_r(yang_count(h)) == i)
        lines.append(
            f'  {c}{lx} = {name:12s} = {float(val.real):+.6f}'
            f'{(f"  {float(val.imag):+.6f}i" if abs(val.imag) > 1e-9 else ""):18s}'
            f'  глифов={cnt}{r}'
        )

    lines.append('')
    sv = grp.verify_sum_identity()
    pv = grp.verify_product_identity()
    ok_s = '✓' if sv['ok'] else '✗'
    ok_p = '✓' if pv['ok'] else '✗'
    lines.append(f'  Σ rᵢ = {sv["sum"]:.6f}  (ожидалось 3)  {ok_s}')
    lines.append(f'  Π rᵢ = {pv["product"]:.6f}  (ожидалось 1)  {ok_p}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Композиция r_i ∘ r_{antipode_j}
# ---------------------------------------------------------------------------

def render_compose(w: float = _PHI, color: bool = True) -> str:
    """8×8 сетка: индекс rᵢ ∘ rⱼ, где i=yang(h), j=yang(antipode(h)).

    Антипод(h) = h XOR 63, yang(antipod(h)) = 6 − yang(h).
    Таким образом, j = (6 − k) % 6  для k = yang(h).
    """
    grp = _make_group(w)

    lines = _header(
        f'R6: композиция r_{{yang(h)}} ∘ r_{{yang(antipode(h))}}  (w={w:.4f})',
        'Цифра = индекс r при ri ∘ rj  (i+j по таблице Кэли)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            ri = _yang_to_r(k)
            rj = _yang_to_r(6 - k)
            rc = grp.multiply(ri, rj)
            sym = _ELEM_SYM[rc]
            if color:
                c = _R_COLOR[rc]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Таблица Кэли R6 (rᵢ ∘ rⱼ):')
    ct = grp.cayley_table()
    hdr = '      ' + '  '.join(f'r{j}' for j in range(6))
    lines.append(hdr)
    for i, row in enumerate(ct):
        c  = _R_COLOR[i] if color else ''
        r  = _RESET if color else ''
        row_str = '  '.join(
            f'{(_R_COLOR[v] if color else "")}{v}{(_RESET if color else "")}'
            for v in row
        )
        lines.append(f'  {c}r{i}{r}│ {row_str}')
    lines.append('')
    lines.append('  Группа R6 ≅ S₃ (симметрическая группа трёх элементов).')
    lines.append(f'  Порядок группы: 6 = 3!')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Инварианты суммы и произведения
# ---------------------------------------------------------------------------

def render_identities(w: float = _PHI, color: bool = True) -> str:
    """8×8 сетка: частичные суммы r₀..r_{yang(h)} и произведения.

    Ярлык: цифра суммы частичных элементов R6 по ян-слоям (1 знак).
    """
    grp  = _make_group(w)
    elems = grp.elements()

    # Накопленные суммы по rᵢ (до yang=6 → 0..5)
    partial_sums: list[complex] = []
    acc = complex(0.0)
    for i in range(7):
        acc += elems[_yang_to_r(i)]
        partial_sums.append(acc)

    # Частичные произведения
    partial_prods: list[complex] = []
    acc = complex(1.0)
    for i in range(7):
        acc *= elems[_yang_to_r(i)]
        partial_prods.append(acc)

    lines = _header(
        f'R6: инварианты суммы и произведения  (w={w:.4f})',
        'Σrᵢ=3  Πrᵢ=1  Показана частичная сумма по ян-слоям (1 знак реальной части)',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            k = yang_count(h)
            ps = partial_sums[k]
            # Ярлык: знак + целая часть реального компонента
            rv = ps.real
            sym = f'{int(abs(rv)):1d}' if abs(rv) < 10 else '9'
            if color:
                c = _YANG_ANSI[k]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Частичные суммы r₀ + r₁ + ... + rₖ (для yang=k):')
    for k in range(7):
        ri  = _yang_to_r(k)
        val = elems[ri]
        ps  = partial_sums[k]
        c   = _YANG_ANSI[k] if color else ''
        r_  = _RESET if color else ''
        im_str = f'  +{ps.imag:+.4f}i' if abs(ps.imag) > 1e-9 else ''
        lines.append(
            f'  ян={k} r{ri}={float(val.real):+.4f}: '
            f'{c}Σ={float(ps.real):+.6f}{im_str}{r_}'
        )

    sv = grp.verify_sum_identity()
    pv = grp.verify_product_identity()
    lines.append('')
    lines.append(f'  ИТОГ: Σ = {sv["sum"]:.6f}  (= 3?) {"✓" if sv["ok"] else "✗"}')
    lines.append(f'  ИТОГ: Π = {pv["product"]:.6f}  (= 1?) {"✓" if pv["ok"] else "✗"}')
    lines.append('')
    dec = grp.s4_decomposition()
    lines.append('  S4-декомпозиция:')
    for ln in dec.splitlines():
        lines.append(f'    {ln}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Изоморфизм с S₃
# ---------------------------------------------------------------------------

def render_s3map(w: float = _PHI, color: bool = True) -> str:
    """8×8 сетка: перестановка S₃, соответствующая r_{yang(h)}.

    S₃ = {e, (12), (13), (23), (123), (132)}.
    Ярлык — индекс перестановки (0..5).
    """
    grp = _make_group(w)
    iso = grp.isomorphism_to_s3()   # dict {r0: perm, r1: perm, ...}

    # iso keys: 'r0'..'r5'
    perms = [iso.get(f'r{i}', (0, 1, 2)) for i in range(6)]

    lines = _header(
        f'R6 ≅ S₃: изоморфизм R6 → S₃  (w={w:.4f})',
        'Ярлык = индекс (0..5) перестановки σ в S₃',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            k  = yang_count(h)
            ri = _yang_to_r(k)
            sym = str(ri)
            if color:
                c = _R_COLOR[ri]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Изоморфизм R6(w={w:.4f}) → S₃:')
    for i in range(6):
        perm = perms[i]
        perm_str = '(' + ' '.join(str(x) for x in perm) + ')'
        c = _R_COLOR[i] if color else ''
        r = _RESET if color else ''
        name = _ELEM_NAMES[i]
        lines.append(
            f'  {c}r{i} = {name:12s} ↦ {perm_str}{r}'
        )

    lines.append('')
    lines.append('  Структура S₃:')
    lines.append('    e=(0,1,2)  (01)  (02)  (12)  (012)  (021)')
    lines.append('  |S₃| = 6 = 3!  (симметрии треугольника)')
    lines.append('  R6 = группа Мёбиуса 6-го порядка, действующая на ℙ¹.')

    # Klein four-group
    kfg = grp.klein_four_group()
    lines.append('')
    lines.append('  Группа Кляйна V₄ ⊂ S₄ (сохраняет w):')
    for perm in kfg:
        lines.append(f'    {perm}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='crossrat_glyphs',
        description='Q6 глифы через группу перекрёстных отношений R6.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')

    for name, hlp in [
        ('elements',   'значения rᵢ = r_{yang}(w)'),
        ('compose',    'таблица Кэли: rᵢ ∘ rⱼ для пар (h, antipode)'),
        ('identities', 'инварианты Σrᵢ=3 и Πrᵢ=1'),
        ('s3map',      'изоморфизм R6 → S₃'),
    ]:
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument('--w', type=float, default=_PHI, metavar='W',
                        help=f'базовое значение w (умолч. φ≈{_PHI:.4f})')

    args = p.parse_args(argv)
    color = not args.no_color
    w = getattr(args, 'w', _PHI)

    dispatch = {
        'elements':   lambda c: render_elements(w, c),
        'compose':    lambda c: render_compose(w, c),
        'identities': lambda c: render_identities(w, c),
        's3map':      lambda c: render_s3map(w, c),
    }
    if args.cmd in dispatch:
        print(dispatch[args.cmd](color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
