"""hexnumderiv/numderiv_glyphs.py — Q6 глифы через арифметическую производную.

Арифметическая производная ∂n определяется формулой:
    ∂1 = 1
    ∂p = 2 для простого p (1 + сумма_делителей = 1 + 1 = 2)
    ∂n = 1 + Σ{правильные делители n}  (общий случай)

Правильные делители числа n — это делители < n, включая 1.

Примеры для h ∈ 0..63:
    ∂1  = 1   ∂2  = 2   ∂3  = 2   ∂4  = 4 (совершенное: ∂4=4)
    ∂6  = 7   ∂12 = 16  ∂24 = 36

Классификация:
    «совершенные» (perfect):  ∂n = n    (фиксированная точка)
    «сверхсовершенные» (super): цепочка n → ∂n → ... растёт
    «обычные» (ordinary):     цепочка убывает или фиксируется

Правило Лейбница (для взаимно простых k, m):
    ∂(k·m) = k·∂m + m·∂k + ∂k·∂m

Для h = 0 нет смысла (0 не натуральное); берётся ∂(h+1) для h=0.

Визуализация (8×8, Gray-код Q6):
  deriv    — значение ∂h (нормировано для отображения)
  chain    — длина цепочки h → ∂h → ∂²h → ...
  classify — класс (P=perfect, S=super, O=ordinary)
  leibniz  — проверка правила Лейбница для пар (h, h⊕1)

Команды CLI:
  deriv
  chain
  classify
  leibniz
"""

from __future__ import annotations
import sys
import argparse
import math
from math import gcd

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexnumderiv.hexnumderiv import NumberDerivative
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

_ND = NumberDerivative()

_GRAY3 = [i ^ (i >> 1) for i in range(8)]


def _n(h: int) -> int:
    """Натуральное число, соответствующее глифу h (h=0 → n=64)."""
    return h if h > 0 else 64


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
# 1. Значение производной ∂h
# ---------------------------------------------------------------------------

def render_deriv(color: bool = True) -> str:
    """8×8 сетка: арифметическая производная ∂n(h).

    Ярлык: последняя десятичная цифра ∂n.
    Цвет: ян-слой yang_count(h).
    """
    _PERFECT_COLOR = '\033[38;5;226m'   # жёлтый = ∂n=n (совершенное)

    lines = _header(
        'NumDeriv: арифметическая производная ∂h',
        'Цифра = ∂h mod 10  (последняя цифра производной)',
    )

    deriv_vals: list[int] = []
    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h  = (row_g << 3) | col_g
            n  = _n(h)
            dn = _ND.derivative(n)
            deriv_vals.append((h, n, dn))
            is_perfect = (dn == n)
            sym = str(dn % 10)
            if color:
                c = '\033[38;5;226m' if is_perfect else _YANG_ANSI[yang_count(h)]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Замечательные значения ∂h:')
    perfect = [(h, n, dn) for h, n, dn in deriv_vals if dn == n]
    primes  = [(h, n, dn) for h, n, dn in deriv_vals if dn == 2 and n != 2]
    lines.append(f'  Простых h (∂h=2): {len(primes)}  '
                 f'({", ".join(str(h) for h,_,_ in primes[:8])}...)')
    lines.append(f'  Совершенных h (∂h=h): {len(perfect)}  '
                 f'({", ".join(str(h) for h,_,_ in perfect)})')

    lines.append('')
    lines.append('  Топ-10 по значению ∂h:')
    top10 = sorted(deriv_vals, key=lambda x: x[2], reverse=True)[:10]
    for h, n, dn in top10:
        c = _YANG_ANSI[yang_count(h)] if color else ''
        r = _RESET if color else ''
        lines.append(f'    h={h:2d}: {c}∂{n}={dn}{r}')

    lines.append('')
    lines.append('  Формула: ∂n = 1 + Σ{d | n, d < n}')
    lines.append('  Для простого p: ∂p = 1 + 1 = 2  (единств. правильный делитель = 1)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Длина цепочки
# ---------------------------------------------------------------------------

def render_chain(color: bool = True) -> str:
    """8×8 сетка: длина цепочки n → ∂n → ∂²n → ... → фикс.точка.

    Цепочка обрывается при фиксированной точке (∂^k n = ∂^(k+1) n) или
    при обнаружении цикла.
    """
    # Диапазон длин для нормировки
    chain_lens: list[tuple[int, int]] = []
    for h in range(64):
        n  = _n(h)
        ch = _ND.chain(n)
        chain_lens.append((h, len(ch)))

    max_len = max(l for _, l in chain_lens)

    lines = _header(
        'NumDeriv: длина цепочки h → ∂h → ∂²h → ...',
        f'Ярлык = длина (нормировано в 0..9)  max={max_len}',
    )

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h   = (row_g << 3) | col_g
            cl  = chain_lens[h][1]
            sym = str(min(9, cl - 1))   # -1 т.к. начало ∈ цепочке
            if color:
                # Цвет по длине (нормируем в 0..6)
                norm = min(6, int((cl - 1) * 6 / max(max_len - 1, 1)))
                c = _YANG_ANSI[norm]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Примеры цепочек:')
    for h in [1, 4, 6, 12, 24, 36, 48, 60]:
        if h >= 64:
            continue
        n  = _n(h)
        ch = _ND.chain(n)
        c  = _YANG_ANSI[yang_count(h)] if color else ''
        r  = _RESET if color else ''
        chain_str = ' → '.join(str(x) for x in ch[:7])
        if len(ch) > 7:
            chain_str += ' → ...'
        lines.append(f'    h={h:2d}: {c}{chain_str}{r}  (длина={len(ch)})')

    lines.append('')
    lines.append('  Цепочки либо стабилизируются (∂ⁿh = ∂ⁿ⁺¹h),')
    lines.append('  либо уходят в бесконечность (super-числа).')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Классификация
# ---------------------------------------------------------------------------

def render_classify(color: bool = True) -> str:
    """8×8 сетка: класс числа n(h).

    P = perfect (∂n = n)
    S = super    (цепочка возрастает)
    O = ordinary (цепочка убывает)
    """
    _P_COLOR = '\033[38;5;226m'   # жёлтый = perfect
    _S_COLOR = '\033[38;5;196m'   # красный = super
    _O_COLOR = '\033[38;5;238m'   # серый   = ordinary

    lines = _header(
        'NumDeriv: классификация P/S/O',
        'P=perfect(∂n=n,жёлт.)  S=super(возраст.,красн.)  O=ordinary(убыв.,серый)',
    )

    counts = {'P': 0, 'S': 0, 'O': 0}
    class_data: list[str] = []
    for h in range(64):
        n  = _n(h)
        cl = _ND.classify(n)
        sym = cl[0].upper()  # 'P', 'S', 'O'
        counts[sym] = counts.get(sym, 0) + 1
        class_data.append(sym)

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h   = (row_g << 3) | col_g
            sym = class_data[h]
            if color:
                c = {'P': _P_COLOR, 'S': _S_COLOR, 'O': _O_COLOR}[sym]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    for sym, label in [('P', 'perfect  (∂n=n)'),
                       ('S', 'super    (растёт)'),
                       ('O', 'ordinary (убывает)')]:
        c = {'P': _P_COLOR, 'S': _S_COLOR, 'O': _O_COLOR}[sym] if color else ''
        r = _RESET if color else ''
        lines.append(f'  {c}{sym}{r}: {counts.get(sym,0):3d} глифов  — {label}')

    lines.append('')
    lines.append('  Совершенные числа Нивена: ∂n = n')
    perfect = [h for h in range(64) if _ND.derivative(_n(h)) == _n(h)]
    for h in perfect:
        n  = _n(h)
        c  = '\033[38;5;226m' if color else ''
        r  = _RESET if color else ''
        lines.append(f'    h={h:2d}: {c}n={n} ∂n={_ND.derivative(n)}{r}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Правило Лейбница
# ---------------------------------------------------------------------------

def render_leibniz(color: bool = True) -> str:
    """8×8 сетка: выполняется ли правило Лейбница для пары (n(h), n(antipode(h))).

    ∂(k·m) = k·∂m + m·∂k + ∂k·∂m  (для взаимно простых k, m).

    Для h и h⊕63 (антипод): числа n(h) и n(antipode(h)) не всегда взаимно просты,
    поэтому вместо этого проверяем правило Лейбница для пар (1..6, 7..12):
    берём пары (h%6+1, h//6+1) если они взаимно просты, иначе '·'.
    """
    _OK_COLOR  = '\033[38;5;82m'    # зелёный = Лейбниц выполнен
    _NA_COLOR  = '\033[38;5;238m'   # серый   = пара не взаимно проста
    _ERR_COLOR = '\033[38;5;196m'   # красный = не выполнен (не должно быть)

    lines = _header(
        'NumDeriv: правило Лейбница ∂(k·m) = k·∂m + m·∂k + ∂k·∂m',
        'Y=выполнено(зел.)  ·=не взаимно просты  (k=h%7+1, m=h//7+1)',
    )

    results: list[str] = []
    for h in range(64):
        k = h % 7 + 1
        m = h // 7 + 1
        if gcd(k, m) != 1:
            results.append('·')
        else:
            try:
                res = _ND.leibniz_rule(k, m)
                results.append('Y' if res['holds'] else '!')
            except Exception:
                results.append('·')

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h   = (row_g << 3) | col_g
            sym = results[h]
            if color:
                c = {
                    'Y': _OK_COLOR,
                    '·': _NA_COLOR,
                    '!': _ERR_COLOR,
                }.get(sym, _NA_COLOR)
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    n_ok = results.count('Y')
    n_na = results.count('·')
    n_er = results.count('!')
    lines.append('')
    lines.append(f'  Правило Лейбница выполнено: {n_ok}/64')
    lines.append(f'  Пары не взаимно просты      : {n_na}/64')
    lines.append(f'  Нарушения (не должно быть)  : {n_er}/64')

    lines.append('')
    lines.append('  Примеры (взаимно простые пары):')
    ex = [(h, h % 7 + 1, h // 7 + 1) for h in range(64)
          if gcd(h % 7 + 1, h // 7 + 1) == 1][:5]
    for h, k, m in ex:
        res = _ND.leibniz_rule(k, m)
        c   = _OK_COLOR if color else ''
        r   = _RESET if color else ''
        lines.append(
            f'    h={h:2d}: k={k} m={m}  ∂(k·m)={res["lhs"]}  '
            f'rhs={res["rhs"]}  {c}{"✓" if res["holds"] else "✗"}{r}'
        )

    lines.append('')
    lines.append('  Формула: ∂(k·m) = k·∂m + m·∂k + ∂k·∂m  (gcd(k,m)=1)')
    lines.append('  Это аналог правила произведения d(fg) = f\'g + fg\'.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='numderiv_glyphs',
        description='Q6 глифы через арифметическую производную ∂n.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('deriv',    help='значение ∂h для каждого глифа')
    sub.add_parser('chain',    help='длина цепочки h→∂h→∂²h→...')
    sub.add_parser('classify', help='класс P/S/O (perfect/super/ordinary)')
    sub.add_parser('leibniz',  help='правило Лейбница для пар')
    args = p.parse_args(argv)
    color = not args.no_color

    dispatch = {
        'deriv':    render_deriv,
        'chain':    render_chain,
        'classify': render_classify,
        'leibniz':  render_leibniz,
    }
    if args.cmd in dispatch:
        print(dispatch[args.cmd](color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
