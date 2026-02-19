"""unique_glyphs — Порядки групп с единственностью через глифы Q6.

Каждый глиф h (0..63) интерпретируется как порядок группы.
Классификация UVW по Герману:
  E — чётное: h ≡ 0 (mod 2)
  U — нечётное, h ≡ 1 (mod 6): самые «простые» нечётные
  V — нечётное, h ≡ 3 (mod 6): «тройные»
  W — нечётное, h ≡ 5 (mod 6): «пятёрочные»

Порядки с единственной группой (unique-order):
  n — unique-order, если ровно одна группа порядка n (с точн. до изоморфизма).
  Среди 0..63: {15, 33, 35, 51, 65→нет, 69→нет} → {15, 33, 35, 51}

Теорема: n — unique-order ⟺ n = p₁^a₁ · ... · pₖ^aₖ и gcd(n, φ(n)) = 1.

Визуализация:
  uvw      — UVW-классификация глифов 0..63
  unique   — выделить unique-order числа среди 0..63
  coprime  — gcd(h, φ(h)) = 1 проверка (условие единственности)
  factors  — число простых делителей каждого глифа

Команды CLI:
  uvw
  unique
  coprime
  factors
"""

from __future__ import annotations
import sys
import argparse
import math
from functools import reduce

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexuniqgrp.hexuniqgrp import (
    classify_uvw, uvw_product, UniqueGroups,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_UVW_ANSI = {
    'E': '\033[38;5;238m',   # серый  — чётные
    'U': '\033[38;5;27m',    # синий  — U
    'V': '\033[38;5;82m',    # зелёный — V
    'W': '\033[38;5;196m',   # красный — W
}
_UVW_BG = {
    'E': '\033[48;5;238m',
    'U': '\033[48;5;27m',
    'V': '\033[48;5;82m',
    'W': '\033[48;5;196m',
}

_ug = UniqueGroups()


def _euler_phi(n: int) -> int:
    """Функция Эйлера φ(n)."""
    if n <= 0:
        return 0
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def _prime_factors(n: int) -> list[int]:
    """Простые делители n (с повторениями)."""
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


def _unique_order_check(n: int) -> bool:
    """Проверить, является ли n unique-order: gcd(n, φ(n)) = 1."""
    if n <= 1:
        return True
    phi = _euler_phi(n)
    return math.gcd(n, phi) == 1


# ---------------------------------------------------------------------------
# 1. UVW-классификация
# ---------------------------------------------------------------------------

def render_uvw(color: bool = True) -> str:
    """
    8×8 сетка: раскраска глифов по UVW-классификации.

    E (even/чётный): h mod 2 = 0  — серый
    U (нечётный, ≡1 mod 6): синий
    V (нечётный, ≡3 mod 6): зелёный
    W (нечётный, ≡5 mod 6): красный
    """
    from collections import Counter
    uvw_all = [classify_uvw(h) for h in range(64)]
    uvw_dist = Counter(uvw_all)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  UVW-классификация Германа для чисел 0..63')
    lines.append('  E=чётные  U=нечётные≡1(mod6)  V=нечётные≡3(mod6)  W=нечётные≡5(mod6)')
    lines.append(f'  E={uvw_dist["E"]}  U={uvw_dist["U"]}  V={uvw_dist["V"]}  W={uvw_dist["W"]}  (сумма=64)')
    lines.append('  Таблица умножения: U·U=U, W·W=U, U·W=W, V·V=V, E·X=E')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            uvw = uvw_all[h]
            rows3 = render_glyph(h)
            if color:
                c = _UVW_ANSI.get(uvw, _YANG_ANSI[0])
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            uvw = uvw_all[h]
            if color:
                c = _UVW_ANSI.get(uvw, _YANG_ANSI[0])
                lbl.append(f'{c}{uvw}{_RESET}')
            else:
                lbl.append(uvw)
        lines.append('  ' + '    '.join(lbl))
        lines.append('')

    # Таблица умножения UVW
    types = ['U', 'V', 'W', 'E']
    lines.append('  Таблица умножения UVW:')
    header = '     ' + '  '.join(f'{t:2s}' for t in types)
    if color:
        lines.append(f'  {_YANG_ANSI[2]}{header}{_RESET}')
    else:
        lines.append(f'  {header}')
    for t1 in types:
        row_str = f'  {t1}  | '
        for t2 in types:
            prod = uvw_product(t1, t2)
            if color:
                c = _UVW_ANSI.get(prod, _YANG_ANSI[0])
                row_str += f'{c}{prod:2s}{_RESET}  '
            else:
                row_str += f'{prod:2s}  '
        lines.append(row_str)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Unique-order числа
# ---------------------------------------------------------------------------

def render_unique(color: bool = True) -> str:
    """
    8×8 сетка: выделить unique-order числа среди 0..63.

    n — unique-order, если существует ровно одна группа порядка n.
    Условие: gcd(n, φ(n)) = 1 (теорема Сциэффера-Холла).
    """
    unique_64 = [h for h in range(64) if _unique_order_check(h) and h > 1]
    not_unique = [h for h in range(2, 64) if not _unique_order_check(h)]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Unique-order числа среди 0..63: единственная группа порядка n')
    lines.append('  Теорема: n — unique-order ⟺ gcd(n, φ(n)) = 1')
    lines.append(f'  Unique-order в [2..63]: {sorted(unique_64)}')
    lines.append(f'  Всего unique-order: {len(unique_64)}')
    lines.append('  Жирный = unique-order  (только одна группа)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            is_unique = (h in unique_64)
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_BG[yc] + _BOLD if is_unique else _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            is_unique = (h in unique_64)
            uvw = classify_uvw(h)
            if color:
                c = _UVW_ANSI.get(uvw, _YANG_ANSI[0])
                lbl.append(f'{c}{"U!" if is_unique else uvw+" "}{_RESET}')
            else:
                lbl.append('U!' if is_unique else uvw + ' ')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Первые 10 unique-order
    lines.append('  Первые unique-order числа и их свойства:')
    for h in sorted(unique_64)[:10]:
        phi = _euler_phi(h)
        g = math.gcd(h, phi)
        factors = _prime_factors(h)
        uvw = classify_uvw(h)
        if color:
            c = _UVW_ANSI.get(uvw, _YANG_ANSI[2])
            lines.append(f'  {c}  n={h:2d}: φ(n)={phi:2d}  gcd={g}'
                         f'  факторы={factors}  UVW={uvw}{_RESET}')
        else:
            lines.append(f'    n={h:2d}: φ(n)={phi:2d}  gcd={g}'
                         f'  факторы={factors}  UVW={uvw}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. gcd(h, φ(h))
# ---------------------------------------------------------------------------

def render_coprime(color: bool = True) -> str:
    """
    8×8 сетка: значение gcd(h, φ(h)) для каждого глифа.

    gcd(h, φ(h)) = 1 ↔ unique-order (одна группа).
    gcd(h, φ(h)) > 1 → несколько неизоморфных групп порядка h.
    """
    gcds = [math.gcd(h, _euler_phi(h)) if h > 1 else 0 for h in range(64)]
    max_gcd = max(gcds)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  gcd(h, φ(h)): мера «неединственности» группы порядка h')
    lines.append('  gcd=1 → unique-order (только Z_h)')
    lines.append('  gcd>1 → есть несколько групп (напр., Z_n и Z_p ⋊ Z_q)')
    lines.append(f'  Максимальный gcd в 0..63: {max_gcd}')
    lines.append('  Цвет = yang_count(h),  жирный = gcd=1 (unique)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            g = gcds[h]
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                is_unique = (g == 1 and h > 1)
                c = _YANG_BG[yc] + _BOLD if is_unique else _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            g = gcds[h]
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}g{g:2d}{_RESET}')
            else:
                lbl.append(f'g{g:2d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Распределение gcd
    from collections import Counter
    gcd_dist = Counter(gcds[h] for h in range(2, 64))
    lines.append('  Распределение gcd(h,φ(h)) для h=2..63:')
    for g_val in sorted(gcd_dist)[:8]:
        cnt = gcd_dist[g_val]
        tag = ' ← unique-order!' if g_val == 1 else ''
        if color:
            c = _YANG_ANSI[min(g_val, 6)]
            lines.append(f'  {c}  gcd={g_val:3d}: {cnt:2d} чисел{tag}{_RESET}')
        else:
            lines.append(f'    gcd={g_val:3d}: {cnt:2d} чисел{tag}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Число простых множителей
# ---------------------------------------------------------------------------

def render_factors(color: bool = True) -> str:
    """
    8×8 сетка: раскраска по числу простых делителей ω(h).

    ω(h) = число различных простых делителей h.
    Unique-order: обычно малое ω (произведение попарно взаимно простых).
    """
    def omega(n: int) -> int:
        if n <= 1:
            return 0
        return len(set(_prime_factors(n)))

    omegas = [omega(h) for h in range(64)]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Число различных простых делителей ω(h) для h=0..63')
    lines.append('  ω(h) = |{p prime: p|h}|')
    lines.append('  Unique-order часто имеют ω=2 (два разных простых делителя)')
    lines.append('  Цвет = ω(h) (0..6)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            om = omegas[h]
            rows3 = render_glyph(h)
            if color:
                c = _YANG_ANSI[om]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            om = omegas[h]
            uvw = classify_uvw(h)
            is_unique = _unique_order_check(h) and h > 1
            if color:
                c = _YANG_ANSI[om]
                lbl.append(f'{c}{"U!" if is_unique else "ω="+str(om)}{_RESET}')
            else:
                lbl.append('U!' if is_unique else f'ω={om}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Список unique-order + их ω
    unique_64 = sorted(h for h in range(2, 64) if _unique_order_check(h))
    lines.append('  Unique-order числа 2..63:')
    for h in unique_64:
        om = omega(h)
        factors = _prime_factors(h)
        uvw = classify_uvw(h)
        if color:
            c = _UVW_ANSI.get(uvw, _YANG_ANSI[2])
            lines.append(f'  {c}  {h:2d} = {" · ".join(str(f) for f in factors)}'
                         f'   ω={om}   UVW={uvw}{_RESET}')
        else:
            lines.append(f'    {h:2d} = {" · ".join(str(f) for f in factors)}'
                         f'   ω={om}   UVW={uvw}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='unique_glyphs',
        description='Порядки групп с единственностью (unique-order) через глифы',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('uvw',     help='UVW-классификация чисел 0..63')
    sub.add_parser('unique',  help='выделить unique-order числа')
    sub.add_parser('coprime', help='gcd(h, φ(h)): мера неединственности')
    sub.add_parser('factors', help='ω(h) = число различных простых делителей')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'uvw':
        print(render_uvw(color))
    elif args.cmd == 'unique':
        print(render_unique(color))
    elif args.cmd == 'coprime':
        print(render_coprime(color))
    elif args.cmd == 'factors':
        print(render_factors(color))


if __name__ == '__main__':
    main()
