"""pack_glyphs — Упаковки замкнутых клеточных полей через глифы Q6.

Каждый глиф h (0..63) — клетка кольца Q6 периода P=64=2^6.
Алгоритм упаковки Германа: ring[n*(n-1)/2 % 64] = n.

Визуализация (8×8 сетка, Gray-код Q6):
  ring     — значение ring[h]: какое число стоит в клетке h
  antipode — антиподальные пары: ring[h]+ring[h⊕32]=65
  fixpoint — для каждого старта m: фиксированные точки
  packable — тест для произвольного P: является ли 2^k?
  magic    — магический квадрат из поля P=2^(2k)
  periods  — таблица периодов P(n) с маркировкой 2^k и простых

Команды CLI:
  ring
  antipode
  fixpoint [--start M]
  packable <P>
  magic <k>
  periods [--max N]
"""
from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexpack.hexpack import (
    PackedRing, MagicSquare, Q6_RING,
    period, valid_periods, prime_periods, _is_power_of_two,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_GRAY3 = [i ^ (i >> 1) for i in range(8)]

# ─── цвет по значению числа ────────────────────────────────────────────────────

def _val_color(v: int, P: int = 64) -> str:
    """Цвет клетки по значению v: первая/последняя четверть — особые."""
    if v == 1 or v == P:
        return '\033[1;93m'    # золотой — крайние числа
    if v <= P // 4:
        return _YANG_ANSI[1]   # нижняя четверть
    if v >= 3 * P // 4:
        return _YANG_ANSI[5]   # верхняя четверть
    yang = yang_count(v - 1)
    return _YANG_ANSI[yang]


def _header(title: str, subtitle: str = '') -> list[str]:
    lines = ['═' * 70, f'  {title}']
    if subtitle:
        lines.append(f'  {subtitle}')
    lines.append('═' * 70)
    col_hdr = '  '.join(f'{g:03b}' for g in _GRAY3)
    lines.append(f'        {col_hdr}')
    lines.append('        ' + '─' * len(col_hdr))
    return lines


# ─── 1. ring ───────────────────────────────────────────────────────────────────

def render_ring() -> list[str]:
    """Показать содержимое кольца Q6: ring[h] = число в клетке h."""
    lines = _header(
        'КОЛЬЦО Q6 — Алгоритм упаковки Германа  P=64=2⁶',
        'ring[h] = n  где  pos(n) = n(n-1)/2 mod 64'
    )
    ring = Q6_RING.as_list()
    for ri, row_g in enumerate(_GRAY3):
        row_parts = []
        for ci, col_g in enumerate(_GRAY3):
            h = (row_g << 3) | col_g
            v = ring[h]
            glyph = render_glyph(h)
            col = _val_color(v)
            row_parts.append(f'{col}{v:3d}{_RESET}')
        lines.append(f'  {ri:03b}│  ' + '  '.join(row_parts))
    lines.append('─' * 70)
    lines.append(f'  Теорема: полная упаковка без пробелов ↔ P = 2^k')
    lines.append(f'  P=64=2⁶ ✓   ring заполнен числами 1..64 без коллизий')
    lines.append(f'  Исключение (нет фикс. точек): старт m=32')
    return lines


# ─── 2. antipode ───────────────────────────────────────────────────────────────

def render_antipode() -> list[str]:
    """Показать антиподальные пары: ring[h] + ring[h⊕32] = 65."""
    lines = _header(
        'АНТИПОДАЛЬНЫЕ ПАРЫ  — Следствие 2 Германа',
        'ring[h] + ring[h ⊕ 32] = P+1 = 65  для всех h'
    )
    ring = Q6_RING.as_list()
    pairs = Q6_RING.antipodal_pairs()
    # Показать первые 16 пар компактно
    lines.append(f'  {"h":>4}  ring[h]  h⊕32  ring[h⊕32]  Сумма')
    lines.append(f'  {"─"*50}')
    for k, k2, v1, v2, s in pairs[:16]:
        yang1 = yang_count(k)
        yang2 = yang_count(k2)
        c1 = _YANG_ANSI[yang1]; c2 = _YANG_ANSI[yang2]
        lines.append(
            f'  {c1}{k:>4}{_RESET}   {v1:>5}    '
            f'{c2}{k2:>4}{_RESET}      {v2:>5}     '
            f'{_BOLD}{s:>5}{_RESET}'
        )
    lines.append(f'  ... (всего 32 пары)')
    lines.append('─' * 70)
    ok = Q6_RING.verify_antipodal()
    lines.append(f'  Все 32 пары суммируются в 65: {"✓" if ok else "✗"}')
    lines.append(f'  Q6-антипод: h ↔ h⊕63 (инверсия всех битов)')
    lines.append(f'  yang(h) + yang(h⊕63) = 6  всегда  (аналог Инь-Ян)')
    return lines


# ─── 3. fixpoint ───────────────────────────────────────────────────────────────

def render_fixpoint(start: int = 0) -> list[str]:
    """Показать фиксированные точки при нумерации с позиции start."""
    fps = Q6_RING.fixed_points(start)
    exc = Q6_RING.exceptional_start()
    lines = _header(
        f'ФИКСИРОВАННЫЕ ТОЧКИ  — Следствие 1 Германа',
        f'Нумерация с позиции m={start}: ring[(m+n-1) % 64] == n'
    )
    ring = Q6_RING.as_list()
    lines.append(f'  Старт m={start}:')
    if fps:
        for n in fps:
            pos = (start + n - 1) % 64
            yang = yang_count(pos)
            col = _YANG_ANSI[yang]
            lines.append(f'    {col}Клетка {pos:>2} (yang={yang}): число {n} = ordinal {n}{_RESET}  ← собственный номер')
    else:
        lines.append(f'    Нет фиксированных точек — это исключительный старт!')
    lines.append('')
    lines.append(f'  Исключительная позиция (Следствие 1): m={exc}')
    lines.append(f'  Для ВСЕХ других 63 стартов: ровно 1 фиксированная точка')
    lines.append('─' * 70)
    lines.append(f'  Аналог в Q6: h=0 → ровно одна гексаграмма "на своём месте"')
    return lines


# ─── 4. packable ───────────────────────────────────────────────────────────────

def render_packable(P: int) -> list[str]:
    """Проверить, упаковываемо ли поле периода P."""
    is_2k = _is_power_of_two(P)
    k = (P - 1).bit_length() if is_2k else -1
    lines = ['═' * 70,
             f'  ТЕСТ УПАКОВЫВАЕМОСТИ  P = {P}',
             '═' * 70]
    if is_2k:
        lines.append(f'  P = {P} = 2^{k}  ✓  ПОЛЕ УПАКОВЫВАЕМО')
        r = PackedRing(P)
        lines.append(f'  Проверка: все {P} позиций заняты без коллизий: {"✓" if r.packable else "✗"}')
        lines.append(f'  Исключительный старт: m={r.exceptional_start()}')
        lines.append(f'  Магический квадрат: {"доступен" if k % 2 == 0 else f"нет (k={k} нечётное)"}')
        if k > 0 and k % 2 == 0:
            ms = MagicSquare(k // 2)
            lines.append(f'  Размер квадрата: {ms.side}×{ms.side}, константа столбцов = {ms.magic_constant}')
    else:
        # Найти ближайшие 2^k
        k_lo = (P - 1).bit_length() - 1
        k_hi = k_lo + 1
        lines.append(f'  P = {P}  ✗  НЕ УПАКОВЫВАЕМО')
        lines.append(f'  Ближайшие допустимые: 2^{k_lo}={2**k_lo}, 2^{k_hi}={2**k_hi}')
    return lines


# ─── 5. magic ──────────────────────────────────────────────────────────────────

def render_magic(k: int) -> list[str]:
    """Показать магический квадрат из поля P = 2^(2k)."""
    P = 2 ** (2 * k)
    lines = ['═' * 70,
             f'  МАГИЧЕСКИЙ КВАДРАТ  P={P}=2^{2*k},  side={2**k}',
             f'  Сумма столбцов = (P+1)×side/2 = {(P+1)*2**k//2}',
             '═' * 70]
    try:
        ms = MagicSquare(k)
        lines.append(ms.format())
        lines.append('─' * 70)
        lines.append(f'  Суммы столбцов: {ms.column_sums()}')
        lines.append(f'  Магическая константа: {ms.magic_constant}  ✓' if ms.is_magic() else '  ✗')
    except Exception as e:
        lines.append(f'  Ошибка: {e}')
    return lines


# ─── 6. periods ────────────────────────────────────────────────────────────────

def render_periods(max_n: int = 64) -> list[str]:
    """Таблица периодов P(n) для n=1..max_n с маркировкой 2^k и простых."""
    from math import gcd as _gcd
    def is_prime(p):
        if p < 2: return False
        if p == 2: return True
        if p % 2 == 0: return False
        return all(p % i != 0 for i in range(3, int(p**0.5)+1, 2))

    lines = ['═' * 70,
             f'  ПЕРИОДЫ ПОЛЕЙ  P(n) = n(n-1)/2 + 1  для n=1..{max_n}',
             f'  ★ = P=2^k (упаковываемо)   • = P простое',
             '═' * 70]
    lines.append(f'  {"n":>4}  {"P(n)":>8}  {"Признак":>12}  {"yang(P mod 64)":>14}')
    lines.append(f'  {"─"*52}')
    for n in range(1, max_n + 1):
        P = period(n)
        marks = []
        if _is_power_of_two(P):
            k = P.bit_length() - 1
            col = _YANG_ANSI[5]
            marks.append(f'★ 2^{k}')
        if is_prime(P):
            col = '\033[1;92m' if not marks else col
            marks.append('• prime')
        col = '\033[0m' if not marks else col if marks else _YANG_ANSI[yang_count(P % 64)]
        mark_str = ', '.join(marks) if marks else ''
        yang_p = yang_count(P % 64)
        lines.append(f'  {n:>4}  {P:>8}  {mark_str:>12}  yang={yang_p}')
    return lines


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog='hexpack',
        description='Упаковки замкнутых клеточных полей (Герман) на Q6'
    )
    sub = p.add_subparsers(dest='cmd')

    sub.add_parser('ring',    help='Содержимое кольца P=64')
    sub.add_parser('antipode', help='Антиподальные пары: ring[h]+ring[h⊕32]=65')

    fp = sub.add_parser('fixpoint', help='Фиксированные точки')
    fp.add_argument('--start', type=int, default=0, metavar='M',
                    help='Начальная позиция нумерации (0..63)')

    pk = sub.add_parser('packable', help='Тест упаковываемости поля P')
    pk.add_argument('P', type=int, help='Период поля')

    mg = sub.add_parser('magic', help='Магический квадрат из P=2^(2k)')
    mg.add_argument('k', type=int, help='k: квадрат 2^k × 2^k (k=1→2×2, k=2→4×4)')

    pr = sub.add_parser('periods', help='Таблица периодов P(n)')
    pr.add_argument('--max', type=int, default=64, dest='max_n', metavar='N')

    args = p.parse_args(argv)

    dispatch = {
        'ring':     lambda: render_ring(),
        'antipode': lambda: render_antipode(),
        'fixpoint': lambda: render_fixpoint(args.start if hasattr(args, 'start') else 0),
        'packable': lambda: render_packable(args.P),
        'magic':    lambda: render_magic(args.k),
        'periods':  lambda: render_periods(args.max_n if hasattr(args, 'max_n') else 64),
    }

    if args.cmd not in dispatch:
        p.print_help()
        return 1

    for line in dispatch[args.cmd]():
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
