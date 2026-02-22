"""order_glyphs — Булева решётка B₆ через систему глифов Q6.

Каждый глиф (0..63) — подмножество {0,...,5}, соответствующее элементу
булевой решётки B₆ с частичным порядком ⊆ (включение подмножеств).

Структура решётки B₆:
  • 7 уровней ранга 0..6  (размеры C(6,k): 1,6,15,20,15,6,1)
  • Нижний элемент: 0 = ∅,  верхний: 63 = {0,...,5}
  • Функция Мёбиуса: μ(x,y) = (−1)^{rank(y)−rank(x)} (булева решётка)
  • Характеристический полином: p(t) = Π_{k=0}^{5} (t − 2^k)

Визуализация:
  • layers  — 7 уровней решётки, каждый глиф раскрашен по рангу
  • mobius  — функция Мёбиуса μ(0, x) = (−1)^{rank(x)} для каждого глифа
  • chains  — наибольшие цепи 0 = x₀ ⊂ x₁ ⊂ ... ⊂ x₆ = 63 (6! = 720 цепей)
  • hasse   — диаграмма Хассе с глифами узлов (уровни 0..6)

Команды CLI:
  layers   — сетка 7 уровней B₆
  mobius   — значения μ(0,x) по всем глифам
  chains   — подсчёт и примеры максимальных цепей
  hasse    — диаграмма Хассе с раскраской по рангу
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexlat.hexlat import (
    rank, mobius, covers, hasse_edges,
    maximal_chains, whitney_numbers,
    characteristic_polynomial,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Уровень ранга = yang_count (число единичных битов = |подмножество|)
# Цвет уровня = _YANG_ANSI[rank]
# ---------------------------------------------------------------------------


def _rank_color(r: int, highlight: bool = False) -> str:
    c = _YANG_ANSI[r]
    return (_BOLD + _YANG_BG[r]) if highlight else c


# ---------------------------------------------------------------------------
# 1. Уровни решётки
# ---------------------------------------------------------------------------

def render_layers(color: bool = True) -> str:
    """
    Показать все 64 глифа, сгруппированных по 7 уровням ранга.

    Уровень 0: {∅} — 1 глиф.
    Уровень 6: {{0,1,2,3,4,5}} — 1 глиф.
    Уровень k: все k-элементные подмножества {0,...,5} — C(6,k) глифов.
    """
    wn = whitney_numbers()   # [1, 6, 15, 20, 15, 6, 1]
    cp = characteristic_polynomial()  # коэфф. x^6 - 21x^5 + ...

    lines: list[str] = []
    lines.append('╔' + '═' * 62 + '╗')
    lines.append('║  Булева решётка B₆: 7 уровней ранга' + ' ' * 25 + '║')
    lines.append('║  rank(x) = |x| = popcount(x)   μ(0,x) = (−1)^rank(x)' + ' ' * 8 + '║')
    lines.append('╚' + '═' * 62 + '╝')
    lines.append('')

    for r in range(7):
        elems = [h for h in range(64) if yang_count(h) == r]
        mu_val = (-1) ** r
        lines.append(f'  Ранг {r}:  {len(elems)} элементов   '
                     f'μ(0,·)={mu_val:+d}   '
                     f'W_{r}=C(6,{r})={wn[r]}')

        # Глифы горизонтально
        cols = len(elems)
        glyph_rows: list[list[str]] = [render_glyph(h) for h in elems]
        if color:
            is_extreme = (r == 0 or r == 6)
            c = (_YANG_BG[r] + _BOLD) if is_extreme else _YANG_ANSI[r]
            glyph_rows = [[c + row + _RESET for row in g] for g in glyph_rows]

        for ri in range(3):
            lines.append('    ' + '  '.join(g[ri] for g in glyph_rows))
        # Подпись: двоичное представление
        lbl = []
        for h in elems:
            bits = format(h, '06b')
            if color:
                c = _YANG_ANSI[r]
                lbl.append(f'{c}{bits}{_RESET}')
            else:
                lbl.append(bits)
        lines.append('    ' + '  '.join(lbl))
        lines.append('')

    # Числа Уитни
    lines.append('  Числа Уитни W_k = C(6,k):  ' + ', '.join(str(w) for w in wn))
    lines.append('  Σ W_k = 64   (все элементы B₆)')
    lines.append('')
    lines.append('  Характеристический полином p(t) = Π(t − 2^k) для k=0..5:')
    cp_str = '  p(t) = ' + '·'.join(f'(t−{2**k})' for k in range(6))
    lines.append(cp_str)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Функция Мёбиуса μ(0, x)
# ---------------------------------------------------------------------------

def render_mobius(color: bool = True) -> str:
    """
    8×8 сетка глифов, раскрашенных по μ(0, x) ∈ {−1, 0, +1}.

    Для булевой решётки: μ(0, x) = (−1)^{rank(x)}.
    Нечётные ранги → −1 (синий), чётные → +1 (красный), ранг 0 → +1.
    """
    lines: list[str] = []
    lines.append('═' * 64)
    lines.append('  Функция Мёбиуса μ(0, x) на B₆')
    lines.append('  μ(0, x) = (−1)^{rank(x)} ∈ {−1, +1}')
    lines.append('  Синий = +1,  красный = −1')
    lines.append('═' * 64)

    _MU_COLORS = {
        +1: _YANG_ANSI[2],   # голубой
        -1: _YANG_ANSI[5],   # красный
    }

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            r = yang_count(h)
            mu_val = (-1) ** r
            rows3 = render_glyph(h)
            if color:
                is_extreme = (r == 0 or r == 6)
                c = (_YANG_BG[r] + _BOLD) if is_extreme else _MU_COLORS[mu_val]
                rows3 = [c + row_s + _RESET for row_s in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            r = yang_count(h)
            mu_val = (-1) ** r
            if color:
                c = _MU_COLORS[mu_val]
                lbl.append(f'{c}μ={mu_val:+d}(r{r}){_RESET}')
            else:
                lbl.append(f'μ={mu_val:+d}(r{r})')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Формулы инверсии Мёбиуса
    lines.append('  Формула инверсии Мёбиуса: если g(x) = Σ_{y≤x} f(y),')
    lines.append('  то f(x) = Σ_{y≤x} μ(y,x) g(y)')
    lines.append('')
    lines.append('  Проверка: Σ_{x} μ(0,x) = ?')
    total = sum((-1) ** yang_count(h) for h in range(64))
    lines.append(f'  Σ μ(0,x) = {total}   [= 0 — принцип включений-исключений]')
    lines.append('')

    # Числа Эйлера: Σ по уровням
    lines.append('  По уровням:')
    for r in range(7):
        cnt = math.comb(6, r)
        mu = (-1) ** r
        contrib = cnt * mu
        if color:
            c = _YANG_ANSI[r]
            lines.append(f'  {c}  ранг {r}: C(6,{r})={cnt:2d}  ×  μ={mu:+d}  = {contrib:+3d}{_RESET}')
        else:
            lines.append(f'    ранг {r}: C(6,{r})={cnt:2d}  ×  μ={mu:+d}  = {contrib:+3d}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Максимальные цепи
# ---------------------------------------------------------------------------

def render_chains(color: bool = True) -> str:
    """
    Максимальные цепи 0 = x₀ ⊂ x₁ ⊂ ... ⊂ x₆ = 63 в B₆.

    Число таких цепей = 6! = 720 (каждый шаг добавляет ровно 1 элемент).
    Каждая цепь соответствует перестановке {0,...,5} (порядку добавления элементов).
    """
    all_chains = maximal_chains()
    n_chains = len(all_chains)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append('  Максимальные цепи в B₆')
    lines.append(f'  Число цепей: {n_chains} = 6! = 720')
    lines.append('  Каждая цепь = перестановка {0,...,5}')
    lines.append('═' * 64)
    lines.append('')

    # Показываем 6 первых цепей
    lines.append(f'  Первые 6 цепей (из {n_chains}):')
    lines.append('')

    for idx, chain in enumerate(all_chains[:6]):
        # Цепь — список из 7 элементов: 0, ..., 63
        glyphs = [render_glyph(h) for h in chain]
        if color:
            colored = []
            for i, h in enumerate(chain):
                r = yang_count(h)
                c = _YANG_ANSI[r]
                colored.append([c + row + _RESET for row in glyphs[i]])
            glyphs = colored

        lines.append(f'  Цепь {idx+1}:')
        for ri in range(3):
            lines.append('    ' + ' → '.join(g[ri] for g in glyphs))
        # Подпись — добавляемые биты
        added = []
        for i in range(1, len(chain)):
            bit = int(math.log2(chain[i] ^ chain[i - 1]))
            added.append(str(bit))
        lines.append('    Биты: ' + ' → '.join(added))
        lines.append('')

    lines.append(f'  Всего цепей: {n_chains}')
    lines.append('  Каждой перестановке σ ∈ S₆ соответствует единственная цепь:')
    lines.append('  x_k = {σ(0), ..., σ(k−1)}   для k = 0, ..., 6')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Диаграмма Хассе (текстовая)
# ---------------------------------------------------------------------------

def render_hasse(color: bool = True) -> str:
    """
    Текстовая диаграмма Хассе B₆ — глифы каждого уровня.

    Рёбра Хассе: x < y  и  rank(y) = rank(x) + 1 — прямое покрытие.
    """
    edges = hasse_edges()
    wn = whitney_numbers()

    lines: list[str] = []
    lines.append('╔' + '═' * 62 + '╗')
    lines.append('║  Диаграмма Хассе B₆ — уровни 0..6' + ' ' * 27 + '║')
    lines.append('║  Рёбра: x ─ y если y покрывает x (rank(y)=rank(x)+1)' + ' ' * 9 + '║')
    lines.append('╚' + '═' * 62 + '╝')
    lines.append('')

    for r in range(6, -1, -1):   # сверху вниз
        elems = sorted(h for h in range(64) if yang_count(h) == r)
        line_glyphs = [render_glyph(h) for h in elems]
        if color:
            line_glyphs = [
                [_YANG_ANSI[r] + row + _RESET for row in g]
                for g in line_glyphs
            ]

        lbl = f'  Ранг {r}  ({wn[r]} элементов)'
        lines.append(lbl)
        for ri in range(3):
            lines.append('  ' + '  '.join(g[ri] for g in line_glyphs))
        # Числовые метки
        nums = '  '.join(
            (_YANG_ANSI[r] + f'{h:2d}' + _RESET if color else f'{h:2d}')
            for h in elems
        )
        lines.append('  ' + nums)
        lines.append('')

    # Статистика рёбер Хассе
    lines.append(f'  Рёбра Хассе: {len(edges)}')
    lines.append('  (каждый элемент ранга k покрывает (6−k) элементов)')
    lines.append('  Σ C(6,k)·(6−k) = 6·2^5 = 192  — совпадает с |E(Q6)|')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='order_glyphs',
        description='Булева решётка B₆ через глифы гексаграмм',
    )
    p.add_argument('--no-color', action='store_true', help='без ANSI-цветов')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('layers', help='7 уровней ранга булевой решётки')
    sub.add_parser('mobius', help='функция Мёбиуса μ(0,x) для каждого глифа')
    sub.add_parser('chains', help='максимальные цепи 0 ⊂ ... ⊂ 63')
    sub.add_parser('hasse', help='диаграмма Хассе с глифами')

    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'layers':
        print(render_layers(color))
    elif args.cmd == 'mobius':
        print(render_mobius(color))
    elif args.cmd == 'chains':
        print(render_chains(color))
    elif args.cmd == 'hasse':
        print(render_hasse(color))


if __name__ == '__main__':
    main()
