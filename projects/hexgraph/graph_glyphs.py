"""graph_glyphs — Теория графов Q6 через глифы.

Каждый глиф (0..63) — вершина 6-мерного гиперкуба Q6.
Q6 — 6-регулярный двудольный граф: 64 вершины, 192 ребра.

Граф Q6:
  • Двудольный: части — чётные/нечётные yang_count
  • Регулярный степени 6: каждая вершина — 6 соседей
  • Диаметр 6,  обхват 4  (кратчайший цикл = квадрат)
  • Число независимости α(Q6) = 32  (одна доля)
  • Число доминирования γ(Q6) ≤ 10
  • Спектр: {6¹, 4⁶, 2¹⁵, 0²⁰, −2¹⁵, −4⁶, −6¹}  = {6−2k}·C(6,k)

Визуализация:
  bipartite  — двудольное разбиение (части A и B)
  independent — максимальное независимое множество
  domination  — минимальное доминирующее множество
  balls [--center h] [--r r] — шары Хэмминга от центра h

Команды CLI:
  bipartite
  independent
  domination
  balls [--center h] [--r r]
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexgraph.hexgraph import (
    q6_full, q6_ball, analyze, Subgraph,
)
from libs.hexcore.hexcore import yang_count, neighbors, hamming
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# 1. Двудольное разбиение
# ---------------------------------------------------------------------------

def render_bipartite(color: bool = True) -> str:
    """
    8×8 сетка: раскраска по доле в двудольном разбиении.

    Часть A: yang_count(h) чётное  (0,2,4,6) — 32 вершины
    Часть B: yang_count(h) нечётное (1,3,5)   — 32 вершины

    Все рёбра Q6 идут строго из A в B (гиперкуб = чётно-нечётный граф).
    """
    g = q6_full()
    a_info = analyze(g)

    # Двухцветная раскраска
    coloring = g.two_coloring()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Двудольное разбиение Q6: части A (чётный yang) и B (нечётный yang)')
    lines.append(f'  |Q6| = {a_info["order"]}   рёбер = {a_info["size"]}   регулярность = 6')
    lines.append(f'  Двудольный: {a_info["bipartite"]}   Диаметр: {a_info["diameter"]}   Обхват: {a_info["girth"]}')
    lines.append('  Часть A = yang∈{0,2,4,6}: синяя    Часть B = yang∈{1,3,5}: жёлтая')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            part = coloring.get(h, 0)
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                if part == 0:
                    # Часть A: чётный yang
                    c = _YANG_ANSI[yc]
                else:
                    # Часть B: нечётный yang — яркий фон
                    c = _YANG_BG[yc] + _BOLD
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            part = coloring.get(h, 0)
            yc = yang_count(h)
            tag = 'A' if part == 0 else 'B'
            if color:
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}{tag}{yc}{_RESET}')
            else:
                lbl.append(f'{tag}{yc}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Спектр Q6
    lines.append('  Спектр Q6: λₖ = 6−2k  (кратность C(6,k))')
    lines.append('  ' + '  '.join(
        f'λ={6-2*k}×C(6,{k})={6-2*k}×{__import__("math").comb(6,k)}'
        for k in range(4)))
    lines.append('  Сумма квадратов λ²: Σ C(6,k)·(6−2k)² = 64·6/2 = 192  (= число рёбер · 2)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Максимальное независимое множество
# ---------------------------------------------------------------------------

def render_independent(color: bool = True) -> str:
    """
    8×8 сетка: максимальное независимое множество Q6.

    α(Q6) = 32 = |Q6|/2 (одна доля двудольного графа).
    Вершины IS выделены цветом, остальные приглушены.
    """
    g = q6_full()
    mis = set(g.max_independent_set())
    alpha = g.independence_number()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Максимальное независимое множество Q6  α(Q6) = {alpha}')
    lines.append('  IS = одна доля двудольного Q6 (вершины с чётным yang_count)')
    lines.append('  Покрытие вершин (теорема Галлаи): τ(Q6) = |Q6|−α = 64−32 = 32')
    lines.append('  Выделены: вершины IS.   Приглушены: вершины вне IS.')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            in_is = h in mis
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_BG[yc] + _BOLD if in_is else _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            in_is = h in mis
            yc = yang_count(h)
            if color:
                c = _YANG_ANSI[yc] if in_is else _YANG_ANSI[0]
                lbl.append(f'{c}{"IS" if in_is else "  "}{_RESET}')
            else:
                lbl.append('IS' if in_is else '  ')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Подробности
    sorted_mis = sorted(mis)
    lines.append(f'  IS ({len(sorted_mis)} вершин): {sorted_mis[:16]} ...')
    lines.append('  Проверка: никакие две вершины IS не соседи — все yang нечётные')
    lines.append(f'  Дополнение IS: вершины с нечётным yang_count ({alpha} вершин)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Минимальное доминирующее множество
# ---------------------------------------------------------------------------

def render_domination(color: bool = True) -> str:
    """
    8×8 сетка: минимальное доминирующее множество Q6.

    DS: каждая вершина Q6 либо ∈ DS, либо имеет соседа в DS.
    γ(Q6) = число вершин в min DS (≈ 10–12 для Q6).
    """
    g = q6_full()
    ds = set(g.min_dominating_set())
    gamma = g.domination_number()

    # Проверить доминирование
    dominated = set(ds)
    for h in ds:
        for nb in neighbors(h):
            dominated.add(nb)
    not_dominated = set(range(64)) - dominated

    # Соседи DS
    ds_neighbors: dict[int, list[int]] = {}
    for h in range(64):
        if h not in ds:
            nbs_in_ds = [nb for nb in neighbors(h) if nb in ds]
            ds_neighbors[h] = nbs_in_ds

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Минимальное доминирующее множество Q6  γ(Q6) = {gamma}')
    lines.append(f'  DS: каждая вершина ∈ DS или имеет соседа ∈ DS')
    lines.append(f'  Найдено |DS| = {len(ds)}  вершин DS  (покрывают всё Q6)')
    lines.append(f'  Недоминированных вершин: {len(not_dominated)} (должно быть 0)')
    lines.append('  Жирный = в DS,   Цвет = yang_count')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            in_ds = h in ds
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_BG[yc] + _BOLD if in_ds else _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            in_ds = h in ds
            nb_cnt = len(ds_neighbors.get(h, []))
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}{"DS" if in_ds else f"n{nb_cnt}"}{_RESET}')
            else:
                lbl.append('DS' if in_ds else f'n{nb_cnt}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    sorted_ds = sorted(ds)
    lines.append(f'  DS = {sorted_ds}')
    lines.append(f'  Нижняя граница: |Q6| / (δ+1) = 64/7 ≈ {64/7:.2f}  → γ ≥ 10')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Шары Хэмминга
# ---------------------------------------------------------------------------

def render_balls(center: int = 0, max_r: int = 3, color: bool = True) -> str:
    """
    8×8 сетка: раскраска по расстоянию Хэмминга от центра h.

    B(h, r) = {v : dist(h,v) ≤ r}
    S(h, r) = {v : dist(h,v) = r}  — сфера Хэмминга
    """
    dists = [hamming(center, v) for v in range(64)]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Шары Хэмминга B(h={center}, r) в Q6')
    lines.append(f'  Центр: {center} = {format(center,"06b")}   yang={yang_count(center)}')
    lines.append(f'  |S(h,k)| = C(6,k):  ' +
                 '  '.join(f'S{k}={__import__("math").comb(6,k)}' for k in range(7)))
    lines.append(f'  Показаны расстояния 0..{max_r}')
    lines.append('  Цвет = расстояние Хэмминга до центра')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            d = dists[h]
            rows3 = render_glyph(h)
            if color:
                if h == center:
                    c = _YANG_BG[0] + _BOLD
                elif d <= max_r:
                    c = _YANG_ANSI[d]
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            d = dists[h]
            if color:
                c = _YANG_ANSI[min(d, 6)]
                lbl.append(f'{c}d={d}{_RESET}')
            else:
                lbl.append(f'd={d}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Суммарно
    from collections import Counter
    import math
    dist_cnt = Counter(dists)
    lines.append('  Сферы S(h, k):')
    cumul = 0
    for k in range(7):
        cnt = dist_cnt.get(k, 0)
        cumul += cnt
        binom = math.comb(6, k)
        ok = '✓' if cnt == binom else '✗'
        if color:
            c = _YANG_ANSI[k]
            lines.append(f'  {c}  k={k}: |S|={cnt}={binom} {ok}   |B(h,{k})|={cumul}{_RESET}')
        else:
            lines.append(f'    k={k}: |S|={cnt}={binom} {ok}   |B(h,{k})|={cumul}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='graph_glyphs',
        description='Теория графов Q6 через глифы гексаграмм',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('bipartite',   help='двудольное разбиение Q6')
    sub.add_parser('independent', help='максимальное независимое множество')
    sub.add_parser('domination',  help='минимальное доминирующее множество')

    s = sub.add_parser('balls', help='шары Хэмминга от центра')
    s.add_argument('--center', type=int, default=0, metavar='H',
                   help='центральная вершина (0..63, default=0)')
    s.add_argument('--r', type=int, default=3, metavar='R',
                   help='максимальный радиус показа (default=3)')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'bipartite':
        print(render_bipartite(color))
    elif args.cmd == 'independent':
        print(render_independent(color))
    elif args.cmd == 'domination':
        print(render_domination(color))
    elif args.cmd == 'balls':
        print(render_balls(center=args.center, max_r=args.r, color=color))


if __name__ == '__main__':
    main()
