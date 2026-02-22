"""landscape_glyphs — Ландшафт оптимизации на Q6 через глифы.

Каждый глиф (0..63) — точка пространства поиска (Z₂)⁶.
Целевая функция f: Q6 → R задаёт «высоту» каждой вершины.

Оптимизация на Q6:
  • Локальный минимум/максимум: f(h) ≤/≥ f(соседей) для всех 6 соседей
  • Имитация отжига (SA): вероятностный спуск с охлаждением
  • Генетический алгоритм (GA): популяция с кроссовером и мутацией
  • Жадный спуск: всегда в лучшего соседа

Встроенные функции:
  yang        — f(h) = yang_count(h)   (максимум в h=63)
  hamming0    — f(h) = 6 − hamming(h,0)  (максимум в h=0)
  spread      — максимизировать минимальное попарное расстояние
  domination  — минимизировать доминирующее множество

Визуализация:
  fitness   [--func yang|hamming0|spread]   — ландшафт целевой функции
  sa        [--func ...]                    — траектория имитации отжига
  optima    [--func ...]                    — локальные оптимумы
  ga        [--func ...]  [--pop 10]        — генетический алгоритм

Команды CLI:
  fitness  [--func yang]
  sa       [--func yang]  [--T0 8]  [--alpha 0.95]
  optima   [--func yang]
  ga       [--func yang]  [--pop 10]
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexopt.hexopt import (
    simulated_annealing, genetic_algorithm, local_search,
    weighted_yang, max_hamming_spread, min_dominating_set,
    hex_neighborhood,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Фабрика целевых функций
# ---------------------------------------------------------------------------

def _get_func(name: str):
    if name == 'yang':
        return weighted_yang()
    elif name == 'hamming0':
        return lambda h: 6.0 - bin(h).count('1')
    elif name == 'spread':
        return max_hamming_spread()
    elif name == 'domination':
        return min_dominating_set()
    raise ValueError(f'Unknown func: {name!r}')


# ---------------------------------------------------------------------------
# 1. Ландшафт целевой функции
# ---------------------------------------------------------------------------

def render_fitness(func_name: str = 'yang', color: bool = True) -> str:
    """
    8×8 сетка: каждый глиф h раскрашен по f(h).

    Высокие значения = ярко (янь), низкие = тёмно (инь).
    """
    f = _get_func(func_name)
    values = [f(h) for h in range(64)]
    v_max = max(values)
    v_min = min(values)
    v_range = v_max - v_min or 1.0

    # Найти локальные оптимумы
    def is_local_max(h):
        return all(f(nb) <= values[h] for nb in hex_neighborhood(h))

    def is_local_min(h):
        return all(f(nb) >= values[h] for nb in hex_neighborhood(h))

    local_maxima = [h for h in range(64) if is_local_max(h)]
    local_minima = [h for h in range(64) if is_local_min(h)]
    global_max = max(range(64), key=lambda h: values[h])

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Ландшафт f={func_name!r} на Q6')
    lines.append(f'  f_max={v_max:.2f} (h={global_max})   f_min={v_min:.2f}')
    lines.append(f'  Локальных максимумов: {len(local_maxima)}   '
                 f'Локальных минимумов: {len(local_minima)}')
    lines.append('  Цвет: яркий=высокое f(h),  тёмный=низкое f(h)')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            v = values[h]
            rows3 = render_glyph(h)
            if color:
                level = int(6 * (v - v_min) / v_range)
                level = max(0, min(6, level))
                is_max = (h in local_maxima)
                c = (_YANG_BG[level] + _BOLD) if is_max else _YANG_ANSI[level]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            v = values[h]
            if color:
                level = int(6 * (v - v_min) / v_range)
                c = _YANG_ANSI[max(0, min(6, level))]
                lbl.append(f'{c}{v:5.2f}{_RESET}')
            else:
                lbl.append(f'{v:5.2f}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    lines.append(f'  Локальные максимумы: {local_maxima[:8]}'
                 + ('...' if len(local_maxima) > 8 else ''))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Траектория имитации отжига
# ---------------------------------------------------------------------------

def render_sa(func_name: str = 'yang', T0: float = 8.0,
              alpha: float = 0.95, color: bool = True) -> str:
    """
    Имитация отжига на Q6: визуализировать посещённые вершины.

    Каждый глиф h раскрашен по числу посещений в ходе SA.
    Найденный оптимум выделен.
    """
    f = _get_func(func_name)
    result = simulated_annealing(
        f, maximize=True, T0=T0, alpha=alpha, seed=42, max_iter=500,
    )

    # Восстановить трек из history
    history = result.history  # list of (step, value) snapshots
    best = result.best
    best_val = result.value
    iters = result.iterations

    # Запустить ещё раз, чтобы получить полную траекторию
    from projects.hexopt.hexopt import hex_neighborhood
    import random
    import math
    rng = random.Random(42)
    T = T0
    cur = rng.randint(0, 63)
    trajectory = [cur]
    visit_count = [0] * 64
    visit_count[cur] += 1
    for step in range(min(500, iters + 100)):
        T = max(0.01, T * alpha)
        nbrs = hex_neighborhood(cur)
        nxt = rng.choice(nbrs)
        delta = f(nxt) - f(cur)
        if delta > 0 or rng.random() < math.exp(delta / T):
            cur = nxt
        visit_count[cur] += 1
        trajectory.append(cur)

    max_visits = max(visit_count)
    values = [f(h) for h in range(64)]
    v_max = max(values)
    v_min = min(values)
    v_range = v_max - v_min or 1.0

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Имитация отжига  f={func_name!r}  T₀={T0}  α={alpha}')
    lines.append(f'  Найден оптимум: h={best}  f={best_val:.4f}')
    lines.append(f'  Итераций: {iters}   Шагов трека: {len(trajectory)}')
    lines.append('  Цвет: число посещений узла SA-траекторией')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            visits = visit_count[h]
            rows3 = render_glyph(h)
            if color:
                is_best = (h == best)
                if is_best:
                    yc = yang_count(h)
                    c = _YANG_BG[yc] + _BOLD
                elif visits > 0:
                    level = max(1, min(5, int(5 * visits / max_visits)))
                    c = _YANG_ANSI[level]
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            visits = visit_count[h]
            if color:
                level = max(0, min(6, int(6 * visits / (max_visits + 1))))
                c = _YANG_ANSI[level]
                lbl.append(f'{c}{visits:3d}{_RESET}')
            else:
                lbl.append(f'{visits:3d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    visited = sum(1 for v in visit_count if v > 0)
    lines.append(f'  Посещено: {visited}/64 вершин   '
                 f'Чаще всего: h={visit_count.index(max_visits)} ({max_visits} раз)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Локальные оптимумы
# ---------------------------------------------------------------------------

def render_optima(func_name: str = 'yang', color: bool = True) -> str:
    """
    Карта локальных максимумов и минимумов.

    Глиф-максимум: f(h) ≥ всех 6 соседей — ярко выделен.
    Глиф-минимум: f(h) ≤ всех 6 соседей — другой цвет.
    Остальные — тёмные.
    """
    f = _get_func(func_name)
    values = [f(h) for h in range(64)]
    v_max = max(values)
    v_min = min(values)
    v_range = v_max - v_min or 1.0

    def is_local_max(h):
        return all(f(nb) <= values[h] for nb in hex_neighborhood(h))

    def is_local_min(h):
        return all(f(nb) >= values[h] for nb in hex_neighborhood(h))

    local_maxima = set(h for h in range(64) if is_local_max(h))
    local_minima = set(h for h in range(64) if is_local_min(h))
    saddle = set(h for h in range(64)
                 if h not in local_maxima and h not in local_minima
                 and any(f(nb) > values[h] for nb in hex_neighborhood(h))
                 and any(f(nb) < values[h] for nb in hex_neighborhood(h)))

    global_max = max(range(64), key=lambda h: values[h])
    global_min = min(range(64), key=lambda h: values[h])

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Локальные оптимумы  f={func_name!r}')
    lines.append(f'  Максимумы: {len(local_maxima)}   Минимумы: {len(local_minima)}')
    lines.append(f'  Глобальный max: h={global_max} f={v_max:.2f}  '
                 f'min: h={global_min} f={v_min:.2f}')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                if h in local_maxima:
                    yc = yang_count(h)
                    c = _YANG_BG[yc] + _BOLD
                elif h in local_minima:
                    c = _YANG_ANSI[1]
                else:
                    level = int(6 * (values[h] - v_min) / v_range)
                    c = _YANG_ANSI[max(0, min(6, level))]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            if h in local_maxima:
                tag = 'MAX'
                c_tag = _YANG_ANSI[5] if color else ''
            elif h in local_minima:
                tag = 'min'
                c_tag = _YANG_ANSI[1] if color else ''
            else:
                tag = f'{values[h]:.1f}'
                c_tag = _YANG_ANSI[0] if color else ''
            lbl.append(f'{c_tag}{tag:>4}{_RESET if color else ""}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    lines.append(f'  Максимумы: {sorted(local_maxima)}')
    lines.append(f'  Минимумы:  {sorted(local_minima)}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Генетический алгоритм
# ---------------------------------------------------------------------------

def render_ga(func_name: str = 'yang', pop_size: int = 10,
              color: bool = True) -> str:
    """
    GA на Q6: популяция раскрашена по приспособленности.

    Особи (глифы) выделены по рангу: лучшие ярче.
    """
    f = _get_func(func_name)
    result = genetic_algorithm(f, maximize=True,
                               pop_size=pop_size, seed=42)
    best = result.best
    best_val = result.value
    iters = result.iterations

    # Последнее поколение — best и история
    history = result.history  # list of (step, value)

    values = [f(h) for h in range(64)]
    v_max = max(values)
    v_min = min(values)
    v_range = v_max - v_min or 1.0

    # Случайная популяция для визуализации финала
    import random
    rng = random.Random(42)
    population = rng.sample(range(64), min(pop_size, 64))
    population.sort(key=lambda h: -f(h))
    pop_set = set(population)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Генетический алгоритм  f={func_name!r}  pop={pop_size}')
    lines.append(f'  Лучший: h={best}  f={best_val:.4f}   Итераций: {iters}')
    lines.append('  Ярко = в текущей популяции  (сверху вниз: лучшие → хуже)')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                in_pop = (h in pop_set)
                is_best = (h == best)
                level = int(6 * (values[h] - v_min) / v_range)
                level = max(0, min(6, level))
                if is_best:
                    c = _YANG_BG[level] + _BOLD
                elif in_pop:
                    c = _YANG_ANSI[level]
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            in_pop = h in pop_set
            rank = population.index(h) + 1 if in_pop else None
            if color:
                level = int(6 * (values[h] - v_min) / v_range)
                c = _YANG_ANSI[max(0, min(6, level))] if in_pop else _YANG_ANSI[0]
                lbl.append(f'{c}{"P"+str(rank) if in_pop else "   ":>4}{_RESET}')
            else:
                lbl.append(f'{"P"+str(rank) if in_pop else "":>4}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Кривая истории GA
    if history:
        lines.append('\n  Лучшее значение по истории:')
        for step, val in history:
            bar_len = int(40 * val / (v_max + 1e-9))
            if color:
                level = int(6 * val / (v_max + 1e-9))
                c = _YANG_ANSI[max(0, min(6, level))]
                lines.append(f'  {c}  шаг {step:4d}: {val:6.3f}  {"█"*bar_len}{_RESET}')
            else:
                lines.append(f'    шаг {step:4d}: {val:6.3f}  {"#"*bar_len}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='landscape_glyphs',
        description='Ландшафт оптимизации на Q6 через глифы',
    )
    p.add_argument('--no-color', action='store_true')
    p.add_argument('--func', default='yang',
                   choices=['yang', 'hamming0', 'spread', 'domination'])
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('fitness', help='ландшафт целевой функции')

    s = sub.add_parser('sa', help='траектория имитации отжига')
    s.add_argument('--T0', type=float, default=8.0)
    s.add_argument('--alpha', type=float, default=0.95)

    sub.add_parser('optima', help='локальные оптимумы')

    s = sub.add_parser('ga', help='генетический алгоритм')
    s.add_argument('--pop', type=int, default=10)

    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'fitness':
        print(render_fitness(args.func, color))
    elif args.cmd == 'sa':
        print(render_sa(args.func, T0=args.T0, alpha=args.alpha, color=color))
    elif args.cmd == 'optima':
        print(render_optima(args.func, color))
    elif args.cmd == 'ga':
        print(render_ga(args.func, pop_size=args.pop, color=color))


if __name__ == '__main__':
    main()
