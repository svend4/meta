"""routing_glyphs — Маршрутизация в гиперкубической сети Q6 через глифы.

Каждый глиф (0..63) — узел-процессор в 64-узловой сети Q6.
Рёбра: два узла соединены ↔ hamming(u,v) = 1 (6 соседей у каждого).

Ключевые сетевые характеристики Q6:
  Диаметр = 6   Степень = 6   Ширина бисекции = 32
  Узловая связность = 6 (надо отключить 6 узлов для изоляции)

Алгоритм E-cube (dimension-ordered):
  Маршрут src → dst: исправлять биты по очереди i=0..5.
  Длина пути = hamming(src, dst).  Детерминированный, без циклов.

Визуализация:
  route     — E-cube маршрут src→dst: глифы вдоль пути
  traffic   — нагрузка на рёбра: 8×8 карта узлов по суммарной нагрузке
  broadcast — дерево широковещания из корня: уровни BFS
  fault     — маршрут с отказавшими узлами (выделены красным)

Команды CLI:
  route     <src> <dst>
  traffic
  broadcast [--root r]
  fault     <src> <dst> [--fail n n ...]
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexnet.hexnet import (
    ecube_route, broadcast_tree, ecube_traffic_load,
    fault_tolerant_route, network_diameter, bisection_width,
    node_connectivity, broadcast_steps, average_path_length,
    count_shortest_paths,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# 1. E-cube маршрут
# ---------------------------------------------------------------------------

def render_route(src: int, dst: int, color: bool = True) -> str:
    """
    Визуализировать E-cube маршрут src → dst.

    Каждый глиф вдоль маршрута показывает бит, который изменяется на следующем шаге.
    Расстояние = hamming(src, dst), маршрут единственный для E-cube.
    """
    path = ecube_route(src, dst)
    dist = len(path) - 1

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  E-cube маршрут: {src} ({format(src,"06b")}) → '
                 f'{dst} ({format(dst,"06b")})')
    lines.append(f'  Длина пути = {dist} = hamming({src},{dst})')
    lines.append(f'  E-cube: последовательное исправление битов i=0..5')
    lines.append('═' * 64)

    # Глифы пути
    glyphs = [render_glyph(h) for h in path]
    if color:
        colored = []
        for i, h in enumerate(path):
            yc = yang_count(h)
            is_src = (i == 0)
            is_dst = (i == len(path) - 1)
            if is_src or is_dst:
                c = _YANG_BG[yc] + _BOLD
            else:
                c = _YANG_ANSI[yc]
            colored.append([c + r + _RESET for r in glyphs[i]])
        glyphs = colored

    lines.append('')
    for ri in range(3):
        lines.append('  ' + ' → '.join(g[ri] for g in glyphs))

    # Метки: какой бит меняется
    changes = []
    for i in range(len(path) - 1):
        diff = path[i] ^ path[i + 1]
        bit = diff.bit_length() - 1
        changes.append(f'↑bit{bit}')
    lbl = ([f'{src:02d}'] + [f'{path[i+1]:02d}' for i in range(len(changes))])
    lines.append('  ' + '      '.join(lbl))
    lines.append('        ' + '       '.join(changes))

    lines.append('')
    lines.append(f'  Статистика сети Q6:')
    lines.append(f'    Диаметр = {network_diameter()}   Бисекция = {bisection_width()}')
    lines.append(f'    Узловая связность = {node_connectivity()}')
    lines.append(f'    Среднее расстояние = {average_path_length():.4f}')
    lines.append(f'    Шагов до широковещания = {broadcast_steps()}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Нагрузка на рёбра (traffic)
# ---------------------------------------------------------------------------

def render_traffic(color: bool = True) -> str:
    """
    8×8 карта узлов: каждый глиф h раскрашен по суммарной
    нагрузке на все рёбра, исходящие из h.

    Нагрузка ребра (u,v) = число E-cube маршрутов, проходящих через него.
    Для Q6 с равномерным трафиком: нагрузка = 64 для всех рёбер (симметрия).
    """
    traffic = ecube_traffic_load()

    # Для каждого узла h — суммарная нагрузка его рёбер
    node_load = [0] * 64
    for edge, load in traffic.items():
        u, v = tuple(edge)
        node_load[u] += load
        node_load[v] += load

    max_load = max(node_load)
    min_load = min(node_load)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append('  Нагрузка на узлы Q6 при равномерном E-cube трафике')
    lines.append(f'  Нагрузка ребра (u,v) = число маршрутов через него')
    lines.append(f'  Нагрузка узла h = Σ нагрузок его 6 рёбер')
    lines.append(f'  max={max_load}  min={min_load}  '
                 f'{"равномерная!" if max_load == min_load else "неравномерная"}')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            load = node_load[h]
            rows3 = render_glyph(h)
            if color:
                if max_load > min_load:
                    level = int(6 * (load - min_load) / (max_load - min_load))
                else:
                    level = 3
                level = max(0, min(6, level))
                c = _YANG_ANSI[level]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            load = node_load[h]
            if color:
                if max_load > min_load:
                    level = int(6 * (load - min_load) / (max_load - min_load))
                else:
                    level = 3
                c = _YANG_ANSI[max(0, min(6, level))]
                lbl.append(f'{c}{load:4d}{_RESET}')
            else:
                lbl.append(f'{load:4d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    lines.append('  Q6 — вершинно-транзитивный граф: все узлы эквивалентны.')
    lines.append('  Нагрузка равномерна: каждый узел маршрутизирует одинаково.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Дерево широковещания
# ---------------------------------------------------------------------------

def render_broadcast(root: int = 0, color: bool = True) -> str:
    """
    Дерево BFS-широковещания из корня.

    Уровень 0: корень.  Уровень k: узлы, достигнутые за k шагов.
    За 6 шагов все 64 узла получают сообщение.
    """
    tree = broadcast_tree(root)   # dict: node → parent (root → None)

    # Построить уровни BFS
    levels: list[list[int]] = [[] for _ in range(7)]
    for node, parent in tree.items():
        if parent is None:
            levels[0].append(node)
        else:
            # Найти уровень (= расстояние от root)
            dist = bin(node ^ root).count('1')
            levels[dist].append(node)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  BFS-дерево широковещания  корень={root}')
    lines.append(f'  За 6 шагов все 64 узла получают сообщение')
    lines.append(f'  Число шагов = диаметр Q6 = {network_diameter()}')
    lines.append('═' * 64)

    for lvl, members in enumerate(levels):
        if not members:
            continue
        members_sorted = sorted(members)
        if color:
            c = _YANG_ANSI[min(6, lvl)]
            lines.append(f'\n  {c}Шаг {lvl}: {len(members_sorted)} узлов{_RESET}')
        else:
            lines.append(f'\n  Шаг {lvl}: {len(members_sorted)} узлов')

        # Глифы (до 16)
        shown = members_sorted[:16]
        glyphs = [render_glyph(h) for h in shown]
        if color:
            lvl_c = _YANG_ANSI[min(6, lvl)]
            is_root_level = (lvl == 0)
            glyphs = [
                [(_YANG_BG[yang_count(h)] + _BOLD if is_root_level else lvl_c) + r + _RESET
                 for r in g]
                for h, g in zip(shown, glyphs)
            ]
        for ri in range(3):
            lines.append('    ' + '  '.join(g[ri] for g in glyphs))
        nums = '  '.join(f'{h:02d}' for h in shown)
        lines.append('    ' + nums + ('  ...' if len(members_sorted) > 16 else ''))

    lines.append('')
    lines.append(f'  Итого: 64 узла за 6 шагов = broadcast_steps() = {broadcast_steps()}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Маршрут с отказами
# ---------------------------------------------------------------------------

def render_fault(src: int, dst: int, failed: list[int],
                 color: bool = True) -> str:
    """
    Адаптивный маршрут src→dst в сети с отказавшими узлами.

    Отказавшие узлы выделены красным.  Обходной маршрут — синим.
    """
    failed_set = set(failed)
    path = fault_tolerant_route(src, dst, set(failed))

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Маршрут с отказами: {src} → {dst}')
    lines.append(f'  Отказавшие узлы: {sorted(failed_set)}')
    if path:
        lines.append(f'  Найден маршрут длиной {len(path)-1}')
    else:
        lines.append('  МАРШРУТ НЕ НАЙДЕН (сеть разбита отказами)')
    lines.append('═' * 64)

    # Карта 8×8: отказавшие узлы красные, путь синий, остальные тёмные
    path_set = set(path) if path else set()

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                if h in failed_set:
                    c = _YANG_ANSI[5] + _BOLD   # красный
                elif h in path_set:
                    yc = yang_count(h)
                    c = _YANG_BG[yc] + _BOLD    # яркий
                else:
                    c = _YANG_ANSI[0]           # тёмный
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            if h in failed_set:
                tag = 'FAIL'
                c_tag = _YANG_ANSI[5] if color else ''
            elif h in path_set:
                idx = path.index(h) if path else -1
                tag = f'P{idx:2d}'
                c_tag = _YANG_ANSI[2] if color else ''
            else:
                tag = f'{h:4d}'
                c_tag = _YANG_ANSI[0] if color else ''
            lbl.append(f'{c_tag}{tag}{_RESET if color else ""}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    if path:
        lines.append(f'  Маршрут: {" → ".join(str(h) for h in path)}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='routing_glyphs',
        description='Маршрутизация в гиперкубической сети Q6 через глифы',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('route', help='E-cube маршрут src→dst')
    s.add_argument('src', type=int)
    s.add_argument('dst', type=int)

    sub.add_parser('traffic', help='нагрузка на узлы при равномерном трафике')

    s = sub.add_parser('broadcast', help='BFS-дерево широковещания')
    s.add_argument('--root', type=int, default=0)

    s = sub.add_parser('fault', help='маршрут с отказавшими узлами')
    s.add_argument('src', type=int)
    s.add_argument('dst', type=int)
    s.add_argument('--fail', type=int, nargs='+', default=[],
                   metavar='N', help='отказавшие узлы')

    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'route':
        print(render_route(args.src, args.dst, color))
    elif args.cmd == 'traffic':
        print(render_traffic(color))
    elif args.cmd == 'broadcast':
        print(render_broadcast(root=args.root, color=color))
    elif args.cmd == 'fault':
        print(render_fault(args.src, args.dst, args.fail, color))


if __name__ == '__main__':
    main()
