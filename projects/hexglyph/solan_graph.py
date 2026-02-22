"""solan_graph.py — Граф орбитального сходства Q6-лексикона.

Слова = вершины; ребро (w1, w2) добавляется если
orbital_distance(w1, w2) < threshold.

Возможности:
  • build_graph()         — список смежности (undirected)
  • connected_components() — связные компоненты (BFS), отсортированы по убыванию
  • degree()              — степень каждой вершины
  • hub_words()           — топ-N вершин с наибольшим числом связей
  • print_graph()         — терминальная визуализация компонент + хабов
  • print_adjacency()     — ASCII-матрица смежности (● / ·)
  • graph_stats()         — сводная статистика

Запуск:
    python3 -m projects.hexglyph.solan_graph
    python3 -m projects.hexglyph.solan_graph --threshold 0.05
    python3 -m projects.hexglyph.solan_graph --adjacency
    python3 -m projects.hexglyph.solan_graph --hubs --n 8
    python3 -m projects.hexglyph.solan_graph --threshold 0.20 --no-color
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_matrix import distance_matrix, nearest_pairs
from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_ca import _RST, _BOLD, _DIM

# ── Граф ─────────────────────────────────────────────────────────────────────

def build_graph(
    words:     list[str] | None = None,
    threshold: float = 0.10,
    width:     int   = 16,
) -> dict[str, list[str]]:
    """Список смежности (неориентированный граф).

    Ребро (w1, w2) добавляется если orbital_distance(w1, w2) < threshold
    и w1 ≠ w2.  Соседи отсортированы по расстоянию (ближайшие первые).
    """
    if words is None:
        words = LEXICON
    mat   = distance_matrix(words, width=width)
    graph: dict[str, list[tuple[str, float]]] = {w: [] for w in words}
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if j <= i:
                continue
            d = mat[(w1, w2)]
            if d == d and d < threshold:   # not NaN and below threshold
                graph[w1].append((w2, d))
                graph[w2].append((w1, d))
    # Sort neighbors by distance (ascending)
    return {w: [n for n, _ in sorted(graph[w], key=lambda x: x[1])]
            for w in words}


def connected_components(graph: dict[str, list[str]]) -> list[list[str]]:
    """Связные компоненты (BFS). Сортируются по убыванию размера."""
    visited: set[str] = set()
    components: list[list[str]] = []
    for start in graph:
        if start in visited:
            continue
        comp: list[str] = []
        q: deque[str] = deque([start])
        visited.add(start)
        while q:
            w = q.popleft()
            comp.append(w)
            for nb in graph[w]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        components.append(comp)
    components.sort(key=len, reverse=True)
    return components


def degree(graph: dict[str, list[str]]) -> dict[str, int]:
    """Степень каждой вершины."""
    return {w: len(nb) for w, nb in graph.items()}


def hub_words(
    graph: dict[str, list[str]],
    n:     int = 5,
) -> list[tuple[str, int]]:
    """Топ-N вершин с наибольшей степенью."""
    deg = degree(graph)
    return sorted(deg.items(), key=lambda x: x[1], reverse=True)[:n]


def graph_stats(graph: dict[str, list[str]]) -> dict:
    """Сводная статистика графа."""
    deg   = degree(graph)
    edges = sum(deg.values()) // 2
    comps = connected_components(graph)
    iso   = [c[0] for c in comps if len(c) == 1]
    return {
        'nodes':      len(graph),
        'edges':      edges,
        'components': len(comps),
        'isolated':   len(iso),
        'isolated_words': iso,
        'largest_comp_size': len(comps[0]) if comps else 0,
        'max_degree': max(deg.values()) if deg else 0,
        'avg_degree': sum(deg.values()) / len(deg) if deg else 0.0,
    }


# ── Вывод ─────────────────────────────────────────────────────────────────────

# ANSI-цвета для компонент (до 10 различных цветов)
_COMP_COLORS = [
    '\033[38;5;75m',   # голубой
    '\033[38;5;120m',  # зелёный
    '\033[38;5;220m',  # жёлтый
    '\033[38;5;213m',  # розовый
    '\033[38;5;208m',  # оранжевый
    '\033[38;5;147m',  # лавандовый
    '\033[38;5;46m',   # ярко-зелёный
    '\033[38;5;196m',  # красный
    '\033[38;5;51m',   # циан
    '\033[38;5;226m',  # ярко-жёлтый
]


def print_graph(
    words:     list[str] | None = None,
    threshold: float = 0.10,
    width:     int   = 16,
    color:     bool  = True,
    hubs_n:    int   = 5,
) -> None:
    """Вывести компоненты, связи и хабы.

    Формат:
      Компонента 1 (12 слов): ВОДА НОРА ЛУНА ...
        ВОДА ── НОРА ЛУНА ДУГА ...
        НОРА ── ВОДА ЛУНА ...
      ...
      Хабы (топ-5): ВОДА(11) ГОРА(3) ...
    """
    if words is None:
        words = LEXICON
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    graph = build_graph(words, threshold=threshold, width=width)
    comps = connected_components(graph)
    stats = graph_stats(graph)

    print(bold + f"  ◈ Граф орбитального сходства Q6"
          f"  threshold={threshold:.2f}  width={width}" + reset)
    print(f"  Узлов: {stats['nodes']}   "
          f"Рёбер: {stats['edges']}   "
          f"Компонент: {stats['components']}   "
          f"Изолированных: {stats['isolated']}")
    print()

    for ci, comp in enumerate(comps):
        col = (_COMP_COLORS[ci % len(_COMP_COLORS)]) if color else ''
        plural = {1: 'слово', 2: 'слова', 3: 'слова', 4: 'слова'}.get(
            len(comp), 'слов')
        header = (bold + col
                  + f"  Компонента {ci+1}  ({len(comp)} {plural})"
                  + reset)
        print(header)

        if len(comp) == 1:
            print(f"    {dim}{comp[0]}  (изолирована){reset}")
        else:
            for w in comp:
                nb_str = '  '.join(graph[w])
                print(f"    {col}{w:10s}{reset}{dim} ─ {reset}"
                      f"{col}{nb_str}{reset}")
        print()

    # Хабы
    hubs = hub_words(graph, n=hubs_n)
    max_deg = hubs[0][1] if hubs else 1
    print(bold + f"  Хабы (топ-{hubs_n}):" + reset)
    for w, d in hubs:
        bar_len = 20
        filled  = int(d / max_deg * bar_len) if max_deg else 0
        bar = '█' * filled + '░' * (bar_len - filled)
        col = _COMP_COLORS[0] if color else ''
        print(f"    {col}{w:10s}{reset}  {dim}[{bar}]{reset}  {d} связей")
    print()


def print_adjacency(
    words:     list[str] | None = None,
    threshold: float = 0.10,
    width:     int   = 16,
    color:     bool  = True,
) -> None:
    """ASCII-матрица смежности: ● если есть ребро, · если нет, ■ диагональ."""
    if words is None:
        words = LEXICON
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    graph = build_graph(words, threshold=threshold, width=width)
    n     = len(words)
    abbr  = [w[:4].ljust(4) for w in words]

    print(bold + f"  Матрица смежности  threshold={threshold:.2f}  ({n}×{n})" + reset)
    print()

    # Шапка (4 строки)
    row_label_w = 9
    for li in range(4):
        head = ' ' * row_label_w
        for ab in abbr:
            head += (dim + ab[li] + reset) if color else ab[li]
        print('  ' + head)
    print()

    for w in words:
        row_lbl = f"{w[:7]:7s} "
        line = (bold + row_lbl + reset) if color else row_lbl
        nb_set = set(graph[w])
        for w2 in words:
            if w == w2:
                line += (dim + '■' + reset) if color else '■'
            elif w2 in nb_set:
                line += ('\033[38;5;75m' if color else '') + '●' + reset
            else:
                line += (dim + '·' + reset) if color else '·'
        print('  ' + line)

    print()
    stats = graph_stats(graph)
    print(f"  {dim}Рёбер: {stats['edges']}   "
          f"Компонент: {stats['components']}{reset}")



# ── Сводка ──────────────────────────────────────────────────────────────────

def graph_summary(
    words:     list[str] | None = None,
    threshold: float = 0.10,
    width:     int   = 16,
) -> dict:
    """JSON-friendly graph stats for the orbital similarity network."""
    g = build_graph(words, threshold=threshold, width=width)
    d = graph_stats(g)
    d.update({'threshold': threshold, 'width': width})
    return d


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Граф орбитального сходства Q6-лексикона')
    parser.add_argument('--words', nargs='+', metavar='WORD',
                        help='список слов (default: весь лексикон)')
    parser.add_argument('--threshold', '-t', type=float, default=0.10,
                        help='порог расстояния (default: 0.10)')
    parser.add_argument('--width', type=int, default=16,
                        help='ширина CA (default: 16)')
    parser.add_argument('--adjacency', action='store_true',
                        help='показать матрицу смежности')
    parser.add_argument('--hubs', action='store_true',
                        help='показать только хабы')
    parser.add_argument('--n', type=int, default=5,
                        help='число хабов (default: 5)')
    parser.add_argument('--stats', action='store_true',
                        help='только статистика')
    parser.add_argument('--no-color', action='store_true',
                        help='без ANSI-цветов')
    parser.add_argument('--json',     action='store_true',
                        help='JSON output')
    args = parser.parse_args()

    _words = args.words if args.words else None
    _color = not args.no_color

    if args.json:
        import json as _json
        print(_json.dumps(graph_summary(_words, threshold=args.threshold, width=args.width), ensure_ascii=False, indent=2))
        import sys; sys.exit(0)
    if args.stats:
        g = build_graph(_words, threshold=args.threshold, width=args.width)
        st = graph_stats(g)
        print(f"Узлов:      {st['nodes']}")
        print(f"Рёбер:      {st['edges']}")
        print(f"Компонент:  {st['components']}")
        print(f"Изолиров.:  {st['isolated']}")
        print(f"Макс. степ.: {st['max_degree']}")
        print(f"Ср. степень: {st['avg_degree']:.2f}")
    elif args.adjacency:
        print_adjacency(_words, threshold=args.threshold,
                        width=args.width, color=_color)
    elif args.hubs:
        g = build_graph(_words, threshold=args.threshold, width=args.width)
        hubs = hub_words(g, n=args.n)
        for w, d in hubs:
            print(f"  {w:12s}  {d} связей")
    else:
        print_graph(_words, threshold=args.threshold,
                    width=args.width, color=_color, hubs_n=args.n)
