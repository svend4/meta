"""solan_dendrogram.py — Иерархическая кластеризация орбит Q6 (UPGMA).

Строит дендрограмму (бинарное дерево слияний) для лексикона Q6 методом
среднего расстояния (UPGMA).  Высота узла слияния = среднее расстояние
между объединяемыми кластерами.

Возможности:
  • build_dendrogram()    — UPGMA, возвращает (nodes, root_id)
  • leaf_order()         — порядок листьев (DFS, меньший кластер — влево)
  • print_dendrogram()   — ASCII-дерево с box-drawing символами
  • print_flat_clusters() — плоские кластеры по высоте отсечения
  • dendrogram_dict()    — dict-представление для JSON-экспорта

Запуск:
    python3 -m projects.hexglyph.solan_dendrogram
    python3 -m projects.hexglyph.solan_dendrogram --cut 0.10
    python3 -m projects.hexglyph.solan_dendrogram --no-color
    python3 -m projects.hexglyph.solan_dendrogram --words ГОРА УДАР ВОДА НОРА
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_matrix  import distance_matrix
from projects.hexglyph.solan_ca      import _RST, _BOLD, _DIM

# ── Node type ─────────────────────────────────────────────────────────────────
# Node: dict with keys:
#   label  : str | None   — word for leaf, None for internal
#   height : float        — merge height (0.0 for leaves)
#   left   : int | None   — left child id
#   right  : int | None   — right child id
#   size   : int          — number of leaves in subtree

_Node = dict[str, Any]


# ── UPGMA algorithm ───────────────────────────────────────────────────────────

def _upgma(words: list[str], mat: dict[tuple[str, str], float]) \
        -> tuple[dict[int, _Node], int]:
    """Average-linkage agglomerative clustering (UPGMA).

    Returns (nodes_dict, root_id) where each node is a _Node dict.
    Node ids are integers; leaves 0..n-1, internals n..2n-2.
    """
    n        = len(words)
    nodes: dict[int, _Node] = {}
    w2id: dict[str, int]    = {}

    # Initialise leaves
    for i, w in enumerate(words):
        nodes[i] = {'label': w, 'height': 0.0, 'left': None,
                    'right': None, 'size': 1}
        w2id[w] = i

    active: set[int] = set(range(n))
    # Working distance dict: (min_id, max_id) → float
    dm: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dm[(i, j)] = mat[(words[i], words[j])]

    next_id = n

    while len(active) > 1:
        # Find minimum-distance active pair
        min_d  = float('inf')
        min_ij: tuple[int, int] = (-1, -1)
        for (i, j), d in dm.items():
            if i in active and j in active and d < min_d:
                min_d  = d
                min_ij = (i, j)

        ci, cj = min_ij
        si, sj = nodes[ci]['size'], nodes[cj]['size']

        # Create merged node
        nid = next_id
        next_id += 1
        nodes[nid] = {
            'label':  None,
            'height': min_d,
            'left':   ci,
            'right':  cj,
            'size':   si + sj,
        }

        # Update distances (UPGMA weighted average)
        active.discard(ci)
        active.discard(cj)
        for k in active:
            ki = (min(k, ci), max(k, ci))
            kj = (min(k, cj), max(k, cj))
            d_ki = dm.get(ki, 0.0)
            d_kj = dm.get(kj, 0.0)
            new_d = (d_ki * si + d_kj * sj) / (si + sj)
            new_key = (min(k, nid), max(k, nid))
            dm[new_key] = new_d
        active.add(nid)

    return nodes, list(active)[0]


# ── Public API ────────────────────────────────────────────────────────────────

def build_dendrogram(
    words: list[str] | None = None,
    width: int = 16,
) -> tuple[dict[int, _Node], int]:
    """Построить UPGMA-дендрограмму для заданного набора слов.

    Возвращает (nodes, root_id).
    """
    if words is None:
        words = LEXICON
    mat = distance_matrix(words, width=width)
    return _upgma(words, mat)


def leaf_order(nodes: dict[int, _Node], root_id: int) -> list[str]:
    """Порядок листьев: DFS, меньший дочерний узел идёт первым (слева)."""
    def _dfs(nid: int) -> list[str]:
        nd = nodes[nid]
        if nd['label'] is not None:
            return [nd['label']]
        li, ri = nd['left'], nd['right']
        # Put smaller subtree on the left for a balanced look
        if nodes[li]['size'] > nodes[ri]['size']:
            li, ri = ri, li
        return _dfs(li) + _dfs(ri)
    return _dfs(root_id)


def flat_clusters(
    nodes:     dict[int, _Node],
    root_id:   int,
    cut:       float,
) -> list[list[str]]:
    """Плоские кластеры: срезаем дерево на высоте cut.

    Возвращает список групп слов.
    """
    result: list[list[str]] = []

    def _collect(nid: int) -> None:
        nd = nodes[nid]
        if nd['label'] is not None:
            result.append([nd['label']])
        elif nd['height'] <= cut:
            # Collect all leaves of this subtree
            def _leaves(sid: int) -> list[str]:
                sn = nodes[sid]
                if sn['label'] is not None:
                    return [sn['label']]
                return _leaves(sn['left']) + _leaves(sn['right'])
            result.append(_leaves(nid))
        else:
            _collect(nd['left'])
            _collect(nd['right'])

    _collect(root_id)
    result.sort(key=len, reverse=True)
    return result


def dendrogram_dict(
    nodes:   dict[int, _Node],
    root_id: int,
) -> dict:
    """Компактный dict для JSON-экспорта и viewer.html."""
    lo = leaf_order(nodes, root_id)
    nd_export = {
        str(nid): {
            'label':  nd['label'],
            'height': round(nd['height'], 6),
            'left':   nd['left'],
            'right':  nd['right'],
            'size':   nd['size'],
        }
        for nid, nd in nodes.items()
    }
    return {
        'nodes':      nd_export,
        'root':       root_id,
        'leaf_order': lo,
        'max_height': round(nodes[root_id]['height'], 6),
    }


# ── ANSI colour helper ────────────────────────────────────────────────────────

def _height_ansi(h: float, max_h: float) -> str:
    if max_h < 1e-9:
        return '\033[38;5;240m'
    t   = min(1.0, h / max_h)
    # green(82)→yellow(226)→red(196)
    if t < 0.5:
        r = round(t * 2 * 5); g = 5
    else:
        r = 5; g = round((1.0 - (t - 0.5) * 2) * 5)
    return f'\033[38;5;{16+36*r+6*g}m'


# ── Terminal rendering ────────────────────────────────────────────────────────

def print_dendrogram(
    words:  list[str] | None = None,
    width:  int  = 16,
    color:  bool = True,
    max_depth: int = 0,          # 0 = unlimited
) -> None:
    """Вывести ASCII-дендрограмму (box-drawing символы)."""
    if words is None:
        words = LEXICON
    nodes, root_id = build_dendrogram(words, width=width)
    max_h = nodes[root_id]['height']

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    print(bold + f"  ◈ Дендрограмма Q6 (UPGMA)  n={len(words)}" + reset)
    print(f"  Максимальная высота слияния: {max_h:.4f}")
    print()

    def _render(nid: int, prefix: str, is_last: bool, depth: int) -> None:
        nd  = nodes[nid]
        con = '└── ' if is_last else '├── '
        cnt = '    ' if is_last else '│   '

        if nd['label'] is not None:
            col = _height_ansi(0.0, max_h) if color else ''
            print(prefix + (dim if color else '') + con
                  + reset + col + nd['label'] + reset)
        else:
            col = _height_ansi(nd['height'], max_h) if color else ''
            h_s = f"{nd['height']:.4f}"
            n_s = f"({nd['size']})"
            print(prefix + (dim if color else '') + con
                  + reset + col + f"[d={h_s}]  {dim}{n_s}{reset}")

            if max_depth and depth >= max_depth:
                print(prefix + cnt + dim + '...' + reset)
                return

            li, ri = nd['left'], nd['right']
            if nodes[li]['size'] > nodes[ri]['size']:
                li, ri = ri, li
            _render(li, prefix + cnt, False, depth + 1)
            _render(ri, prefix + cnt, True,  depth + 1)

    _render(root_id, '  ', True, 0)
    print()


def print_flat_clusters(
    words:     list[str] | None = None,
    width:     int   = 16,
    cut:       float = 0.10,
    color:     bool  = True,
) -> None:
    """Вывести плоские кластеры при заданной высоте отсечения."""
    if words is None:
        words = LEXICON
    nodes, root_id = build_dendrogram(words, width=width)
    max_h = nodes[root_id]['height']
    clusters = flat_clusters(nodes, root_id, cut=cut)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    print(bold + f"  Плоские кластеры Q6  cut={cut:.3f}  n={len(words)}" + reset)
    print(f"  Кластеров: {len(clusters)}")
    print()

    palette = [
        '\033[38;5;75m', '\033[38;5;120m', '\033[38;5;220m',
        '\033[38;5;213m', '\033[38;5;208m', '\033[38;5;147m',
    ]
    for ci, clust in enumerate(clusters):
        col     = (palette[ci % len(palette)]) if color else ''
        plural  = {1: 'слово', 2: 'слова', 3: 'слова', 4: 'слова'}.get(
            len(clust), 'слов')
        words_s = '  '.join(clust)
        print(f"  {col}{ci+1:2d}  ({len(clust):2d} {plural}){reset}  "
              f"{dim}{words_s}{reset}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UPGMA-дендрограмма орбитальных расстояний Q6')
    parser.add_argument('--words', nargs='+', metavar='WORD',
                        help='список слов (default: весь лексикон)')
    parser.add_argument('--cut', type=float, default=0.0,
                        help='высота отсечения для плоских кластеров')
    parser.add_argument('--json', action='store_true',
                        help='экспорт в JSON (stdout)')
    parser.add_argument('--width', type=int, default=16,
                        help='ширина CA (default: 16)')
    parser.add_argument('--depth', type=int, default=0,
                        help='максимальная глубина вывода (0=unlimited)')
    parser.add_argument('--no-color', action='store_true',
                        help='без ANSI-цветов')
    args = parser.parse_args()

    _words = args.words if args.words else None
    _color = not args.no_color

    if args.json:
        nd, rid = build_dendrogram(_words, width=args.width)
        print(json.dumps(dendrogram_dict(nd, rid), ensure_ascii=False, indent=2))
    elif args.cut > 0:
        print_flat_clusters(_words, width=args.width,
                            cut=args.cut, color=_color)
    else:
        print_dendrogram(_words, width=args.width,
                         color=_color, max_depth=args.depth)
