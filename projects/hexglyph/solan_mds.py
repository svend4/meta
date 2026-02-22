"""solan_mds.py — Классическое MDS-вложение орбитального пространства Q6.

Проецирует матрицу попарных расстояний (n×n) в ℝ² методом классического
MDS (cMDS): двойное центрирование → матрица Грама → top-2 собственных вектора
через степенные итерации + дефляция.  Без внешних зависимостей.

Метрика качества: нормированный стресс Kruskal-1.

Возможности:
  • build_mds()         — (words, coords_2d, stress)
  • mds_dict()          — dict для viewer.html / JSON
  • print_mds()         — координаты + ASCII scatter plot
  • print_stress_info() — стресс + интерпретация

Запуск:
    python3 -m projects.hexglyph.solan_mds
    python3 -m projects.hexglyph.solan_mds --words ГОРА УДАР ВОДА НОРА ЛУНА
    python3 -m projects.hexglyph.solan_mds --json
    python3 -m projects.hexglyph.solan_mds --no-color
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_matrix  import distance_matrix
from projects.hexglyph.solan_ca      import _RST, _BOLD, _DIM


# ── Linear algebra helpers (pure Python, no numpy) ───────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(_dot(v, v))


def _matvec(M: list[list[float]], v: list[float]) -> list[float]:
    return [_dot(row, v) for row in M]


def _normalize(v: list[float]) -> list[float]:
    n = _norm(v)
    return [x / n for x in v] if n > 1e-14 else v[:]


def _axpy(M: list[list[float]], lam: float, u: list[float], v: list[float]) \
        -> list[list[float]]:
    """M -= lam * outer(u, v)  (in-place deflation)."""
    n = len(M)
    for i in range(n):
        for j in range(n):
            M[i][j] -= lam * u[i] * v[j]
    return M


def _power_iter(B: list[list[float]], k: int = 2,
                iters: int = 200) -> list[tuple[float, list[float]]]:
    """Top-k (eigenvalue, eigenvector) pairs via power iteration + deflation.

    Works on a copy of B so the original is not modified.
    Returns list of (lambda, vec) sorted by |lambda| descending.
    """
    n   = len(B)
    rem = [row[:] for row in B]          # working copy
    result: list[tuple[float, list[float]]] = []

    for _ in range(k):
        # Initial vector: alternating signs for diversity
        v = _normalize([(1 if i % 2 == 0 else -1) * (1.0 + i * 0.01)
                        for i in range(n)])
        lam_prev = float('inf')
        for __ in range(iters):
            w   = _matvec(rem, v)
            lam = _dot(w, v)
            nw  = _norm(w)
            if nw < 1e-14:
                break
            v = [x / nw for x in w]
            if abs(lam - lam_prev) < 1e-12 * (abs(lam) + 1e-12):
                break
            lam_prev = lam
        lam = _dot(_matvec(rem, v), v)
        result.append((lam, v[:]))
        # Deflate: B_rem -= lam * v vᵀ
        _axpy(rem, lam, v, v)

    return result


# ── Double-centring → Gram matrix ─────────────────────────────────────────────

def _gram(dists: list[list[float]]) -> list[list[float]]:
    """Classical MDS: B = -½ H D² H   (H = I - 1/n 11ᵀ)."""
    n      = len(dists)
    D2     = [[dists[i][j] ** 2 for j in range(n)] for i in range(n)]
    rmeans = [sum(D2[i]) / n for i in range(n)]
    cmeans = [sum(D2[i][j] for i in range(n)) / n for j in range(n)]
    grand  = sum(cmeans) / n
    return [[-0.5 * (D2[i][j] - rmeans[i] - cmeans[j] + grand)
             for j in range(n)]
            for i in range(n)]


# ── Public API ────────────────────────────────────────────────────────────────

def build_mds(
    words: list[str] | None = None,
    width: int = 16,
    k: int = 2,
) -> tuple[list[str], list[list[float]], float]:
    """Построить MDS-вложение.

    Возвращает (words, coords, stress):
      • coords — список n k-мерных векторов [[x1,y1], ...]
      • stress — нормированный стресс Kruskal-1
    """
    if words is None:
        words = LEXICON
    mat  = distance_matrix(words, width=width)
    n    = len(words)
    dmat = [[mat[(words[i], words[j])] for j in range(n)] for i in range(n)]

    B         = _gram(dmat)
    eigpairs  = _power_iter(B, k=k)

    # coords[i][d] = eigvec_d[i] * sqrt(max(0, eigenvalue_d))
    coords = [[0.0] * k for _ in range(n)]
    for d, (lam, vec) in enumerate(eigpairs):
        scale = math.sqrt(max(0.0, lam))
        for i in range(n):
            coords[i][d] = vec[i] * scale

    stress = mds_stress(dmat, coords)
    return words, coords, stress


def mds_stress(dmat: list[list[float]], coords: list[list[float]]) -> float:
    """Kruskal stress-1 = sqrt(Σ(d_orig - d_mds)² / Σ d_orig²)."""
    n   = len(coords)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d_o = dmat[i][j]
            d_m = math.sqrt(sum((coords[i][d] - coords[j][d]) ** 2
                                for d in range(len(coords[i]))))
            num += (d_o - d_m) ** 2
            den += d_o ** 2
    return math.sqrt(num / den) if den > 1e-14 else 0.0


def mds_dict(
    words:   list[str],
    coords:  list[list[float]],
    stress:  float,
    dmat:    list[list[float]] | None = None,
) -> dict:
    """Dict-представление MDS-вложения для viewer.html / JSON."""
    # Normalise to [-1, 1]
    for d in range(len(coords[0])):
        vals = [c[d] for c in coords]
        vmax = max(abs(v) for v in vals) or 1.0
        for c in coords:
            c[d] /= vmax

    data: dict = {
        'words':  words,
        'coords': [[round(c[0], 6), round(c[1], 6)] for c in coords],
        'stress': round(stress, 6),
    }
    if dmat is not None:
        data['dmat'] = [[round(dmat[i][j], 4) for j in range(len(words))]
                        for i in range(len(words))]
    return data


# ── ASCII scatter plot ────────────────────────────────────────────────────────

def _ascii_scatter(
    words:  list[str],
    coords: list[list[float]],
    W: int = 56,
    H: int = 22,
) -> list[str]:
    """Render 2-D scatter plot as ASCII text (W×H chars)."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0

    grid = [['·'] * W for _ in range(H)]

    def to_col(x: float) -> int:
        return min(W - 1, max(0, int((x - x0) / dx * (W - 1))))

    def to_row(y: float) -> int:
        # y increases upward in MDS but downward in terminal
        return min(H - 1, max(0, int((y1 - y) / dy * (H - 1))))

    placed: dict[tuple[int, int], str] = {}
    for w, c in zip(words, coords):
        col = to_col(c[0])
        row = to_row(c[1])
        label = w[:2]
        key   = (row, col)
        if key not in placed:
            grid[row][col]     = label[0]
            if col + 1 < W and len(label) > 1:
                grid[row][col + 1] = label[1]
            placed[key] = w
        else:
            # Nudge one column right
            col2 = min(W - 1, col + 2)
            key2 = (row, col2)
            if key2 not in placed:
                grid[row][col2]     = label[0]
                if col2 + 1 < W and len(label) > 1:
                    grid[row][col2 + 1] = label[1]
                placed[key2] = w

    # Axis
    for r in range(H):
        grid[r][0] = '│'
    for c in range(W):
        grid[H - 1][c] = '─'
    grid[H - 1][0] = '└'

    return ['  ' + ''.join(row) for row in grid]


# ── Terminal output ───────────────────────────────────────────────────────────

_STRESS_LEVELS = [
    (0.05, 'отлично'),
    (0.10, 'хорошо'),
    (0.20, 'приемлемо'),
    (0.30, 'слабо'),
    (1.00, 'плохо'),
]


def _stress_label(s: float) -> str:
    for threshold, label in _STRESS_LEVELS:
        if s <= threshold:
            return label
    return 'плохо'


def print_mds(
    words: list[str] | None = None,
    width: int = 16,
    color: bool = True,
) -> None:
    """Вывести MDS-координаты и ASCII scatter plot."""
    if words is None:
        words = LEXICON
    wl, coords, stress = build_mds(words, width=width)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    print(bold + f"  ◈ MDS-вложение Q6  n={len(wl)}  "
          + f"stress={stress:.4f} ({_stress_label(stress)})" + reset)
    print()

    # Scatter plot
    for line in _ascii_scatter(wl, coords):
        print(dim + line + reset)
    print()

    # Coordinate table
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    print(f"  Диапазон X: [{x0:+.4f}, {x1:+.4f}]  "
          f"Y: [{y0:+.4f}, {y1:+.4f}]")
    print()

    # Top rows sorted by x
    sorted_idx = sorted(range(len(wl)), key=lambda i: coords[i][0])
    print(bold + "  Слово              x        y" + reset)
    for i in sorted_idx:
        col = '\033[38;5;117m' if color else ''
        print(f"  {col}{wl[i]:<14}{reset}  "
              f"{coords[i][0]:+.4f}  {coords[i][1]:+.4f}")
    print()


def print_stress_info(
    words: list[str] | None = None,
    width: int = 16,
    color: bool = True,
) -> None:
    """Вывести только значение стресса с интерпретацией."""
    if words is None:
        words = LEXICON
    _, _, stress = build_mds(words, width=width)
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    print(bold + f"  Stress Kruskal-1: {stress:.4f}  "
          f"({_stress_label(stress)})" + reset)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MDS-вложение орбитального пространства Q6')
    parser.add_argument('--words', nargs='+', metavar='WORD')
    parser.add_argument('--json',     action='store_true')
    parser.add_argument('--stress',   action='store_true')
    parser.add_argument('--width',    type=int, default=16)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _words = args.words if args.words else None
    _color = not args.no_color

    if args.json:
        from projects.hexglyph.solan_matrix import distance_matrix
        ws = _words or LEXICON
        mat  = distance_matrix(ws, width=args.width)
        n    = len(ws)
        dmat = [[mat[(ws[i], ws[j])] for j in range(n)] for i in range(n)]
        wl, coords, stress = build_mds(ws, width=args.width)
        print(json.dumps(mds_dict(wl, coords, stress, dmat),
                         ensure_ascii=False, indent=2))
    elif args.stress:
        print_stress_info(_words, width=args.width, color=_color)
    else:
        print_mds(_words, width=args.width, color=_color)
