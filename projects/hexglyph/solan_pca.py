"""solan_pca.py — Principal Component Analysis of Q6 CA Orbit Trajectories.

The P orbit states {x_0, …, x_{P−1}} ⊂ R^N (N = 16 cells) form a point
cloud in Q6 state space.  PCA decomposes the variance of this cloud into
orthogonal principal directions (PCs), answering:

    • Which cells jointly drive the most temporal variation of the orbit?
    • How many independent "modes of variation" does the orbit have?
    • Is the orbit low-dimensional (variance concentrated in few PCs)?

Method
──────────────────────────────────────────────────────────────────────────────
  Centre: X̃[t] = x_t − μ   (μ = mean over t)

  Because P ≤ 8 ≪ N = 16 the P×P Gram matrix G is computed instead of the
  N×N covariance:

      G[t₁][t₂] = ⟨X̃[t₁], X̃[t₂]⟩ / P

  Eigendecomposition of G (via power-iteration + deflation) yields the same
  non-zero eigenvalues as the covariance matrix C = X̃ᵀX̃/P.

  Cell loadings of the k-th PC:
      v_k[i] = Σ_t X̃[t][i] · u_k[t] / √(P · λ_k)

  where u_k is the k-th eigenvector of G with eigenvalue λ_k.
  v_k is normalised to unit length.

Key discoveries
──────────────────────────────────────────────────────────────────────────────
  P = 2 words (20/49 lexicon words, XOR3):
      rank = 1 exactly.  PC₁ explains 100 % of variance — the orbit is a
      two-point oscillation, trivially 1-dimensional in N-space.

  P = 8 words (29/49), XOR3:
      rank = 7 (P − 1) in all cases.  PC₁ explains 28 – 53 % of variance:
      • ТОННА  XOR3: PC₁ = 52.9 %  (most concentrated orbit)
      • ЗАВОД  XOR3: PC₁ = 27.8 %  (most spread, many near-equal modes)

  РАБОТА XOR3  (P=8):
      PC₁ dominant cell = 1, loading = +0.666
      → Cell 1 drives 66 % of the first mode of variation.
      Cross-check: cell 1 also has the maximum turn-count (6 turns) in
      solan_run.py — cells with high temporal volatility dominate PC₁.

  НИТРО XOR3  (P=8):
      PC₁ dominant cell = 5, loading = −0.640  (second-highest magnitude)

  XOR rule (P = 1, fixed point):
      rank = 0, total_var = 0 — no variance at all.

Запуск:
    python3 -m projects.hexglyph.solan_pca --word РАБОТА --rule xor3
    python3 -m projects.hexglyph.solan_pca --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_pca --table --rule xor3
    python3 -m projects.hexglyph.solan_pca --json --word РАБОТА
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES          = tuple(_ALL_RULES)
_DEFAULT_WIDTH = 16
_EIG_ITERS     = 400
_EIG_TOL       = 1e-12
_EVR_THRESH    = 0.05   # "meaningful" PC threshold


# ── Eigendecomposition (power iteration + deflation) ─────────────────────────

def _mat_vec(A: list[list[float]], v: list[float]) -> list[float]:
    P = len(A)
    return [sum(A[i][j] * v[j] for j in range(P)) for i in range(P)]


def _dot(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def _norm(v: list[float]) -> float:
    return math.sqrt(_dot(v, v))


def _normalise(v: list[float]) -> list[float]:
    n = _norm(v)
    return [x / n for x in v] if n > 1e-20 else v[:]


def _power_eigvec(
    A: list[list[float]],
    start: list[float],
    n_iter: int = _EIG_ITERS,
    tol: float  = _EIG_TOL,
) -> list[float]:
    """Dominant eigenvector of symmetric matrix A via power iteration."""
    v = _normalise(start)
    for _ in range(n_iter):
        vn = _normalise(_mat_vec(A, v))
        if _norm([vn[i] - v[i] for i in range(len(v))]) < tol:
            return vn
        v = vn
    return v


def gram_eig(
    G:      list[list[float]],
    n_iter: int   = _EIG_ITERS,
    tol:    float = _EIG_TOL,
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors of symmetric P×P matrix G.

    Returns (eigenvalues, eigenvectors) sorted by eigenvalue descending.
    Uses power iteration with deflation.  Works correctly for P ≤ 8.
    """
    P = len(G)
    if P == 0:
        return [], []
    vals:   list[float]       = []
    vecs:   list[list[float]] = []
    Ac: list[list[float]] = [row[:] for row in G]

    for k in range(P):
        # Different starting vector for each component
        start = [1.0 if i == k else 0.1 * (i + 1) / P for i in range(P)]
        u = _power_eigvec(Ac, start, n_iter, tol)
        lam = _dot(u, _mat_vec(Ac, u))
        lam = max(0.0, lam)          # clip numerical noise
        vals.append(lam)
        vecs.append(u)
        # Deflation: Ac -= lam * u @ uᵀ
        for i in range(P):
            for j in range(P):
                Ac[i][j] -= lam * u[i] * u[j]

    # Sort descending by eigenvalue
    pairs = sorted(zip(vals, vecs), key=lambda x: -x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ── Core computation ──────────────────────────────────────────────────────────

def orbit_pca(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """PCA of the P×N orbit matrix.

    Returns raw dict with keys:
        period, n_cells, eigenvalues, eigenvecs,
        total_var, evr, cumevr, loadings, mean_state
    """
    from projects.hexglyph.solan_transfer import get_orbit

    orbit = get_orbit(word, rule, width)
    P, N  = len(orbit), width

    # Fixed point: no variation
    if P == 1:
        return {
            'period': P, 'n_cells': N,
            'eigenvalues': [], 'eigenvecs': [],
            'total_var': 0.0, 'evr': [], 'cumevr': [],
            'loadings': [[0.0] * N] * P,
            'mean_state': [float(orbit[0][i]) for i in range(N)],
        }

    # Build centred orbit matrix X (P × N)
    raw    = [[float(orbit[t][i]) for i in range(N)] for t in range(P)]
    mean_s = [sum(raw[t][i] for t in range(P)) / P for i in range(N)]
    X      = [[raw[t][i] - mean_s[i] for i in range(N)] for t in range(P)]

    # P×P Gram matrix G[t1][t2] = <X̃[t1], X̃[t2]> / P
    G: list[list[float]] = [
        [sum(X[t1][i] * X[t2][i] for i in range(N)) / P for t2 in range(P)]
        for t1 in range(P)
    ]

    vals, uvecs = gram_eig(G)
    total_var = sum(vals)

    evr: list[float] = (
        [v / total_var for v in vals] if total_var > 1e-20 else [0.0] * P
    )
    cumevr: list[float] = []
    cum = 0.0
    for e in evr:
        cum += e
        cumevr.append(min(cum, 1.0))

    # Cell loadings for each PC: v_k[i] = Σ_t X[t][i]*u_k[t] / sqrt(P*λ_k)
    loadings: list[list[float]] = []
    for k, (lam, uk) in enumerate(zip(vals, uvecs)):
        if lam < 1e-10:
            loadings.append([0.0] * N)
            continue
        vk = [sum(X[t][i] * uk[t] for t in range(P)) for i in range(N)]
        nm = _norm(vk)
        if nm > 1e-20:
            vk = [x / nm for x in vk]
        loadings.append(vk)

    return {
        'period':      P,
        'n_cells':     N,
        'eigenvalues': vals,
        'eigenvecs':   uvecs,
        'total_var':   total_var,
        'evr':         evr,
        'cumevr':      cumevr,
        'loadings':    loadings,
        'mean_state':  mean_s,
    }


# ── Per-word summary ──────────────────────────────────────────────────────────

def pca_summary(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """PCA orbit statistics for one word/rule.

    Keys
    ────
    word, rule, period, n_cells
    eigenvalues         : list[float]   — sorted descending (length = period)
    explained_var_ratio : list[float]   — fraction each PC explains
    cumulative_evr      : list[float]   — running sum of EVR
    total_var           : float         — total orbit variance
    orbit_rank          : int           — non-zero eigenvalues (≥ 0.01)
    n_components_95     : int           — PCs needed for 95% cumulative EVR

    pc1_loadings        : list[float]   — N cell loadings of first PC
    pc1_dom_cell        : int           — cell with max |loading| in PC1
    pc1_dom_loading     : float         — that loading value
    pc1_var_ratio       : float         — EVR of PC1
    n_pcs_meaningful    : int           — PCs with EVR > 5%

    all_loadings        : list[list[float]]  — loadings for all PCs
    mean_state          : list[float]        — mean Q6 value per cell
    """
    raw = orbit_pca(word, rule, width)
    P   = raw['period']
    N   = raw['n_cells']

    evals = raw['eigenvalues']
    evr   = raw['evr']
    cum   = raw['cumevr']

    orbit_rank = sum(1 for v in evals if v > 0.01 * (raw['total_var'] / max(P, 1)))
    n95 = next((i + 1 for i, c in enumerate(cum) if c >= 0.95), len(cum))

    pc1_load = raw['loadings'][0] if raw['loadings'] else [0.0] * N
    dom_cell = max(range(N), key=lambda i: abs(pc1_load[i]))

    return {
        'word':    word,
        'rule':    rule,
        'period':  P,
        'n_cells': N,

        'eigenvalues':         evals,
        'explained_var_ratio': evr,
        'cumulative_evr':      cum,
        'total_var':           round(raw['total_var'], 4),
        'orbit_rank':          orbit_rank,
        'n_components_95':     n95,

        'pc1_loadings':    pc1_load,
        'pc1_dom_cell':    dom_cell,
        'pc1_dom_loading': round(pc1_load[dom_cell], 6),
        'pc1_var_ratio':   round(evr[0], 6) if evr else 0.0,
        'n_pcs_meaningful': sum(1 for e in evr if e > _EVR_THRESH),

        'all_loadings':   raw['loadings'],
        'mean_state':     raw['mean_state'],
    }


def all_pca(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict[str, Any]]:
    """pca_summary for all 4 CA rules."""
    return {r: pca_summary(word, r, width) for r in RULES}


def build_pca_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Full PCA analysis for the lexicon."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)
    return {
        'words': list(words),
        'data':  {w: {r: pca_summary(w, r, width) for r in RULES}
                  for w in words},
    }


def pca_dict(s: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of pca_summary."""
    return {
        k: ([round(v, 8) for v in val] if isinstance(val, list) and
            val and isinstance(val[0], float) else val)
        for k, val in s.items()
        if k != 'all_loadings'  # omit full loading matrix from default export
    } | {'all_loadings': [[round(x, 8) for x in row]
                           for row in s['all_loadings']]}


# ── Terminal output ───────────────────────────────────────────────────────────

_BAR_WIDTH = 24


def _loading_bar(v: float, color: bool, pos_col: str, neg_col: str) -> str:
    """Visualise a loading value as a centred bar."""
    frac = min(abs(v), 1.0)
    half = _BAR_WIDTH // 2
    filled = max(1, round(frac * half)) if frac > 0.02 else 0
    if v >= 0:
        bar = ' ' * half + (pos_col if color else '') + '█' * filled + ('\033[0m' if color else '')
    else:
        bar = ' ' * (half - filled) + (neg_col if color else '') + '█' * filled + ('\033[0m' if color else '') + ' ' * half
    return bar + f' {v:+.3f}'


def print_pca(
    word:  str,
    rule:  str,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print PCA orbit analysis for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    s = pca_summary(word, rule, width)
    P = s['period']
    N = s['n_cells']

    print(bold + f"  ◈ Orbit PCA  {word.upper()}  "
          + col + lbl + reset + bold + f"  (P={P})" + reset)
    print()

    if P == 1:
        print(f"  Fixed point (P=1): no temporal variance.")
        print()
        return

    # Eigenvalue / EVR table
    evr = s['explained_var_ratio']
    cum = s['cumulative_evr']
    print(f"  {'PC':>3}  {'eigenval':>10}  {'EVR':>7}  {'cumEVR':>7}  bar")
    print('  ' + '─' * 52)
    hi_col = '\033[38;5;214m' if color else ''
    for k, (lam, e, c) in enumerate(zip(s['eigenvalues'], evr, cum)):
        bar_len = max(1, round(e * 30)) if e > 0.01 else 0
        bar  = '█' * bar_len
        bc   = (hi_col if e == max(evr) else dim) if color else ''
        star = ' ←' if e > _EVR_THRESH else ''
        print(f"  {k+1:3d}  {lam:10.2f}  {e:7.3f}  {c:7.3f}  "
              f"{bc}{bar}{reset}{star}")

    print()
    print(f"  orbit_rank      : {s['orbit_rank']}")
    print(f"  n_meaningful PCs: {s['n_pcs_meaningful']}  (EVR > {_EVR_THRESH:.0%})")
    print(f"  n_PCs for 95%%   : {s['n_components_95']}")
    print(f"  total variance  : {s['total_var']:.2f}")
    print()

    # PC1 cell loadings
    pos_col = '\033[38;5;214m' if color else ''
    neg_col = '\033[38;5;39m'  if color else ''
    dom     = s['pc1_dom_cell']
    print(f"  PC₁ cell loadings  (EVR = {s['pc1_var_ratio']:.1%}):")
    print(f"  {'cell':>4}  {'loading bar':>48s}  dominant?")
    print('  ' + '─' * 56)
    for i, v in enumerate(s['pc1_loadings']):
        bar  = _loading_bar(v, color, pos_col, neg_col)
        star = ' ★' if i == dom else ''
        print(f"  {i:4d}  {bar}{star}")
    print()
    print(f"  Dominant cell : {dom}  (loading = {s['pc1_dom_loading']:+.4f})")
    print()


def print_pca_table(
    words: list[str] | None = None,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Summary table: PCA stats for all lexicon words."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Orbit PCA Summary ({lbl}, n={len(words)})" + reset)
    print()
    print(f"  {'Слово':12s}  {'P':>3}  {'rank':>4}  {'PC1%':>5}  "
          f"{'n_mPCs':>6}  {'n95':>3}  {'dom_cell':>8}  {'dom_load':>8}")
    print('  ' + '─' * 68)

    for word in words:
        s = pca_summary(word, rule, width)
        if s['period'] == 1:
            print(f"  {word.upper():12s}  {1:>3}  {'0':>4}  {'n/a':>5}  "
                  f"{'—':>6}  {'—':>3}  {'—':>8}  {'—':>8}")
            continue
        print(f"  {word.upper():12s}  {s['period']:>3}  "
              f"{s['orbit_rank']:>4}  {s['pc1_var_ratio']:>5.1%}  "
              f"{s['n_pcs_meaningful']:>6}  {s['n_components_95']:>3}  "
              f"{s['pc1_dom_cell']:>8}  {s['pc1_dom_loading']:>+8.4f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PCA of Q6 CA orbit trajectories')
    parser.add_argument('--word',   metavar='WORD', default='РАБОТА')
    parser.add_argument('--rule',   choices=list(RULES), default='xor3')
    parser.add_argument('--table',  action='store_true')
    parser.add_argument('--json',   action='store_true')
    parser.add_argument('--width',  type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = pca_summary(args.word.upper(), args.rule, args.width)
        print(json.dumps(pca_dict(s), ensure_ascii=False, indent=2))
    elif args.table:
        print_pca_table(rule=args.rule, width=args.width, color=_color)
    else:
        print_pca(args.word.upper(), args.rule, args.width, _color)
