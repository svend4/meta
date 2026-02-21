"""solan_transfer.py — Transfer Entropy (TE) analysis for Q6 CA.

Transfer entropy T(X→Y) measures the directed information flow from cell X
to cell Y along the periodic attractor orbit.  It quantifies "how much does
knowing x_{t-1} reduce uncertainty about y_t beyond what y_{t-1} already
reveals?":

    T(X→Y) = H(Y_t | Y_{t-1}) − H(Y_t | Y_{t-1}, X_{t-1})
           = I(Y_t ; X_{t-1} | Y_{t-1})

Computation is carried out independently on each of the 6 bit-planes of Q6
(bits 0–5) and then summed, giving units of bits:

    T(i→j) = Σ_{b=0}^{5} T_b(cell_i → cell_j)

Empirical distributions are estimated from the P transition tuples on the
periodic attractor (cyclic: index modulo P).

Interpretation per rule:
    XOR  s_j(t+1) = s_{j-1} ⊕ s_{j+1}   → only ±1 neighbours carry TE to j
                    (but P=1 → zero attractor → TE everywhere = 0)
    XOR3 s_j(t+1) = s_{j-1} ⊕ s_j ⊕ s_{j+1}
                                           → TE(j-1→j), TE(j→j), TE(j+1→j) ≠ 0
    AND  s_j(t+1) = s_{j-1} & s_{j+1}    → only ±1 neighbours
    OR   s_j(t+1) = s_{j-1} | s_{j+1}    → only ±1 neighbours

For period-1 attractors (e.g. all-zeros for XOR/AND or all-ones for OR) all
entropies vanish and TE = 0 uniformly.  The most informative cases are
XOR3 ТУМАН (P=8) and XOR3 ГОРА (P=2).

Функции:
    get_orbit(word, rule, width)               → list[tuple[int, …]]
    bit_te(y_bits, x_bits)                     → float
    cell_te(orbit, i, j, n_bits)               → float
    te_matrix(word, rule, width)               → list[list[float]]
    te_asymmetry(mat)                          → list[list[float]]
    te_summary(word, rule, width)              → dict   (≡ te_dict)
    te_dict(word, rule, width)                 → dict
    all_te(word, width)                        → dict[str, dict]
    build_te_data(words, width)                → dict
    print_te(word, rule, width, color)         → None
    print_te_stats(words, width, color)        → None

Запуск:
    python3 -m projects.hexglyph.solan_transfer --word ТУМАН --rule xor3
    python3 -m projects.hexglyph.solan_transfer --word ГОРА --all-rules --no-color
    python3 -m projects.hexglyph.solan_transfer --stats
    python3 -m projects.hexglyph.solan_transfer --word ТУМАН --json
    python3 -m projects.hexglyph.solan_transfer --table
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    step, find_orbit,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16
_N_BITS        = 6
_DEFAULT_WORDS = list(LEXICON)


# ── Attractor orbit ───────────────────────────────────────────────────────────

def get_orbit(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[tuple]:
    """Return ordered list of state tuples on the periodic attractor.

    Length = max(period, 1).
    """
    cells = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells[:], rule)
    period = max(period, 1)
    c = cells[:]
    for _ in range(transient):
        c = step(c, rule)
    states: list[tuple] = []
    for _ in range(period):
        states.append(tuple(c))
        c = step(c, rule)
    return states


# ── Bit-level transfer entropy ────────────────────────────────────────────────

def _h(counts: list[int], total: int) -> float:
    """Shannon entropy in bits from a list of counts."""
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def bit_te(y_bits: list[int], x_bits: list[int]) -> float:
    """Transfer entropy T(X→Y) in bits for two binary time series.

    Series are treated as cyclic (index modulo P).
    Returns a non-negative float (clamped to 0 for numerical noise).

    T(X→Y) = H(Y_t, Y_{t−1}) − H(Y_{t−1}) − H(Y_t, Y_{t−1}, X_{t−1}) + H(Y_{t−1}, X_{t−1})
    """
    P = len(y_bits)
    if P < 2:
        return 0.0

    # Frequency tables indexed by (y_{t-1},), (y_{t-1}, y_t),
    # (y_{t-1}, x_{t-1}), (y_{t-1}, x_{t-1}, y_t)
    c_y1     = [0, 0]          # y_{t-1}
    c_y1_yt  = [[0, 0], [0, 0]]  # [y_{t-1}][y_t]
    c_y1_x1  = [[0, 0], [0, 0]]  # [y_{t-1}][x_{t-1}]
    c_y1x1yt = [[[0, 0], [0, 0]],
                [[0, 0], [0, 0]]]  # [y_{t-1}][x_{t-1}][y_t]

    for t in range(P):
        yt   = y_bits[t]
        yt_1 = y_bits[(t - 1) % P]
        xt_1 = x_bits[(t - 1) % P]
        c_y1[yt_1]            += 1
        c_y1_yt[yt_1][yt]     += 1
        c_y1_x1[yt_1][xt_1]   += 1
        c_y1x1yt[yt_1][xt_1][yt] += 1

    h_y1     = _h(c_y1, P)
    h_y1_yt  = _h([c_y1_yt[a][b]  for a in range(2) for b in range(2)], P)
    h_y1_x1  = _h([c_y1_x1[a][b]  for a in range(2) for b in range(2)], P)
    h_y1x1yt = _h([c_y1x1yt[a][b][c]
                   for a in range(2) for b in range(2) for c in range(2)], P)

    te = (h_y1_yt - h_y1) - (h_y1x1yt - h_y1_x1)
    return max(0.0, round(te, 9))


# ── Cell-pair TE (summed over bits) ──────────────────────────────────────────

def cell_te(
    orbit:  list[tuple],
    i:      int,
    j:      int,
    n_bits: int = _N_BITS,
) -> float:
    """Total TE from cell *i* to cell *j*, summed over all *n_bits* bit-planes."""
    total = 0.0
    for b in range(n_bits):
        y_bits = [(s[j] >> b) & 1 for s in orbit]
        x_bits = [(s[i] >> b) & 1 for s in orbit]
        total += bit_te(y_bits, x_bits)
    return round(total, 8)


# ── Full 16×16 TE matrix ──────────────────────────────────────────────────────

def te_matrix(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> list[list[float]]:
    """Return width×width matrix M where M[i][j] = T(cell_i → cell_j).

    Diagonal entry M[i][i] = T(cell_i → itself) = "self-influence".
    """
    orbit = get_orbit(word, rule, width)
    mat: list[list[float]] = [[0.0] * width for _ in range(width)]
    for j in range(width):
        for i in range(width):
            mat[i][j] = cell_te(orbit, i, j)
    return mat


def te_asymmetry(mat: list[list[float]]) -> list[list[float]]:
    """Antisymmetric part: A[i][j] = M[i][j] − M[j][i].

    Positive → more flow i→j than j→i.
    """
    N = len(mat)
    return [[round(mat[i][j] - mat[j][i], 8) for j in range(N)]
            for i in range(N)]


# ── Summary dict ─────────────────────────────────────────────────────────────

def te_dict(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """JSON-serialisable TE analysis for one word + rule.

    Returns dict:
        word          : str
        rule          : str
        width         : int
        period        : int            (attractor period)
        matrix        : list[list[float]]    (width × width)
        max_te        : float
        mean_te       : float          (mean over off-diagonal)
        self_te       : list[float]    (diagonal, self-influence per cell)
        right_te      : list[float]    (M[i+1][i] per cell — rightward influence)
        left_te       : list[float]    (M[i-1][i] per cell — leftward influence)
        asymmetry     : list[list[float]]
        mean_right    : float
        mean_left     : float
        lr_asymmetry  : float          (mean_right − mean_left)
    """
    orbit = get_orbit(word, rule, width)
    period = len(orbit)
    mat    = te_matrix(word, rule, width)
    asym   = te_asymmetry(mat)

    all_vals = [mat[i][j] for i in range(width) for j in range(width)
                if i != j]
    max_te   = round(max(mat[i][j] for i in range(width)
                         for j in range(width)), 8)
    mean_te  = round(sum(all_vals) / len(all_vals), 8) if all_vals else 0.0

    self_te  = [mat[i][i] for i in range(width)]
    right_te = [mat[(i + 1) % width][i] for i in range(width)]  # i+1 → i
    left_te  = [mat[(i - 1) % width][i] for i in range(width)]  # i-1 → i

    mean_right   = round(sum(right_te) / width, 8)
    mean_left    = round(sum(left_te)  / width, 8)
    lr_asymmetry = round(mean_right - mean_left, 8)

    return {
        'word':         word.upper(),
        'rule':         rule,
        'width':        width,
        'period':       period,
        'matrix':       mat,
        'max_te':       max_te,
        'mean_te':      mean_te,
        'self_te':      self_te,
        'right_te':     right_te,
        'left_te':      left_te,
        'asymmetry':    asym,
        'mean_right':   mean_right,
        'mean_left':    mean_left,
        'lr_asymmetry': lr_asymmetry,
    }


def te_summary(
    word:  str,
    rule:  str = 'xor3',
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """Alias for te_dict — standard *_summary convention."""
    return te_dict(word, rule, width)


def all_te(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict]:
    """te_summary for all 4 rules."""
    return {r: te_summary(word, r, width) for r in _ALL_RULES}


def build_te_data(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
) -> dict:
    """TE summary for the full lexicon × 4 rules.

    Returns dict:
        words       : list[str]
        per_rule    : {rule: {word: {max_te, mean_te, lr_asymmetry, period}}}
        ranking     : {rule: [(word, max_te), …]}   descending
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = te_dict(word, rule, width)
            per_rule[rule][word] = {
                'max_te':       d['max_te'],
                'mean_te':      d['mean_te'],
                'lr_asymmetry': d['lr_asymmetry'],
                'period':       d['period'],
            }
    ranking: dict[str, list] = {}
    for rule in _ALL_RULES:
        ranking[rule] = sorted(
            ((w, v['max_te']) for w, v in per_rule[rule].items()),
            key=lambda x: -x[1],
        )
    return {'words': words, 'per_rule': per_rule, 'ranking': ranking}


# ── ASCII display ─────────────────────────────────────────────────────────────

_SHADE = ' ░▒▓█'


def _shade(v: float, vmax: float) -> str:
    if vmax <= 0:
        return _SHADE[0]
    ratio = v / vmax
    idx   = min(int(ratio * (len(_SHADE) - 1) + 0.5), len(_SHADE) - 1)
    return _SHADE[idx]


def print_te(
    word:  str,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print TE matrix as a shaded ASCII heatmap."""
    d    = te_dict(word, rule, width)
    mat  = d['matrix']
    vmax = d['max_te']
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    print(f"{bold}  ◈ Transfer Entropy Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  P={d['period']}  max={vmax:.4f}  "
          f"lr_asym={d['lr_asymmetry']:+.4f}")
    print(f"  {' '*3}{''.join(f'{j%10}' for j in range(width))}")
    for i in range(width):
        row = ''.join(
            (col if mat[i][j] >= vmax * 0.5 else '') +
            _shade(mat[i][j], vmax) +
            (rst if color and mat[i][j] >= vmax * 0.5 else '')
            for j in range(width)
        )
        print(f"  {i:2d} {row}  ∑={sum(mat[i]):.3f}")
    print(f"  {'─' * 30}")
    print(f"  R→  {''.join(_shade(v, vmax) for v in d['right_te'])}  "
          f"mean={d['mean_right']:.4f}")
    print(f"  L→  {''.join(_shade(v, vmax) for v in d['left_te'])}  "
          f"mean={d['mean_left']:.4f}")
    print(f"  self{''.join(_shade(v, vmax) for v in d['self_te'])}  "
          f"mean={sum(d['self_te'])/width:.4f}")
    print()


def print_te_stats(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
    color: bool             = True,
) -> None:
    """Summary table: max_te for each word × rule."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Transfer Entropy Q6 — max TE (бит){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = te_dict(word, rule, width)
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{d['max_te']:>8.4f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import json as _json
    parser = argparse.ArgumentParser(description='Transfer Entropy Q6 CA')
    parser.add_argument('--word',      default='ТУМАН', help='Русское слово')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--table',     action='store_true', help='Lexicon TE table')
    parser.add_argument('--json',      action='store_true', help='JSON output')
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.json:
        d = te_summary(args.word, args.rule, args.width)
        print(_json.dumps(d, ensure_ascii=False, indent=2))
    elif args.stats or args.table:
        print_te_stats(color=color, width=args.width)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_te(args.word, rule, args.width, color)
    else:
        print_te(args.word, args.rule, args.width, color)


if __name__ == '__main__':
    _main()
