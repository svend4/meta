"""
solan_coact.py — Bit Co-activation & Pearson Correlation Analysis of Q6 CA.

For a cell with temporal series [v_0, …, v_{P−1}] the 6-bit values define
6 binary indicator processes:  X_b(t) = (v_t >> b) & 1  ∈ {0, 1}.

Two matrices characterise their joint behaviour:

  Joint probability   J[b][b'] = P(X_b=1 AND X_{b'}=1)
                                = (# steps with both bits = 1) / P

  Pearson correlation  r[b][b'] = Cov(X_b, X_{b'}) / sqrt(Var_b * Var_{b'})
                                  (= 0 when either bit is frozen, var = 0)

Diagonal  r[b][b] = 1 unless bit b is frozen in some cells
  (aggregate diagonal < 1 signals that some cells have frozen bit b).

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      All bits frozen at 0 → J = 0, r = 0 everywhere.

  ГОРА AND  (P=2, 47↔1)
      Bits 0 and 4 are frozen (var=0) → r = 0 for any pair involving them.
      Bits {1,2,3,5} all move together (all ON in 47, all OFF in 1) →
        r[b][b'] = +1.0 for any pair b, b' ∈ {1, 2, 3, 5}.
      Result: 4×4 block of perfect correlations; frozen rows/cols = 0.

  ТУМАН XOR3  (P=8)
      ★ Hidden dependency: bit 0 = bit 1 in EVERY value of the orbit
        → r[0][1] = +1.0 (across ALL 16 cells).
      Aggregate diagonal:
        r[0][0]=r[1][1]=1.00  (active, no frozen cells)
        r[2][2]=r[4][4]=r[5][5]≈0.88 (2/16 cells have frozen bit, contribute 0)
        r[3][3]=1.00
      Weak negative correlations: r[0][5]≈−0.15, r[0][3]≈−0.10 (anti-correlated)
      Result: b0–b1 block visible; partial structure from parity constraint.

Interpretation
  r = +1   : bits always active together (co-expressed)
  r = −1   : bits perfectly anti-correlated (mutually exclusive activation)
  r =  0   : bits statistically independent, or one is frozen
  agg diag < 1 : some cells have a frozen bit, reducing the aggregate

Functions
─────────
  bit_joint_prob(series)                          → list[list[float]]  6×6
  bit_pearson_corr(series)                        → list[list[float]]  6×6
  cell_coact_stats(series)                        → dict
  aggregate_joint_prob(word, rule, width)         → list[list[float]]  6×6
  aggregate_pearson(word, rule, width)            → list[list[float]]  6×6
  top_corr_pairs(matrix, n, abs_threshold)        → list[tuple]
  coact_summary(word, rule, width)                → dict
  all_coact(word, width)                          → dict[str, dict]
  build_coact_data(words, width)                  → dict
  print_coact(word, rule, color)                  → None
  print_coact_stats(words, color)                 → None

Запуск
──────
  python3 -m projects.hexglyph.solan_coact --word ГОРА --rule and
  python3 -m projects.hexglyph.solan_coact --word ТУМАН --all-rules --no-color
  python3 -m projects.hexglyph.solan_coact --stats --no-color
"""

from __future__ import annotations
import sys
import argparse
import math

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_NBITS:     int = 6
_EPS:       float = 1e-12


# ── Core per-cell computation ─────────────────────────────────────────────────

def bit_joint_prob(series: list[int]) -> list[list[float]]:
    """
    6×6 joint probability matrix: J[b][b'] = P(bit b = 1 AND bit b' = 1).

    Diagonal J[b][b] = P(bit b = 1) = balance_b.
    J is symmetric: J[b][b'] = J[b'][b].
    """
    P   = len(series)
    mat = [[0.0] * _NBITS for _ in range(_NBITS)]
    if P == 0:
        return mat
    for v in series:
        for b in range(_NBITS):
            if (v >> b) & 1:
                for b2 in range(_NBITS):
                    if (v >> b2) & 1:
                        mat[b][b2] += 1
    return [[x / P for x in row] for row in mat]


def bit_pearson_corr(series: list[int]) -> list[list[float]]:
    """
    6×6 Pearson correlation matrix of the 6 bit indicator processes.

    r[b][b'] = Cov(X_b, X_{b'}) / sqrt(Var_b * Var_{b'}).
    Set to 0.0 when either bit has zero variance (frozen bit).
    Diagonal r[b][b] = 1.0 for active bits, 0.0 for frozen bits.
    """
    P   = len(series)
    if P == 0:
        return [[0.0] * _NBITS for _ in range(_NBITS)]

    # Balance (means)
    bal = [sum(1 for v in series if (v >> b) & 1) / P for b in range(_NBITS)]
    jnt = bit_joint_prob(series)

    mat = []
    for b in range(_NBITS):
        var_b = bal[b] * (1 - bal[b])
        row = []
        for b2 in range(_NBITS):
            var_b2 = bal[b2] * (1 - bal[b2])
            if var_b < _EPS or var_b2 < _EPS:
                row.append(0.0)   # frozen bit → no meaningful correlation
            else:
                cov = jnt[b][b2] - bal[b] * bal[b2]
                row.append(cov / math.sqrt(var_b * var_b2))
        mat.append(row)
    return mat


def cell_coact_stats(series: list[int]) -> dict:
    """Full co-activation statistics for a single cell's series."""
    jnt  = bit_joint_prob(series)
    corr = bit_pearson_corr(series)
    # Off-diagonal pairs (b < b')
    pairs = [(corr[b][b2], b, b2)
             for b in range(_NBITS) for b2 in range(b + 1, _NBITS)]
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    max_corr_pair = pairs[0] if pairs else (0.0, 0, 1)
    min_corr_pair = pairs[-1] if pairs else (0.0, 0, 1)
    n_positive  = sum(1 for r, _, _ in pairs if r > 0.1)
    n_negative  = sum(1 for r, _, _ in pairs if r < -0.1)
    n_dependent = sum(1 for r, _, _ in pairs if abs(r) > 0.99)
    return {
        'joint_prob':  jnt,
        'pearson':     corr,
        'max_corr':    round(max_corr_pair[0], 6),
        'max_pair':    (max_corr_pair[1], max_corr_pair[2]),
        'min_corr':    round(min_corr_pair[0], 6),
        'min_pair':    (min_corr_pair[1], min_corr_pair[2]),
        'n_positive':  n_positive,
        'n_negative':  n_negative,
        'n_dependent': n_dependent,
    }


# ── Orbit helper ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int):
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate_joint_prob(word: str, rule: str,
                         width: int = _DEFAULT_W) -> list[list[float]]:
    """Mean joint probability matrix over all cells."""
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    agg   = [[0.0] * _NBITS for _ in range(_NBITS)]
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        jnt    = bit_joint_prob(series)
        for b in range(_NBITS):
            for b2 in range(_NBITS):
                agg[b][b2] += jnt[b][b2]
    return [[round(v / width, 6) for v in row] for row in agg]


def aggregate_pearson(word: str, rule: str,
                      width: int = _DEFAULT_W) -> list[list[float]]:
    """Mean Pearson correlation matrix over all cells."""
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    agg   = [[0.0] * _NBITS for _ in range(_NBITS)]
    for i in range(width):
        series = [orbit[t][i] for t in range(P)]
        corr   = bit_pearson_corr(series)
        for b in range(_NBITS):
            for b2 in range(_NBITS):
                agg[b][b2] += corr[b][b2]
    return [[round(v / width, 6) for v in row] for row in agg]


# ── Top correlated pairs ───────────────────────────────────────────────────────

def top_corr_pairs(matrix: list[list[float]], n: int = 6,
                   abs_threshold: float = 0.0) -> list[tuple]:
    """
    Top n off-diagonal pairs by |r|, sorted descending.

    Returns list of (r, b, b') with b < b'.
    """
    pairs = [(matrix[b][b2], b, b2)
             for b in range(_NBITS) for b2 in range(b + 1, _NBITS)
             if abs(matrix[b][b2]) >= abs_threshold]
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    return pairs[:n]


# ── Summary ───────────────────────────────────────────────────────────────────

def coact_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Comprehensive bit co-activation statistics for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    period = traj['period']

    agg_jp   = aggregate_joint_prob(word, rule, width)
    agg_corr = aggregate_pearson(word, rule, width)

    pairs = top_corr_pairs(agg_corr, n=15, abs_threshold=0.0)
    n_positive  = sum(1 for r, _, _ in pairs if r > 0.1)
    n_negative  = sum(1 for r, _, _ in pairs if r < -0.1)
    n_dependent = sum(1 for r, _, _ in pairs if abs(r) > 0.99)

    # Diagonal: < 1 signals frozen bits in some cells
    diag = [round(agg_corr[b][b], 6) for b in range(_NBITS)]
    n_frozen_bits = sum(1 for d in diag if d < 0.5)

    return {
        'word':        word.upper(),
        'rule':        rule,
        'period':      period,
        'agg_joint':   agg_jp,
        'agg_pearson': agg_corr,
        'top_pairs':   pairs,
        'n_positive':  n_positive,
        'n_negative':  n_negative,
        'n_dependent': n_dependent,
        'diagonal':    diag,
        'n_frozen_bits': n_frozen_bits,
    }


def all_coact(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """coact_summary for all 4 rules."""
    return {rule: coact_summary(word, rule, width) for rule in _RULES}


def build_coact_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact co-activation data (no cell_stats)."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = coact_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in
                ('period', 'agg_pearson', 'n_positive', 'n_negative',
                 'n_dependent', 'diagonal', 'n_frozen_bits')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'


def _corr_char(r: float) -> str:
    if r > 0.9:  return '██'
    if r > 0.5:  return '▓▓'
    if r > 0.2:  return '▒▒'
    if r > 0.05: return '░░'
    if r < -0.9: return '▼▼'
    if r < -0.5: return '▽▽'
    if r < -0.2: return '↓↓'
    if r < -0.05: return '↕↕'
    return '  '


def print_coact(word: str = 'ГОРА', rule: str = 'and',
                color: bool = True) -> None:
    d   = coact_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    print(f'  {c}◈ CoAct  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'n_dep={d["n_dependent"]}  n_pos={d["n_positive"]}  '
          f'n_neg={d["n_negative"]}{r}')
    print('  ' + '─' * 62)

    # Pearson matrix
    print(f'  Aggregate Pearson correlation (b0..b5 × b0..b5):')
    print(f'         ' + ''.join(f'  b{b}  ' for b in range(_NBITS)))
    for b in range(_NBITS):
        row = d['agg_pearson'][b]
        row_str = ''.join(f'{v:+6.2f}' for v in row)
        print(f'    b{b}   {row_str}')

    # Diagonal
    print(f'\n  Diagonal (aggregate): {[round(v, 2) for v in d["diagonal"]]}')
    print(f'  Frozen-bit indicators (diag < 0.5): {d["n_frozen_bits"]} bits')

    # Top pairs
    print(f'\n  Top correlated pairs (|r| ≥ 0.1):')
    for rval, b, b2 in d['top_pairs']:
        if abs(rval) >= 0.1:
            ch = _corr_char(rval)
            print(f'    b{b}—b{b2}: r={rval:+.4f}  {ch}')
    print()


def print_coact_stats(words: list[str] | None = None,
                      color: bool = True) -> None:
    WORDS = words or ['ТУМАН', 'ГОРА']
    for word in WORDS:
        for rule in _RULES:
            print_coact(word, rule, color)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Bit co-activation and Pearson correlation')
    p.add_argument('--word',      default='ГОРА')
    p.add_argument('--rule',      default='and', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--stats',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.stats:
        print_coact_stats(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_coact(args.word, rule, color)
    else:
        print_coact(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
