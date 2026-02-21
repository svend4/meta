"""solan_bitplane.py — Phonetic Bit-Plane Analysis of Q6 CA Orbits.

Each Q6 cell value v ∈ {0…63} encodes six binary phonetic features:

    bit 0 (T , value  1) — tip / apical
    bit 1 (B , value  2) — bottom / bilabial
    bit 2 (L , value  4) — left / labial
    bit 3 (R , value  8) — right / dorsal
    bit 4 (D1, value 16) — diameter feature 1
    bit 5 (D2, value 32) — diameter feature 2

For a period-P orbit of width N=16, the six bit planes are:

    plane[b][t][i] = (orbit[t][i] >> b) & 1     b ∈ {0…5}

Each bit plane is itself a binary CA trajectory (P×N matrix of 0/1 values)
with its own period p_b | P.

Frozen Planes
─────────────
A plane with p_b = 1 is "frozen": its binary pattern does not change over
the orbit.  Two sub-types:
  uniform_0 / uniform_1 — all cells identical (fully locked feature)
  patterned             — cells differ but are each constant over time

  МАТ   XOR3 (P=8): D1 plane is uniform_1 — every cell has D1=1 at ALL steps.
  ГОРА  XOR3 (P=2): T  plane is uniform_1 — T=1 always.
  РУЛОН XOR3 (P=8): T  plane is uniform_1 — T=1 always.
  ДОБРО XOR3 (P=8): B  plane is uniform_1 — B=1 always.

Perfect Coupling
────────────────
Two bit planes are "perfectly coupled" (Pearson r = +1.0) when one is
identical to the other across all P×N entries.  "Anti-coupled" means
r = −1.0 (bitwise complement).

Universal pattern across the lexicon:
  L ↔ R  coupling (bit2 == bit3): 25+ words — L/R features almost always
          change in phase; together they encode place-of-articulation.
  T ↔ B  coupling (bit0 == bit1): ~15 words — apical/bilabial co-activate.

Anti-coupling (r = −1):
  B != D1 in ДОБРО, ГОРН, ЗУБР, ГОРОД — bilabial anti-correlates with D1.

Key results (width = 16)
─────────────────────────
  ТУМАН XOR3  (P=8): bit0(T)==bit1(B)  [r=1.0];  no frozen planes
  МАТ   XOR3  (P=8): bit0==bit1==bit2  [T=B=L]; D1 plane uniform_1
  ГОРА  XOR3  (P=2): B==L==R; T plane uniform_1
  ДОБРО XOR3  (P=8): L==R; B plane uniform_1; L != D1 (r=−1)
  ТУНДРА XOR3 (P=8): L==R  AND  D1==D2

Запуск:
    python3 -m projects.hexglyph.solan_bitplane --word ТУМАН --rule xor3
    python3 -m projects.hexglyph.solan_bitplane --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_bitplane --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_bitplane --table --no-color
    python3 -m projects.hexglyph.solan_bitplane --json --word МАТ
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES     = tuple(_ALL_RULES)
BIT_NAMES = ['T', 'B', 'L', 'R', 'D1', 'D2']
_DEFAULT_WIDTH = 16
_COUPLE_THRESH = 1.0 - 1e-9


# ── Bit-plane extraction ──────────────────────────────────────────────────────

def get_bit_plane(
    orbit: list,
    b:     int,
) -> list[tuple[int, ...]]:
    """Return the binary P×N plane for bit b of the orbit."""
    return [tuple((int(v) >> b) & 1 for v in state) for state in orbit]


def plane_period(plane: list[tuple[int, ...]]) -> int:
    """Period of a bit-plane trajectory (minimum recurrence)."""
    seen: dict[tuple[int, ...], int] = {}
    for t, state in enumerate(plane):
        if state in seen:
            return t - seen[state]
        seen[state] = t
    return len(plane)


def frozen_type(plane: list[tuple[int, ...]]) -> str:
    """Classify a frozen (period=1) plane.

    Returns one of:
      'uniform_0'  — all cells 0 at all steps
      'uniform_1'  — all cells 1 at all steps
      'patterned'  — cells may differ but each is constant over time
      'active'     — plane is not frozen (period > 1)
    """
    p = plane_period(plane)
    if p > 1:
        return 'active'
    row = plane[0]
    if all(v == 0 for v in row):
        return 'uniform_0'
    if all(v == 1 for v in row):
        return 'uniform_1'
    return 'patterned'


def cell_activity(plane: list[tuple[int, ...]]) -> list[float]:
    """Fraction of orbit steps where each cell's bit is 1."""
    P = len(plane)
    N = len(plane[0]) if plane else 0
    if P == 0:
        return []
    return [sum(plane[t][i] for t in range(P)) / P for i in range(N)]


def plane_hamming(plane: list[tuple[int, ...]]) -> list[int]:
    """Consecutive Hamming distances (orbit Hamming for this bit plane)."""
    P = len(plane)
    return [
        sum(1 for i in range(len(plane[0])) if plane[t][i] != plane[(t + 1) % P][i])
        for t in range(P)
    ]


# ── Correlation between bit planes ────────────────────────────────────────────

def _pearson(xs: list[int], ys: list[int]) -> float | None:
    """Pearson r between two sequences; returns None if either is constant."""
    n  = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    da  = sum((a - mx) ** 2 for a in xs) ** 0.5
    db  = sum((b - my) ** 2 for b in ys) ** 0.5
    if da < 1e-9 or db < 1e-9:
        return None
    return num / (da * db)


def coupling_matrix(orbit: list) -> list[list[float | None]]:
    """6×6 Pearson correlation matrix between all bit-plane pairs.

    mat[b1][b2] = Pearson r of (plane_b1 flat) vs (plane_b2 flat).
    Diagonal = 1.0.  Entry is None if either plane is constant.
    """
    P     = len(orbit)
    planes = [get_bit_plane(orbit, b) for b in range(6)]
    flats  = [
        [int(planes[b][t][i]) for t in range(P) for i in range(len(orbit[0]))]
        for b in range(6)
    ]
    mat: list[list[float | None]] = [[None] * 6 for _ in range(6)]
    for b1 in range(6):
        mat[b1][b1] = 1.0
        for b2 in range(b1 + 1, 6):
            r = _pearson(flats[b1], flats[b2])
            mat[b1][b2] = r
            mat[b2][b1] = r
    return mat


# ── Per-word summary ──────────────────────────────────────────────────────────

def bitplane_summary(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Bit-plane analysis for one word/rule combination.

    Keys
    ────
    word, rule, period, n_cells
    bit_periods     : list[int]        — period of each bit plane (0..5)
    frozen_types    : list[str]        — 'uniform_0'|'uniform_1'|'patterned'|'active'
    n_active        : int              — bit planes with period > 1
    n_frozen_uniform: int              — uniform_0 or uniform_1 planes
    n_frozen_patterned: int            — patterned frozen planes
    frozen_uniform_bits : list[int]    — indices of uniform frozen planes
    frozen_bit_values   : list[int|None] — 0 or 1 for uniform planes, None otherwise
    coupling        : list[list[float|None]]  — 6×6 Pearson r matrix
    coupled_pairs   : list[tuple[int,int]]    — pairs with r > threshold
    anti_coupled_pairs: list[tuple[int,int]]  — pairs with r < -threshold
    cell_activity   : list[list[float]]   — 6 × N per-cell mean bit activity
    plane_hamming   : list[list[int]]     — 6 × P consecutive Hamming per plane
    """
    from projects.hexglyph.solan_perm import get_orbit

    orbit = get_orbit(word, rule, width)
    P     = len(orbit)
    N     = width

    planes = [get_bit_plane(orbit, b) for b in range(6)]
    periods = [plane_period(pl) for pl in planes]
    ftypes  = [frozen_type(pl) for pl in planes]

    n_active           = sum(1 for ft in ftypes if ft == 'active')
    n_frozen_uniform   = sum(1 for ft in ftypes if ft in ('uniform_0', 'uniform_1'))
    n_frozen_patterned = sum(1 for ft in ftypes if ft == 'patterned')

    frozen_uniform_bits = [b for b, ft in enumerate(ftypes)
                           if ft in ('uniform_0', 'uniform_1')]
    frozen_bit_values = [
        (0 if ftypes[b] == 'uniform_0' else 1 if ftypes[b] == 'uniform_1' else None)
        for b in range(6)
    ]

    cmat = coupling_matrix(orbit)

    coupled_pairs: list[tuple[int, int]] = []
    anti_coupled_pairs: list[tuple[int, int]] = []
    for b1 in range(6):
        for b2 in range(b1 + 1, 6):
            r = cmat[b1][b2]
            if r is None:
                continue
            if r >= _COUPLE_THRESH:
                coupled_pairs.append((b1, b2))
            elif r <= -_COUPLE_THRESH:
                anti_coupled_pairs.append((b1, b2))

    c_activity = [cell_activity(planes[b]) for b in range(6)]
    p_hamming  = [plane_hamming(planes[b]) for b in range(6)]

    return {
        'word':                word,
        'rule':                rule,
        'period':              P,
        'n_cells':             N,
        'bit_periods':         periods,
        'frozen_types':        ftypes,
        'n_active':            n_active,
        'n_frozen_uniform':    n_frozen_uniform,
        'n_frozen_patterned':  n_frozen_patterned,
        'frozen_uniform_bits': frozen_uniform_bits,
        'frozen_bit_values':   frozen_bit_values,
        'coupling':            cmat,
        'coupled_pairs':       coupled_pairs,
        'anti_coupled_pairs':  anti_coupled_pairs,
        'cell_activity':       c_activity,
        'plane_hamming':       p_hamming,
    }


def all_bitplane(
    word:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, dict[str, Any]]:
    """Bit-plane summary for all 4 CA rules."""
    return {r: bitplane_summary(word, r, width) for r in RULES}


def build_bitplane_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict[str, Any]:
    """Full bit-plane analysis for the lexicon."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)
    return {
        'words': list(words),
        'data':  {w: {r: bitplane_summary(w, r, width) for r in RULES}
                  for w in words},
    }


def bitplane_dict(s: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of bitplane_summary."""
    cmat = s['coupling']
    ser_cmat = [[round(v, 6) if v is not None else None for v in row]
                for row in cmat]
    return {
        'word':                s['word'],
        'rule':                s['rule'],
        'period':              s['period'],
        'n_cells':             s['n_cells'],
        'bit_periods':         s['bit_periods'],
        'frozen_types':        s['frozen_types'],
        'n_active':            s['n_active'],
        'n_frozen_uniform':    s['n_frozen_uniform'],
        'n_frozen_patterned':  s['n_frozen_patterned'],
        'frozen_uniform_bits': s['frozen_uniform_bits'],
        'frozen_bit_values':   s['frozen_bit_values'],
        'coupling_matrix':     ser_cmat,
        'coupled_pairs':       [[b1, b2] for b1, b2 in s['coupled_pairs']],
        'anti_coupled_pairs':  [[b1, b2] for b1, b2 in s['anti_coupled_pairs']],
        'cell_activity':       [[round(v, 4) for v in row]
                                for row in s['cell_activity']],
        'plane_hamming':       s['plane_hamming'],
    }


# ── Terminal output ───────────────────────────────────────────────────────────

def _r_str(r: float | None) -> str:
    if r is None:
        return '  n/a '
    return f'{r:+.3f}'


def print_bitplane(
    word:  str,
    rule:  str,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Print bit-plane analysis for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    col   = (_RULE_COLOR.get(rule, '') if color else '')
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    s   = bitplane_summary(word, rule, width)
    P   = s['period']
    N   = s['n_cells']

    print(bold + f"  ◈ Bit-Plane Analysis  {word.upper()}  "
          + col + lbl + reset + bold + f"  (P={P})" + reset)
    print()

    # Per-plane table
    frozen_col = '\033[38;5;220m' if color else ''
    pat_col    = '\033[38;5;117m' if color else ''
    active_col = '\033[38;5;120m' if color else ''

    print(f"  {'Bit':5s}  {'Period':7s}  {'Type':12s}  {'Hamming seq':32s}  Cell activity (first 8)")
    print('  ' + '─' * 90)
    for b in range(6):
        ft   = s['frozen_types'][b]
        pp   = s['bit_periods'][b]
        hd   = s['plane_hamming'][b]
        act  = s['cell_activity'][b]
        hd_s = '[' + ','.join(f'{x:2d}' for x in hd) + ']'
        act_s = ' '.join(f'{a:.2f}' for a in act[:8]) + '…'

        if ft == 'active':
            tc = active_col
        elif 'uniform' in ft:
            tc = frozen_col
        else:
            tc = pat_col

        fval = s['frozen_bit_values'][b]
        ft_display = ft if fval is None else f"{ft}  [{fval}]"
        print(f"  {bold}bit{b}({BIT_NAMES[b]:2s}){reset}  "
              f"p={pp:4d}   {tc}{ft_display:<18}{reset}  "
              f"{dim}{hd_s:32s}{reset}  {dim}{act_s}{reset}")

    print()

    # Coupling matrix
    cmat = s['coupling']
    print(f"  Pearson coupling matrix (bit planes):")
    header = '  ' + ' ' * 12 + ''.join(f'  {BIT_NAMES[b]:3s}' for b in range(6))
    print(header)
    for b1 in range(6):
        row = f"  bit{b1}({BIT_NAMES[b1]:2s})  "
        for b2 in range(6):
            r = cmat[b1][b2]
            if r is None:
                cell = '  ---'
            elif b1 == b2:
                cell = ' 1.00' if color else ' 1.00'
            elif r >= _COUPLE_THRESH:
                cell = ('\033[38;5;120m' if color else '') + ' +1.0' + reset
            elif r <= -_COUPLE_THRESH:
                cell = ('\033[38;5;203m' if color else '') + ' -1.0' + reset
            else:
                cell = f'{r:+.2f}'
            row += f'  {cell:4s}'
        print(row)
    print()

    # Summary
    cp  = s['coupled_pairs']
    acp = s['anti_coupled_pairs']
    fu  = s['frozen_uniform_bits']
    fp  = [b for b, ft in enumerate(s['frozen_types']) if ft == 'patterned']

    if cp:
        cp_str = '  '.join(f"bit{b1}({BIT_NAMES[b1]})==bit{b2}({BIT_NAMES[b2]})"
                           for b1, b2 in cp)
        print(f"  Coupled (+1.0)  : {cp_str}")
    if acp:
        acp_str = '  '.join(f"bit{b1}({BIT_NAMES[b1]})!=bit{b2}({BIT_NAMES[b2]})"
                            for b1, b2 in acp)
        print(f"  Anti-coupled    : {acp_str}")
    if fu:
        fu_str = '  '.join(f"bit{b}({BIT_NAMES[b]})={'0' if s['frozen_types'][b]=='uniform_0' else '1'}"
                           for b in fu)
        print(f"  Frozen uniform  : {fu_str}")
    if fp:
        fp_str = '  '.join(f"bit{b}({BIT_NAMES[b]})" for b in fp)
        print(f"  Frozen patterned: {fp_str}")

    print(f"  Active planes   : {s['n_active']} / 6")
    print()


def print_bitplane_table(
    words: list[str] | None = None,
    rule:  str  = 'xor3',
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Summary table: bit-plane stats for all lexicon words."""
    from projects.hexglyph.solan_lexicon import LEXICON
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''
    lbl   = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Bit-Plane Summary ({lbl}, n={len(words)})" + reset)
    print()
    print(f"  {'Слово':12s}  {'P':>3}  "
          f"{'act':>3}  {'frz':>3}  {'pat':>3}  "
          f"{'Coupled pairs':30s}  Frozen uniform")
    print('  ' + '─' * 86)

    for word in words:
        s  = bitplane_summary(word, rule, width)
        P  = s['period']
        cp_s = '  '.join(
            f"{BIT_NAMES[b1]}={BIT_NAMES[b2]}" for b1, b2 in s['coupled_pairs']
        )
        acp_s = '  '.join(
            f"{BIT_NAMES[b1]}≠{BIT_NAMES[b2]}" for b1, b2 in s['anti_coupled_pairs']
        )
        full_coupling = cp_s
        if acp_s:
            full_coupling += ('  ' if cp_s else '') + dim + acp_s + reset
        fu_s = ' '.join(
            f"{BIT_NAMES[b]}={'0' if s['frozen_types'][b]=='uniform_0' else '1'}"
            for b in s['frozen_uniform_bits']
        )
        print(f"  {word.upper():12s}  {P:>3}  "
              f"{s['n_active']:>3}  {s['n_frozen_uniform']:>3}  "
              f"{s['n_frozen_patterned']:>3}  "
              f"{full_coupling:30s}  {dim}{fu_s}{reset}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bit-plane phonetic analysis of Q6 CA orbits')
    parser.add_argument('--word',  metavar='WORD', default='ТУМАН')
    parser.add_argument('--rule',  choices=list(RULES), default='xor3')
    parser.add_argument('--table', action='store_true',
                        help='Summary table for full lexicon')
    parser.add_argument('--json',  action='store_true')
    parser.add_argument('--width', type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = bitplane_summary(args.word.upper(), args.rule, args.width)
        print(json.dumps(bitplane_dict(s), ensure_ascii=False, indent=2))
    elif args.table:
        print_bitplane_table(rule=args.rule, width=args.width, color=_color)
    else:
        print_bitplane(args.word.upper(), args.rule, args.width, color=_color)
