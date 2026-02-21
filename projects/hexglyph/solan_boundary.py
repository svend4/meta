"""
solan_boundary.py — Spatial XOR-Boundary Analysis of Q6 CA Attractors.

For each orbit step t the *boundary pattern* B[t] is defined as the
cell-wise XOR of each cell with its right neighbour (periodic ring):

    B[t][i] = orbit[t][i]  ⊕  orbit[t][(i+1) mod N]      ∈ {0..63}

Interpretation
──────────────
  B[t][i] = 0   → cell i and cell i+1 carry the SAME Q6 value (no boundary).
  B[t][i] ≠ 0   → the two cells differ; popcount(B[t][i]) counts the number
                    of bit positions where they disagree.

  High n_active = many spatial domain boundaries  (heterogeneous state).
  n_active = 0  → all cells equal (uniform state, e.g. all-0 or all-63).
  n_active = N  → every adjacent pair differs (maximally anti-correlated).

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      B[0] = (0, 0, …, 0).  n_active=0.  Uniform zero state has no boundaries.
      vocab_nz={}  b_period=1  uniform=True

  ГОРА AND  (P=2, anti-phase 47↔1)
      ★ Period compression: b_period=1 < P=2.
        Although the value orbit oscillates (47→1→47→…), the BOUNDARY orbit
        is CONSTANT: B[t] = (46, 46, …, 46) at every step.
        47 XOR 1 = 0b101110 = 46 and 1 XOR 47 = 46 as well.
      n_active=16 (max, every adjacent pair always differs).
      vocab_nz={46}  uniform=True.
      Bit constraints: b₀=b₄; b₁=b₂=b₃=b₅  (from 46 = 0b101110).

  ГОРА XOR3  (P=2, 4-cluster pattern)
      b_period=2=P (no compression).  n_active=16 at both steps.
      B[0]=(30,32,48,14,…)  B[1]=(48,14,30,32,…)  — a phase-shifted copy.
      vocab_nz={14, 30, 32, 48}.
      ★ Bit constraint: b₁=b₂=b₃ in ALL boundary values.
        (14=001110, 30=011110, 32=100000, 48=110000 — bits 1,2,3 always agree.)

  ГОРА OR  (P=1, all=63)
      B[0] = (0, 0, …, 0).  n_active=0.  Uniform saturated state, no boundaries.

  ТУМАН XOR3  (P=8)
      b_period=8=P.  n_active varies: [15, 16, 16, 16, 16, 16, 16, 13].
      Steps 0 and 7 have one and three zero-boundary slots respectively.
      vocab_nz contains 15 distinct non-zero values.
      ★ Bit constraint: b₀=b₁ in ALL 16 boundary values (including 0).
        This follows from the coact constraint (solan_coact) that b₀=b₁ in
        every Q6 orbit value of ТУМАН XOR3 — the XOR of two such values
        also has b₀=b₁, so the same constraint propagates to boundaries.

Functions
─────────
  boundary_step(state)                          → tuple[int,...]
  boundary_orbit(word, rule, width)             → list[tuple[int,...]]
  boundary_period(word, rule, width)            → int
  period_compressed(word, rule, width)          → bool
  n_active_boundaries(word, rule, width)        → list[int]
  n_zero_boundaries(word, rule, width)          → list[int]
  boundary_vocab_nz(word, rule, width)          → list[int]
  boundary_vocab_all(word, rule, width)         → list[int]
  boundary_uniform(word, rule, width)           → bool
  boundary_bit_constraints(word, rule, width)   → list[tuple[int,int]]
  boundary_summary(word, rule, width)           → dict
  all_boundary(word, width)                     → dict[str, dict]
  build_boundary_data(words, width)             → dict
  print_boundary(word, rule, color)             → None
  print_boundary_table(words, color)            → None

Запуск
──────
  python3 -m projects.hexglyph.solan_boundary --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_boundary --table --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


def _get_orbit(word: str, rule: str, width: int) -> list[tuple[int, ...]]:
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


# ── Core boundary functions ────────────────────────────────────────────────────

def boundary_step(state: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Compute boundary pattern for a single CA state.

    B[i] = state[i] XOR state[(i+1) % N].
    B[i] = 0   → cell i and i+1 are identical (no spatial boundary).
    B[i] ≠ 0   → cells differ; popcount gives the number of differing bits.
    """
    N = len(state)
    return tuple(state[i] ^ state[(i + 1) % N] for i in range(N))


def boundary_orbit(word: str, rule: str,
                   width: int = _DEFAULT_W) -> list[tuple[int, ...]]:
    """Boundary orbit: B[t] = boundary_step(orbit[t]) for t = 0..P−1."""
    orbit = _get_orbit(word, rule, width)
    return [boundary_step(state) for state in orbit]


def boundary_period(word: str, rule: str,
                    width: int = _DEFAULT_W) -> int:
    """Period of the boundary orbit {B[t]}.

    The boundary orbit period divides the value-orbit period P, but may
    be strictly shorter (period compression).  See ГОРА AND example.
    """
    b_orbit = boundary_orbit(word, rule, width)
    P = len(b_orbit)
    for period in range(1, P + 1):
        if all(b_orbit[t] == b_orbit[t % period] for t in range(P)):
            return period
    return P


def period_compressed(word: str, rule: str,
                      width: int = _DEFAULT_W) -> bool:
    """True when the boundary period is strictly less than the value period."""
    from projects.hexglyph.solan_traj import word_trajectory
    P = word_trajectory(word.upper(), rule, width)['period']
    return boundary_period(word, rule, width) < P


def n_active_boundaries(word: str, rule: str,
                         width: int = _DEFAULT_W) -> list[int]:
    """Number of non-zero boundary slots at each orbit step.

    n_active[t] = |{i : B[t][i] ≠ 0}| ∈ {0..N}.
    0 → uniform step; N → every adjacent pair differs.
    """
    b_orbit = boundary_orbit(word, rule, width)
    return [sum(1 for x in b if x != 0) for b in b_orbit]


def n_zero_boundaries(word: str, rule: str,
                       width: int = _DEFAULT_W) -> list[int]:
    """Number of zero boundary slots at each orbit step (same-neighbour pairs)."""
    b_orbit = boundary_orbit(word, rule, width)
    return [sum(1 for x in b if x == 0) for b in b_orbit]


def boundary_vocab_nz(word: str, rule: str,
                       width: int = _DEFAULT_W) -> list[int]:
    """Sorted list of distinct non-zero boundary values across the whole orbit."""
    b_orbit = boundary_orbit(word, rule, width)
    return sorted(set(x for b in b_orbit for x in b if x != 0))


def boundary_vocab_all(word: str, rule: str,
                        width: int = _DEFAULT_W) -> list[int]:
    """Sorted list of all distinct boundary values (including 0 if present)."""
    b_orbit = boundary_orbit(word, rule, width)
    return sorted(set(x for b in b_orbit for x in b))


def boundary_uniform(word: str, rule: str,
                      width: int = _DEFAULT_W) -> bool:
    """True when the boundary pattern is the same at every orbit step.

    Equivalent to boundary_period == 1 AND n_active uniform.
    ГОРА AND: True (constant (46,46,…,46)).
    ТУМАН XOR: True (constant (0,0,…,0)).
    """
    b_orbit = boundary_orbit(word, rule, width)
    return len(set(b_orbit)) == 1


def boundary_bit_constraints(word: str, rule: str,
                              width: int = _DEFAULT_W) -> list[tuple[int, int]]:
    """Return bit-pairs (b, b') where bit b always equals bit b' in ALL boundary values.

    This reveals hidden algebraic structure in the boundary orbit.
    Example: ТУМАН XOR3 → [(0, 1)] because b₀=b₁ in every boundary value,
    a consequence of the coact constraint (solan_coact) that b₀=b₁ in the
    value orbit itself.
    """
    all_vals: set[int] = set(x for b in boundary_orbit(word, rule, width)
                             for x in b)
    constraints: list[tuple[int, int]] = []
    for b1 in range(6):
        for b2 in range(b1 + 1, 6):
            if all(((v >> b1) & 1) == ((v >> b2) & 1) for v in all_vals):
                constraints.append((b1, b2))
    return constraints


# ── Summary functions ──────────────────────────────────────────────────────────

def boundary_summary(word: str, rule: str,
                     width: int = _DEFAULT_W) -> dict:
    """Full boundary summary for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    b_orbit  = boundary_orbit(word, rule, width)
    P        = len(b_orbit)
    b_period = boundary_period(word, rule, width)
    n_act    = [sum(1 for x in b if x != 0) for b in b_orbit]
    n_zer    = [sum(1 for x in b if x == 0) for b in b_orbit]
    vocab_nz = sorted(set(x for b in b_orbit for x in b if x != 0))
    vocab_al = sorted(set(x for b in b_orbit for x in b))
    uniform  = len(set(b_orbit)) == 1
    all_vals = set(x for b in b_orbit for x in b)
    constraints = []
    for b1 in range(6):
        for b2 in range(b1 + 1, 6):
            if all(((v >> b1) & 1) == ((v >> b2) & 1) for v in all_vals):
                constraints.append((b1, b2))
    return {
        'word':               word.upper(),
        'rule':               rule,
        'period':             P,
        'b_period':           b_period,
        'period_compressed':  b_period < P,
        'n_active':           n_act,
        'n_zero':             n_zer,
        'mean_n_active':      round(sum(n_act) / P, 6) if P else 0.0,
        'max_n_active':       max(n_act) if n_act else 0,
        'min_n_active':       min(n_act) if n_act else 0,
        'vocab_nz':           vocab_nz,
        'vocab_nz_size':      len(vocab_nz),
        'vocab_all':          vocab_al,
        'uniform':            uniform,
        'bit_constraints':    constraints,
        'b_orbit':            b_orbit,
    }


def all_boundary(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """boundary_summary for all 4 rules."""
    return {rule: boundary_summary(word, rule, width) for rule in _RULES}


def build_boundary_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact boundary data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = boundary_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'b_period', 'period_compressed',
                                   'n_active', 'mean_n_active',
                                   'max_n_active', 'min_n_active',
                                   'vocab_nz_size', 'vocab_nz',
                                   'uniform', 'bit_constraints')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'


def print_boundary(word: str = 'ТУМАН', rule: str = 'xor3',
                   color: bool = True) -> None:
    d   = boundary_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',  'or':   'OR |'}.get(rule, rule)

    compress_flag = ' ★compressed' if d['period_compressed'] else ''
    print(f'  {c}◈ Boundary  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'b_period={d["b_period"]}{compress_flag}{r}')
    print('  ' + '─' * 64)
    print(f'  n_active={d["n_active"]}  mean={d["mean_n_active"]:.2f}')
    print(f'  uniform={d["uniform"]}  '
          f'vocab_nz_size={d["vocab_nz_size"]}')
    if d['vocab_nz']:
        entries = '  '.join(f'{v}={v:06b}({bin(v).count("1")}Δ)'
                             for v in d['vocab_nz'][:8])
        print(f'  vocab_nz: {entries}')
    if d['bit_constraints']:
        print(f'  bit_constraints: {d["bit_constraints"]}')
    print()
    # Print boundary orbit rows
    for t, row in enumerate(d['b_orbit']):
        cells = ''.join(
            ('·' if v == 0
             else str(bin(v).count('1')))
            for v in row
        )
        print(f'  t={t}: [{cells}]')
    print()


def print_boundary_table(words: list[str] | None = None,
                          color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    head = '  '.join(
        (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5s} bp cmp n_act vc' + R
        for rl in _RULES)
    print(f'  {"Слово":10s}  {head}')
    print('  ' + '─' * 82)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = _RCOL.get(rule, '') if color else ''
            d   = boundary_summary(word, rule)
            cmp = '★' if d['period_compressed'] else ' '
            na  = d['mean_n_active']
            parts.append(f'{col}{d["b_period"]:>2d} {cmp} '
                          f'{na:>4.1f} {d["vocab_nz_size"]:>2d}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Spatial XOR-Boundary Analysis of Q6 CA')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args  = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.table:
        print_boundary_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_boundary(args.word, rule, color)
    else:
        print_boundary(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
