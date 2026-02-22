"""
solan_symm.py — Rotational Symmetry of Q6 CA Attractor States.

A state S = (S_0, …, S_{N−1}) on a ring of N=16 cells is *rotationally
symmetric* if there exists k ∈ {1, …, N} such that

    rotate(S, k) = S    (i.e. S_i ≡ S_{(i+k) mod N} for all i)

The *rotational period* is the smallest such k:

    rot_period(S) = min{ k ≥ 1 : rotate(S, k) = S }

The *rotational order* is the number of distinct symmetry-preserving
rotations:

    rot_order(S) = N / rot_period(S)

High rot_order means high spatial symmetry; rot_order=1 means no
rotational symmetry (the state is maximally asymmetric).

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, all=0)
      rot_period=1, rot_order=16 — maximum symmetry.
      The uniform zero state is invariant under all 16 rotations.

  ГОРА AND  (P=2, anti-phase alternation {47, 1})
      rot_period=2, rot_order=8 — 8-fold symmetry.
      State is invariant under rotations by even numbers of steps.

  ГОРА XOR3  (P=2, 4 spatial clusters repeating period 4)
      rot_period=4, rot_order=4 — 4-fold symmetry.
      ★  All XOR3 period-2 words share this 4-fold ring symmetry.

  ТУМАН XOR3  (P=8)
      rot_period=16, rot_order=1 for every step —
      zero rotational symmetry; each state is maximally asymmetric.

Symmetry hierarchy  (rot_order)
  16 → constant state        (XOR fixed point)
   8 → binary alternation    (AND anti-phase)
   4 → 4-cluster pattern     (XOR3 period-2 universally)
   2 → half-period pattern   (rarely)
   1 → fully asymmetric      (XOR3 period-8 states)

Functions
─────────
  rot_period(state)                         → int
  rot_order(state)                          → int
  orbit_rot_periods(word, rule, width)      → list[int]
  orbit_rot_orders(word, rule, width)       → list[int]
  min_rot_period(word, rule, width)         → int
  max_rot_order(word, rule, width)          → int
  symm_summary(word, rule, width)           → dict
  all_symm(word, width)                     → dict[str, dict]
  build_symm_data(words, width)             → dict
  print_symm(word, rule, color)             → None
  print_symm_table(words, color)            → None

Запуск
──────
  python3 -m projects.hexglyph.solan_symm --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_symm --table --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16


# ── Core functions ─────────────────────────────────────────────────────────────

def rot_period(state: list[int]) -> int:
    """Smallest k ≥ 1 such that rotate(state, k) equals state.

    Returns N if state has no rotational symmetry shorter than a full rotation.
    """
    N = len(state)
    if N == 0:
        return 0
    for k in range(1, N + 1):
        if state[k:] + state[:k] == state:
            return k
    return N          # full period (but this branch is never reached: k=N always matches)


def rot_order(state: list[int]) -> int:
    """Number of distinct symmetry-preserving rotations = N // rot_period."""
    N = len(state)
    if N == 0:
        return 0
    return N // rot_period(state)


# ── Orbit helpers ──────────────────────────────────────────────────────────────

def _get_orbit(word: str, rule: str, width: int) -> list[list[int]]:
    from projects.hexglyph.solan_perm import get_orbit
    return [list(s) for s in get_orbit(word.upper(), rule, width)]


def orbit_rot_periods(word: str, rule: str,
                      width: int = _DEFAULT_W) -> list[int]:
    """Rotational period for each attractor state [t=0, …, P−1]."""
    orbit = _get_orbit(word, rule, width)
    return [rot_period(orbit[t]) for t in range(len(orbit))]


def orbit_rot_orders(word: str, rule: str,
                     width: int = _DEFAULT_W) -> list[int]:
    """Rotational order for each attractor state [t=0, …, P−1]."""
    orbit = _get_orbit(word, rule, width)
    return [rot_order(orbit[t]) for t in range(len(orbit))]


def min_rot_period(word: str, rule: str, width: int = _DEFAULT_W) -> int:
    """Characteristic rotational period: min over all attractor steps.

    A lower value means at least one state has high spatial symmetry.
    """
    periods = orbit_rot_periods(word, rule, width)
    return min(periods) if periods else width


def max_rot_order(word: str, rule: str, width: int = _DEFAULT_W) -> int:
    """Maximum rotational order over all attractor steps."""
    orders = orbit_rot_orders(word, rule, width)
    return max(orders) if orders else 1


# ── Summary ────────────────────────────────────────────────────────────────────

def symm_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full rotational-symmetry summary for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj     = word_trajectory(word.upper(), rule, width)
    period   = traj['period']
    periods  = orbit_rot_periods(word, rule, width)
    orders   = orbit_rot_orders(word, rule, width)

    min_rp   = min(periods) if periods else width
    max_ro   = max(orders)  if orders  else 1
    # Are all states at the same symmetry level?
    uniform  = len(set(periods)) == 1

    # Classify symmetry level
    if max_ro >= width:
        level = 'maximum'      # uniform/constant state
    elif max_ro >= width // 2:
        level = 'high'         # binary alternation
    elif max_ro >= width // 4:
        level = 'moderate'     # 4-cluster
    elif max_ro >= 2:
        level = 'low'
    else:
        level = 'none'         # fully asymmetric

    return {
        'word':            word.upper(),
        'rule':            rule,
        'period':          period,
        'rot_periods':     periods,
        'rot_orders':      orders,
        'min_rot_period':  min_rp,
        'max_rot_order':   max_ro,
        'uniform_symmetry': uniform,
        'symmetry_level':  level,
    }


def all_symm(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """symm_summary for all 4 rules."""
    return {rule: symm_summary(word, rule, width) for rule in _RULES}


def build_symm_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact symmetry data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = symm_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'rot_periods', 'rot_orders',
                                   'min_rot_period', 'max_rot_order',
                                   'uniform_symmetry', 'symmetry_level')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m', 'and': '\033[91m', 'or': '\033[33m'}
_RST  = '\033[0m'
_DIM  = '\033[2m'

_SYM_CHARS = {1: '·', 2: '▪', 4: '◆', 8: '★', 16: '●'}


def print_symm(word: str = 'ГОРА', rule: str = 'xor3',
               color: bool = True) -> None:
    d   = symm_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    dim = _DIM if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3', 'and': 'AND &', 'or': 'OR |'}.get(rule, rule)

    print(f'  {c}◈ Symm  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'level={d["symmetry_level"]}{r}')
    print('  ' + '─' * 62)

    N = 16
    print(f'  Rotational symmetry per attractor step:')
    for t in range(len(d['rot_periods'])):
        rp  = d['rot_periods'][t]
        ro  = d['rot_orders'][t]
        bar = '█' * ro + '░' * (N - ro)
        sym = _SYM_CHARS.get(ro, '?')
        print(f'    t={t:2d}  rot_period={rp:2d}  rot_order={ro:2d}  '
              f'|{bar}|  {sym}')

    print(f'\n  Summary:')
    print(f'    min_rot_period  = {d["min_rot_period"]}  '
          f'(highest symmetry: order={d["max_rot_order"]})')
    print(f'    uniform_symmetry = {d["uniform_symmetry"]}  '
          f'(all steps at same level)')
    print(f'    symmetry_level   = {d["symmetry_level"]}')
    print()


def print_symm_table(words: list[str] | None = None,
                     color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    c_xor  = _RCOL.get('xor',  '') if color else ''
    c_xor3 = _RCOL.get('xor3', '') if color else ''
    c_and  = _RCOL.get('and',  '') if color else ''
    c_or   = _RCOL.get('or',   '') if color else ''
    R = _RST if color else ''

    print(f'  {"Слово":10s}  '
          f'{c_xor}XOR rot_ord{R}  {c_xor3}XOR3 rot_ord{R}  '
          f'{c_and}AND rot_ord{R}  {c_or}OR rot_ord{R}')
    print('  ' + '─' * 60)
    for word in WORDS:
        parts = []
        for rule, col in [('xor',c_xor),('xor3',c_xor3),('and',c_and),('or',c_or)]:
            ro = max_rot_order(word, rule)
            parts.append(f'{col}{ro:>12}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Rotational Symmetry of Q6 CA Attractors')
    p.add_argument('--word',      default='ГОРА')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(symm_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.table:
        print_symm_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_symm(args.word, rule, color)
    else:
        print_symm(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
