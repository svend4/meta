"""
solan_width.py — Period vs. Ring Width Scaling of Q6 CA Attractors.

For each word × rule, sweep the ring width N and measure the orbit period P(N).
The scaling P(N) reveals fundamental structural properties:

    P = const      : orbit structure independent of ring size
    P ∝ N          : linear growth (period set by ring circumference)
    P = 1 for 2^k  : nilpotent structure at power-of-two sizes
    P irregular    : complex dependence (sensitive to IC / word encoding)

Default width sweep: N ∈ {4, 8, 12, 16, 20, 24, 32, 48, 64}  (all even).

Key results
───────────
  ГОРА AND   (P=2  for all tested N)
      The anti-phase {47, 1} pattern satisfies AND with ANY even ring size.
      Both values have all 1-bits shared between alternating cells, so
      l AND r = {(47 AND 47)=47, (1 AND 1)=1} regardless of N.
      ★  P(N) ≡ 2  — universally constant.

  ГОРА XOR3  (P=2  for all tested N)
      The 4-fold spatial cluster structure (repating pattern of 4 values)
      locks XOR3 into period 2 for any ring width divisible by 4.
      ★  P(N) ≡ 2  — universal period-2 orbit.

  ЛУНА XOR3  (P=2  for all tested N)
      Same universally-constant P=2 as ГОРА XOR3.

  ТУМАН XOR  (P=1 for N=4,8,16,32,64;  irregular otherwise)
      XOR is a linear CA over GF(2); on a ring of size 2^k the
      transition matrix is nilpotent → orbit converges to 0 in ≤ N steps,
      giving P=1.  For N not a power of 2 (e.g. N=12 → P=4, N=20 → P=3),
      the ring geometry introduces non-trivial periodicity.

  ТУМАН XOR3 (P grows with N, non-monotonically)
      N=4→P=2, N=8→P=4, N=12→P=2, N=16→P=8, N=20→P=3,
      N=24→P=4, N=32→P=16, N=48→P=8, N=64→P=32.
      Rough trend P ≈ N/2 for powers-of-2 widths (N=8,16,32,64).

Functions
─────────
  period_at_width(word, rule, width)         → int
  width_series(word, rule, widths)            → list[int]
  width_summary(word, rule, widths)           → dict
  is_constant_period(word, rule, widths)      → bool
  constant_period_value(word, rule, widths)   → int | None
  max_width_period(word, rule, widths)        → tuple[int, int]
  build_width_data(words, widths)             → dict
  print_width(word, rule, widths, color)      → None
  print_width_table(words, widths, color)     → None

Запуск
──────
  python3 -m projects.hexglyph.solan_width --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_width --table --no-color
  python3 -m projects.hexglyph.solan_width --word ТУМАН --rule xor3 --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:          list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTHS: list[int] = [4, 8, 12, 16, 20, 24, 32, 48, 64]
_DEFAULT_W:      int = 16


# ── Core computation ───────────────────────────────────────────────────────────

def period_at_width(word: str, rule: str, width: int) -> int:
    """Orbit period for the word × rule at a specific ring width N."""
    from projects.hexglyph.solan_perm import get_orbit
    return len(get_orbit(word.upper(), rule, width))


def width_series(word: str, rule: str,
                 widths: list[int] = _DEFAULT_WIDTHS) -> list[int]:
    """List of orbit periods for each ring width in widths."""
    return [period_at_width(word, rule, w) for w in widths]


# ── Analysis ───────────────────────────────────────────────────────────────────

def is_constant_period(word: str, rule: str,
                       widths: list[int] = _DEFAULT_WIDTHS) -> bool:
    """True when the orbit period is the same for all tested widths."""
    ps = width_series(word, rule, widths)
    return len(set(ps)) == 1


def constant_period_value(word: str, rule: str,
                          widths: list[int] = _DEFAULT_WIDTHS) -> int | None:
    """The constant period value if is_constant_period, else None."""
    ps = width_series(word, rule, widths)
    s  = set(ps)
    return s.pop() if len(s) == 1 else None


def max_width_period(word: str, rule: str,
                     widths: list[int] = _DEFAULT_WIDTHS) -> tuple[int, int]:
    """(width, period) pair with the maximum period over the sweep."""
    ps  = width_series(word, rule, widths)
    idx = ps.index(max(ps))
    return (widths[idx], ps[idx])


def width_summary(word: str, rule: str,
                  widths: list[int] = _DEFAULT_WIDTHS) -> dict:
    """Full period-vs-width summary for word × rule."""
    ps      = width_series(word, rule, widths)
    const   = len(set(ps)) == 1
    cval    = ps[0] if const else None
    mxw, mxp = max_width_period(word, rule, widths)
    mnp     = min(ps)
    # Ratio P/N for each width
    pn_ratio = [round(p / w, 4) for p, w in zip(ps, widths)]
    # Is period always a power of 2?
    def _is_pow2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0
    all_pow2 = all(_is_pow2(p) for p in ps)
    # Does period always equal 1?
    all_one  = all(p == 1 for p in ps)
    # Does period always equal 2?
    all_two  = all(p == 2 for p in ps)

    return {
        'word':              word.upper(),
        'rule':              rule,
        'widths':            widths,
        'periods':           ps,
        'pn_ratio':          pn_ratio,
        'is_constant':       const,
        'constant_value':    cval,
        'min_period':        mnp,
        'max_period':        mxp,
        'max_period_width':  mxw,
        'all_pow2':          all_pow2,
        'all_one':           all_one,
        'all_two':           all_two,
        'n_distinct':        len(set(ps)),
    }


def build_width_data(words: list[str],
                     widths: list[int] = _DEFAULT_WIDTHS) -> dict:
    """Period-vs-width data for all words × all 4 rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = width_summary(word, rule, widths)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('periods', 'pn_ratio', 'is_constant',
                                   'constant_value', 'min_period', 'max_period',
                                   'all_pow2', 'all_one', 'all_two', 'n_distinct')
            }
    return {'words': [w.upper() for w in words], 'widths': widths,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'


def print_width(word: str = 'ТУМАН', rule: str = 'xor3',
                widths: list[int] = _DEFAULT_WIDTHS,
                color: bool = True) -> None:
    d   = width_summary(word, rule, widths)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',   'or':   'OR |'}.get(rule, rule)
    const_str = f' → const P={d["constant_value"]}' if d['is_constant'] else ''

    print(f'  {c}◈ Width  {word.upper()}  |  {lbl}  '
          f'n_distinct={d["n_distinct"]}  max_P={d["max_period"]}@N={d["max_period_width"]}'
          f'{const_str}{r}')
    print('  ' + '─' * 62)
    bar_max = max(d['periods']) if d['periods'] else 1
    for w, p, ratio in zip(d['widths'], d['periods'], d['pn_ratio']):
        bar = '█' * int(p / bar_max * 20) + '░' * (20 - int(p / bar_max * 20))
        print(f'  N={w:3d}: P={p:4d}  P/N={ratio:.3f}  {bar}')
    print(f'  all_pow2={d["all_pow2"]}  all_two={d["all_two"]}  '
          f'all_one={d["all_one"]}')
    print()


def print_width_table(words: list[str] | None = None,
                      widths: list[int] = _DEFAULT_WIDTHS,
                      color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    print(f'  {"Слово":10s}  '
          + '  '.join(
              (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5s}  const  max_P' + R
              for rl in _RULES))
    print('  ' + '─' * 68)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col  = (_RCOL.get(rule, '') if color else '')
            d    = width_summary(word, rule, widths)
            flag = ('≡' + str(d['constant_value'])) if d['is_constant'] else '~'
            maxp = d['max_period']
            parts.append(f'{col}{flag:>4s}  {maxp:4d}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='Period vs Ring Width of Q6 CA')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        print(_json.dumps(width_summary(args.word, args.rule), ensure_ascii=False, indent=2))
    elif args.table:
        print_width_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_width(args.word, rule, color=color)
    else:
        print_width(args.word, args.rule, color=color)


if __name__ == '__main__':
    _cli()
