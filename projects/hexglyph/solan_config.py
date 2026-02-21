"""
solan_config.py — 1-Bit Neighborhood Configuration Coverage of Q6 CA Attractors.

Because Q6 rules are bitwise, each bit plane b evolves as a 1-bit CA with
rule: new_b(i) = rule_1bit(l_b, c_b, r_b)

A *configuration* is a 1-bit neighborhood triplet (l, c, r) ∈ {0,1}³ — 8 possible.
The *active configurations* for bit b are all (l,c,r) triplets that appear
in the orbit transitions:

    {(bit_b(orbit[t][i−1]), bit_b(orbit[t][i]), bit_b(orbit[t][i+1]))
     : t ∈ {0..P−1}, i ∈ {0..N−1}}

Since the rule is deterministic each config maps to exactly one output.
n_active(b) ∈ {1..8} measures how much of the rule table is used by bit b.

Key results  (N = width = 16)
──────────────────────────────
  ТУМАН XOR  (P=1, all=0)
      All 6 planes: 1/8 configs active — only (0,0,0)→0.
      coverage_vector=[1,1,1,1,1,1]  mean_coverage=0.125.

  ГОРА AND  (P=2, {47,1} anti-phase)
      Bit 0 frozen-1: (1,1,1)→1 only.
      Bit 4 frozen-0: (0,0,0)→0 only.
      Bits 1,2,3,5 alternating: exactly 2/8 configs — (0,1,0)→0 and (1,0,1)→1.
        (AND rule: l AND r; alternating IC forces l=r=0 or l=r=1.)
      coverage_vector=[1,2,2,2,1,2]  mean_coverage=0.208.

  ГОРА XOR3  (P=2, 4 spatial clusters)
      Bit 0 frozen-1: (1,1,1)→1 only.
      Bits 1,2,3,5: ALL 8/8 configs active — despite only P×N=32 observations.
        XOR3's structural richness causes full rule-table coverage.
      Bit 4: exactly 4/8 configs — only l≠r configs:
        (0,0,1)→1, (0,1,1)→0, (1,0,0)→1, (1,1,0)→0.
        ★ Bit 4 of ГОРА only encounters anti-symmetric neighborhoods (l≠r);
          same-side neighborhoods (l=r) never appear for bit 4.
      coverage_vector=[1,8,8,8,4,8]  mean_coverage=0.771.

  ТУМАН XOR3  (P=8)
      ALL 6 bits: 8/8 configs active — FULL RULE TABLE COVERAGE.
      coverage_vector=[8,8,8,8,8,8]  mean_coverage=1.000.
      ★ The attractor visits every possible 1-bit neighborhood for every bit plane.

Coverage hierarchy (mean_coverage)
  ТУМАН XOR : 0.125  (minimal — frozen orbits)
  ГОРА  AND : 0.208  (low — anti-phase with 2 configs per active bit)
  ГОРА  XOR3: 0.771  (high — full for 4/6 bits, partial for bit 4)
  ТУМАН XOR3: 1.000  (maximal — ergodic, explores full rule table)

Functions
─────────
  active_configs(word, rule, bit, width)     → dict[tuple, int]
  n_active_configs(word, rule, bit, width)   → int
  coverage_fraction(word, rule, bit, width)  → float
  coverage_vector(word, rule, width)         → list[int]
  mean_coverage(word, rule, width)           → float
  full_coverage_bits(word, rule, width)      → list[int]
  minimal_coverage_bits(word, rule, width)   → list[int]
  config_transition_table(word, rule, bit, width) → dict[tuple, int]
  config_summary(word, rule, width)          → dict
  all_config(word, width)                    → dict[str, dict]
  build_config_data(words, width)            → dict
  print_config(word, rule, color)            → None
  print_config_table(words, color)           → None

Запуск
──────
  python3 -m projects.hexglyph.solan_config --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_config --table --no-color
"""

from __future__ import annotations
import sys
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
_RULES:     list[str] = ['xor', 'xor3', 'and', 'or']
_DEFAULT_W: int = 16
_N_CONFIGS: int = 8   # 2^3 possible 1-bit neighborhood configurations
# Canonical order of all 8 1-bit configs (l, c, r)
_ALL_CONFIGS: list[tuple[int, int, int]] = [
    (l, c, r) for l in range(2) for c in range(2) for r in range(2)
]


# ── Core computation ───────────────────────────────────────────────────────────

def active_configs(word: str, rule: str, bit: int,
                   width: int = _DEFAULT_W) -> dict[tuple[int, int, int], int]:
    """Active (l,c,r) 1-bit neighborhood configs for bit plane b.

    Returns a dict mapping each observed (l,c,r) triplet to its output bit.
    Because the rule is deterministic, each config maps to exactly one value.
    """
    from projects.hexglyph.solan_perm import get_orbit
    orbit = get_orbit(word.upper(), rule, width)
    P, N  = len(orbit), width
    cfg: dict[tuple[int, int, int], int] = {}
    for t in range(P):
        for i in range(N):
            l   = (orbit[t][(i - 1) % N] >> bit) & 1
            c   = (orbit[t][i]           >> bit) & 1
            r   = (orbit[t][(i + 1) % N] >> bit) & 1
            out = (orbit[(t + 1) % P][i] >> bit) & 1
            cfg[(l, c, r)] = out
    return cfg


def n_active_configs(word: str, rule: str, bit: int,
                     width: int = _DEFAULT_W) -> int:
    """Number of distinct 1-bit configs active in bit plane b (0..8)."""
    return len(active_configs(word, rule, bit, width))


def coverage_fraction(word: str, rule: str, bit: int,
                      width: int = _DEFAULT_W) -> float:
    """Fraction of the 8 possible 1-bit configs active in bit plane b."""
    return round(n_active_configs(word, rule, bit, width) / _N_CONFIGS, 6)


def config_transition_table(word: str, rule: str, bit: int,
                             width: int = _DEFAULT_W) -> dict[tuple[int, int, int], int]:
    """Full 1-bit transition table for the given rule (all 8 configs, not just orbit-active).

    Maps every possible (l, c, r) triplet to its output using the rule.
    """
    def _apply(l: int, c: int, r: int) -> int:
        if rule == 'xor':
            return l ^ r
        elif rule == 'xor3':
            return l ^ c ^ r
        elif rule == 'and':
            return l & r
        else:   # or
            return l | r
    return {(l, c, r): _apply(l, c, r) for l, c, r in _ALL_CONFIGS}


# ── Aggregate over 6 bit planes ────────────────────────────────────────────────

def coverage_vector(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """n_active_configs for each bit b = 0..5.  Values ∈ {1..8}."""
    return [n_active_configs(word, rule, b, width) for b in range(6)]


def mean_coverage(word: str, rule: str, width: int = _DEFAULT_W) -> float:
    """Mean fraction of rule table explored across all 6 bit planes."""
    return round(sum(coverage_vector(word, rule, width)) / (6 * _N_CONFIGS), 6)


def full_coverage_bits(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """Bit indices with all 8 configs active (ergodic bit planes)."""
    return [b for b in range(6) if n_active_configs(word, rule, b, width) == _N_CONFIGS]


def minimal_coverage_bits(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """Bit indices with only 1 config active (frozen or nearly-frozen bit planes)."""
    return [b for b in range(6) if n_active_configs(word, rule, b, width) == 1]


# ── Summary ────────────────────────────────────────────────────────────────────

def config_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full neighborhood configuration coverage summary for word × rule."""
    from projects.hexglyph.solan_traj import word_trajectory
    traj   = word_trajectory(word.upper(), rule, width)
    P      = traj['period']
    cv     = coverage_vector(word, rule, width)
    mc     = mean_coverage(word, rule, width)
    full   = full_coverage_bits(word, rule, width)
    mini   = minimal_coverage_bits(word, rule, width)
    # Per-bit active config mappings
    per_bit = [active_configs(word, rule, b, width) for b in range(6)]
    # Full rule table (independent of IC/orbit)
    rule_table = config_transition_table(word, rule, 0, width)  # same for all bits

    # Unique output patterns among active configs per bit
    # (e.g., bit always maps config to 0 → "frozen output")
    output_diversity = [len(set(per_bit[b].values())) if per_bit[b] else 0
                        for b in range(6)]

    return {
        'word':              word.upper(),
        'rule':              rule,
        'period':            P,
        'coverage_vector':   cv,
        'coverage_fractions': [round(v / _N_CONFIGS, 6) for v in cv],
        'mean_coverage':     mc,
        'full_coverage_bits': full,
        'minimal_coverage_bits': mini,
        'n_full_coverage':   len(full),
        'n_minimal':         len(mini),
        'per_bit_configs':   [sorted(per_bit[b].keys()) for b in range(6)],
        'per_bit_outputs':   [per_bit[b] for b in range(6)],
        'output_diversity':  output_diversity,
        'rule_table':        rule_table,
    }


def all_config(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """config_summary for all 4 rules."""
    return {rule: config_summary(word, rule, width) for rule in _RULES}


def build_config_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact config data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = config_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'coverage_vector',
                                   'mean_coverage', 'full_coverage_bits',
                                   'minimal_coverage_bits',
                                   'n_full_coverage', 'n_minimal',
                                   'output_diversity')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'n_configs': _N_CONFIGS, 'all_configs': _ALL_CONFIGS,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'
_BCOLS = ['\033[34m', '\033[36m', '\033[32m', '\033[33m', '\033[31m', '\033[35m']
_CFG_NAMES = {
    (0,0,0): '000', (0,0,1): '001', (0,1,0): '010', (0,1,1): '011',
    (1,0,0): '100', (1,0,1): '101', (1,1,0): '110', (1,1,1): '111',
}


def print_config(word: str = 'ГОРА', rule: str = 'xor3',
                 color: bool = True) -> None:
    d   = config_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',   'or':   'OR |'}.get(rule, rule)

    print(f'  {c}◈ Config  {word.upper()}  |  {lbl}  P={d["period"]}  '
          f'mean_cov={d["mean_coverage"]:.3f}  '
          f'full={d["full_coverage_bits"]}  '
          f'minimal={d["minimal_coverage_bits"]}{r}')
    print('  ' + '─' * 62)
    # Header: 8 config names
    print('      ' + '  '.join(_CFG_NAMES[cfg] for cfg in _ALL_CONFIGS))
    print('      ' + '  '.join('→' + str(d['rule_table'][cfg]) for cfg in _ALL_CONFIGS))
    print('      ' + '─' * 44)
    for b in range(6):
        bc  = _BCOLS[b] if color else ''
        n   = d['coverage_vector'][b]
        row = []
        for cfg in _ALL_CONFIGS:
            out = d['per_bit_outputs'][b].get(cfg)
            if out is None:
                row.append(' . ')
            else:
                row.append(f' {out} ')
        print(f'  {bc}b{b}{r}  {"".join(row)}  {n}/8={n/_N_CONFIGS:.2f}')
    print(f'\n  coverage_vector: {d["coverage_vector"]}')
    print(f'  output_diversity: {d["output_diversity"]}')
    print()


def print_config_table(words: list[str] | None = None,
                       color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    print(f'  {"Слово":10s}  '
          + '  '.join(
              (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5s}  cov' + R
              for rl in _RULES))
    print('  ' + '─' * 56)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = (_RCOL.get(rule, '') if color else '')
            mc  = mean_coverage(word, rule)
            cv  = coverage_vector(word, rule)
            parts.append(f'{col}[{"".join(str(v) for v in cv)}]  {mc:.3f}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description='1-Bit Neighborhood Config Coverage')
    p.add_argument('--word',      default='ГОРА')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--no-color',  action='store_true')
    args = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.table:
        print_config_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_config(args.word, rule, color)
    else:
        print_config(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
