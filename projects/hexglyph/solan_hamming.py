"""
solan_hamming.py — Consecutive-step Hamming Distances & Cell Mobility.

For an attractor orbit [s₀, s₁, …, s_{P−1}] (periodic ring of width N):

  H(t)  = Hamming(sₜ, s_{(t+1) mod P})
         = |{i ∈ 0..N−1 : sₜ[i] ≠ s_{t+1}[i]}|
         = number of cells that CHANGE VALUE at step t

  flip_mask[t][i] = 1 if sₜ[i] ≠ s_{t+1}[i], else 0

  cell_mobility(i)  = (1/P) Σ_t flip_mask[t][i]
                    = fraction of orbit transitions at which cell i changes

Contrast with related modules
───────────────────────────────
  solan_dist:     pairwise Hamming between ALL orbit step PAIRS (not just
                  consecutive); finds maximum and minimum separation.
  solan_boundary: spatial XOR between ADJACENT CELLS at one step (domain walls).
  solan_hamming:  TEMPORAL Hamming between CONSECUTIVE STATES (dynamics).

Key results  (width = 16)
──────────────────────────
  ТУМАН XOR  (P=1, fixed point)
      H = [0]  mean=0  ★ frozen: no cell ever changes (fixed-point orbit).

  ГОРА OR    (P=1, fixed point)
      H = [0]  mean=0  ★ same: trivially frozen.

  ГОРА AND   (P=2, anti-phase)
      H = [16, 16]  mean=16  ★ maximally turbulent: ALL cells flip at EVERY step.
      mobility = [1, 1, …, 1] (all cells flip at each of the two transitions).

  ГОРА XOR3  (P=2, 4-periodic pattern)
      H = [16, 16]  mean=16  ★ same as AND: all cells flip at every step.
      mobility = [1, 1, …, 1]

  ТУМАН XOR3  (P=8)
      H = [16, 16, 10, 12, 14, 16, 16, 14]   mean=14.25
      ★ Steps 0→1, 1→2, 5→6, 6→7: all 16 cells change (H=N).
      ★ Step 2→3: MINIMUM — only 10 cells change; 6 cells stay still:
          cells {0,1,2} and {13,14,15} form FROZEN EDGES.
      ★ Step 3→4: 12 cells change; 4 cells stay still: {0,1} and {14,15}.
      ★ Step 4→5: 14 cells change; 2 cells stay still: {0} and {15}.
      ★ "Shrinking frozen-edge" pattern: the static zone contracts from 6
          cells at step 2→3 to 4 cells at 3→4 to 2 cells at 4→5.

      Cell mobility (symmetric profile):
          cell 0,15 : 0.500  (change at 4/8 steps — LEAST mobile)
          cell 1,14 : 0.750  (change at 6/8 steps)
          cell 2,13 : 0.875  (change at 7/8 steps)
          cell 3..12: 1.000  (change at 8/8 steps — MAXIMALLY mobile)

      ★ Mobility is SYMMETRIC: mob[i] = mob[N−1−i] for all i.
      ★ Mobility OPPOSES entropy at the extremes:
          edge cells (0,15): LOW mobility (0.5) but INTERMEDIATE H_c (1.75)
          inner cells (7,8): MAX mobility (1.0) but LOWEST H_c (1.56)
          → inner cells always move but cycle through fewer distinct values;
            outer cells pause longer but visit more varied states.

Flip-mask pattern for ТУМАН XOR3 (1=changes, 0=stays):
    t=0→1: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  (all 16)
    t=1→2: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  (all 16)
    t=2→3: 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0  (10; frozen edges: 0,1,2 & 13,14,15)
    t=3→4: 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0  (12; frozen: 0,1 & 14,15)
    t=4→5: 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0  (14; frozen: 0 & 15)
    t=5→6: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  (all 16)
    t=6→7: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  (all 16)
    t=7→0: 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0  (14; frozen: 0 & 15)

Functions
─────────
  hamming_dist(a, b)                      → int
  consecutive_hamming(orbit)              → list[int]
  flip_mask(orbit)                        → list[list[int]]
  cell_mobility(orbit)                    → list[float]
  hamming_profile(word, rule, width)      → list[int]
  flip_mask_word(word, rule, width)       → list[list[int]]
  cell_mobility_word(word, rule, width)   → list[float]
  mean_hamming(word, rule, width)         → float
  max_hamming(word, rule, width)          → int
  min_hamming(word, rule, width)          → int
  hamming_summary(word, rule, width)      → dict
  all_hamming(word, width)                → dict[str, dict]
  build_hamming_data(words, width)        → dict
  print_hamming(word, rule, color)        → None
  print_hamming_table(words, color)       → None

Запуск
──────
  python3 -m projects.hexglyph.solan_hamming --word ТУМАН --rule xor3 --no-color
  python3 -m projects.hexglyph.solan_hamming --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_hamming --table --no-color
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


# ── Core functions ─────────────────────────────────────────────────────────────

def hamming_dist(a: tuple[int, ...] | list[int],
                 b: tuple[int, ...] | list[int]) -> int:
    """Number of positions where a[i] ≠ b[i]."""
    return sum(x != y for x, y in zip(a, b))


def consecutive_hamming(orbit: list[tuple[int, ...]]) -> list[int]:
    """H(t) = Hamming(orbit[t], orbit[(t+1) % P]) for each orbit step.

    H(t) = number of cells that change value at transition t → t+1.
    Returns a list of P values (one per orbit step).
    """
    P = len(orbit)
    return [hamming_dist(orbit[t], orbit[(t + 1) % P]) for t in range(P)]


def flip_mask(orbit: list[tuple[int, ...]]) -> list[list[int]]:
    """Binary flip indicator for each (step, cell) pair.

    flip_mask[t][i] = 1 if orbit[t][i] ≠ orbit[(t+1)%P][i], else 0.
    Shape: P × N.
    """
    P = len(orbit)
    N = len(orbit[0])
    return [[1 if orbit[t][i] != orbit[(t + 1) % P][i] else 0
             for i in range(N)]
            for t in range(P)]


def cell_mobility(orbit: list[tuple[int, ...]]) -> list[float]:
    """For each cell i, fraction of orbit transitions at which it changes.

    cell_mobility[i] = (1/P) Σ_t flip_mask[t][i] ∈ [0.0, 1.0].

    0.0 → cell never changes (frozen / fixed cell).
    1.0 → cell changes at every orbit step (maximally mobile).
    """
    P = len(orbit)
    N = len(orbit[0])
    masks = flip_mask(orbit)
    return [round(sum(masks[t][i] for t in range(P)) / P, 6) for i in range(N)]


# ── Per-word functions ─────────────────────────────────────────────────────────

def hamming_profile(word: str, rule: str, width: int = _DEFAULT_W) -> list[int]:
    """Consecutive Hamming distance at each orbit step."""
    return consecutive_hamming(_get_orbit(word, rule, width))


def flip_mask_word(word: str, rule: str,
                   width: int = _DEFAULT_W) -> list[list[int]]:
    """Flip mask (P × N binary array) for the orbit."""
    return flip_mask(_get_orbit(word, rule, width))


def cell_mobility_word(word: str, rule: str,
                       width: int = _DEFAULT_W) -> list[float]:
    """Per-cell mobility fraction for the orbit."""
    return cell_mobility(_get_orbit(word, rule, width))


def mean_hamming(word: str, rule: str, width: int = _DEFAULT_W) -> float:
    """Mean consecutive Hamming distance (= mean_H over orbit steps)."""
    h = hamming_profile(word, rule, width)
    return round(sum(h) / len(h), 6)


def max_hamming(word: str, rule: str, width: int = _DEFAULT_W) -> int:
    """Maximum consecutive Hamming distance across orbit steps."""
    return max(hamming_profile(word, rule, width))


def min_hamming(word: str, rule: str, width: int = _DEFAULT_W) -> int:
    """Minimum consecutive Hamming distance across orbit steps."""
    return min(hamming_profile(word, rule, width))


# ── Summary ────────────────────────────────────────────────────────────────────

def hamming_summary(word: str, rule: str, width: int = _DEFAULT_W) -> dict:
    """Full Hamming/mobility summary for word × rule."""
    orbit = _get_orbit(word, rule, width)
    P     = len(orbit)
    N     = width
    H     = consecutive_hamming(orbit)
    mob   = cell_mobility(orbit)
    masks = flip_mask(orbit)
    # Frozen cells: mobility = 0 (for P>1 fixed points these are all cells)
    frozen   = [i for i, m in enumerate(mob) if m == 0.0]
    maxmob   = [i for i, m in enumerate(mob) if m == 1.0]
    mob_sym  = all(abs(mob[i] - mob[N - 1 - i]) < 1e-9 for i in range(N // 2))
    # Steps with minimum H (most frozen transitions)
    min_h     = min(H)
    frozen_steps = [t for t, h in enumerate(H) if h == min_h]
    return {
        'word':             word.upper(),
        'rule':             rule,
        'period':           P,
        'n_cells':          N,
        # Hamming profile
        'hamming':          H,
        'mean_hamming':     round(sum(H) / P, 6),
        'max_hamming':      max(H),
        'min_hamming':      min_h,
        'hamming_range':    max(H) - min_h,
        # Mobility
        'mobility':         mob,
        'mean_mobility':    round(sum(mob) / N, 6),
        'max_mobility':     max(mob),
        'min_mobility':     min(mob),
        'frozen_cells':     frozen,
        'maxmobile_cells':  maxmob,
        'mobile_symmetric': mob_sym,
        # Flags
        'all_frozen':       all(h == 0 for h in H),
        'all_max':          all(h == N for h in H),
        'min_hamming_steps': frozen_steps,
        'flip_mask':        masks,
    }


def all_hamming(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """hamming_summary for all 4 rules."""
    return {rule: hamming_summary(word, rule, width) for rule in _RULES}


def build_hamming_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact Hamming data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = hamming_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period', 'hamming', 'mean_hamming',
                                   'max_hamming', 'min_hamming', 'hamming_range',
                                   'mobility', 'mean_mobility',
                                   'max_mobility', 'min_mobility',
                                   'frozen_cells', 'maxmobile_cells',
                                   'mobile_symmetric', 'all_frozen', 'all_max',
                                   'min_hamming_steps')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Print helpers ──────────────────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_RST  = '\033[0m'
_BAR_W = 16   # bar width in chars


def _bar(val: int | float, max_val: int | float, w: int = _BAR_W) -> str:
    frac   = val / max_val if max_val > 0 else 0.0
    filled = round(frac * w)
    return '█' * filled + '░' * (w - filled)


def print_hamming(word: str = 'ТУМАН', rule: str = 'xor3',
                  color: bool = True) -> None:
    d   = hamming_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',  'or':   'OR |'}.get(rule, rule)
    af  = ' ★all_frozen'  if d['all_frozen'] else ''
    am  = ' ★all_max'     if d['all_max']    else ''
    sym = ' ★mob_sym'     if d['mobile_symmetric'] else ''
    print(f'  {c}◈ Hamming  {word.upper()}  |  {lbl}  P={d["period"]}'
          f'  mean_H={d["mean_hamming"]:.2f}'
          f'  mean_mob={d["mean_mobility"]:.3f}{af}{am}{sym}{r}')
    print('  ' + '─' * 68)
    N = d['n_cells']
    print(f'  {"step":>4}  {"H":>3}/{N:<2}  bar             '
          f'  frozen cells')
    for t, (h, mask) in enumerate(zip(d['hamming'], d['flip_mask'])):
        frozen = [i for i, f in enumerate(mask) if f == 0]
        bar = _bar(h, N)
        fr  = str(frozen) if frozen else '—'
        print(f'  t={t:>2}  {h:>3}     [{bar}]  {fr}')
    print()
    print(f'  Cell mobility  (range [{d["min_mobility"]:.3f}, {d["max_mobility"]:.3f}]):')
    mob = d['mobility']
    max_m = max(mob) if max(mob) > 0 else 1.0
    for i, m in enumerate(mob):
        bar = _bar(m, max_m)
        tag = ''
        if i in d['frozen_cells']:    tag = ' ★frozen'
        if i in d['maxmobile_cells']: tag = ' ★max'
        print(f'  cell {i:>2}  {m:.4f}  [{bar}]{tag}')
    print()


def print_hamming_table(words: list[str] | None = None,
                         color: bool = True) -> None:
    from projects.hexglyph.solan_lexicon import LEXICON
    WORDS = words or LEXICON
    R = _RST if color else ''
    head = '  '.join(
        (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>5} mH mMob af' + R
        for rl in _RULES)
    print(f'  {"Слово":10s}  {head}')
    print('  ' + '─' * 72)
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = _RCOL.get(rule, '') if color else ''
            d   = hamming_summary(word, rule)
            af  = '★' if d['all_frozen'] or d['all_max'] else ' '
            parts.append(f'{col}{d["mean_hamming"]:>4.1f}'
                         f' {d["mean_mobility"]:>5.3f} {af}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description='Consecutive-step Hamming Distances & Cell Mobility')
    p.add_argument('--word',      default='ТУМАН')
    p.add_argument('--rule',      default='xor3', choices=_RULES)
    p.add_argument('--all-rules', action='store_true')
    p.add_argument('--table',     action='store_true')
    p.add_argument('--json',      action='store_true', help='JSON output')
    p.add_argument('--no-color',  action='store_true')
    args  = p.parse_args()
    color = not args.no_color and sys.stdout.isatty()
    if args.json:
        import json as _json
        d = hamming_summary(args.word, args.rule)
        print(_json.dumps(d, ensure_ascii=False, indent=2))
    elif args.table:
        print_hamming_table(color=color)
    elif args.all_rules:
        for rule in _RULES:
            print_hamming(args.word, rule, color)
    else:
        print_hamming(args.word, args.rule, color)


if __name__ == '__main__':
    _cli()
