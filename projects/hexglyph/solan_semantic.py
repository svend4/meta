"""solan_semantic.py — Semantic Orbit Trajectory of Q6 CA Attractors.

For a period-P attractor each orbit state orbit[t] (t = 0…P−1) is a
16-cell Q6 configuration.  By comparing it against the encoded ICs of all
49 lexicon words (using cell-level Hamming distance), we map each orbit step
to its nearest "semantic neighbour" in the lexicon.

    nearest_word(t)  = argmin_{w ∈ LEXICON}  H(orbit[t], encode(w))
    nearest_dist(t)  = min H(orbit[t], encode(w))

The sequence of nearest words across P steps forms the **semantic trajectory**:
it reveals the "vocabulary" the orbit visits as it evolves.

Semantic Void
─────────────
When nearest_dist(t) = N = 16 (maximum), the orbit state at step t is
equidistant from every lexicon word — a "semantic void".  This occurs:
  • All P=2 XOR3 orbits at t=1 (complement state is not in lexicon)
  • AND orbits for many words at t=0 (attractor IC not in lexicon)
  БОЛТ AND: BOTH t=0 and t=1 are voids (fully outside lexicon)

Self-Echo Pattern (ТУМАН XOR3)
────────────────────────────────
ТУМАН is the nearest word at t=0 (d=0), t=3 (d=6), t=6 (d=12) — three
"echoes" across the 8-step orbit with increasing Hamming distance.
At t=1,2,4,5,7 other words are nearer.  Mean nearest dist ≈ 12.5.

Запуск:
    python3 -m projects.hexglyph.solan_semantic --word ТУМАН --rule xor3
    python3 -m projects.hexglyph.solan_semantic --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_semantic --word МАТ --rule xor3
    python3 -m projects.hexglyph.solan_semantic --table --no-color
    python3 -m projects.hexglyph.solan_semantic --json --word ТУМАН
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_ca      import (
    _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES,
)

RULES = tuple(_ALL_RULES)   # ('xor', 'xor3', 'and', 'or')
_DEFAULT_WIDTH = 16


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _encode_lex(width: int = _DEFAULT_WIDTH) -> dict[str, list[int]]:
    """Padded Q6 ICs for every lexicon word (cached within the call)."""
    from projects.hexglyph.solan_word import encode_word, pad_to
    return {w: pad_to(encode_word(w), width) for w in LEXICON}


def hamming_dist(a: list[int], b: list[int]) -> int:
    """Cell-level Q6 Hamming distance."""
    return sum(1 for x, y in zip(a, b) if x != y)


# ── Core computation ──────────────────────────────────────────────────────────

def nearest_in_lex(
    state:    list[int],
    lex_ics:  dict[str, list[int]],
    top_n:    int = 3,
) -> list[tuple[str, int]]:
    """Top-N nearest lexicon words to an orbit state (sorted by Hamming dist)."""
    dists = [(w, hamming_dist(state, ic)) for w, ic in lex_ics.items()]
    dists.sort(key=lambda x: (x[1], x[0]))
    return dists[:top_n]


def dist_to_self(
    word:     str,
    orbit:    list,
    width:    int = _DEFAULT_WIDTH,
) -> list[int]:
    """Hamming distance from each orbit state to the word's own IC."""
    from projects.hexglyph.solan_word import encode_word, pad_to
    ic = pad_to(encode_word(word), width)
    return [hamming_dist(list(s), ic) for s in orbit]


# ── Per-word summary ──────────────────────────────────────────────────────────

def semantic_summary(
    word:     str,
    rule:     str,
    width:    int = _DEFAULT_WIDTH,
    *,
    top_n:    int = 3,
    _lex_ics: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Semantic orbit trajectory for one word/rule.

    Keys
    ────
    word, rule, period, n_cells
    nearest          : list[list[(word, dist)]]  — top-N per orbit step
    nearest_word     : list[str]    — closest word at each step
    nearest_dist     : list[int]    — Hamming to closest word at each step
    dist_to_self     : list[int]    — Hamming(orbit[t], word_IC) per step
    self_nearest_steps : list[int]  — steps where word itself is nearest
    void_steps       : list[int]    — steps where nearest_dist = N (max)
    unique_words     : list[str]    — distinct nearest words across orbit
    n_unique_words   : int
    mean_nearest_dist: float
    min_nearest_dist : int
    max_nearest_dist : int
    self_is_nearest_t0 : bool       — True when transient=0 (word starts on attractor)
    """
    from projects.hexglyph.solan_perm import get_orbit

    orbit   = get_orbit(word, rule, width)
    P       = len(orbit)
    N       = width

    lex_ics = _lex_ics if _lex_ics is not None else _encode_lex(width)

    nearest_all: list[list[tuple[str, int]]] = []
    near_word:   list[str]  = []
    near_dist:   list[int]  = []

    for state in orbit:
        nbs = nearest_in_lex(list(state), lex_ics, top_n=top_n)
        nearest_all.append(nbs)
        near_word.append(nbs[0][0] if nbs else '')
        near_dist.append(nbs[0][1] if nbs else N)

    d_self = dist_to_self(word, orbit, width)

    self_nearest_steps = [
        t for t in range(P) if nearest_all[t] and nearest_all[t][0][0] == word
    ]
    void_steps = [t for t in range(P) if near_dist[t] >= N]

    unique = list(dict.fromkeys(near_word))  # order-preserving dedup
    mean_d = sum(near_dist) / P if P else 0.0

    return {
        'word':              word,
        'rule':              rule,
        'period':            P,
        'n_cells':           N,
        'nearest':           nearest_all,
        'nearest_word':      near_word,
        'nearest_dist':      near_dist,
        'dist_to_self':      d_self,
        'self_nearest_steps': self_nearest_steps,
        'void_steps':        void_steps,
        'unique_words':      unique,
        'n_unique_words':    len(unique),
        'mean_nearest_dist': round(mean_d, 4),
        'min_nearest_dist':  min(near_dist) if near_dist else 0,
        'max_nearest_dist':  max(near_dist) if near_dist else 0,
        'self_is_nearest_t0': bool(nearest_all) and bool(nearest_all[0])
                              and nearest_all[0][0][0] == word,
    }


def all_semantic(
    word:  str,
    width: int = _DEFAULT_WIDTH,
    top_n: int = 3,
) -> dict[str, dict[str, Any]]:
    """Semantic summary for all 4 rules, sharing the lexicon IC cache."""
    lex_ics = _encode_lex(width)
    return {
        r: semantic_summary(word, r, width, top_n=top_n, _lex_ics=lex_ics)
        for r in RULES
    }


def build_semantic_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
    top_n: int = 3,
) -> dict[str, Any]:
    """Full semantic analysis for the lexicon."""
    if words is None:
        words = list(LEXICON)
    lex_ics = _encode_lex(width)
    return {
        'words': list(words),
        'data': {
            w: {r: semantic_summary(w, r, width, top_n=top_n, _lex_ics=lex_ics)
                for r in RULES}
            for w in words
        },
    }


def semantic_dict(s: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable version of semantic_summary."""
    return {
        'word':              s['word'],
        'rule':              s['rule'],
        'period':            s['period'],
        'n_cells':           s['n_cells'],
        'nearest_word':      s['nearest_word'],
        'nearest_dist':      s['nearest_dist'],
        'dist_to_self':      s['dist_to_self'],
        'self_nearest_steps': s['self_nearest_steps'],
        'void_steps':        s['void_steps'],
        'unique_words':      s['unique_words'],
        'n_unique_words':    s['n_unique_words'],
        'mean_nearest_dist': s['mean_nearest_dist'],
        'min_nearest_dist':  s['min_nearest_dist'],
        'max_nearest_dist':  s['max_nearest_dist'],
        'self_is_nearest_t0': s['self_is_nearest_t0'],
        'nearest_top':       [
            [{'word': w, 'dist': d} for w, d in step_nbs]
            for step_nbs in s['nearest']
        ],
    }


# ── Terminal output ───────────────────────────────────────────────────────────

def print_semantic(
    word:  str,
    rule:  str,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
    top_n: int  = 3,
) -> None:
    """Print semantic orbit trajectory for one word/rule."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    s   = semantic_summary(word, rule, width, top_n=top_n)
    P   = s['period']
    N   = s['n_cells']
    col = (_RULE_COLOR.get(rule, '') if color else '')
    lbl = _RULE_NAMES.get(rule, rule.upper())

    print(bold + f"  ◈ Semantic Orbit  {word.upper()}  "
          + col + lbl + reset + bold + f"  (P={P})" + reset)
    print()

    if P == 1:
        nd = s['nearest_dist'][0]
        nw = s['nearest_word'][0]
        ds = s['dist_to_self'][0]
        if nd == 0:
            marker = bold + '★ self' + reset
        elif nd >= N:
            marker = '\033[38;5;220mvoid\033[0m' if color else 'void'
        else:
            marker = ''
        print(f"  t0: {col}{nw:<14}{reset} d={nd:2d} "
              f"| d_self={ds:2d}  {marker}")
        print()
        return

    # Trajectory table
    header = (f"  {'t':2s}  {'Nearest word':14s}  {'d':>3s}  "
              f"{'d_self':>6s}  Top-3 neighbours")
    print(header)
    print('  ' + '─' * 70)

    void_col  = '\033[38;5;220m' if color else ''
    self_col  = '\033[38;5;120m' if color else ''
    other_col = '\033[38;5;117m' if color else ''

    for t in range(P):
        nw  = s['nearest_word'][t]
        nd  = s['nearest_dist'][t]
        ds  = s['dist_to_self'][t]
        nbs = s['nearest']

        if nd >= N:
            wc = void_col
            marker = ' ← void'
        elif nw == word:
            wc = self_col
            marker = ' ← self'
        else:
            wc = other_col
            marker = ''

        top_str = '  '.join(
            f"{w}({d})" for w, d in nbs[t]
        )
        print(f"  t{t}: {wc}{nw:<14}{reset}  {nd:2d}  "
              f"{dim}d_self={ds:2d}{reset}  "
              f"{dim}{top_str}{reset}{marker}")

    print()
    print(f"  Semantic void steps   : {s['void_steps']} "
          f"({len(s['void_steps'])} / {P})")
    print(f"  Self-nearest steps    : {s['self_nearest_steps']} "
          f"({len(s['self_nearest_steps'])} / {P})")
    print(f"  Unique nearest words  : {s['n_unique_words']}  "
          f"{dim}[{', '.join(s['unique_words'])}]{reset}")
    print(f"  Mean nearest dist     : {s['mean_nearest_dist']:.4f}")
    print(f"  Min/Max nearest dist  : {s['min_nearest_dist']} / {s['max_nearest_dist']}")
    print()


def print_semantic_table(
    words: list[str] | None = None,
    width: int  = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Table: semantic orbit stats for all words, XOR3 rule."""
    if words is None:
        words = list(LEXICON)

    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    print(bold + f"  ◈ Semantic Orbit Summary (XOR3, n={len(words)})" + reset)
    print()
    print(f"  {'Слово':12s}  {'P':>3}  {'voids':>5}  {'self':>4}  "
          f"{'μ-dist':>6}  {'unique':>6}  Nearest words")
    print('  ' + '─' * 78)

    lex_ics = _encode_lex(width)
    for word in words:
        s    = semantic_summary(word, 'xor3', width, _lex_ics=lex_ics)
        P    = s['period']
        col  = (_RULE_COLOR.get('xor3', '') if color else '')
        void_s = f"{len(s['void_steps'])}/{P}"
        self_s = f"{len(s['self_nearest_steps'])}/{P}"
        uniq   = '  '.join(s['unique_words'][:4])
        if len(s['unique_words']) > 4:
            uniq += '…'
        print(f"  {word.upper():12s}  {P:>3}  {void_s:>5}  {self_s:>4}  "
              f"{s['mean_nearest_dist']:>6.2f}  {s['n_unique_words']:>6}  "
              f"{dim}{uniq}{reset}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Semantic orbit trajectory for Q6 CA attractors')
    parser.add_argument('--word',  metavar='WORD', default='ТУМАН')
    parser.add_argument('--rule',  choices=list(RULES), default='xor3')
    parser.add_argument('--table', action='store_true',
                        help='XOR3 summary table for full lexicon')
    parser.add_argument('--json',  action='store_true')
    parser.add_argument('--width', type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--top',   type=int, default=3, dest='top_n')
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        s = semantic_summary(args.word.upper(), args.rule, args.width,
                             top_n=args.top_n)
        print(json.dumps(semantic_dict(s), ensure_ascii=False, indent=2))
    elif args.table:
        print_semantic_table(width=args.width, color=_color)
    else:
        print_semantic(args.word.upper(), args.rule, args.width,
                       color=_color, top_n=args.top_n)
