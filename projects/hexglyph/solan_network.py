"""solan_network.py — Information Flow Network of Q6 CA.

Builds a directed weighted graph from the Transfer Entropy (TE) matrix
M[i][j] = T(cell_i → cell_j) on the periodic attractor.

Node metrics
─────────────
  in_weight(j)   = Σ_i M[i][j]    total TE arriving at cell j
  out_weight(i)  = Σ_j M[i][j]    total TE leaving  cell i
  net_flow(i)    = out − in        net information production at cell i
                                   > 0 → source,  < 0 → sink

Graph algorithms (no external dependencies)
────────────────────────────────────────────
  pagerank    power-iteration PageRank on the TE-normalised random walk
  hub_score / auth_score   HITS algorithm (Kleinberg)
  scc         strongly connected components via Tarjan's algorithm
              edge (i→j) exists when M[i][j] > threshold (default 0)

Network summary
───────────────
  total_te     sum of all M[i][j]   (global information flow)
  n_sccs       number of SCCs
  largest_scc  size of the biggest SCC
  top_sources  cells with highest net_flow (info producers)
  top_sinks    cells with lowest  net_flow (info consumers)
  top_pr       cells with highest PageRank (most visited in random walk)

Interpretation
──────────────
  XOR/AND (period-1 all-zeros): TE = 0 everywhere → trivial network,
    all nodes equal PageRank 1/N, no SCCs with edges.
  XOR3 (period-8): TE reveals nearest-neighbour influence;
    net_flow reflects the XOR3 parity asymmetry.

Функции
───────
  network_dict(word, rule, width)  → dict
  all_network(word)                → dict[rule, dict]
  build_network_data(words)        → dict
  print_network(word, rule, color) → None

Запуск
──────
  python3 -m projects.hexglyph.solan_network --word ТУМАН --rule xor3
  python3 -m projects.hexglyph.solan_network --word ГОРА --all-rules --no-color
  python3 -m projects.hexglyph.solan_network --stats --no-color
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    find_orbit,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_transfer import te_matrix, get_orbit
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16


# ── Node metrics ──────────────────────────────────────────────────────────────

def in_weights(mat: list[list[float]]) -> list[float]:
    """Total TE arriving at each node j: Σ_i M[i][j]."""
    N = len(mat)
    return [round(sum(mat[i][j] for i in range(N)), 8) for j in range(N)]


def out_weights(mat: list[list[float]]) -> list[float]:
    """Total TE leaving each node i: Σ_j M[i][j]."""
    return [round(sum(row), 8) for row in mat]


def net_flow(mat: list[list[float]]) -> list[float]:
    """net_flow[i] = out_weight[i] − in_weight[i] (+ → source, − → sink)."""
    iw = in_weights(mat)
    ow = out_weights(mat)
    return [round(ow[i] - iw[i], 8) for i in range(len(mat))]


# ── PageRank ──────────────────────────────────────────────────────────────────

def pagerank(
    mat:     list[list[float]],
    damping: float = 0.85,
    iters:   int   = 100,
) -> list[float]:
    """PageRank via power iteration on the TE-normalised random walk.

    The transition matrix T[i][j] = M[i][j] / Σ_k M[i][k]
    (dangling nodes → uniform distribution).
    """
    N = len(mat)
    # build row-normalised transition matrix
    trans: list[list[float]] = []
    for row in mat:
        s = sum(row)
        if s > 0:
            trans.append([v / s for v in row])
        else:
            trans.append([1.0 / N] * N)
    # power iteration
    pr = [1.0 / N] * N
    base = (1.0 - damping) / N
    for _ in range(iters):
        new_pr = [base] * N
        for i in range(N):
            pi = pr[i]
            ti = trans[i]
            for j in range(N):
                new_pr[j] += damping * pi * ti[j]
        pr = new_pr
    return [round(v, 10) for v in pr]


# ── HITS ──────────────────────────────────────────────────────────────────────

def hits(
    mat:   list[list[float]],
    iters: int = 50,
) -> tuple[list[float], list[float]]:
    """HITS algorithm (Kleinberg 1999).

    hub_score[i]    — cell i points to high-authority cells
    auth_score[j]   — cell j is pointed to by high-hub cells
    Returns (hub_scores, auth_scores) L1-normalised.
    """
    N = len(mat)
    h = [1.0 / N] * N
    a = [1.0 / N] * N
    for _ in range(iters):
        # authority update: a[j] += h[i] * M[i][j]
        new_a = [0.0] * N
        for i in range(N):
            hi = h[i]
            for j in range(N):
                new_a[j] += hi * mat[i][j]
        # hub update: h[i] += a[j] * M[i][j]
        new_h = [0.0] * N
        for i in range(N):
            for j in range(N):
                new_h[i] += a[j] * mat[i][j]
        # L1-normalise
        sa = sum(new_a) or 1.0
        sh = sum(new_h) or 1.0
        a  = [v / sa for v in new_a]
        h  = [v / sh for v in new_h]
    return (
        [round(v, 10) for v in h],
        [round(v, 10) for v in a],
    )


# ── Tarjan SCC ────────────────────────────────────────────────────────────────

def tarjan_scc(
    mat:       list[list[float]],
    threshold: float = 0.0,
) -> list[list[int]]:
    """Strongly connected components via Tarjan's algorithm.

    Edge i→j exists when mat[i][j] > threshold.
    Returns list-of-lists, sorted by component size (largest first).
    """
    N     = len(mat)
    index_of = [-1] * N
    lowlink  = [0]  * N
    on_stack = [False] * N
    stack    = []
    counter  = [0]
    sccs: list[list[int]] = []

    def _connect(v: int) -> None:
        index_of[v] = lowlink[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in range(N):
            if mat[v][w] > threshold:
                if index_of[w] < 0:
                    _connect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index_of[w])
        if lowlink[v] == index_of[v]:
            scc: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in range(N):
        if index_of[v] < 0:
            _connect(v)
    return sorted(sccs, key=len, reverse=True)


# ── Full analysis ─────────────────────────────────────────────────────────────

def network_dict(
    word:  str,
    rule:  str,
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """Full information-flow network analysis for one word + rule.

    Returns dict with keys:
        word, rule, width
        period, transient
        te_mat          : list[list[float]]  N×N TE matrix
        in_weight       : list[float]  per-node incoming TE
        out_weight      : list[float]  per-node outgoing TE
        net_flow        : list[float]  per-node net TE (out − in)
        total_te        : float        sum of all TE entries
        pagerank        : list[float]  PageRank scores (sum ≈ 1)
        hub_score       : list[float]  HITS hub scores
        auth_score      : list[float]  HITS authority scores
        sccs            : list[list[int]]  SCCs sorted by size desc
        n_sccs          : int
        largest_scc     : int
        top_sources     : list[int]  top-3 cell indices by net_flow desc
        top_sinks       : list[int]  top-3 cell indices by net_flow asc
        top_pr          : list[int]  top-3 cell indices by PageRank desc
    """
    cells             = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells[:], rule)
    period            = max(period, 1)

    mat = te_matrix(word, rule, width)
    iw  = in_weights(mat)
    ow  = out_weights(mat)
    nf  = net_flow(mat)
    tt  = round(sum(sum(row) for row in mat), 8)
    pr  = pagerank(mat)
    hs, as_ = hits(mat)
    sc  = tarjan_scc(mat)

    argsort = lambda lst, rev=False: sorted(range(len(lst)), key=lambda k: lst[k], reverse=rev)
    top_src = argsort(nf,  rev=True)[:3]
    top_snk = argsort(nf,  rev=False)[:3]
    top_pr_ = argsort(pr,  rev=True)[:3]

    return {
        'word':        word.upper(),
        'rule':        rule,
        'width':       width,
        'period':      period,
        'transient':   transient,
        'te_mat':      mat,
        'in_weight':   iw,
        'out_weight':  ow,
        'net_flow':    nf,
        'total_te':    tt,
        'pagerank':    pr,
        'hub_score':   hs,
        'auth_score':  as_,
        'sccs':        sc,
        'n_sccs':      len(sc),
        'largest_scc': max(len(s) for s in sc) if sc else 0,
        'top_sources': top_src,
        'top_sinks':   top_snk,
        'top_pr':      top_pr_,
    }


def all_network(word: str) -> dict[str, dict]:
    """network_dict for all 4 rules."""
    return {r: network_dict(word, r) for r in _ALL_RULES}


def build_network_data(
    words: list[str] | None = None,
) -> dict:
    """Network summary across the lexicon × 4 rules.

    Returns:
        words, per_rule: {rule: {word: {total_te, n_sccs, largest_scc}}}
        ranking: {rule: [(word, total_te), …] sorted descending}
    """
    words = words if words is not None else list(LEXICON)
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            d = network_dict(word, rule)
            per_rule[rule][word] = {
                'total_te':    d['total_te'],
                'n_sccs':      d['n_sccs'],
                'largest_scc': d['largest_scc'],
            }
    ranking = {
        r: sorted(
            ((w, v['total_te']) for w, v in per_rule[r].items()),
            key=lambda x: -x[1],
        )
        for r in _ALL_RULES
    }
    return {'words': words, 'per_rule': per_rule, 'ranking': ranking}


# ── ASCII / ANSI display ───────────────────────────────────────────────────────

def print_network(
    word:  str  = 'ТУМАН',
    rule:  str  = 'xor3',
    color: bool = True,
) -> None:
    d    = network_dict(word, rule)
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    print(f"{bold}  ◈ Сеть информационного потока Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  P={d['period']}  total_TE={d['total_te']:.4f}")
    print(f"  SCCs={d['n_sccs']}  largest_SCC={d['largest_scc']}")

    nf = d['net_flow']
    iw = d['in_weight']
    ow = d['out_weight']
    pr = d['pagerank']
    max_abs_nf = max(abs(v) for v in nf) or 1.0
    print(f"  {'Cell':>4}  {'in_TE':>7}  {'out_TE':>7}  {'net_flow':>9}  "
          f"{'PageRank':>9}  bar")
    for i in range(d['width']):
        bar_len = int(abs(nf[i]) / max_abs_nf * 14)
        bar_ch  = '▶' if nf[i] >= 0 else '◀'
        bar     = bar_ch * bar_len
        print(f"  {i:>4}  {iw[i]:>7.4f}  {ow[i]:>7.4f}  {nf[i]:>+9.4f}  "
              f"{pr[i]:>9.5f}  {bar}")
    print(f"  sources={d['top_sources']}  sinks={d['top_sinks']}  "
          f"top_PR={d['top_pr']}")
    print()


def print_network_stats(
    words: list[str] | None = None,
    color: bool             = True,
) -> None:
    """Table: total TE per word × rule."""
    words = words if words is not None else list(LEXICON)
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Суммарный поток TE (total_TE) по сети{rst}")
    print('  ' + '─' * (len(header) + 2))
    print('  ' + header)
    print('  ' + '─' * (len(header) + 2))
    for word in sorted(words):
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            d   = network_dict(word, rule)
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{d['total_te']:>8.4f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Information Flow Network Q6 CA')
    parser.add_argument('--word',      default='ТУМАН')
    parser.add_argument('--rule',      default='xor3', choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_network_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_network(args.word, rule, color)
    else:
        print_network(args.word, args.rule, color)


if __name__ == '__main__':
    _main()
