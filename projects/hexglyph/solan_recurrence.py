"""solan_recurrence.py — Рекуррентный анализ (RQA) CA-траекторий Q6.

Рекуррентная матрица:

    R[t1,t2] = 1  если  H(state(t1), state(t2)) ≤ ε
    H = суммарное Q6-хэмминг-расстояние (число различающихся битов по всем клеткам)

По умолчанию ε = 0: точное совпадение состояний.

RQA-метрики (вычисляются по внедиагональным элементам):
    RR  — Recurrence Rate: доля рекуррентных пар (исключая главную диагональ)
    DET — Determinism: доля рекуррентных точек, входящих в диагонали длиной ≥ L_min
    L   — средняя длина диагональных линий ≥ L_min
    LAM — Laminarity: доля рекуррентных точек в вертикальных линиях ≥ V_min
    TT  — Trapping Time: средняя длина вертикальных линий ≥ V_min

Типичные результаты для Q6 (ε=0, n_cycles=4):
    XOR  → RR=1.0, DET=1.0 (аттрактор = нулевой вектор, все строки идентичны)
    XOR3 → RR зависит от слова; DET высок при длинном периоде (длинные диагонали)
    AND  → RR=1.0 при P=1; шахматная структура при P=2
    OR   → аналогично AND

Функции:
    state_hamming(row1, row2)                         → int
    recurrence_matrix(rows, eps)                      → list[list[int]]
    rqa_metrics(R, min_line)                          → dict
    trajectory_recurrence(word, rule, width, eps, n_cycles) → dict
    all_recurrences(word, width, eps)                 → dict[str, dict]
    build_recurrence_data(words, width, eps)          → dict
    recurrence_dict(word, width, eps)                 → dict
    print_recurrence(word, rule, width, eps, color)   → None
    print_rqa_stats(words, width, color)              → None

Запуск:
    python3 -m projects.hexglyph.solan_recurrence --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_recurrence --word ТУМАН --all-rules --no-color
    python3 -m projects.hexglyph.solan_recurrence --stats
"""
from __future__ import annotations

import argparse
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
_N_BITS        = 6           # bits per Q6 cell
_DEFAULT_WIDTH = 16
_DEFAULT_EPS   = 0           # exact recurrence
_N_CYCLES      = 4           # attractor cycles shown in recurrence plot
_DEFAULT_WORDS = list(LEXICON)


# ── Distance primitive ────────────────────────────────────────────────────────

def state_hamming(row1: list[int], row2: list[int]) -> int:
    """Суммарное Q6-хэмминг-расстояние между двумя состояниями CA."""
    return sum(bin(a ^ b).count('1') for a, b in zip(row1, row2))


# ── Recurrence matrix ─────────────────────────────────────────────────────────

def recurrence_matrix(
    rows: list[list[int]],
    eps:  int = _DEFAULT_EPS,
) -> list[list[int]]:
    """Рекуррентная матрица R[i][j] = 1 iff H(row_i, row_j) ≤ ε."""
    N = len(rows)
    R: list[list[int]] = []
    for i in range(N):
        r: list[int] = []
        for j in range(N):
            r.append(1 if state_hamming(rows[i], rows[j]) <= eps else 0)
        R.append(r)
    return R


# ── RQA metrics ───────────────────────────────────────────────────────────────

def rqa_metrics(
    R:        list[list[int]],
    min_line: int = 2,
) -> dict:
    """Вычислить RQA-метрики из рекуррентной матрицы.

    Главная диагональ (i==j) исключается из всех метрик.

    Возвращает dict: N, RR, DET, L, LAM, TT.
    """
    N = len(R)
    if N == 0:
        return {'N': 0, 'RR': 0.0, 'DET': 0.0, 'L': 0.0, 'LAM': 0.0, 'TT': 0.0}

    # ── Total off-diagonal recurrent points ──────────────────────────────────
    total = sum(R[i][j] for i in range(N) for j in range(N) if i != j)
    RR = total / (N * (N - 1)) if N > 1 else 0.0

    # ── Diagonal lines (all offsets ≠ 0) ─────────────────────────────────────
    diag_pts = 0
    all_diag_lens: list[int] = []

    for offset in range(1, N):
        for indices in [
            [(k, k + offset) for k in range(N - offset)],      # upper diagonal
            [(k + offset, k) for k in range(N - offset)],      # lower diagonal
        ]:
            run = 0
            for i, j in indices:
                if R[i][j]:
                    run += 1
                else:
                    if run >= min_line:
                        diag_pts += run
                        all_diag_lens.append(run)
                    run = 0
            if run >= min_line:
                diag_pts += run
                all_diag_lens.append(run)

    DET = diag_pts / total if total > 0 else 0.0
    L   = sum(all_diag_lens) / len(all_diag_lens) if all_diag_lens else 0.0

    # ── Vertical lines (per column, upper and lower halves separately) ────────
    # Split each column at the main diagonal to exclude R[j][j].
    vert_pts = 0
    all_vert_lens: list[int] = []

    for j in range(N):
        for seg_range in [range(j), range(j + 1, N)]:
            run = 0
            for i in seg_range:
                if R[i][j]:
                    run += 1
                else:
                    if run >= min_line:
                        vert_pts += run
                        all_vert_lens.append(run)
                    run = 0
            if run >= min_line:
                vert_pts += run
                all_vert_lens.append(run)

    LAM = vert_pts / total if total > 0 else 0.0
    TT  = sum(all_vert_lens) / len(all_vert_lens) if all_vert_lens else 0.0

    return {
        'N':   N,
        'RR':  round(RR,  4),
        'DET': round(DET, 4),
        'L':   round(L,   4),
        'LAM': round(LAM, 4),
        'TT':  round(TT,  4),
    }


# ── Per-word trajectory recurrence ────────────────────────────────────────────

def _traj_rows(
    word:     str,
    rule:     str,
    width:    int,
    n_cycles: int,
) -> tuple[list[list[int]], int, int]:
    """Строки CA-траектории (транзиент + n_cycles × период)."""
    cells = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells, rule)
    period = max(period, 1)          # guard against period=0
    n_steps = transient + n_cycles * period
    rows: list[list[int]] = []
    c = cells[:]
    for _ in range(n_steps):
        rows.append(c[:])
        c = step(c, rule)
    return rows, transient, period


def trajectory_recurrence(
    word:     str,
    rule:     str   = 'xor3',
    width:    int   = _DEFAULT_WIDTH,
    eps:      int   = _DEFAULT_EPS,
    n_cycles: int   = _N_CYCLES,
) -> dict:
    """Рекуррентный анализ CA-траектории слова при одном правиле.

    Возвращает dict:
        word        : str
        rule        : str
        width       : int
        eps         : int
        transient   : int
        period      : int
        n_steps     : int
        R           : list[list[int]]  — рекуррентная матрица (n_steps × n_steps)
        rqa         : dict             — {N, RR, DET, L, LAM, TT}
    """
    rows, transient, period = _traj_rows(word, rule, width, n_cycles)
    R   = recurrence_matrix(rows, eps)
    rqa = rqa_metrics(R)
    return {
        'word':      word.upper(),
        'rule':      rule,
        'width':     width,
        'eps':       eps,
        'transient': transient,
        'period':    period,
        'n_steps':   len(rows),
        'R':         R,
        'rqa':       rqa,
    }


def recurrence_summary(
    word:     str,
    rule:     str   = 'xor3',
    width:    int   = _DEFAULT_WIDTH,
    eps:      int   = _DEFAULT_EPS,
    n_cycles: int   = _N_CYCLES,
) -> dict:
    """Alias for trajectory_recurrence — standard *_summary convention."""
    return trajectory_recurrence(word, rule, width, eps, n_cycles)


def all_recurrences(
    word:     str,
    width:    int = _DEFAULT_WIDTH,
    eps:      int = _DEFAULT_EPS,
) -> dict[str, dict]:
    """Рекуррентный анализ по всем 4 правилам."""
    return {r: trajectory_recurrence(word, r, width, eps) for r in _ALL_RULES}


# ── Full dataset ───────────────────────────────────────────────────────────────

def build_recurrence_data(
    words:    list[str] | None = None,
    width:    int              = _DEFAULT_WIDTH,
    eps:      int              = _DEFAULT_EPS,
) -> dict:
    """RQA-метрики для всего лексикона × 4 правила.

    Возвращает dict:
        words     : list[str]
        width     : int
        eps       : int
        per_rule  : {rule: {word: rqa_dict}}
        ranking   : {rule: [(word, RR), ...]}  — убыв. по RR
        max_rr    : {rule: (word, RR)}
        min_rr    : {rule: (word, RR)}
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            tr = trajectory_recurrence(word, rule, width, eps)
            per_rule[rule][word] = tr['rqa']

    ranking: dict[str, list] = {}
    max_rr:  dict[str, tuple] = {}
    min_rr:  dict[str, tuple] = {}
    for rule in _ALL_RULES:
        by_rr = sorted(
            ((w, d['RR']) for w, d in per_rule[rule].items()),
            key=lambda x: -x[1],
        )
        ranking[rule] = by_rr
        max_rr[rule]  = by_rr[0]
        min_rr[rule]  = by_rr[-1]

    return {
        'words':    words,
        'width':    width,
        'eps':      eps,
        'per_rule': per_rule,
        'ranking':  ranking,
        'max_rr':   max_rr,
        'min_rr':   min_rr,
    }


# ── JSON export ────────────────────────────────────────────────────────────────

def recurrence_dict(
    word:  str,
    width: int = _DEFAULT_WIDTH,
    eps:   int = _DEFAULT_EPS,
) -> dict:
    """JSON-совместимый словарь RQA по всем правилам (матрица не включается)."""
    result: dict[str, object] = {
        'word':  word.upper(),
        'width': width,
        'eps':   eps,
        'rules': {},
    }
    for rule in _ALL_RULES:
        tr = trajectory_recurrence(word, rule, width, eps)
        result['rules'][rule] = {         # type: ignore[index]
            'transient': tr['transient'],
            'period':    tr['period'],
            'n_steps':   tr['n_steps'],
            **tr['rqa'],
        }
    return result


# ── ASCII display ──────────────────────────────────────────────────────────────

_DIAG_CH  = '■'   # main diagonal
_REC_CH   = '█'   # recurrent (off-diagonal)
_NOREC_CH = '·'   # non-recurrent


def print_recurrence(
    word:     str,
    rule:     str  = 'xor3',
    width:    int  = _DEFAULT_WIDTH,
    eps:      int  = _DEFAULT_EPS,
    color:    bool = True,
) -> None:
    """Напечатать рекуррентный граф (ASCII) и RQA-метрики."""
    tr   = trajectory_recurrence(word, rule, width, eps)
    R    = tr['R']
    rqa  = tr['rqa']
    N    = rqa['N']
    col  = _RULE_COLOR.get(rule, '') if color else ''
    name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    print(f"{bold}  ◈ Рекуррентный граф Q6  {word.upper()}  |  "
          f"{col}{name}{rst}  "
          f"(T={tr['transient']}, P={tr['period']}, N={N}, ε={eps})")
    print(f"  {'─' * (N + 6)}")

    for i, row in enumerate(R):
        line_parts: list[str] = []
        for j, v in enumerate(row):
            if i == j:
                line_parts.append(f"{dim}{_DIAG_CH}{rst}")
            elif v:
                line_parts.append(f"{col}{_REC_CH}{rst}")
            else:
                line_parts.append(_NOREC_CH)
        print(f"  {i:2d} {''.join(line_parts)}")

    print(f"  {'─' * (N + 6)}")
    # RQA summary
    print(f"  RR={rqa['RR']:.3f}  DET={rqa['DET']:.3f}  L={rqa['L']:.2f}  "
          f"LAM={rqa['LAM']:.3f}  TT={rqa['TT']:.2f}")
    print()


def print_rqa_stats(
    words: list[str] | None = None,
    width: int              = _DEFAULT_WIDTH,
    color: bool             = True,
) -> None:
    """Сводная таблица RQA-метрик (RR) для лексикона × 4 правила."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Рекуррентность аттрактора Q6 — RR (ε=0){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")
    for word in sorted(words):
        row_parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            tr  = trajectory_recurrence(word, rule, width)
            rr  = tr['rqa']['RR']
            col = _RULE_COLOR.get(rule, '') if color else ''
            row_parts.append(f"  {col}{rr:>8.3f}{rst}")
        print('  ' + ''.join(row_parts))


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Рекуррентный анализ Q6 CA')
    parser.add_argument('--word',      default='ГОРА',  help='Русское слово')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--eps',       type=int, default=_DEFAULT_EPS)
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--json',      action='store_true', help='JSON output')
    parser.add_argument('--no-color',  action='store_true')
    args   = parser.parse_args()
    color  = not args.no_color
    if args.json:
        import json as _json
        print(_json.dumps(recurrence_dict(args.word, args.width, args.eps),
                          ensure_ascii=False, indent=2))
    elif args.stats:
        print_rqa_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_recurrence(args.word, rule, args.width, args.eps, color)
    else:
        print_recurrence(args.word, args.rule, args.width, args.eps, color)


if __name__ == '__main__':
    _main()
