"""solan_traj.py — Пространственно-временные траектории Q6-автомата.

Для каждого слова и правила CA вычисляет полную траекторию:
  начальное состояние → транзиент → один цикл аттрактора

Функции:
    word_trajectory(word, rule, width) → {rows, transient, period}
    all_word_trajectories(word, width) → {rule: traj_dict, ...}
    traj_stats(word, rule, width) → {entropy, attr_entropy, mean_q6, ...}
    build_trajectory_data(words, width) → сводный анализ по всему лексикону
    trajectory_similarity(t1, t2) → float
    print_trajectory(word, rule, width, color) — ASCII-тепловая карта
    trajectory_dict(word, width) → JSON-совместимый словарь

Запуск:
    python3 -m projects.hexglyph.solan_traj --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_traj --word ТУМАН --rule and --no-color
    python3 -m projects.hexglyph.solan_traj --stats
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to, find_orbit
from projects.hexglyph.solan_ca import (
    step, render_row_char,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16
_DEFAULT_WORDS = list(LEXICON)


# ── Core trajectory ─────────────────────────────────────────────────────────

def word_trajectory(word: str, rule: str = 'xor3', width: int = _DEFAULT_WIDTH) -> dict:
    """Вычислить полную CA-траекторию слова.

    Возвращает dict:
        rows      : list[list[int]]  — строки (t=0..transient+period-1)
        transient : int              — длина транзиента
        period    : int              — длина цикла аттрактора
        n_rows    : int              — итого строк (= transient + period)
        word      : str
        rule      : str
        width     : int
    """
    cells = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells, rule)
    n = transient + period
    rows: list[list[int]] = []
    c = cells[:]
    for _ in range(n):
        rows.append(c[:])
        c = step(c, rule)
    return {
        'rows': rows,
        'transient': transient,
        'period': period,
        'n_rows': n,
        'word': word.upper(),
        'rule': rule,
        'width': width,
    }


def all_word_trajectories(word: str, width: int = _DEFAULT_WIDTH) -> dict[str, dict]:
    """Вычислить траектории по всем 4 правилам."""
    return {r: word_trajectory(word, r, width) for r in _ALL_RULES}


# ── Statistics ───────────────────────────────────────────────────────────────

def _shannon(values: list[int]) -> float:
    """Нормированная энтропия Шеннона (бит)."""
    if not values:
        return 0.0
    n = len(values)
    freq: dict[int, int] = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    h = -sum(c / n * math.log2(c / n) for c in freq.values() if c > 0)
    return max(0.0, h)  # avoid -0.0 floating-point artifact


def traj_stats(word: str, rule: str = 'xor3', width: int = _DEFAULT_WIDTH) -> dict:
    """Статистика траектории: энтропия, среднее Q6, уникальные состояния.

    Возвращает dict:
        transient       : int
        period          : int
        trans_entropy   : float  — H(Q6) транзиентных строк
        attr_entropy    : float  — H(Q6) аттракторных строк
        total_entropy   : float  — H(Q6) всей траектории
        mean_q6         : float  — среднее значение Q6 по аттрактору
        unique_cells    : int    — уникальных Q6-значений в аттракторе
        unique_states   : int    — уникальных строк в аттракторе (= period)
    """
    traj = word_trajectory(word, rule, width)
    rows, tr, per = traj['rows'], traj['transient'], traj['period']
    trans_vals = [v for row in rows[:tr] for v in row]
    attr_vals  = [v for row in rows[tr:] for v in row]
    all_vals   = [v for row in rows      for v in row]
    return {
        'transient':     tr,
        'period':        per,
        'trans_entropy': _shannon(trans_vals) if trans_vals else 0.0,
        'attr_entropy':  _shannon(attr_vals)  if attr_vals  else 0.0,
        'total_entropy': _shannon(all_vals),
        'mean_q6':       sum(attr_vals) / len(attr_vals) if attr_vals else 0.0,
        'unique_cells':  len(set(attr_vals)),
        'unique_states': per,
    }


# ── Similarity ───────────────────────────────────────────────────────────────

def trajectory_similarity(t1: dict, t2: dict) -> float:
    """Нормированное расстояние Хэмминга между двумя траекториями.

    Траектории выравниваются по длине (более короткая — циклически дополняется).
    Каждая клетка сравнивается побитово: нормируется на (ширина × 6 бит).
    Возвращает 0.0 (идентичны) … 1.0 (максимально различны).
    """
    rows1, rows2 = t1['rows'], t2['rows']
    n = max(len(rows1), len(rows2))
    width = t1.get('width', 16)
    if n == 0 or width == 0:
        return 0.0
    total_bits = n * width * 6  # Q6 = 6 бит
    hamming = 0
    for i in range(n):
        r1 = rows1[i % len(rows1)]
        r2 = rows2[i % len(rows2)]
        for a, b in zip(r1, r2):
            hamming += bin(a ^ b).count('1')
    return hamming / total_bits


# ── Build full dataset ────────────────────────────────────────────────────────

def build_trajectory_data(words: list[str] | None = None, width: int = _DEFAULT_WIDTH) -> dict:
    """Сводный анализ траекторий для всего лексикона.

    Возвращает dict:
        words      : list[str]
        width      : int
        per_rule   : {rule: {word: stats_dict}}
        max_entropy: {rule: (word, entropy)}   — самая сложная траектория
        min_entropy: {rule: (word, entropy)}   — самая простая траектория
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            per_rule[rule][word] = traj_stats(word, rule, width)
    # Find extrema by total_entropy
    max_e: dict[str, tuple[str, float]] = {}
    min_e: dict[str, tuple[str, float]] = {}
    for rule in _ALL_RULES:
        entries = [(w, d['total_entropy']) for w, d in per_rule[rule].items()]
        max_e[rule] = max(entries, key=lambda x: x[1])
        min_e[rule] = min(entries, key=lambda x: x[1])
    return {
        'words':       words,
        'width':       width,
        'per_rule':    per_rule,
        'max_entropy': max_e,
        'min_entropy': min_e,
    }


# ── JSON export ───────────────────────────────────────────────────────────────

def trajectory_dict(word: str, width: int = _DEFAULT_WIDTH) -> dict:
    """JSON-совместимый словарь траекторий по всем правилам."""
    result: dict[str, object] = {'word': word.upper(), 'width': width, 'rules': {}}
    for rule in _ALL_RULES:
        traj = word_trajectory(word, rule, width)
        st   = traj_stats(word, rule, width)
        result['rules'][rule] = {  # type: ignore[index]
            'transient':     traj['transient'],
            'period':        traj['period'],
            'n_rows':        traj['n_rows'],
            'rows':          traj['rows'],
            'attr_entropy':  round(st['attr_entropy'],  4),
            'total_entropy': round(st['total_entropy'], 4),
            'mean_q6':       round(st['mean_q6'], 2),
            'unique_cells':  st['unique_cells'],
        }
    return result


# ── ASCII display ─────────────────────────────────────────────────────────────

_TRANS_MARK  = '\033[38;5;208m'  # orange
_ATTR_MARK   = '\033[38;5;81m'   # cyan
_BOUND_MARK  = '\033[38;5;226m'  # yellow


def print_trajectory(
    word: str,
    rule: str = 'xor3',
    width: int = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Напечатать CA-траекторию в виде ASCII тепловой карты.

    Каждая строка — один шаг CA; каждая клетка — один Solan-символ.
    Транзиент выделен оранжевым, аттрактор — голубым.
    """
    traj = word_trajectory(word, rule, width)
    rows, tr, per = traj['rows'], traj['transient'], traj['period']
    rule_col = _RULE_COLOR.get(rule, '') if color else ''
    rst = _RST if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    # Header
    rule_name = _RULE_NAMES.get(rule, rule.upper())
    tc = _TRANS_MARK if color else ''
    ac = _ATTR_MARK  if color else ''
    print(f"{bold}  ◈ Траектория Q6  {word.upper()}  |  {rule_col}{rule_name}{rst}{bold}"
          f"  |  T={tc}{tr}{rst}{bold}  P={ac}{per}{rst}")
    print(f"  {'─' * (width + 12)}")

    for i, row in enumerate(rows):
        phase = 'T' if i < tr else 'A'
        if color:
            if i < tr:
                ph_col = _TRANS_MARK
                ph_sym = 't'
            else:
                ph_col = _ATTR_MARK
                ph_sym = 'a' if per > 1 else 'A'
        else:
            ph_col, ph_sym = '', ('t' if i < tr else 'a')

        # Boundary line
        if i == tr and tr > 0:
            sep_col = _BOUND_MARK if color else ''
            print(f"  {sep_col}{'╌' * (width + 8)}{rst}")

        label = f"  {ph_col}{ph_sym}{i:02d}{rst}│"
        rendered = render_row_char(row, color)
        cycle_marker = ''
        if i >= tr:
            cyc = i - tr
            cycle_marker = f"  {dim}[{cyc % per + 1}/{per}]{rst}"
        print(label + ' ' + rendered + cycle_marker)

    print()
    # Stats
    st = traj_stats(word, rule, width)
    print(f"  {dim}  H_attr={st['athr_entropy']:.3f}  H_total={st['total_entropy']:.3f}"
          f"  ⟨Q6⟩={st['mean_q6']:.1f}  unique_cells={st['unique_cells']}{rst}")


def _print_traj_fixed(word, rule, width, color):
    """Internal — typo-safe wrapper for print_trajectory stats line."""
    traj = word_trajectory(word, rule, width)
    rows, tr, per = traj['rows'], traj['transient'], traj['period']
    rule_col = _RULE_COLOR.get(rule, '') if color else ''
    rst = _RST if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    rule_name = _RULE_NAMES.get(rule, rule.upper())
    tc = _TRANS_MARK if color else ''
    ac = _ATTR_MARK  if color else ''
    print(f"{bold}  ◈ Траектория Q6  {word.upper()}  |  {rule_col}{rule_name}{rst}{bold}"
          f"  |  T={tc}{tr}{rst}{bold}  P={ac}{per}{rst}")
    print(f"  {'─' * (width + 12)}")

    for i, row in enumerate(rows):
        if color:
            ph_col = _TRANS_MARK if i < tr else _ATTR_MARK
            ph_sym = 't' if i < tr else 'a'
        else:
            ph_col, ph_sym = '', ('t' if i < tr else 'a')

        if i == tr and tr > 0:
            sep_col = _BOUND_MARK if color else ''
            print(f"  {sep_col}{'╌' * (width + 8)}{rst}")

        label = f"  {ph_col}{ph_sym}{i:02d}{rst}│"
        rendered = render_row_char(row, color)
        cycle_marker = ''
        if i >= tr:
            cyc = i - tr
            cycle_marker = f"  {dim}[{cyc % per + 1}/{per}]{rst}"
        print(label + ' ' + rendered + cycle_marker)

    print()
    st = traj_stats(word, rule, width)
    print(f"  {dim}  H_attr={st['attr_entropy']:.3f}  H_total={st['total_entropy']:.3f}"
          f"  ⟨Q6⟩={st['mean_q6']:.1f}  unique_cells={st['unique_cells']}{rst}")


def print_all_trajectories(word: str, width: int = _DEFAULT_WIDTH, color: bool = True) -> None:
    """Напечатать траектории по всем 4 правилам."""
    for rule in _ALL_RULES:
        _print_traj_fixed(word, rule, width, color)
        print()


def print_trajectory_stats(words: list[str] | None = None, width: int = _DEFAULT_WIDTH,
                           color: bool = True) -> None:
    """Таблица энтропий траекторий для лексикона."""
    words = words if words is not None else _DEFAULT_WORDS
    rst = _RST if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:8s}{rst}" for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Энтропия траекторий Q6 (аттрактор){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")

    for word in sorted(words):
        row_parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            st = traj_stats(word, rule, width)
            h = st['attr_entropy']
            col = _RULE_COLOR.get(rule, '') if color else ''
            row_parts.append(f"  {col}{h:8.3f}{rst}")
        print('  ' + ''.join(row_parts))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Траектории CA Q6')
    parser.add_argument('--word', default='ГОРА', help='Русское слово')
    parser.add_argument('--rule', default='xor3', choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true', help='Все 4 правила')
    parser.add_argument('--stats', action='store_true', help='Таблица энтропий лексикона')
    parser.add_argument('--width', type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    color = not args.no_color
    if args.stats:
        print_trajectory_stats(color=color)
    elif args.all_rules:
        print_all_trajectories(args.word, args.width, color)
    else:
        _print_traj_fixed(args.word, args.rule, args.width, color)


if __name__ == '__main__':
    _main()
