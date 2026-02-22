"""solan_complexity.py — Сложность Лемпеля–Зива (LZ76) CA-траекторий Q6.

Двоичная строка, полученная из CA-траектории (Q6-значения → 6-битный код
× ширина × шаги), парсится алгоритмом LZ76.  Число фраз c(n) нормируется:

    C_norm = c(n) · log₂(n) / n   ∈ [0, ~1]

  C_norm → 1   для случайных / несжимаемых строк
  C_norm → 0   для постоянных / строго периодических строк

Применяется к:
  «полная траектория»  — транзиент + аттрактор (все шаги × ширина × 6 бит)
  «только аттрактор»   — один период аттрактора

Это даёт скалярную «информационную сложность» слова при каждом правиле.

Функции:
    to_bits(values)                         → list[int]
    lz76_phrases(s)                         → int        — число фраз LZ76
    lz76_norm(s)                            → float      — норм. сложность
    trajectory_complexity(word, rule, width)→ dict
    all_complexities(word, width)           → dict[str, dict]
    build_complexity_data(words, width)     → dict
    complexity_dict(word, width)            → dict
    print_complexity(word, width, color)
    print_complexity_ranking(words, width, color)

Запуск:
    python3 -m projects.hexglyph.solan_complexity --word ГОРА
    python3 -m projects.hexglyph.solan_complexity --ranking --no-color
    python3 -m projects.hexglyph.solan_complexity --word ТУМАН --all-rules --no-color
"""
from __future__ import annotations

import argparse
import math
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
_DEFAULT_WORDS = list(LEXICON)


# ── Binary representation ─────────────────────────────────────────────────────

def to_bits(values: list[int], n_bits: int = _N_BITS) -> list[int]:
    """Преобразовать список целых в список бит (MSB first)."""
    bits: list[int] = []
    for v in values:
        for b in range(n_bits - 1, -1, -1):
            bits.append((v >> b) & 1)
    return bits


# ── LZ76 complexity ───────────────────────────────────────────────────────────

def lz76_phrases(s: list[int]) -> int:
    """Число фраз LZ76 для бинарной строки s.

    Парсинг: на каждом шаге расширяем текущую фразу s[i:i+k], пока она
    встречается как подстрока s[0:i+k-1]; при неудаче — фиксируем фразу.

    Сложность O(n² · k_max), приемлемо для строк длиной ≤ 5000.
    """
    n = len(s)
    if n == 0:
        return 0
    c = 1   # number of phrases (seed = 1)
    i = 0   # start of current phrase
    k = 1   # current phrase length
    while i + k <= n:
        # Is s[i:i+k] a substring of s[0 : i+k-1]?
        phrase  = s[i:i + k]
        end     = i + k - 1         # search up to (not including) end
        found   = False
        ph_len  = k
        for start in range(end - ph_len + 1):
            if s[start:start + ph_len] == phrase:
                found = True
                break
        if found:
            k += 1
        else:
            c += 1
            i += k
            k  = 1
    return c


def lz76_norm(s: list[int]) -> float:
    """Нормированная сложность LZ76 ∈ [0, ≈1].

    C_norm = c(n) · log₂(n) / n

    Для случайного источника C_norm → 1 при n → ∞.
    Для постоянной строки C_norm → 0.
    """
    n = len(s)
    if n <= 1:
        return 0.0
    c = lz76_phrases(s)
    return c * math.log2(n) / n


# ── Per-word trajectory complexity ────────────────────────────────────────────

def _traj_rows(word: str, rule: str, width: int) -> tuple[list[list[int]], int, int]:
    """Вернуть (все строки траектории, transient, period)."""
    cells = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells, rule)
    rows: list[list[int]] = []
    c = cells[:]
    for _ in range(transient + period):
        rows.append(c[:])
        c = step(c, rule)
    return rows, transient, period


def trajectory_complexity(
    word:  str,
    rule:  str = 'xor3',
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """Сложность LZ76 CA-траектории слова при одном правиле.

    Возвращает dict:
        word          : str
        rule          : str
        transient     : int
        period        : int
        traj_bits     : int          — длина полной битовой строки
        attr_bits     : int          — длина битовой строки аттрактора
        traj_phrases  : int          — число фраз LZ76 (полная)
        attr_phrases  : int          — число фраз LZ76 (аттрактор)
        traj_norm     : float        — норм. сложность полной траектории
        attr_norm     : float        — норм. сложность аттрактора
    """
    rows, transient, period = _traj_rows(word, rule, width)
    # Full trajectory bits
    full_flat  = [v for row in rows for v in row]
    attr_flat  = [v for row in rows[transient:] for v in row]
    traj_s     = to_bits(full_flat)
    attr_s     = to_bits(attr_flat)
    traj_ph    = lz76_phrases(traj_s)
    attr_ph    = lz76_phrases(attr_s) if attr_s else 0
    traj_norm  = lz76_norm(traj_s)
    attr_norm  = lz76_norm(attr_s) if attr_s else 0.0
    return {
        'word':         word.upper(),
        'rule':         rule,
        'transient':    transient,
        'period':       period,
        'traj_bits':    len(traj_s),
        'attr_bits':    len(attr_s),
        'traj_phrases': traj_ph,
        'attr_phrases': attr_ph,
        'traj_norm':    round(traj_norm, 4),
        'attr_norm':    round(attr_norm, 4),
    }


def all_complexities(word: str, width: int = _DEFAULT_WIDTH) -> dict[str, dict]:
    """Сложность по всем 4 правилам."""
    return {r: trajectory_complexity(word, r, width) for r in _ALL_RULES}


# ── Full dataset ──────────────────────────────────────────────────────────────

def build_complexity_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """Сводный анализ сложности для всего лексикона.

    Возвращает dict:
        words         : list[str]
        width         : int
        per_rule      : {rule: {word: {traj_norm, attr_norm, period}}}
        ranking_traj  : {rule: list[(word, traj_norm)]}  — убыв. по traj_norm
        ranking_attr  : {rule: list[(word, attr_norm)]}
        most_complex  : {rule: (word, traj_norm)}
        least_complex : {rule: (word, traj_norm)}
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            tc = trajectory_complexity(word, rule, width)
            per_rule[rule][word] = {
                'traj_norm': tc['traj_norm'],
                'attr_norm': tc['attr_norm'],
                'period':    tc['period'],
                'transient': tc['transient'],
            }
    ranking_traj:  dict[str, list] = {}
    ranking_attr:  dict[str, list] = {}
    most_complex:  dict[str, tuple] = {}
    least_complex: dict[str, tuple] = {}
    for rule in _ALL_RULES:
        by_traj = sorted(
            ((w, d['traj_norm']) for w, d in per_rule[rule].items()),
            key=lambda x: -x[1]
        )
        by_attr = sorted(
            ((w, d['attr_norm']) for w, d in per_rule[rule].items()),
            key=lambda x: -x[1]
        )
        ranking_traj[rule]  = by_traj
        ranking_attr[rule]  = by_attr
        most_complex[rule]  = by_traj[0]
        least_complex[rule] = by_traj[-1]
    return {
        'words':         words,
        'width':         width,
        'per_rule':      per_rule,
        'ranking_traj':  ranking_traj,
        'ranking_attr':  ranking_attr,
        'most_complex':  most_complex,
        'least_complex': least_complex,
    }


# ── JSON export ───────────────────────────────────────────────────────────────

def complexity_dict(word: str, width: int = _DEFAULT_WIDTH) -> dict:
    """JSON-совместимый словарь по всем правилам."""
    result: dict[str, object] = {
        'word':  word.upper(),
        'width': width,
        'rules': {},
    }
    for rule in _ALL_RULES:
        tc = trajectory_complexity(word, rule, width)
        result['rules'][rule] = {  # type: ignore[index]
            'traj_norm':    tc['traj_norm'],
            'attr_norm':    tc['attr_norm'],
            'traj_phrases': tc['traj_phrases'],
            'attr_phrases': tc['attr_phrases'],
            'traj_bits':    tc['traj_bits'],
            'attr_bits':    tc['attr_bits'],
            'period':       tc['period'],
            'transient':    tc['transient'],
        }
    return result


# ── ASCII display ─────────────────────────────────────────────────────────────

_BAR_CHARS = '▏▎▍▌▋▊▉█'
_BAR_MAX_W = 32


def _frac_bar(val: float, max_val: float = 1.0, width: int = _BAR_MAX_W) -> str:
    """Горизонтальная полоска от 0 до max_val."""
    if max_val == 0:
        return ' ' * width
    frac  = min(1.0, max(0.0, val / max_val))
    full  = int(frac * width)
    rem   = frac * width - full
    idx   = int(rem * len(_BAR_CHARS))
    bar   = '█' * full
    if idx > 0 and full < width:
        bar += _BAR_CHARS[idx - 1]
    return (bar + ' ' * width)[:width]


def print_complexity(
    word:  str,
    width: int = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Распечатать нормированные сложности для всех 4 правил."""
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''
    print(f"{bold}  ◈ LZ76-сложность Q6  {word.upper()}{rst}")
    print(f"  {'─' * 62}")
    print(f"  {'Правило':12s}  {'T':>4}  {'P':>4}  {'полн.':>6}  {'атт.':>6}  {'гистограмма (полная)'}")
    print(f"  {'─' * 62}")
    for rule in _ALL_RULES:
        tc  = trajectory_complexity(word, rule, width)
        col = _RULE_COLOR.get(rule, '') if color else ''
        bar = _frac_bar(tc['traj_norm'])
        # Colour intensity by complexity
        if tc['traj_norm'] > 0.6:
            nc = '\033[38;5;196m' if color else ''
        elif tc['traj_norm'] > 0.3:
            nc = '\033[38;5;226m' if color else ''
        else:
            nc = dim
        name = _RULE_NAMES.get(rule, rule)
        print(f"  {col}{name:12s}{rst}  {tc['transient']:>4}  {tc['period']:>4}  "
              f"{nc}{tc['traj_norm']:>6.3f}{rst}  "
              f"{dim}{tc['attr_norm']:>6.3f}{rst}  "
              f"{nc}{bar}{rst}")
    print()


def print_complexity_ranking(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Таблица нормированных сложностей, отсортированная по убыванию."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>10s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ LZ76-сложность (норм.) — ранжировка по XOR3{rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")
    # Build rows
    rows_data = []
    for word in words:
        vals = {r: trajectory_complexity(word, r, width)['traj_norm']
                for r in _ALL_RULES}
        rows_data.append((word, vals))
    # Sort by xor3 descending
    rows_data.sort(key=lambda x: -x[1].get('xor3', 0))
    for word, vals in rows_data:
        parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            col = _RULE_COLOR.get(rule, '') if color else ''
            parts.append(f"  {col}{vals[rule]:>10.3f}{rst}")
        print('  ' + ''.join(parts))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='LZ76-сложность CA-траекторий Q6')
    parser.add_argument('--word',     default='ГОРА',  help='Русское слово')
    parser.add_argument('--ranking',  action='store_true')
    parser.add_argument('--width',    type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--json',     action='store_true', help='JSON output')
    args = parser.parse_args()
    color = not args.no_color
    if args.json:
        import json as _json
        print(_json.dumps(complexity_dict(args.word, args.width), ensure_ascii=False, indent=2))
        return
    if args.ranking:
        print_complexity_ranking(color=color, width=args.width)
    else:
        print_complexity(args.word, args.width, color)


if __name__ == '__main__':
    _main()
