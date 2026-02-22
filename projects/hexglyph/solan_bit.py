"""solan_bit.py — Побитовое разложение Q6-автомата.

Q6-значение = 6-битное число (h = T×1 + B×2 + L×4 + R×8 + D1×16 + D2×32).
Все 4 правила CA (xor, xor3, and, or) — побитовые операции, поэтому
каждый из 6 битовых слоёв (T/B/L/R/D1/D2) эволюционирует независимо
как 1-битный клеточный автомат.

Функции:
    extract_bit_plane(cells, bit) → list[int]
    bit_step(bits, rule) → list[int]
    bit_plane_trajectory(word, bit, rule, width) → dict
    word_bit_planes(word, rule, width) → dict[int, dict]
    attractor_activity(word, rule, width) → dict[int, dict]
    bit_plane_signature(word, width) → dict
    build_bit_plane_data(words, width) → dict
    bit_plane_dict(word, width) → dict
    print_bit_planes(word, rule, width, color)
    print_bit_plane_summary(words, width, color)

Запуск:
    python3 -m projects.hexglyph.solan_bit --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_bit --word ТУМАН --all-rules --no-color
    python3 -m projects.hexglyph.solan_bit --stats
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

# h = T×1 + B×2 + L×4 + R×8 + D1×16 + D2×32
_SEGMENT_NAMES = ['T', 'B', 'L', 'R', 'D1', 'D2']  # индексы битов 0..5

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16
_DEFAULT_WORDS = list(LEXICON)


# ── Bit-plane primitives ─────────────────────────────────────────────────────

def extract_bit_plane(cells: list[int], bit: int) -> list[int]:
    """Извлечь битовый слой: список 0/1 для бита `bit` из каждой клетки."""
    return [(c >> bit) & 1 for c in cells]


def bit_step(bits: list[int], rule: str) -> list[int]:
    """Один шаг 1-битного CA.

    Правила идентичны Q6-шагу, но значения = 0/1 (бит-слой).
    Тороидальные граничные условия.
    """
    n = len(bits)
    if rule == 'xor':
        return [bits[(i - 1) % n] ^ bits[(i + 1) % n] for i in range(n)]
    if rule == 'xor3':
        return [bits[(i - 1) % n] ^ bits[i] ^ bits[(i + 1) % n]
                for i in range(n)]
    if rule == 'and':
        return [bits[(i - 1) % n] & bits[(i + 1) % n] for i in range(n)]
    if rule == 'or':
        return [bits[(i - 1) % n] | bits[(i + 1) % n] for i in range(n)]
    raise ValueError(f"Неизвестное правило: {rule!r}")


def _find_bit_orbit(bits: list[int], rule: str) -> tuple[int, int]:
    """Floyd-орбита 1-битного CA → (transient, period)."""
    seen: dict[tuple[int, ...], int] = {}
    cur = bits[:]
    t = 0
    while True:
        key = tuple(cur)
        if key in seen:
            return seen[key], t - seen[key]
        seen[key] = t
        cur = bit_step(cur, rule)
        t += 1


# ── Per-bit trajectory ────────────────────────────────────────────────────────

def bit_plane_trajectory(
    word: str,
    bit: int,
    rule: str = 'xor3',
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """Траектория 1-битного CA для одного бита слова.

    Возвращает dict:
        bit          : int               — индекс бита (0=T … 5=D2)
        segment      : str               — имя сегмента
        rule         : str
        word         : str
        rows         : list[list[int]]   — строки 0/1 (transient + period шагов)
        transient    : int
        period       : int
        active       : bool              — аттрактор не нулевой
        mean_attr    : float             — средняя плотность 1-бит в аттракторе
    """
    cells = pad_to(encode_word(word.upper()), width)
    init  = extract_bit_plane(cells, bit)
    transient, period = _find_bit_orbit(init, rule)
    n   = transient + period
    rows: list[list[int]] = []
    cur = init[:]
    for _ in range(n):
        rows.append(cur[:])
        cur = bit_step(cur, rule)
    attr_vals = [v for row in rows[transient:] for v in row]
    active    = any(v for v in attr_vals)
    mean_attr = sum(attr_vals) / len(attr_vals) if attr_vals else 0.0
    return {
        'bit':       bit,
        'segment':   _SEGMENT_NAMES[bit],
        'rule':      rule,
        'word':      word.upper(),
        'rows':      rows,
        'transient': transient,
        'period':    period,
        'active':    active,
        'mean_attr': mean_attr,
    }


def word_bit_planes(
    word: str,
    rule: str = 'xor3',
    width: int = _DEFAULT_WIDTH,
) -> dict[int, dict]:
    """Траектории по всем 6 битам при одном правиле.

    Возвращает {0: plane_T, 1: plane_B, ..., 5: plane_D2}.
    """
    return {b: bit_plane_trajectory(word, b, rule, width) for b in range(6)}


# ── Activity summary ─────────────────────────────────────────────────────────

def attractor_activity(
    word: str,
    rule: str = 'xor3',
    width: int = _DEFAULT_WIDTH,
) -> dict[int, dict]:
    """Сводная активность аттрактора по каждому биту.

    Возвращает {bit: {active, period, mean_attr, segment}}.
    """
    return {
        b: {
            'active':    p['active'],
            'period':    p['period'],
            'mean_attr': p['mean_attr'],
            'segment':   p['segment'],
        }
        for b, p in word_bit_planes(word, rule, width).items()
    }


# ── 4-rule × 6-bit signature ─────────────────────────────────────────────────

def bit_plane_signature(word: str, width: int = _DEFAULT_WIDTH) -> dict:
    """4×6 матрица активности: {rule: {bit: {active, period, mean_attr, segment}}}.

    Для каждого из 4 правил — какие из 6 битовых слоёв «живые» в аттракторе.
    """
    return {r: attractor_activity(word, r, width) for r in _ALL_RULES}


# ── Full dataset ──────────────────────────────────────────────────────────────

def build_bit_plane_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """Сводный анализ битовых слоёв для всего лексикона.

    Возвращает dict:
        words        : list[str]
        width        : int
        per_rule     : {rule: {word: {bit: {active, period, mean_attr, segment}}}}
        active_count : {rule: {word: int}}   — число активных битов (0..6)
        max_active   : {rule: (word, count)} — слово с наибольшим числом активных бит
        min_active   : {rule: (word, count)} — слово с наименьшим числом активных бит
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            per_rule[rule][word] = attractor_activity(word, rule, width)
    # Active bit counts
    active_count: dict[str, dict[str, int]] = {}
    for rule in _ALL_RULES:
        active_count[rule] = {
            word: sum(1 for d in per_rule[rule][word].values() if d['active'])
            for word in words
        }
    # Extrema
    max_active: dict[str, tuple[str, int]] = {}
    min_active: dict[str, tuple[str, int]] = {}
    for rule in _ALL_RULES:
        entries = list(active_count[rule].items())
        max_active[rule] = max(entries, key=lambda x: x[1])
        min_active[rule] = min(entries, key=lambda x: x[1])
    return {
        'words':        words,
        'width':        width,
        'per_rule':     per_rule,
        'active_count': active_count,
        'max_active':   max_active,
        'min_active':   min_active,
    }


# ── JSON export ───────────────────────────────────────────────────────────────

def bit_plane_dict(word: str, width: int = _DEFAULT_WIDTH) -> dict:
    """JSON-совместимый словарь битовых слоёв по всем правилам."""
    result: dict[str, object] = {
        'word':          word.upper(),
        'width':         width,
        'segment_names': _SEGMENT_NAMES,
        'rules':         {},
    }
    for rule in _ALL_RULES:
        planes = word_bit_planes(word, rule, width)
        result['rules'][rule] = {  # type: ignore[index]
            str(b): {
                'segment':   p['segment'],
                'transient': p['transient'],
                'period':    p['period'],
                'active':    p['active'],
                'mean_attr': round(p['mean_attr'], 4),
                'rows':      p['rows'],
            }
            for b, p in planes.items()
        }
    return result


# ── ASCII display ─────────────────────────────────────────────────────────────

_BIT_COLORS = [
    '\033[38;5;196m',  # bit 0 T  — красный
    '\033[38;5;208m',  # bit 1 B  — оранжевый
    '\033[38;5;226m',  # bit 2 L  — жёлтый
    '\033[38;5;46m',   # bit 3 R  — зелёный
    '\033[38;5;51m',   # bit 4 D1 — голубой
    '\033[38;5;129m',  # bit 5 D2 — фиолетовый
]
_LIVE = '█'
_DEAD = '·'
_TRANS_C = '\033[38;5;208m'
_ATTR_C  = '\033[38;5;81m'
_BOUND_C = '\033[38;5;226m'


def print_bit_planes(
    word: str,
    rule: str = 'xor3',
    width: int = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Распечатать 6 мини-тепловых карт битовых слоёв (параллельно по столбцам).

    Каждый столбец — один бит (T/B/L/R/D1/D2), каждая строка — один шаг CA.
    '█' = 1, '·' = 0; транзиент / аттрактор разделены чертой.
    """
    planes = word_bit_planes(word, rule, width)
    rule_col  = _RULE_COLOR.get(rule, '') if color else ''
    rule_name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    print(f"{bold}  ◈ Битовые плоскости Q6  {word.upper()}  |  "
          f"{rule_col}{rule_name}{rst}")

    # Header: segment names coloured
    hdr = '  row  '
    for b in range(6):
        bc  = _BIT_COLORS[b] if color else ''
        seg = _SEGMENT_NAMES[b]
        # Each plane is `width` chars wide
        pad = (width - len(seg)) // 2
        hdr += f"  {bc}{' ' * pad}{seg}{' ' * (width - len(seg) - pad)}{rst}"
    print(hdr)
    total_dash = 8 + 6 * (width + 2)
    print(f"  {'─' * total_dash}")

    max_rows = max(p['transient'] + p['period'] for p in planes.values())
    max_tr   = max(p['transient'] for p in planes.values())

    for i in range(max_rows):
        # Separator at the transient/attractor boundary (use max_tr)
        if i == max_tr and max_tr > 0:
            sep_c = _BOUND_C if color else ''
            print(f"  {sep_c}{'╌' * total_dash}{rst}")

        label_c = (_TRANS_C if i < max_tr else _ATTR_C) if color else ''
        row_line = f"  {label_c}{i:03d}{rst}  "
        for b in range(6):
            p   = planes[b]
            nr  = p['transient'] + p['period']
            bc  = _BIT_COLORS[b] if color else ''
            if i < nr:
                row = p['rows'][i]
                ph_c = (_TRANS_C if i < p['transient'] else bc) if color else ''
                chars = ''.join((_LIVE if v else _DEAD) for v in row)
                row_line += f"  {ph_c}{chars}{rst}"
            else:
                row_line += f"  {dim}{' ' * width}{rst}"
        print(row_line)

    print()
    # Summary line per bit
    print(f"  {dim}Активные слои:{rst}")
    for b in range(6):
        p  = planes[b]
        bc = _BIT_COLORS[b] if color else ''
        status = f"T={p['transient']} P={p['period']} ρ={p['mean_attr']:.2f}"
        on_off = 'ON ' if p['active'] else 'off'
        print(f"  {bc}  bit{b} {_SEGMENT_NAMES[b]:<2} [{on_off}]  {status}{rst}")
    print()


def print_bit_plane_summary(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Таблица: число активных битовых слоёв для каждого слова × правила."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''

    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Число активных битовых слоёв (из 6){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")

    for word in sorted(words):
        row_parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            act = attractor_activity(word, rule, width)
            cnt = sum(1 for d in act.values() if d['active'])
            col = _RULE_COLOR.get(rule, '') if color else ''
            row_parts.append(f"  {col}{cnt:>8d}{rst}")
        print('  ' + ''.join(row_parts))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Битовые плоскости CA Q6')
    parser.add_argument('--word',      default='ГОРА',  help='Русское слово')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color',  action='store_true')
    args = parser.parse_args()

    color = not args.no_color
    if args.stats:
        print_bit_plane_summary(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_bit_planes(args.word, rule, args.width, color)
    else:
        print_bit_planes(args.word, args.rule, args.width, color)


if __name__ == '__main__':
    _main()
