"""solan_phoneme.py — Фонемный анализ Q6: чувствительность орбитального класса
к однобуквенным заменам.

Для каждой позиции слова вычисляется, сколько из 15 возможных замен фонемы
изменяют орбитальный класс (из 13). Карта замен визуализирует «мутационный
ландшафт» слова в фонемном пространстве.

Функции:
    phoneme_table()           → dict[letter → {q6, binary, hw, segs, char}]
    substitution_matrix(word) → list[PositionDict]   # per-position substitutions
    sensitivity_profile(word) → list[float]          # 0.0–1.0 per position
    critical_positions(word, threshold) → list[int]  # high-sensitivity positions
    neutral_positions(word)   → list[int]            # zero-sensitivity positions
    pair_stats(words)         → dict[(orig,sub) → {rate, count}]  # global stats
    build_phoneme_data(words) → comprehensive analysis dict
    print_phoneme_table(color) — ASCII таблица 16 фонем
    print_substitution(word, color) — ASCII карта замен
    phoneme_dict(word)        → JSON-совместимый словарь

Запуск:
    python3 -m projects.hexglyph.solan_phoneme --table
    python3 -m projects.hexglyph.solan_phoneme --word ГОРА
    python3 -m projects.hexglyph.solan_phoneme --word ТУМАН --no-color
    python3 -m projects.hexglyph.solan_phoneme --pairs
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_phonetic import RUSSIAN_PHONETIC, phonetic_h
from projects.hexglyph.solan_transient import full_key, transient_classes
from projects.hexglyph.solan_ca import _RST, _BOLD, _DIM
from projects.hexglyph.solan_lexicon import LEXICON

_DEFAULT_WIDTH = 16
_DEFAULT_WORDS = list(LEXICON)

# 16-letter Solan alphabet (sorted by Q6 value)
SOLAN_ALPHABET: list[str] = sorted(
    RUSSIAN_PHONETIC.keys(), key=lambda c: RUSSIAN_PHONETIC[c]['h']
)

# Class palette (same as in solan_predict / viewer.html)
_CLASS_COLORS = [
    '\033[38;5;81m',   # 0  cyan
    '\033[38;5;120m',  # 1  green
    '\033[38;5;220m',  # 2  yellow
    '\033[38;5;213m',  # 3  pink
    '\033[38;5;208m',  # 4  orange
    '\033[38;5;155m',  # 5  lime
    '\033[38;5;196m',  # 6  red
    '\033[38;5;117m',  # 7  light blue
    '\033[38;5;229m',  # 8  pale yellow
    '\033[38;5;183m',  # 9  lavender
    '\033[38;5;86m',   # 10 teal
    '\033[38;5;218m',  # 11 salmon
    '\033[38;5;180m',  # 12 tan
]

# Canonical order of 13 classes (same as transient_classes() returns)
_CLASS_KEYS: list[tuple] | None = None

def _get_class_id(key: tuple) -> int:
    """Return 0-based class index for the given full_key tuple."""
    global _CLASS_KEYS
    if _CLASS_KEYS is None:
        _CLASS_KEYS = [c['key'] for c in transient_classes()]
    try:
        return _CLASS_KEYS.index(key)
    except ValueError:
        return -1  # new class


# ── Phoneme table ─────────────────────────────────────────────────────────────

def phoneme_table() -> dict[str, dict]:
    """16-фонемный алфавит Solan с Q6-свойствами.

    Каждый элемент:
        q6      : int   — Q6-значение (h-value, 0–63)
        binary  : str   — 6-битное представление '000000'–'111111'
        hw      : int   — вес Хэмминга (число единичных бит, 0–6)
        segs    : str   — имена активных сегментов 'T+B+L+…'
        char    : str   — Solan-символ (латинская буква)
    """
    result: dict[str, dict] = {}
    for letter, data in RUSSIAN_PHONETIC.items():
        h = data['h']
        result[letter] = {
            'q6':    h,
            'binary': format(h, '06b'),
            'hw':    bin(h).count('1'),
            'segs':  data.get('segs', ''),
            'char':  data.get('char') or '?',
        }
    return result


# ── Substitution analysis ─────────────────────────────────────────────────────

def substitution_matrix(word: str, width: int = _DEFAULT_WIDTH) -> list[dict]:
    """Матрица замен фонем слова.

    Для каждой позиции слова перебирает все 15 оставшихся фонем и записывает,
    изменяется ли орбитальный класс при замене.

    Возвращает список dict'ов (по одному на позицию):
        pos        : int
        letter     : str    — оригинальная буква
        q6         : int    — Q6 оригинала
        orig_key   : tuple  — ключ (5-tuple) оригинального слова
        orig_class : int    — 0-based class id оригинала
        subs       : dict[sub_letter → {new_key, new_class, changed}]
        n_changed  : int    — число замен, изменивших класс
        sensitivity: float  — n_changed / 15
    """
    letters = [c for c in word.upper() if phonetic_h(c) is not None]
    if not letters:
        return []
    orig_key = full_key(''.join(letters))
    orig_class = _get_class_id(orig_key)
    ptable = phoneme_table()
    result: list[dict] = []

    for pos, orig_letter in enumerate(letters):
        subs: dict[str, dict] = {}
        for sub_letter in SOLAN_ALPHABET:
            if sub_letter == orig_letter:
                continue
            new_letters = letters[:pos] + [sub_letter] + letters[pos + 1:]
            new_key = full_key(''.join(new_letters), width)
            new_class = _get_class_id(new_key)
            subs[sub_letter] = {
                'new_key':   new_key,
                'new_class': new_class,
                'changed':   new_key != orig_key,
            }
        n_changed = sum(1 for v in subs.values() if v['changed'])
        result.append({
            'pos':        pos,
            'letter':     orig_letter,
            'q6':         ptable[orig_letter]['q6'],
            'orig_key':   orig_key,
            'orig_class': orig_class,
            'subs':       subs,
            'n_changed':  n_changed,
            'sensitivity': n_changed / max(1, len(subs)),
        })
    return result


def sensitivity_profile(word: str, width: int = _DEFAULT_WIDTH) -> list[float]:
    """Профиль чувствительности: доля замен, изменяющих класс, для каждой позиции."""
    return [p['sensitivity'] for p in substitution_matrix(word, width)]


def critical_positions(word: str, width: int = _DEFAULT_WIDTH,
                        threshold: float = 0.5) -> list[int]:
    """Позиции, где > threshold замен изменяют орбитальный класс."""
    return [p['pos'] for p in substitution_matrix(word, width)
            if p['sensitivity'] > threshold]


def neutral_positions(word: str, width: int = _DEFAULT_WIDTH) -> list[int]:
    """Позиции, где ни одна замена не изменяет класс."""
    return [p['pos'] for p in substitution_matrix(word, width)
            if p['n_changed'] == 0]


# ── Pair statistics ───────────────────────────────────────────────────────────

def pair_stats(words: list[str] | None = None,
               width: int = _DEFAULT_WIDTH) -> dict[tuple[str, str], dict]:
    """Глобальная статистика пар (orig_letter → sub_letter) по всему лексикону.

    Возвращает dict[(orig, sub)] → {count: int, changed: int, rate: float}.
    """
    words = words if words is not None else _DEFAULT_WORDS
    stats: dict[tuple[str, str], dict] = {}

    for word in words:
        for pos_dict in substitution_matrix(word, width):
            orig = pos_dict['letter']
            for sub, info in pos_dict['subs'].items():
                pair = (orig, sub)
                if pair not in stats:
                    stats[pair] = {'count': 0, 'changed': 0}
                stats[pair]['count'] += 1
                if info['changed']:
                    stats[pair]['changed'] += 1

    for v in stats.values():
        v['rate'] = v['changed'] / v['count'] if v['count'] else 0.0
    return stats


# ── Full dataset ──────────────────────────────────────────────────────────────

def build_phoneme_data(words: list[str] | None = None,
                       width: int = _DEFAULT_WIDTH) -> dict:
    """Сводный фонемный анализ по всему лексикону.

    Возвращает dict:
        phoneme_table  : dict (16 phonemes)
        words          : list[str]
        profiles       : dict[word → list[float]]     — sensitivity profiles
        critical       : dict[word → list[int]]        — critical positions
        neutral        : dict[word → list[int]]        — neutral positions
        pairs          : dict[(str,str) → stats_dict] — pair statistics
        most_stable    : str  — word with lowest mean sensitivity
        most_sensitive : str  — word with highest mean sensitivity
    """
    words = words if words is not None else _DEFAULT_WORDS
    pt = phoneme_table()
    profiles: dict[str, list[float]] = {}
    crits: dict[str, list[int]] = {}
    neuts: dict[str, list[int]] = {}

    for w in words:
        profiles[w] = sensitivity_profile(w, width)
        crits[w]    = critical_positions(w, width)
        neuts[w]    = neutral_positions(w, width)

    def mean_sens(w: str) -> float:
        p = profiles[w]
        return sum(p) / len(p) if p else 0.0

    most_stable    = min(words, key=mean_sens)
    most_sensitive = max(words, key=mean_sens)
    ps = pair_stats(words, width)

    return {
        'phoneme_table':  pt,
        'words':          words,
        'profiles':       profiles,
        'critical':       crits,
        'neutral':        neuts,
        'pairs':          ps,
        'most_stable':    most_stable,
        'most_sensitive': most_sensitive,
    }


# ── JSON export ───────────────────────────────────────────────────────────────

def phoneme_dict(word: str, width: int = _DEFAULT_WIDTH) -> dict:
    """JSON-совместимый словарь фонемного анализа слова."""
    matrix = substitution_matrix(word, width)
    return {
        'word':    word.upper(),
        'width':   width,
        'profile': [p['sensitivity'] for p in matrix],
        'positions': [
            {
                'pos':         p['pos'],
                'letter':      p['letter'],
                'q6':          p['q6'],
                'orig_class':  p['orig_class'],
                'n_changed':   p['n_changed'],
                'sensitivity': round(p['sensitivity'], 4),
                'subs': {
                    sub: {
                        'new_class': info['new_class'],
                        'changed':   info['changed'],
                    }
                    for sub, info in p['subs'].items()
                },
            }
            for p in matrix
        ],
    }


# ── ASCII display ─────────────────────────────────────────────────────────────

def print_phoneme_table(color: bool = True) -> None:
    """Напечатать таблицу 16 фонем с Q6-свойствами."""
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    pt = phoneme_table()
    seg_colors = {
        'T': '\033[38;5;196m', 'B': '\033[38;5;208m',
        'L': '\033[38;5;220m', 'R': '\033[38;5;82m',
        'D1': '\033[38;5;39m', 'D2': '\033[38;5;213m',
    }

    print(f"\n{bold}  ◈ Таблица фонем Q6 (16 букв алфавита Solan){rst}")
    print(f"  {'─'*62}")
    print(f"  {'Буква':6s} {'Q6':4s} {'Двоичн':8s} {'HW':3s} {'Символ':8s} {'Сегменты'}")
    print(f"  {'─'*62}")

    for letter in SOLAN_ALPHABET:
        d = pt[letter]
        # Colour the binary string by bit position
        if color:
            bits = d['binary']
            segs_active = set(d['segs'].replace(' ', '').split('+'))
            seg_order = ['T', 'B', 'L', 'R', 'D1', 'D2']  # MSB to LSB
            colored_bits = ''
            for i, (seg, bit) in enumerate(zip(seg_order, bits)):
                c = seg_colors.get(seg, '')
                colored_bits += (c + bit + rst) if bit == '1' else (dim + bit + rst)
            bin_str = colored_bits
        else:
            bin_str = d['binary']

        hw_bar = '█' * d['hw'] + '░' * (6 - d['hw'])
        if color:
            hw_col = _CLASS_COLORS[min(d['hw'] * 2, 12)]
            hw_bar = hw_col + hw_bar + rst
        segs = d['segs']
        char = d['char']
        print(f"  {bold}{letter:6s}{rst} {d['q6']:4d} {bin_str}  {hw_bar}  "
              f"{dim}{char:8s}{rst} {dim}{segs}{rst}")

    print()


def print_substitution(word: str, width: int = _DEFAULT_WIDTH,
                        color: bool = True) -> None:
    """Напечатать карту замен слова: строки = позиции, столбцы = 15 фонем-замен."""
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    matrix = substitution_matrix(word, width)
    if not matrix:
        print(f"  (нет фонетических символов в '{word}')")
        return

    orig_key   = matrix[0]['orig_key'] if matrix else None
    orig_class = matrix[0]['orig_class'] if matrix else -1

    print(f"\n{bold}  ◈ Фонемные замены  {word.upper()}  "
          f"класс={orig_class+1}/13  ключ={orig_key}{rst}")
    print(f"  {'─'*72}")

    # Header: phoneme row
    header = '        '
    for sub_letter in SOLAN_ALPHABET:
        header += f' {sub_letter}'
    print('  ' + header)
    print(f"  {'─'*72}")

    for pd in matrix:
        letter = pd['letter']
        sens = pd['sensitivity']
        n_ch = pd['n_changed']

        # Sensitivity bar
        bar_len = 12
        filled = round(sens * bar_len)
        if color:
            bar_col = ('\033[38;5;196m' if sens > 0.67 else
                       '\033[38;5;208m' if sens > 0.33 else
                       '\033[38;5;82m')
            bar = bar_col + '█' * filled + dim + '░' * (bar_len - filled) + rst
        else:
            bar = '█' * filled + '░' * (bar_len - filled)

        line = f'  {bold}{letter}{rst}[{pd["pos"]}] {bar} {n_ch:2d}/15  '

        for sub_letter in SOLAN_ALPHABET:
            if sub_letter == letter:
                line += f' {dim}·{rst}'
                continue
            info = pd['subs'][sub_letter]
            if info['changed']:
                cid = info['new_class']
                col = _CLASS_COLORS[cid % len(_CLASS_COLORS)] if color else ''
                line += f' {col}▪{rst}'
            else:
                line += f' {dim}={rst}'
        print(line)

    print()
    # Legend
    changed_total = sum(p['n_changed'] for p in matrix)
    total_subs    = sum(len(p['subs']) for p in matrix)
    overall_rate  = changed_total / total_subs if total_subs else 0
    print(f"  {dim}Легенда: {bold}▪{rst}{dim} = смена класса (цвет = новый класс)  "
          f"= = класс сохранён  · = та же буква{rst}")
    print(f"  {dim}Общая чувствительность: {changed_total}/{total_subs} "
          f"({overall_rate:.1%}){rst}")


def print_pair_stats(words: list[str] | None = None, color: bool = True,
                     top: int = 10) -> None:
    """Напечатать топ самых дестабилизирующих и нейтральных замен."""
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    stats = pair_stats(words)
    pairs = [(pair, d) for pair, d in stats.items() if d['count'] >= 3]
    pairs.sort(key=lambda x: -x[1]['rate'])

    print(f"\n{bold}  ◈ Самые дестабилизирующие замены (rate ≥ 3 вхождений){rst}")
    print(f"  {'─'*40}")
    for pair, d in pairs[:top]:
        bar = '█' * round(d['rate'] * 10) + '░' * (10 - round(d['rate'] * 10))
        if color:
            col = '\033[38;5;196m' if d['rate'] > 0.7 else '\033[38;5;208m'
            bar = col + bar + rst
        print(f"  {bold}{pair[0]}→{pair[1]}{rst}  {bar}  "
              f"{d['rate']:.3f}  ({d['count']} вхождений)")

    print(f"\n{bold}  ◈ Самые нейтральные замены (rate ≥ 3 вхождений){rst}")
    print(f"  {'─'*40}")
    for pair, d in sorted(pairs, key=lambda x: x[1]['rate'])[:top]:
        bar = '█' * round(d['rate'] * 10) + '░' * (10 - round(d['rate'] * 10))
        if color:
            bar = dim + bar + rst
        print(f"  {bold}{pair[0]}→{pair[1]}{rst}  {bar}  "
              f"{d['rate']:.3f}  ({d['count']} вхождений)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Фонемный анализ Q6')
    parser.add_argument('--word', default='ГОРА', help='Русское слово')
    parser.add_argument('--table', action='store_true', help='Таблица 16 фонем')
    parser.add_argument('--pairs', action='store_true', help='Пары замен')
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--json',      action='store_true', help='JSON output')
    args = parser.parse_args()

    color = not args.no_color
    if args.json:
        import json as _json
        print(_json.dumps(phoneme_dict(args.word), ensure_ascii=False, indent=2))
        return
    if args.table:
        print_phoneme_table(color)
    elif args.pairs:
        print_pair_stats(color=color)
    else:
        print_substitution(args.word, color=color)


if __name__ == '__main__':
    _main()
