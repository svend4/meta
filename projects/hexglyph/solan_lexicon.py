"""solan_lexicon.py — Орбитальный словарь русских слов в Q6-CA.

Хранит встроенный лексикон из ~45 русских слов, кодируемых
фонетическим алфавитом Solan (16 букв: А Б В Г Д Ж З И Л М Н О Р Т У Ч).
Для каждого слова вычисляется орбитальная сигнатура под 4 правилами CA.

Возможности:
  • Найти орбитально-ближайших «соседей» для любого слова
  • Сгруппировать лексикон по орбитальным кластерам
  • Вывести таблицу с периодами для всех слов

Запуск:
    python3 -m projects.hexglyph.solan_lexicon
    python3 -m projects.hexglyph.solan_lexicon --word РАТОН --neighbors 5
    python3 -m projects.hexglyph.solan_lexicon --clusters --threshold 0.15
    python3 -m projects.hexglyph.solan_lexicon --table
    python3 -m projects.hexglyph.solan_lexicon --no-color
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import (
    encode_word, pad_to, word_signature, sig_distance, Signature,
)
from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_NAMES, _RULE_COLOR,
)

# ── Встроенный лексикон ──────────────────────────────────────────────────────
# Все слова используют только буквы А Б В Г Д Ж З И Л М Н О Р Т У Ч

LEXICON: list[str] = [
    # Природа и окружение
    'ГОРА',     # mountain
    'ВОДА',     # water
    'НОРА',     # burrow
    'ЛУНА',     # moon
    'ВОЛНА',    # wave
    'ТУМАН',    # fog
    'ТУНДРА',   # tundra
    'ДУГА',     # arc
    'ГРУДА',    # pile
    'ЗИМА',     # winter
    'ЛУГА',     # meadows (pl. gen.)
    # Действие и состояние
    'УДАР',     # strike
    'ВЗЛОМ',    # break-in
    'РАБОТА',   # work
    'НАДО',     # must
    'ДОБРО',    # good
    'ОБРАЗ',    # image
    # Предметы
    'ТРОН',     # throne
    'ГОРН',     # bugle
    'БОЛТ',     # bolt
    'ЛИТР',     # litre
    'НОТА',     # note
    'РОТА',     # company (mil.)
    'РУЛОН',    # roll
    'ЗОНТ',     # umbrella
    'ТОННА',    # tonne
    'ЖУРНАЛ',   # journal
    'ВИТРАЖ',   # stained glass
    'МОНТАЖ',   # montage
    'ДОМРА',    # domra (instrument)
    'НАБОР',    # set
    'НИТРО',    # nitro
    # Живое
    'ЗУБР',     # bison
    'НАРОД',    # people
    'ДРОН',     # drone
    # Время
    'МАРТ',     # March
    'УТРО',     # morning
    # Короткие
    'МАТ',      # mat / checkmate
    'РИТМ',     # rhythm
    'ЛИТР',     # litre (duplicate — будет дедуплицировано)
    'ЗИМА',     # winter (duplicate)
    # Прочие
    'ДУМА',     # duma / thought
    'НОРМА',    # norm
    'МОДА',     # fashion
    'ГОРОД',    # city
    'ЗАВОД',    # factory
    'ДРОВА',    # firewood
    'ЖАТВА',    # harvest
    'БИТВА',    # battle
    'ЖИТО',     # grain
    'НАРОД',    # people (duplicate)
    'РАТОН',    # project word
    'ЗУБР',     # bison (duplicate)
]

# Дедуплицировать, сохранив порядок
_seen: set[str] = set()
_dedup: list[str] = []
for _w in LEXICON:
    if _w not in _seen:
        _dedup.append(_w)
        _seen.add(_w)
LEXICON = _dedup


# ── Работа с лексиконом ──────────────────────────────────────────────────────

def all_signatures(
    words:     list[str] | None = None,
    width:     int = 16,
    max_steps: int = 2000,
) -> dict[str, Signature]:
    """Вычислить орбитальные сигнатуры для всех слов лексикона."""
    if words is None:
        words = LEXICON
    return {w: word_signature(w, width=width, max_steps=max_steps)
            for w in words}


def neighbors(
    word:     str,
    sigs:     dict[str, Signature] | None = None,
    n:        int = 5,
    width:    int = 16,
) -> list[tuple[str, float]]:
    """Найти n ближайших соседей по орбитальному расстоянию.

    Возвращает [(word, distance)] отсортированный по distance.
    """
    if sigs is None:
        sigs = all_signatures(width=width)
    target_sig = word_signature(word, width=width)
    dists = [
        (w, sig_distance(target_sig, s))
        for w, s in sigs.items()
        if w != word
    ]
    dists.sort(key=lambda x: (float('inf') if x[1] != x[1] else x[1]))
    return dists[:n]


def orbital_clusters(
    sigs:      dict[str, Signature] | None = None,
    threshold: float = 0.15,
    width:     int   = 16,
) -> list[list[str]]:
    """Сгруппировать слова по орбитальной близости (жадный алгоритм).

    Два слова попадают в один кластер, если их расстояние ≤ threshold.
    """
    if sigs is None:
        sigs = all_signatures(width=width)

    words = list(sigs.keys())
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    for seed in words:
        if seed in assigned:
            continue
        cluster = [seed]
        assigned.add(seed)
        seed_sig = sigs[seed]
        for w in words:
            if w in assigned:
                continue
            d = sig_distance(seed_sig, sigs[w])
            if d <= threshold:
                cluster.append(w)
                assigned.add(w)
        clusters.append(cluster)

    return sorted(clusters, key=len, reverse=True)


# ── Вывод ────────────────────────────────────────────────────────────────────

_ALL_RULES = ['xor', 'xor3', 'and', 'or']


def print_lexicon_table(
    width: int  = 16,
    color: bool = True,
) -> None:
    """Таблица: слово + период P для каждого правила."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    sigs = all_signatures(width=width)

    # Header
    rule_hdrs = '  '.join(
        (_RULE_COLOR.get(r, '') if color else '') + f"P({r.upper():4s})" + reset
        for r in _ALL_RULES
    )
    print(bold + f"  Орбитальный лексикон Q6  width={width}  "
          f"({len(sigs)} слов)" + reset)
    print()
    print(f"  {'Слово':14s}  {rule_hdrs}")
    print('  ' + '─' * (14 + 2 + len(_ALL_RULES) * 12))

    for w in LEXICON:
        if w not in sigs:
            continue
        sig = sigs[w]
        parts = []
        for r in _ALL_RULES:
            _, p = sig[r]
            col  = (_RULE_COLOR.get(r, '') if color else '')
            p_s  = str(p) if p is not None else '?'
            parts.append(f"{col}{p_s:>8s}{reset}")
        print(f"  {w:14s}  {'  '.join(parts)}")


def print_neighbors(
    word:  str,
    n:     int  = 8,
    width: int  = 16,
    color: bool = True,
) -> None:
    """Найти и вывести n ближайших слов по орбитальному расстоянию."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    from projects.hexglyph.solan_word import encode_word
    raw = encode_word(word)
    if not raw:
        print(f"  '{word}' не содержит фонетических символов алфавита Solan.")
        return

    sigs  = all_signatures(width=width)
    nbrs  = neighbors(word, sigs=sigs, n=n, width=width)

    # Собственная сигнатура
    own_sig = word_signature(word, width=width)
    own_parts = []
    for r in _ALL_RULES:
        _, p = own_sig[r]
        col = (_RULE_COLOR.get(r, '') if color else '')
        own_parts.append(f"{col}P={p if p is not None else '?'}{reset}")
    print(bold + f"  {word.upper()}" + reset
          + f"  [{' '.join(own_parts)}]")
    print(f"  {dim}← орбитальная сигнатура слова{reset}")
    print()

    print(bold + f"  {n} ближайших соседей (width={width}):" + reset)
    print()

    max_d = max((d for _, d in nbrs if d == d), default=1.0) or 1.0
    for w, d in nbrs:
        bar_len = 20
        filled  = int(d / max_d * bar_len) if d == d else bar_len
        bar = '█' * filled + '░' * (bar_len - filled)
        d_str = f"{d:.3f}" if d == d else "  NaN"
        sig_n = sigs.get(w, {})
        periods = ' '.join(
            str(sig_n[r][1]) if sig_n.get(r, (None, None))[1] is not None else '?'
            for r in _ALL_RULES
        )
        print(f"  {w:14s}  |{bar}| {d_str}  [{dim}{periods}{reset}]")


def print_clusters(
    threshold: float = 0.15,
    width:     int   = 16,
    color:     bool  = True,
) -> None:
    """Вывести орбитальные кластеры."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    sigs  = all_signatures(width=width)
    clsts = orbital_clusters(sigs=sigs, threshold=threshold, width=width)

    print(bold + f"  Орбитальные кластеры  threshold={threshold}  "
          f"width={width}" + reset)
    print(f"  {dim}{len(clsts)} кластеров из {len(sigs)} слов{reset}")
    print()

    for i, cluster in enumerate(clsts):
        col = '\033[38;5;{}m'.format(75 + (i * 37) % 180) if color else ''
        print(f"  {col}Кластер {i+1}{reset} ({len(cluster)} слов):")
        # По строкам по 8 слов
        for j in range(0, len(cluster), 8):
            print('    ' + '  '.join(f"{w:14s}" for w in cluster[j:j+8]))
        print()


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Орбитальный словарь Q6 — поиск похожих слов')
    parser.add_argument('--word', default='РАТОН',
                        help='целевое слово')
    parser.add_argument('--neighbors', type=int, default=8, metavar='N',
                        help='число ближайших соседей (default: 8)')
    parser.add_argument('--table', action='store_true',
                        help='вывести полную таблицу лексикона')
    parser.add_argument('--clusters', action='store_true',
                        help='вывести орбитальные кластеры')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='порог кластеризации (default: 0.15)')
    parser.add_argument('--width', type=int, default=16,
                        help='ширина CA (default: 16)')
    parser.add_argument('--no-color', action='store_true',
                        help='без ANSI-цветов')
    parser.add_argument('--json',     action='store_true',
                        help='JSON output')
    args = parser.parse_args()

    _color = not args.no_color

    if args.json:
        import json as _json
        nbrs = neighbors(args.word, n=args.neighbors, width=args.width)
        print(_json.dumps({'word': args.word.upper(), 'neighbors': [[w, d] for w, d in nbrs]}, ensure_ascii=False, indent=2))
        import sys; sys.exit(0)
    if args.table:
        print_lexicon_table(width=args.width, color=_color)
    elif args.clusters:
        print_clusters(threshold=args.threshold, width=args.width, color=_color)
    else:
        print_neighbors(args.word, n=args.neighbors,
                        width=args.width, color=_color)
