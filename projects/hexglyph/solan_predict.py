"""solan_predict.py — Предсказание орбитального класса Q6 для произвольного слова.

Для любого русского слова (не обязательно из 49-словного лексикона)
вычисляет полную орбитальную сигнатуру, определяет принадлежность к одному
из 13 транзиентных классов и находит ближайших соседей в лексиконе.

Возможности:
  • predict()        — полное предсказание для одного слова
  • batch_predict()  — батч (общий кэш сигнатур лексикона)
  • predict_text()   — предсказание для каждого слова в тексте
  • print_prediction()  — цветной вывод в терминал

Запуск:
    python3 -m projects.hexglyph.solan_predict --word ГОРА
    python3 -m projects.hexglyph.solan_predict --word КОМПЬЮТЕР
    python3 -m projects.hexglyph.solan_predict --text "ГОРА ЛУНА ЖУРНАЛ"
    python3 -m projects.hexglyph.solan_predict --word ГОРА --json
    python3 -m projects.hexglyph.solan_predict --word ГОРА --no-color
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_ca      import _RST, _BOLD, _DIM, _RULE_COLOR, _RULE_NAMES, _ALL_RULES

RULES = tuple(_ALL_RULES)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _lex_sigs(width: int = 16) -> dict:
    """Кэшированные сигнатуры всего лексикона."""
    from projects.hexglyph.solan_lexicon import all_signatures
    return all_signatures(width=width)


def _get_transient_classes(width: int = 16) -> list[dict]:
    """13 транзиентных классов эквивалентности."""
    from projects.hexglyph.solan_transient import transient_classes
    return transient_classes(width=width)


# ── Public API ────────────────────────────────────────────────────────────────

def predict(
    word:  str,
    width: int = 16,
    *,
    _lex_cache: dict | None = None,
    _cls_cache: list | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Полное предсказание орбитального класса Q6.

    Параметры
    ---------
    word        : произвольное русское слово
    width       : ширина CA (default 16)
    _lex_cache  : предвычисленные сигнатуры лексикона (для батча)
    _cls_cache  : предвычисленные классы (для батча)
    top_n       : число ближайших соседей

    Возвращает dict:
      word, signature, full_key, class_id, class_words, is_new_class,
      neighbors [(word, dist)]
    """
    from projects.hexglyph.solan_word      import word_signature, sig_distance
    from projects.hexglyph.solan_transient import full_key as _full_key

    sig    = word_signature(word, width=width)
    fk     = _full_key(word, width=width)
    lsigs  = _lex_cache  if _lex_cache  is not None else _lex_sigs(width)
    clses  = _cls_cache  if _cls_cache  is not None else _get_transient_classes(width)

    # Find class membership
    class_id    = None
    class_words: list[str] = []
    for ci, cls in enumerate(clses):
        if fk == cls['key']:
            class_id    = ci           # 0-based
            class_words = cls['words']
            break

    is_new_class = class_id is None

    # Nearest neighbors in lexicon (excluding self)
    dists = sorted(
        ((w, sig_distance(sig, s)) for w, s in lsigs.items()),
        key=lambda x: (float('inf') if math.isnan(x[1]) else x[1], x[0]),
    )
    neighbors = [(w, d) for w, d in dists if not math.isnan(d)][:top_n]

    return {
        'word':         word,
        'signature':    sig,
        'full_key':     fk,
        'class_id':     class_id,
        'class_words':  class_words,
        'is_new_class': is_new_class,
        'neighbors':    neighbors,
    }


def batch_predict(
    words:  list[str],
    width:  int = 16,
    top_n:  int = 10,
) -> list[dict[str, Any]]:
    """Предсказание для списка слов с общим кэшем лексикона."""
    lex   = _lex_sigs(width)
    clses = _get_transient_classes(width)
    return [
        predict(w, width=width, _lex_cache=lex, _cls_cache=clses, top_n=top_n)
        for w in words
    ]


def predict_text(
    text:  str,
    width: int = 16,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Токенизировать русский текст и предсказать для каждого уникального слова."""
    tokens = [t.upper() for t in re.findall(r'[А-ЯЁа-яё]+', text)]
    seen: set[str] = set()
    unique = [t for t in tokens if t not in seen and not seen.add(t)]  # type: ignore[func-returns-value]
    return batch_predict(unique, width=width, top_n=top_n)


def prediction_dict(result: dict[str, Any]) -> dict[str, Any]:
    """JSON-сериализуемое представление результата predict()."""
    sig_out: dict[str, dict] = {}
    for rule, (t, p) in result['signature'].items():
        sig_out[rule] = {'transient': t, 'period': p}
    return {
        'word':         result['word'],
        'signature':    sig_out,
        'full_key':     list(result['full_key']),
        'class_id':     result['class_id'],
        'class_words':  result['class_words'],
        'is_new_class': result['is_new_class'],
        'neighbors':    [
            {'word': w, 'dist': round(d, 6)}
            for w, d in result['neighbors']
        ],
    }


# ── Terminal output ───────────────────────────────────────────────────────────

def print_prediction(
    word:  str,
    width: int  = 16,
    color: bool = True,
) -> None:
    """Вывести полное предсказание для слова."""
    bold  = _BOLD if color else ''
    reset = _RST  if color else ''
    dim   = _DIM  if color else ''

    result = predict(word, width=width)
    clses  = _get_transient_classes(width)

    print(bold + f"  ◈ Предсказание Q6  {word.upper()}  (width={width})" + reset)
    print()

    # Signature table
    print(f"  {'Правило':<10}  {'Транзиент':>10}  {'Период':>8}")
    print('  ' + '─' * 32)
    for rule in RULES:
        t, p = result['signature'].get(rule, (None, None))
        col  = (_RULE_COLOR.get(rule, '') if color else '')
        lbl  = _RULE_NAMES.get(rule, rule.upper())
        t_s  = str(t) if t is not None else '?'
        p_s  = str(p) if p is not None else '?'
        print(f"  {col}{lbl:<10}{reset}  {t_s:>10}  {p_s:>8}")
    print()

    # Class
    fk  = result['full_key']
    fk_s = f"({','.join(str(v) for v in fk)})"
    if result['is_new_class']:
        cls_s = (f"\033[38;5;220m" if color else '') + "Новый класс!" + reset
        print(f"  Класс:  {cls_s}  ключ={fk_s}")
    else:
        ci   = result['class_id']
        cw   = result['class_words']
        col  = '\033[38;5;75m' if color else ''
        nt   = len(clses)
        print(f"  Класс:  {col}{ci + 1} / {nt}{reset}  ключ={fk_s}  "
              f"({len(cw)} слов: {dim}{' '.join(cw[:6])}"
              f"{'...' if len(cw) > 6 else ''}{reset})")
    print(f"  Новый?  {'Да' if result['is_new_class'] else 'Нет'}")
    print()

    # Neighbors
    print(bold + f"  Ближайшие соседи (top-{len(result['neighbors'])}):" + reset)
    for rank, (nbr, dist) in enumerate(result['neighbors'], 1):
        col = ('\033[38;5;117m' if color else '')
        d_s = f"{dist:.6f}" if dist > 0 else '0 (одинаковая орбита)'
        print(f"  {rank:>3}. {col}{nbr:<14}{reset}  d={d_s}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Предсказание орбитального класса Q6 для произвольного слова')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--word',  metavar='WORD', help='одно слово')
    group.add_argument('--text',  metavar='TEXT', help='русский текст (несколько слов)')
    parser.add_argument('--json',     action='store_true')
    parser.add_argument('--width',    type=int, default=16)
    parser.add_argument('--top',      type=int, default=10, dest='top_n')
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    _color = not args.no_color

    if args.word:
        if args.json:
            r = predict(args.word.upper(), width=args.width, top_n=args.top_n)
            print(json.dumps(prediction_dict(r), ensure_ascii=False, indent=2))
        else:
            print_prediction(args.word.upper(), width=args.width, color=_color)
    else:
        results = predict_text(args.text, width=args.width, top_n=args.top_n)
        if args.json:
            print(json.dumps([prediction_dict(r) for r in results],
                             ensure_ascii=False, indent=2))
        else:
            for r in results:
                print_prediction(r['word'], width=args.width, color=_color)
