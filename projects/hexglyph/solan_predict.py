"""solan_predict.py — Q6 CA Prediction for Arbitrary Russian Words.

Given any Russian word (including words outside the 49-word lexicon),
classify it into one of the 13 transient classes and find its nearest
lexicon neighbors by orbital distance.

Pipeline:
    word "КОМПЬЮТЕР"
    → encode via solan_word.encode_word()
    → pad to width 16
    → word_signature() : orbit (T, P) for 4 rules
    → full_key()       : (xor_t, and_t, and_p, or_t, or_p)
    → match against 13 known transient classes
    → measure orbital distance to all 49 lexicon words
    → return top-N nearest neighbors

Functions
─────────
  predict(word, width, top_n)           → dict
  batch_predict(words, width, top_n)    → list[dict]
  predict_text(text, width, top_n)      → list[dict]
  print_prediction(word, width, color)  → None
  prediction_dict(result)               → dict
  predict_summary(word, width)          → dict

Запуск
──────
  python3 -m projects.hexglyph.solan_predict --word ГОРА --no-color
  python3 -m projects.hexglyph.solan_predict --word КОМПЬЮТЕР --no-color
  python3 -m projects.hexglyph.solan_predict --text "ГОРА ЛУНА ЖУРНАЛ" --no-color
  python3 -m projects.hexglyph.solan_predict --word ГОРА --json
"""
from __future__ import annotations

import argparse
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_ca import (
    _RST, _BOLD, _DIM, _RULE_NAMES, _RULE_COLOR, _ALL_RULES,
)
from projects.hexglyph.solan_word import word_signature, sig_distance
from projects.hexglyph.solan_transient import full_key, transient_classes
from projects.hexglyph.solan_lexicon import LEXICON

# ── Known class table (computed once at import) ─────────────────────────────

_CLASSES: list[dict] = transient_classes()
_CLASS_KEYS: list[tuple] = [tuple(c['key']) for c in _CLASSES]


# ── Core prediction ─────────────────────────────────────────────────────────

def predict(word: str, width: int = 16, top_n: int = 10) -> dict:
    """Full Q6 prediction for *word*.

    Returns
    ───────
    word          : str              uppercased input word
    width         : int
    signature     : dict[rule → [transient, period]]
    full_key      : tuple[int, ...]  (xor_t, and_t, and_p, or_t, or_p)
    class_id      : int | None       0-based index into the 13 known classes
    class_words   : list[str]        lexicon words sharing the same class
    neighbors     : list[(word, dist)]  top-N nearest, ascending
    is_new_class  : bool             True when key not in any known class
    """
    w   = word.upper()
    sig = word_signature(w, width=width)
    fk  = full_key(w, width=width)

    # Class lookup
    class_id: int | None = None
    class_words: list[str] = []
    if fk in _CLASS_KEYS:
        class_id   = _CLASS_KEYS.index(fk)
        class_words = list(_CLASSES[class_id]['words'])

    # Neighbor distances against the full lexicon
    lex_sigs = {lw: word_signature(lw, width=width) for lw in LEXICON}
    dists = [(lw, sig_distance(sig, lex_sigs[lw])) for lw in LEXICON]
    dists.sort(key=lambda x: x[1])
    neighbors = [(lw, round(d, 6)) for lw, d in dists[:top_n]]

    return {
        'word':        w,
        'width':       width,
        'signature':   {r: list(v) for r, v in sig.items()},
        'full_key':    fk,
        'class_id':    class_id,
        'class_words': class_words,
        'neighbors':   neighbors,
        'is_new_class': class_id is None,
    }


def batch_predict(words: list[str], width: int = 16, top_n: int = 10) -> list[dict]:
    """Predict for multiple words (re-uses shared lexicon signature cache)."""
    lex_sigs = {lw: word_signature(lw, width=width) for lw in LEXICON}

    results = []
    for word in words:
        w   = word.upper()
        sig = word_signature(w, width=width)
        fk  = full_key(w, width=width)

        class_id: int | None = None
        class_words: list[str] = []
        if fk in _CLASS_KEYS:
            class_id   = _CLASS_KEYS.index(fk)
            class_words = list(_CLASSES[class_id]['words'])

        dists = [(lw, sig_distance(sig, lex_sigs[lw])) for lw in LEXICON]
        dists.sort(key=lambda x: x[1])
        neighbors = [(lw, round(d, 6)) for lw, d in dists[:top_n]]

        results.append({
            'word':        w,
            'width':       width,
            'signature':   {r: list(v) for r, v in sig.items()},
            'full_key':    fk,
            'class_id':    class_id,
            'class_words': class_words,
            'neighbors':   neighbors,
            'is_new_class': class_id is None,
        })
    return results


def predict_text(text: str, width: int = 16, top_n: int = 10) -> list[dict]:
    """Tokenise *text* into Russian words, predict each unique word.

    Splits on non-Cyrillic characters; returns predictions for unique words
    in order of first appearance.
    """
    tokens = re.findall(r'[А-ЯЁа-яё]+', text)
    seen: list[str] = []
    for t in tokens:
        t_up = t.upper()
        if t_up not in seen:
            seen.append(t_up)
    return batch_predict(seen, width=width, top_n=top_n)


# ── JSON-friendly wrappers ──────────────────────────────────────────────────

def prediction_dict(result: dict) -> dict:
    """Return a JSON-serialisable copy of a predict() result.

    neighbors are returned as [{'word': w, 'dist': d}, ...] dicts.
    full_key is returned as a list (not tuple).
    """
    neighbors = [{'word': w, 'dist': d} for w, d in result['neighbors']]
    return {
        'word':        result['word'],
        'width':       result['width'],
        'signature':   result['signature'],
        'full_key':    list(result['full_key']),
        'class_id':    result['class_id'],
        'class_count': len(_CLASSES),
        'class_words': result['class_words'],
        'neighbors':   neighbors,
        'is_new_class': result['is_new_class'],
    }


def predict_summary(word: str, width: int = 16) -> dict:
    """JSON-serialisable prediction summary for *word*."""
    return prediction_dict(predict(word, width=width))


# ── Pretty-printing ─────────────────────────────────────────────────────────

def print_prediction(
    word:  str,
    width: int  = 16,
    color: bool = True,
) -> None:
    """Pretty-print the full Q6 prediction for *word*."""
    bold = _BOLD if color else ''
    rst  = _RST  if color else ''

    result = predict(word, width=width)
    w      = result['word']
    sig    = result['signature']
    fk     = result['full_key']

    print(bold + f"  ◈ Предсказание Q6  {w}  (width={width})" + rst)
    print()

    # Signature table
    print(f"  {'Правило':8s}  {'Транзиент':>12s}  {'Период':>8s}")
    print('  ' + '─' * 36)
    for r in _ALL_RULES:
        t, p  = sig[r]
        col   = _RULE_COLOR.get(r, '') if color else ''
        t_str = str(t) if t is not None else '—'
        p_str = str(p) if p is not None else '>2000'
        lbl   = _RULE_NAMES.get(r, r.upper())
        print(f"  {col}{lbl:8s}{rst}  {t_str:>12s}  {p_str:>8s}")
    print()

    # Class info
    cid     = result['class_id']
    cwords  = result['class_words']
    new_str = ('Нет' if not result['is_new_class']
               else bold + 'ДА (новый класс)' + rst)
    key_str = '(' + ','.join(str(x) for x in fk) + ')'

    if cid is not None:
        sample = ' '.join(cwords[:5])
        extra  = (f'  ({len(cwords)} слов: {sample} …)'
                  if len(cwords) > 5
                  else f'  ({len(cwords)} слов: {sample})')
        print(f"  Класс:  {cid + 1} / {len(_CLASSES)}  ключ={key_str}{extra}")
    else:
        print(f"  Класс:  — (новый)  ключ={key_str}")
    print(f"  Новый?  {new_str}")
    print()

    # Neighbors
    print(bold + "  Ближайшие соседи:" + rst)
    for i, (nw, d) in enumerate(result['neighbors'], 1):
        bar = '█' * int((1 - d) * 10)
        print(f"    {i:2d}. {nw:12s}  d={d:.4f}  {bar}")
    print()


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description='Q6 CA Prediction for Arbitrary Russian Words')
    parser.add_argument('--word',  default='ГОРА',
                        help='Russian word to predict')
    parser.add_argument('--text',  default=None,
                        help='Russian text to tokenise and predict')
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--json',     action='store_true', help='JSON output')
    args  = parser.parse_args()
    color = not args.no_color and sys.stdout.isatty()

    if args.json:
        import json as _json
        if args.text:
            results = predict_text(args.text, width=args.width)
            print(_json.dumps([prediction_dict(r) for r in results],
                               ensure_ascii=False, indent=2))
        else:
            print(_json.dumps(predict_summary(args.word, args.width),
                               ensure_ascii=False, indent=2))
        return

    if args.text:
        results = predict_text(args.text, width=args.width)
        for r in results:
            print_prediction(r['word'], width=args.width, color=color)
    else:
        print_prediction(args.word, width=args.width, color=color)


if __name__ == '__main__':
    _cli()
