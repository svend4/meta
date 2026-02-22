"""solan_correlation.py — Пространственная автокорреляция Q6-аттракторов.

Для каждого CA-аттрактора (слово + правило) вычисляется нормированная
пространственная автокорреляционная функция:

    r(d) = C(d) / C(0)  ∈ [−1, 1]

    C(d) = ⟨(s[i] − μ)(s[(i+d) mod N] − μ)⟩   (среднее по i и шагам аттрактора)

Это «пространственный отпечаток»: показывает характерные масштабы корреляции
в аттракторе.  Связан с DFT-спектром через теорему Винера–Хинчина:
FT{r(d)} = мощностной спектр аттрактора.

Частные случаи:
  XOR  → аттрактор нулевой; нет вариации → r(d) ≡ «неопределено» (NaN / 0)
  AND/OR → знакочередующийся аттрактор → r(d) = (−1)^d
  XOR3   → богатая структура, зависит от слова

Функции:
    row_autocorr(row)                         → list[float]
    attractor_autocorr(word, rule, width)     → list[float]
    all_autocorrs(word, width)                → dict[str, list[float]]
    cross_corr(word1, word2, rule, width)     → list[float]
    correlation_length(word, rule, width)     → float
    build_correlation_data(words, width)      → dict
    correlation_dict(word, width)             → dict
    print_correlation(word, rule, width, color)

Запуск:
    python3 -m projects.hexglyph.solan_correlation --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_correlation --word ЛУНА --all-rules --no-color
    python3 -m projects.hexglyph.solan_correlation --stats
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
_DEFAULT_WIDTH = 16
_DEFAULT_WORDS = list(LEXICON)


# ── Core correlation primitive ────────────────────────────────────────────────

def row_autocorr(row: list[int]) -> list[float]:
    """Нормированная пространственная автокорреляция одной строки.

    Возвращает r(d) для d = 0 … N//2 (длина = N//2 + 1).

    Особые случаи:
      - Нулевая дисперсия (все значения одинаковы): r(d) = 1.0 для всех d
        (сигнал тривиально «идеально скоррелирован» с собой при любом сдвиге).
    """
    N  = len(row)
    mu = sum(row) / N
    var = sum((v - mu) ** 2 for v in row) / N
    n_lags = N // 2 + 1
    if var == 0.0:
        return [1.0] * n_lags          # постоянный сигнал
    result: list[float] = []
    for d in range(n_lags):
        cov = sum((row[i] - mu) * (row[(i + d) % N] - mu) for i in range(N)) / N
        result.append(cov / var)
    return result


def _attractor_rows(word: str, rule: str, width: int) -> list[list[int]]:
    """Строки аттрактора (после транзиента)."""
    cells = pad_to(encode_word(word.upper()), width)
    transient, period = find_orbit(cells, rule)
    rows: list[list[int]] = []
    c = cells[:]
    for t in range(transient + period):
        if t >= transient:
            rows.append(c[:])
        c = step(c, rule)
    return rows


# ── Per-word correlations ─────────────────────────────────────────────────────

def attractor_autocorr(
    word:  str,
    rule:  str = 'xor3',
    width: int = _DEFAULT_WIDTH,
) -> list[float]:
    """Усреднённая автокорреляция аттрактора по всем шагам цикла.

    Возвращает r(d) для d = 0 … width//2.
    Для нулевого аттрактора (XOR) возвращает [1.0, 0.0, 0.0, ...].
    """
    rows   = _attractor_rows(word, rule, width)
    n_lags = width // 2 + 1
    if not rows:
        return [1.0] + [0.0] * (n_lags - 1)

    sums = [0.0] * n_lags
    cnt  = 0
    for row in rows:
        c = row_autocorr(row)
        # Only count rows that are non-constant (var > 0)
        # (constant rows give r=1 trivially but bias the mean)
        mu  = sum(row) / len(row)
        var = sum((v - mu) ** 2 for v in row) / len(row)
        if var > 0:
            for i, v in enumerate(c):
                sums[i] += v
            cnt += 1

    if cnt == 0:
        # All attractor rows are constant → zero attractor
        return [1.0] + [0.0] * (n_lags - 1)

    return [s / cnt for s in sums]


def all_autocorrs(word: str, width: int = _DEFAULT_WIDTH) -> dict[str, list[float]]:
    """Автокорреляция аттракторов по всем 4 правилам."""
    return {r: attractor_autocorr(word, r, width) for r in _ALL_RULES}


# ── Cross-correlation between two words ───────────────────────────────────────

def cross_corr(
    word1: str,
    word2: str,
    rule:  str = 'xor3',
    width: int = _DEFAULT_WIDTH,
) -> list[float]:
    """Нормированная взаимная корреляция аттракторов двух слов при каждом лаге d.

    Вычисляется как средняя корреляция между «первым шагом» аттрактора word1
    и «смещённым первым шагом» аттрактора word2.  Обобщает автокорреляцию.

    Возвращает r12(d) для d = 0 … width//2.
    """
    r1 = _attractor_rows(word1, rule, width)
    r2 = _attractor_rows(word2, rule, width)
    n_lags = width // 2 + 1
    if not r1 or not r2:
        return [0.0] * n_lags

    # Use the first attractor row of each word
    row1 = r1[0]
    row2 = r2[0]
    N    = width
    mu1  = sum(row1) / N
    mu2  = sum(row2) / N
    std1 = math.sqrt(sum((v - mu1) ** 2 for v in row1) / N)
    std2 = math.sqrt(sum((v - mu2) ** 2 for v in row2) / N)
    if std1 == 0 or std2 == 0:
        return [0.0] * n_lags
    result: list[float] = []
    for d in range(n_lags):
        cov = sum((row1[i] - mu1) * (row2[(i + d) % N] - mu2) for i in range(N)) / N
        result.append(cov / (std1 * std2))
    return result


# ── Correlation length ────────────────────────────────────────────────────────

def correlation_length(
    word:  str,
    rule:  str = 'xor3',
    width: int = _DEFAULT_WIDTH,
) -> float:
    """Характерная длина корреляции: наименьший лаг d, где |r(d)| ≤ 1/e.

    Если |r(d)| > 1/e для всех d > 0 (долгодействующая корреляция),
    возвращает width//2 (максимально измеримый лаг).
    """
    corr  = attractor_autocorr(word, rule, width)
    e_inv = math.exp(-1)   # ≈ 0.368
    for d in range(1, len(corr)):
        if abs(corr[d]) <= e_inv:
            return float(d)
    return float(width // 2)


# ── Full dataset ──────────────────────────────────────────────────────────────

def build_correlation_data(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
) -> dict:
    """Автокорреляции аттракторов для всего лексикона.

    Возвращает dict:
        words       : list[str]
        width       : int
        n_lags      : int                        — width//2 + 1
        per_rule    : {rule: {word: list[float]}}
        corr_lengths: {rule: {word: float}}
        max_corr_len: {rule: (word, length)}     — слово с наибольшей дл. корр.
        min_corr_len: {rule: (word, length)}
    """
    words = words if words is not None else _DEFAULT_WORDS
    n_lags = width // 2 + 1
    per_rule: dict[str, dict[str, list[float]]] = {r: {} for r in _ALL_RULES}
    corr_lengths: dict[str, dict[str, float]]   = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            ac = attractor_autocorr(word, rule, width)
            per_rule[rule][word]    = ac
            corr_lengths[rule][word] = correlation_length(word, rule, width)
    max_corr: dict[str, tuple[str, float]] = {}
    min_corr: dict[str, tuple[str, float]] = {}
    for rule in _ALL_RULES:
        items = list(corr_lengths[rule].items())
        max_corr[rule] = max(items, key=lambda x: x[1])
        min_corr[rule] = min(items, key=lambda x: x[1])
    return {
        'words':        words,
        'width':        width,
        'n_lags':       n_lags,
        'per_rule':     per_rule,
        'corr_lengths': corr_lengths,
        'max_corr_len': max_corr,
        'min_corr_len': min_corr,
    }


# ── JSON export ───────────────────────────────────────────────────────────────

def correlation_dict(word: str, width: int = _DEFAULT_WIDTH) -> dict:
    """JSON-совместимый словарь автокорреляций по всем правилам."""
    result: dict[str, object] = {
        'word':  word.upper(),
        'width': width,
        'lags':  list(range(width // 2 + 1)),
        'rules': {},
    }
    for rule in _ALL_RULES:
        ac = attractor_autocorr(word, rule, width)
        cl = correlation_length(word, rule, width)
        result['rules'][rule] = {  # type: ignore[index]
            'autocorr':    [round(v, 4) for v in ac],
            'corr_length': round(cl, 2),
        }
    return result


# ── ASCII display ─────────────────────────────────────────────────────────────

_BAR_W = 30


def _signed_bar(val: float, width: int = _BAR_W) -> str:
    """Горизонтальная полоска со знаком: −1 … 0 … +1."""
    half  = width // 2
    frac  = max(-1.0, min(1.0, val))
    pos   = int(round(frac * (half - 1)))   # clamp to half-1 to stay in bounds
    bar   = [' '] * width
    mid   = half
    bar[mid] = '│'
    if pos > 0:
        for i in range(mid + 1, min(width, mid + pos + 1)):
            bar[i] = '█'
    elif pos < 0:
        for i in range(max(0, mid + pos), mid):
            bar[i] = '█'
    return ''.join(bar)


def print_correlation(
    word:  str,
    rule:  str = 'xor3',
    width: int = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Распечатать профиль автокорреляции: r(d) vs d."""
    ac   = attractor_autocorr(word, rule, width)
    cl   = correlation_length(word, rule, width)
    rule_col  = _RULE_COLOR.get(rule, '') if color else ''
    rule_name = _RULE_NAMES.get(rule, rule.upper())
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    print(f"{bold}  ◈ Автокорреляция Q6  {word.upper()}  |  "
          f"{rule_col}{rule_name}{rst}  "
          f"(дл. корр. = {cl:.1f})")
    print(f"  {'─' * 52}")
    print(f"  {'лаг d':>6}  {'r(d)':>6}  {'−1':>4}  {'│':>16}  {'+1'}")
    print(f"  {'─' * 52}")
    e_inv = math.exp(-1)
    for d, rv in enumerate(ac):
        # Colour by significance
        if abs(rv) > e_inv:
            val_c = rule_col if color else ''
        elif abs(rv) > 0.1:
            val_c = '\033[38;5;226m' if color else ''
        else:
            val_c = dim
        bar = _signed_bar(rv)
        marker = ' ←1/e' if d == int(cl) else ''
        print(f"  {d:>6}  {val_c}{rv:>+6.3f}{rst}  {bar}{marker}")
    print()


def print_correlation_stats(
    words: list[str] | None = None,
    width: int = _DEFAULT_WIDTH,
    color: bool = True,
) -> None:
    """Сводная таблица длин корреляции для всего лексикона × 4 правила."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>8s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Длина пространственной корреляции аттрактора{rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")
    for word in sorted(words):
        row_parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            cl  = correlation_length(word, rule, width)
            col = _RULE_COLOR.get(rule, '') if color else ''
            row_parts.append(f"  {col}{cl:>8.1f}{rst}")
        print('  ' + ''.join(row_parts))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Автокорреляция Q6 CA аттракторов')
    parser.add_argument('--word',      default='ГОРА',  help='Русское слово')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color',  action='store_true')
    args = parser.parse_args()
    color = not args.no_color
    if args.stats:
        print_correlation_stats(color=color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_correlation(args.word, rule, args.width, color)
    else:
        print_correlation(args.word, args.rule, args.width, color)


if __name__ == '__main__':
    _main()
