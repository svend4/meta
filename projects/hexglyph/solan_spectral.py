"""solan_spectral.py — Спектральный анализ CA-орбит Q6 (ДПФ аттрактора).

Каждая строка CA-аттрактора — последовательность Q6-значений (0–63);
дискретное преобразование Фурье (ДПФ) выявляет доминирующие пространственные
частоты: насколько «периодична» структура аттрактора по клеткам.

Ключевые наблюдения:
  · Аттрактор period=1 (фиксированная точка) → только DC-компонента (k=0)
  · Аттрактор с N-кратной симметрией → пик на k=N (4-буквенные слова → k=4)
  · XOR-аттрактор = all-zeros → DC=0, все остальные = 0 (вырождение)
  · XOR3 с period>1 → богатый спектр (несколько гармоник)

Функции:
    row_spectrum(cells)          → list[float]  # мощностной спектр строки
    attractor_spectrum(word, rule, width) → SpectrumDict
    all_spectra(word, width)     → dict[rule → SpectrumDict]
    spectral_distance(s1, s2)   → float        # 1 - cosine similarity
    build_spectral_data(words, width) → сводный анализ лексикона
    spectral_fingerprint(word, width) → flat AC vector по всем 4 правилам
    print_spectrum(word, rule, width, color)   — ASCII гистограмма
    spectral_dict(word, width)  → JSON-совместимый словарь

Запуск:
    python3 -m projects.hexglyph.solan_spectral --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_spectral --word ТУМАН --all-rules
    python3 -m projects.hexglyph.solan_spectral --stats
    python3 -m projects.hexglyph.solan_spectral --no-color
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_traj import word_trajectory
from projects.hexglyph.solan_lexicon import LEXICON
from projects.hexglyph.solan_ca import _RST, _BOLD, _DIM, _RULE_NAMES, _RULE_COLOR

_ALL_RULES = ['xor', 'xor3', 'and', 'or']
_DEFAULT_WIDTH = 16
_DEFAULT_WORDS = list(LEXICON)


# ── Core DFT ─────────────────────────────────────────────────────────────────

def row_spectrum(cells: list[int]) -> list[float]:
    """Мощностной спектр строки CA: |DFT[k]|² / N², k=0..N//2.

    Normalised by N² so that DC = (mean_q6)² regardless of length.
    Returns list of length N//2 + 1.
    """
    N = len(cells)
    if N == 0:
        return []
    half = N // 2
    result: list[float] = []
    two_pi_over_N = 2.0 * math.pi / N
    for k in range(half + 1):
        re = sum(cells[n] * math.cos(two_pi_over_N * k * n) for n in range(N))
        im = sum(cells[n] * math.sin(two_pi_over_N * k * n) for n in range(N))
        result.append((re * re + im * im) / (N * N))
    return result


def _avg_spectra(spectra: list[list[float]]) -> list[float]:
    """Component-wise average of a list of equal-length spectra."""
    if not spectra:
        return []
    n_freqs = len(spectra[0])
    avg = [sum(s[k] for s in spectra) / len(spectra) for k in range(n_freqs)]
    return avg


def _ac_normalise(spectrum: list[float]) -> list[float]:
    """AC-normalized spectrum: exclude DC (k=0), normalise sum to 1.

    Returns same-length list with spectrum[0]=0 and sum(rest)=1 (or all-zeros).
    """
    ac_total = sum(spectrum[1:])
    if ac_total < 1e-12:
        return [0.0] * len(spectrum)
    result = [0.0]
    result += [v / ac_total for v in spectrum[1:]]
    return result


# ── Attractor spectrum ────────────────────────────────────────────────────────

def attractor_spectrum(word: str, rule: str = 'xor3',
                       width: int = _DEFAULT_WIDTH) -> dict:
    """Усреднённый мощностной спектр аттрактора слова.

    Возвращает dict:
        rule        : str
        word        : str
        transient   : int
        period      : int
        n_freqs     : int          — N//2 + 1
        wavelengths : list[float]  — N/k для k=0..N//2 (первый = inf)
        power       : list[float]  — усреднённый |DFT|²/N²
        ac_power    : list[float]  — AC-нормированный спектр (DC=0, Σ=1)
        dominant_k  : int          — k наибольшего AC пика (1..N//2)
        dominant_wl : float        — длина волны при dominant_k (N/k)
        dominant_amp: float        — AC амплитуда dominant_k
        dc          : float        — DC компонента (power[0])
    """
    traj = word_trajectory(word, rule, width)
    rows, tr, per = traj['rows'], traj['transient'], traj['period']
    attr_rows = rows[tr:]

    if not attr_rows:
        n_freqs = width // 2 + 1
        return {
            'rule': rule, 'word': word.upper(), 'transient': tr, 'period': per,
            'n_freqs': n_freqs, 'wavelengths': [],
            'power': [0.0] * n_freqs, 'ac_power': [0.0] * n_freqs,
            'dominant_k': 1, 'dominant_wl': float(width),
            'dominant_amp': 0.0, 'dc': 0.0,
        }

    avg = _avg_spectra([row_spectrum(row) for row in attr_rows])
    ac  = _ac_normalise(avg)
    n_freqs = len(avg)

    # Dominant AC frequency (skip k=0)
    dom_k   = max(range(1, n_freqs), key=lambda k: ac[k]) if n_freqs > 1 else 1
    dom_wl  = width / dom_k if dom_k > 0 else float('inf')
    dom_amp = ac[dom_k]

    # Wavelengths list (k=0 → inf → represented as 0)
    wls = [0.0] + [width / k for k in range(1, n_freqs)]

    return {
        'rule':        rule,
        'word':        word.upper(),
        'transient':   tr,
        'period':      per,
        'n_freqs':     n_freqs,
        'wavelengths': wls,
        'power':       avg,
        'ac_power':    ac,
        'dominant_k':  dom_k,
        'dominant_wl': dom_wl,
        'dominant_amp': dom_amp,
        'dc':          avg[0],
    }


def all_spectra(word: str, width: int = _DEFAULT_WIDTH) -> dict[str, dict]:
    """Спектры по всем 4 правилам для одного слова."""
    return {r: attractor_spectrum(word, r, width) for r in _ALL_RULES}


# ── Distance ──────────────────────────────────────────────────────────────────

def spectral_distance(s1: dict, s2: dict) -> float:
    """Косинусное расстояние между AC-спектрами двух слов (0 = одинаковые, 1 = ортогональные)."""
    a = s1['ac_power'][1:]
    b = s2['ac_power'][1:]
    if not a or not b:
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-14 or norm_b < 1e-14:
        return 0.0 if (norm_a < 1e-14 and norm_b < 1e-14) else 1.0
    return max(0.0, 1.0 - dot / (norm_a * norm_b))


# ── Full dataset ──────────────────────────────────────────────────────────────

def spectral_fingerprint(word: str, width: int = _DEFAULT_WIDTH) -> list[float]:
    """Flat AC-спектральный вектор по всем 4 правилам (для сравнения слов).

    Returns list of length 4 × (width//2) (DC excluded).
    """
    result: list[float] = []
    for rule in _ALL_RULES:
        sp = attractor_spectrum(word, rule, width)
        result.extend(sp['ac_power'][1:])  # exclude DC
    return result


def build_spectral_data(words: list[str] | None = None,
                        width: int = _DEFAULT_WIDTH) -> dict:
    """Сводный спектральный анализ лексикона.

    Возвращает dict:
        words       : list[str]
        width       : int
        per_rule    : dict[rule → dict[word → spectrum_dict]]
        dom_freq    : dict[rule → Counter(dominant_k → [words])]
        most_harmonic: dict[rule → (word, dominant_amp)] — richest AC spectrum
        most_dc     : dict[rule → (word, dc_frac)]      — most DC-dominated
    """
    words = words if words is not None else _DEFAULT_WORDS

    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            per_rule[rule][word] = attractor_spectrum(word, rule, width)

    dom_freq: dict[str, dict[int, list[str]]] = {r: {} for r in _ALL_RULES}
    most_harmonic: dict[str, tuple[str, float]] = {}
    most_dc: dict[str, tuple[str, float]] = {}

    for rule in _ALL_RULES:
        rule_data = per_rule[rule]
        for word, sp in rule_data.items():
            k = sp['dominant_k']
            dom_freq[rule].setdefault(k, []).append(word)

        # word with richest AC spectrum (highest dominant_amp)
        best_ac = max(rule_data.items(), key=lambda x: x[1]['dominant_amp'])
        most_harmonic[rule] = (best_ac[0], best_ac[1]['dominant_amp'])

        # word most dominated by DC
        def dc_frac(item: tuple[str, dict]) -> float:
            p = item[1]['power']
            total = sum(p)
            return p[0] / total if total > 1e-14 else 1.0

        best_dc = max(rule_data.items(), key=dc_frac)
        most_dc[rule] = (best_dc[0], dc_frac(best_dc))

    return {
        'words':         words,
        'width':         width,
        'per_rule':      per_rule,
        'dom_freq':      dom_freq,
        'most_harmonic': most_harmonic,
        'most_dc':       most_dc,
    }


# ── JSON export ───────────────────────────────────────────────────────────────

def spectral_dict(word: str, width: int = _DEFAULT_WIDTH) -> dict:
    """JSON-совместимый словарь спектров по всем 4 правилам."""
    result: dict = {'word': word.upper(), 'width': width, 'rules': {}}
    for rule in _ALL_RULES:
        sp = attractor_spectrum(word, rule, width)
        result['rules'][rule] = {
            'transient':    sp['transient'],
            'period':       sp['period'],
            'dominant_k':   sp['dominant_k'],
            'dominant_wl':  round(sp['dominant_wl'], 3),
            'dominant_amp': round(sp['dominant_amp'], 4),
            'dc':           round(sp['dc'], 4),
            'power':        [round(v, 4) for v in sp['power']],
            'ac_power':     [round(v, 4) for v in sp['ac_power']],
        }
    return result


# ── ASCII display ─────────────────────────────────────────────────────────────

_FREQ_COLORS = [
    '\033[38;5;196m',  # k=1 red
    '\033[38;5;208m',  # k=2 orange
    '\033[38;5;220m',  # k=3 yellow
    '\033[38;5;82m',   # k=4 green
    '\033[38;5;51m',   # k=5 cyan
    '\033[38;5;81m',   # k=6 blue
    '\033[38;5;213m',  # k=7 purple
    '\033[38;5;231m',  # k=8 white
]


def print_spectrum(word: str, rule: str = 'xor3', width: int = _DEFAULT_WIDTH,
                   color: bool = True) -> None:
    """ASCII гистограмма AC-спектра аттрактора."""
    sp = attractor_spectrum(word, rule, width)
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''
    rule_col = _RULE_COLOR.get(rule, '') if color else ''
    rule_name = _RULE_NAMES.get(rule, rule.upper())

    print(f"\n{bold}  ◈ Спектр Q6  {word.upper()}  |  {rule_col}{rule_name}{rst}"
          f"{bold}  |  T={sp['transient']}  P={sp['period']}"
          f"  dom_k={sp['dominant_k']}  λ={sp['dominant_wl']:.1f}{rst}")
    print(f"  {'─'*58}")

    ac = sp['ac_power']
    n_freqs = sp['n_freqs']
    bar_max = 40

    for k in range(1, n_freqs):
        amp = ac[k]
        bars = round(amp * bar_max)
        wl = width / k
        fc = (_FREQ_COLORS[(k - 1) % len(_FREQ_COLORS)] if color else '')
        bar = fc + '█' * bars + dim + '░' * (bar_max - bars) + rst
        marker = f' {bold}←{rst}' if k == sp['dominant_k'] else ''
        print(f"  k={k:2d} λ={wl:5.1f} │{bar}│ {amp:6.3f}{marker}")

    dc_total = sum(sp['power'])
    dc_frac = sp['power'][0] / dc_total if dc_total > 1e-12 else 1.0
    print(f"  {dim}DC фракция: {dc_frac:.1%}  |  AC энергия: {1-dc_frac:.1%}{rst}")


def print_all_spectra(word: str, width: int = _DEFAULT_WIDTH,
                      color: bool = True) -> None:
    """Спектры по всем 4 правилам."""
    for rule in _ALL_RULES:
        print_spectrum(word, rule, width, color)
        print()


def print_spectral_stats(words: list[str] | None = None, width: int = _DEFAULT_WIDTH,
                         color: bool = True) -> None:
    """Таблица: доминирующие частоты и волны для лексикона."""
    words = words if words is not None else _DEFAULT_WORDS
    rst  = _RST  if color else ''
    bold = _BOLD if color else ''
    dim  = _DIM  if color else ''

    data = build_spectral_data(words, width)

    print(f"\n{bold}  ◈ Доминирующая AC-частота k (XOR3 | AND | OR){rst}")
    print(f"  {'─'*60}")
    print(f"  {'Слово':10s} {'XOR3':>6s} {'AND':>6s} {'OR':>6s} {'XOR3 λ':>7s}")
    print(f"  {'─'*60}")

    for word in sorted(words):
        parts = []
        for rule in ['xor3', 'and', 'or']:
            sp = data['per_rule'][rule][word]
            fc = (_FREQ_COLORS[(sp['dominant_k']-1) % len(_FREQ_COLORS)]
                  if color else '')
            parts.append(f"{fc}k={sp['dominant_k']}{rst}")
        xor3_sp = data['per_rule']['xor3'][word]
        wl = xor3_sp['dominant_wl']
        print(f"  {word:10s} {parts[0]:>6s} {parts[1]:>6s} {parts[2]:>6s} "
              f" {dim}{wl:5.1f}λ{rst}")

    print(f"\n{bold}  Наиболее гармоничные слова (по правилам):{rst}")
    for rule in _ALL_RULES:
        word, amp = data['most_harmonic'][rule]
        rc = _RULE_COLOR.get(rule, '') if color else ''
        rn = _RULE_NAMES.get(rule, rule)
        print(f"  {rc}{rn:10s}{rst}  {bold}{word}{rst}  dom_amp={amp:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Спектральный анализ Q6')
    parser.add_argument('--word', default='ГОРА', help='Русское слово')
    parser.add_argument('--rule', default='xor3', choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats', action='store_true', help='Таблица лексикона')
    parser.add_argument('--width', type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color', action='store_true')
    args = parser.parse_args()

    color = not args.no_color
    if args.stats:
        print_spectral_stats(color=color)
    elif args.all_rules:
        print_all_spectra(args.word, args.width, color)
    else:
        print_spectrum(args.word, args.rule, args.width, color)


if __name__ == '__main__':
    _main()
