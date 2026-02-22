"""solan_lyapunov.py — Анализ расходимости траекторий CA Q6.

Для каждого слова и каждого правила: однобитовые возмущения начального состояния
(переворот одного бита в одной клетке из 16×6 = 96 вариантов) → Хэмминг-расстояние
между исходной и возмущённой траекториями в каждый момент времени.

Это дискретный аналог показателя Ляпунова: характеризует, насколько «хаотичен»
или «стабилен» данный CA-аттрактор.

Функции:
    q6_hamming(a, b)                     → int    — битовое расстояние между Q6-значениями
    state_distance(cells1, cells2)       → int    — суммарное расстояние между состояниями
    perturb(cells, cell_idx, bit)        → list   — сдвинуть один бит
    divergence_trajectory(word, ci, bit, rule, width) → list[int]
    lyapunov_profile(word, rule, width)  → dict
    lyapunov_summary(word, width)        → dict
    build_lyapunov_data(words, width)    → dict
    lyapunov_dict(word, width)           → dict
    print_lyapunov(word, rule, width, color)

Запуск:
    python3 -m projects.hexglyph.solan_lyapunov --word ГОРА --rule xor3
    python3 -m projects.hexglyph.solan_lyapunov --word ТУМАН --all-rules --no-color
    python3 -m projects.hexglyph.solan_lyapunov --stats
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from projects.hexglyph.solan_word import encode_word, pad_to
from projects.hexglyph.solan_ca import (
    step,
    _RST, _BOLD, _DIM,
    _RULE_NAMES, _RULE_COLOR,
)
from projects.hexglyph.solan_lexicon import LEXICON

_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_N_BITS        = 6          # bits per Q6 cell
_DEFAULT_WIDTH = 16
_DEFAULT_STEPS = 32         # steps to track per perturbation
_DEFAULT_WORDS = list(LEXICON)


# ── Distance primitives ───────────────────────────────────────────────────────

def q6_hamming(a: int, b: int) -> int:
    """Количество отличающихся битов между двумя Q6-значениями (0..63)."""
    return bin(a ^ b).count('1')


def state_distance(cells1: list[int], cells2: list[int]) -> int:
    """Суммарное Q6-Хэмминг-расстояние между двумя CA-состояниями."""
    return sum(q6_hamming(a, b) for a, b in zip(cells1, cells2))


def perturb(cells: list[int], cell_idx: int, bit: int) -> list[int]:
    """Вернуть копию состояния с перевёрнутым битом `bit` в клетке `cell_idx`."""
    new = cells[:]
    new[cell_idx] ^= (1 << bit)
    return new


# ── Single perturbation trajectory ───────────────────────────────────────────

def divergence_trajectory(
    word:      str,
    cell_idx:  int,
    bit:       int,
    rule:      str = 'xor3',
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> list[int]:
    """Отслеживать Хэмминг-расстояние между оригиналом и возмущением.

    Возвращает список d(t) для t = 0 … max_steps (включительно).
    d(0) = 1 (одно перевёрнутое расстояние ровно 1 бит).
    """
    orig = pad_to(encode_word(word.upper()), width)
    pert = perturb(orig, cell_idx, bit)
    distances: list[int] = []
    cur_o, cur_p = orig[:], pert[:]
    for _ in range(max_steps + 1):
        distances.append(state_distance(cur_o, cur_p))
        cur_o = step(cur_o, rule)
        cur_p = step(cur_p, rule)
    return distances


# ── Profile: average over all perturbations ───────────────────────────────────

def lyapunov_profile(
    word:      str,
    rule:      str = 'xor3',
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> dict:
    """Усреднённый профиль расходимости по всем width×6 однобитовым возмущениям.

    Возвращает dict:
        word         : str
        rule         : str
        width        : int
        n_perturb    : int          — число возмущений (width × 6)
        mean_dist    : list[float]  — средн. d(t) в каждый шаг
        max_dist     : list[float]  — макс. d(t) по всем возмущениям
        min_dist     : list[float]  — мин. d(t)
        peak_mean    : float        — пик средн. расстояния
        peak_step    : int          — шаг с максимальным средн. расстоянием
        final_mean   : float        — средн. d(max_steps) = «остаточная» расходимость
        converges    : bool         — final_mean == 0 (все возмущения поглощаются)
        per_perturb  : list[dict]   — [{cell, bit, traj}] для каждого возмущения
    """
    orig = pad_to(encode_word(word.upper()), width)
    n_perturb   = width * _N_BITS
    all_trajs:  list[list[int]] = []
    per_perturb: list[dict] = []

    for ci in range(width):
        for b in range(_N_BITS):
            # Only perturb if that bit is meaningful (cell has room)
            traj = divergence_trajectory(word, ci, b, rule, width, max_steps)
            all_trajs.append(traj)
            per_perturb.append({'cell': ci, 'bit': b, 'traj': traj})

    n_steps = max_steps + 1
    mean_dist = [sum(t[i] for t in all_trajs) / n_perturb for i in range(n_steps)]
    max_dist  = [max(t[i] for t in all_trajs)             for i in range(n_steps)]
    min_dist  = [min(t[i] for t in all_trajs)             for i in range(n_steps)]

    peak_step = int(max(range(n_steps), key=lambda i: mean_dist[i]))
    peak_mean = mean_dist[peak_step]
    final_mean = mean_dist[-1]

    return {
        'word':        word.upper(),
        'rule':        rule,
        'width':       width,
        'n_perturb':   n_perturb,
        'mean_dist':   mean_dist,
        'max_dist':    max_dist,
        'min_dist':    min_dist,
        'peak_mean':   peak_mean,
        'peak_step':   peak_step,
        'final_mean':  final_mean,
        'converges':   final_mean == 0.0,
        'per_perturb': per_perturb,
    }


# ── Summary across all 4 rules ────────────────────────────────────────────────

def lyapunov_summary(
    word:      str,
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> dict:
    """Краткая сводка по всем 4 правилам.

    Возвращает {rule: {peak_mean, peak_step, final_mean, converges}}.
    """
    result: dict[str, dict] = {}
    for rule in _ALL_RULES:
        p = lyapunov_profile(word, rule, width, max_steps)
        result[rule] = {
            'peak_mean':  round(p['peak_mean'],  4),
            'peak_step':  p['peak_step'],
            'final_mean': round(p['final_mean'], 4),
            'converges':  p['converges'],
        }
    return result


# ── Peak-step sensitivity heatmap ─────────────────────────────────────────────

def peak_sensitivity_map(
    word:      str,
    rule:      str = 'xor3',
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> list[list[float]]:
    """Матрица d(peak_step) по всем возмущениям: [cell_idx][bit].

    Показывает, какие клетки/биты наиболее чувствительны на пике расходимости.
    Размер: width × 6.
    """
    prof = lyapunov_profile(word, rule, width, max_steps)
    ps   = prof['peak_step']
    mat: list[list[float]] = [[0.0] * _N_BITS for _ in range(width)]
    for entry in prof['per_perturb']:
        mat[entry['cell']][entry['bit']] = float(entry['traj'][ps])
    return mat


# ── Full dataset ──────────────────────────────────────────────────────────────

def build_lyapunov_data(
    words:     list[str] | None = None,
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> dict:
    """Сводный анализ Ляпунова для всего лексикона.

    Возвращает dict:
        words       : list[str]
        width       : int
        max_steps   : int
        per_rule    : {rule: {word: {peak_mean, peak_step, final_mean, converges}}}
        most_chaotic: {rule: (word, peak_mean)}  — слово с наибольшим пиком расходимости
        most_stable : {rule: (word, peak_mean)}  — слово с наименьшим пиком
    """
    words = words if words is not None else _DEFAULT_WORDS
    per_rule: dict[str, dict[str, dict]] = {r: {} for r in _ALL_RULES}
    for word in words:
        for rule in _ALL_RULES:
            p = lyapunov_profile(word, rule, width, max_steps)
            per_rule[rule][word] = {
                'peak_mean':  round(p['peak_mean'],  4),
                'peak_step':  p['peak_step'],
                'final_mean': round(p['final_mean'], 4),
                'converges':  p['converges'],
            }
    most_chaotic: dict[str, tuple[str, float]] = {}
    most_stable:  dict[str, tuple[str, float]] = {}
    for rule in _ALL_RULES:
        entries = [(w, d['peak_mean']) for w, d in per_rule[rule].items()]
        most_chaotic[rule] = max(entries, key=lambda x: x[1])
        most_stable[rule]  = min(entries, key=lambda x: x[1])
    return {
        'words':        words,
        'width':        width,
        'max_steps':    max_steps,
        'per_rule':     per_rule,
        'most_chaotic': most_chaotic,
        'most_stable':  most_stable,
    }


# ── JSON export ───────────────────────────────────────────────────────────────

def lyapunov_dict(
    word:      str,
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> dict:
    """JSON-совместимый словарь профилей по всем правилам (без per_perturb)."""
    result: dict[str, object] = {
        'word': word.upper(), 'width': width, 'max_steps': max_steps,
        'rules': {},
    }
    for rule in _ALL_RULES:
        p = lyapunov_profile(word, rule, width, max_steps)
        result['rules'][rule] = {  # type: ignore[index]
            'mean_dist':  [round(v, 4) for v in p['mean_dist']],
            'max_dist':   p['max_dist'],
            'min_dist':   p['min_dist'],
            'peak_mean':  round(p['peak_mean'],  4),
            'peak_step':  p['peak_step'],
            'final_mean': round(p['final_mean'], 4),
            'converges':  p['converges'],
        }
    return result


# ── ASCII display ─────────────────────────────────────────────────────────────

_BAR_CHARS = ' ▁▂▃▄▅▆▇█'
_MAX_DIST  = _N_BITS * _DEFAULT_WIDTH   # theoretical max = 6 × 16 = 96


def _bar(val: float, max_val: float, width: int = 24) -> str:
    """Горизонтальная полоска для ASCII-графика."""
    if max_val == 0:
        return ' ' * width
    frac  = min(1.0, val / max_val)
    filled = int(frac * width)
    rem    = frac * width - filled
    bar    = '█' * filled
    idx    = int(rem * (len(_BAR_CHARS) - 1))
    bar   += _BAR_CHARS[idx]
    return (bar + ' ' * width)[:width]


def print_lyapunov(
    word:      str,
    rule:      str = 'xor3',
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
    color:     bool = True,
) -> None:
    """Распечатать профиль расходимости: горизонтальная ASCII-полоска по шагам."""
    prof  = lyapunov_profile(word, rule, width, max_steps)
    rule_col  = _RULE_COLOR.get(rule, '') if color else ''
    rule_name = _RULE_NAMES.get(rule, rule.upper())
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    dim   = _DIM  if color else ''

    max_d  = max(prof['max_dist']) if prof['max_dist'] else 1
    print(f"{bold}  ◈ Ляпунов Q6  {word.upper()}  |  {rule_col}{rule_name}{rst}")
    print(f"  {'─' * 52}")
    print(f"  {'шаг':>4}  {'средн.':>6}  {'макс':>5}  {'гистограмма'}")
    print(f"  {'─' * 52}")

    for i, (mn, mx) in enumerate(zip(prof['mean_dist'], prof['max_dist'])):
        marker = '*' if i == prof['peak_step'] else ' '
        bar_c  = rule_col if i == prof['peak_step'] else (
            ('\033[38;5;196m' if color else '') if mn > max_d * 0.5 else
            ('\033[38;5;208m' if color else '') if mn > max_d * 0.25 else dim
        )
        bar_str = _bar(mn, max_d)
        print(f"  {i:>4}{marker} {mn:>6.2f}  {mx:>5}  {bar_c}{bar_str}{rst}")

    print()
    status = 'сходится' if prof['converges'] else f"остаток={prof['final_mean']:.2f}"
    print(f"  Пик: шаг {prof['peak_step']}  d̄={prof['peak_mean']:.2f}  | {status}")
    print()


def print_lyapunov_stats(
    words:     list[str] | None = None,
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
    color:     bool = True,
) -> None:
    """Таблица пиков расходимости для всего лексикона × 4 правила."""
    words = words if words is not None else _DEFAULT_WORDS
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''

    header = f"{'Слово':10s}" + ''.join(
        f"  {_RULE_COLOR.get(r,'') if color else ''}{_RULE_NAMES[r]:>10s}{rst}"
        for r in _ALL_RULES
    )
    print(f"\n{bold}  ◈ Пик расходимости Ляпунова (средн. Хэмминг-расстояние){rst}")
    print(f"  {'─' * (len(header) + 2)}")
    print('  ' + header)
    print(f"  {'─' * (len(header) + 2)}")

    for word in sorted(words):
        row_parts = [f'{word:10s}']
        for rule in _ALL_RULES:
            p = lyapunov_profile(word, rule, width, max_steps)
            col = _RULE_COLOR.get(rule, '') if color else ''
            cv  = '↓' if p['converges'] else '~'
            row_parts.append(f"  {col}{p['peak_mean']:>8.2f}{cv}{rst}")
        print('  ' + ''.join(row_parts))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description='Анализ расходимости CA Q6 (Ляпунов)')
    parser.add_argument('--word',      default='ГОРА',  help='Русское слово')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true')
    parser.add_argument('--steps',     type=int, default=_DEFAULT_STEPS)
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--no-color',  action='store_true')
    args = parser.parse_args()

    color = not args.no_color
    if args.stats:
        print_lyapunov_stats(color=color, max_steps=args.steps)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_lyapunov(args.word, rule, args.width, args.steps, color)
    else:
        print_lyapunov(args.word, args.rule, args.width, args.steps, color)


if __name__ == '__main__':
    _main()
