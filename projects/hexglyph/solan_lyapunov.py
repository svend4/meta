"""solan_lyapunov.py — Perturbation Sensitivity & Lyapunov-like Analysis (CA Q6).

Two complementary perturbation analyses:

══════════════════════════════════════════════════════════════════════
A. ORIGINAL INTERFACE  (attractor-independent, bit-level Hamming)
══════════════════════════════════════════════════════════════════════

  IC = raw Q6 encoding of the word (pad_to(encode_word(word), width)).
  Perturb one bit → evolve both → measure BIT-LEVEL Hamming per step.

  state_distance(s, s') = Σ_i popcount(s[i] XOR s'[i])   (bit-level)

  Functions:
    q6_hamming(a, b)                    → int   (bits differ in a vs b)
    state_distance(cells1, cells2)      → int
    perturb(cells, ci, bit)             → list
    divergence_trajectory(word, ci, bit, rule, width, max_steps) → list[int]
    lyapunov_profile(word, rule, width, max_steps)  → dict
    lyapunov_summary(word, width, max_steps)        → {rule: {peak_mean,…}}
    peak_sensitivity_map(word, rule, width, max_steps) → list[list[float]]
    build_lyapunov_data(words, width, max_steps)    → dict
    lyapunov_dict(word, width, max_steps)           → dict
    print_lyapunov(word, rule, width, max_steps, color)  → None
    print_lyapunov_stats(words, width, max_steps, color) → None
    _ALL_RULES, _N_BITS, _DEFAULT_STEPS

══════════════════════════════════════════════════════════════════════
B. ORBIT-MODE INTERFACE  (attractor-based, cell-level Hamming)
══════════════════════════════════════════════════════════════════════

  IC = orbit[0] (first state on the attractor from solan_perm.get_orbit).
  Perturb one bit → evolve both → measure CELL-LEVEL Hamming per step.

  d(t) = |{i : s_t[i] ≠ s'_t[i]}|   (cell-level, NOT bit-level)

  Four Lyapunov Modes  (width=16, T=32)
  ─────────────────────────────────────
    absorbs   (mean_d[1]=0): ТУМАН AND, ГОРА OR — erased in 1 step
    stabilizes (d→0 t>1):   ТУМАН XOR — Sierpinski; full recovery at t=8
    plateau    (d→const):   ГОРА AND — linear growth → d=4 plateau
    periodic   (d oscillates): ТУМАН/ГОРА XOR3 — period-8, max d=11

  Functions:
    perturb_profile(ic, j, bit, rule, T)          → list[int]
    perturb_all_profiles(ic, rule, T)             → list[list[int]]
    mean_d_profile(profiles)                      → list[float]
    detect_period(seq)                            → int | None
    classify_mode(mean_prof, T)                   → str
    perturbation_cone(ic, j, rule, T)             → list[list[int]]
    lyapunov_mode_summary(word, rule, width, T)   → dict
    all_lyapunov(word, width)                     → dict[str, dict]
    build_mode_data(words, width)                 → dict
    print_mode(word, rule, color)                 → None
    print_mode_table(words, color)                → None

Запуск
──────
  python3 -m projects.hexglyph.solan_lyapunov --word ГОРА --rule xor3
  python3 -m projects.hexglyph.solan_lyapunov --word ТУМАН --all-rules --no-color
  python3 -m projects.hexglyph.solan_lyapunov --stats
  python3 -m projects.hexglyph.solan_lyapunov --mode --word ТУМАН --rule xor3
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

# ── Shared constants ───────────────────────────────────────────────────────────
_ALL_RULES     = ['xor', 'xor3', 'and', 'or']
_RULES         = _ALL_RULES          # alias for orbit-mode interface
_N_BITS        = 6                   # bits per Q6 cell
_DEFAULT_WIDTH = 16
_DEFAULT_W     = _DEFAULT_WIDTH      # alias
_DEFAULT_STEPS = 32
_DEFAULT_T     = _DEFAULT_STEPS      # alias
_MODES         = ('absorbs', 'stabilizes', 'plateau', 'periodic')
_DEFAULT_WORDS = list(LEXICON)


# ══════════════════════════════════════════════════════════════════════════════
# A. ORIGINAL INTERFACE — bit-level Hamming, raw-encoding IC
# ══════════════════════════════════════════════════════════════════════════════

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


def lyapunov_profile(
    word:      str,
    rule:      str = 'xor3',
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> dict:
    """Усреднённый профиль расходимости по всем width×6 однобитовым возмущениям.

    Возвращает dict:
        word, rule, width, n_perturb,
        mean_dist, max_dist, min_dist,
        peak_mean, peak_step, final_mean, converges,
        per_perturb: [{cell, bit, traj}]
    """
    orig = pad_to(encode_word(word.upper()), width)
    n_perturb    = width * _N_BITS
    all_trajs:   list[list[int]] = []
    per_perturb: list[dict] = []

    for ci in range(width):
        for b in range(_N_BITS):
            traj = divergence_trajectory(word, ci, b, rule, width, max_steps)
            all_trajs.append(traj)
            per_perturb.append({'cell': ci, 'bit': b, 'traj': traj})

    n_steps   = max_steps + 1
    mean_dist = [sum(t[i] for t in all_trajs) / n_perturb for i in range(n_steps)]
    max_dist  = [max(t[i] for t in all_trajs)             for i in range(n_steps)]
    min_dist  = [min(t[i] for t in all_trajs)             for i in range(n_steps)]

    peak_step  = int(max(range(n_steps), key=lambda i: mean_dist[i]))
    peak_mean  = mean_dist[peak_step]
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


def lyapunov_summary(
    word:      str,
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> dict:
    """Краткая сводка по всем 4 правилам: {rule: {peak_mean, peak_step, …}}."""
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


def peak_sensitivity_map(
    word:      str,
    rule:      str = 'xor3',
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> list[list[float]]:
    """Матрица d(peak_step) по всем возмущениям: [cell_idx][bit] (width × 6)."""
    prof = lyapunov_profile(word, rule, width, max_steps)
    ps   = prof['peak_step']
    mat: list[list[float]] = [[0.0] * _N_BITS for _ in range(width)]
    for entry in prof['per_perturb']:
        mat[entry['cell']][entry['bit']] = float(entry['traj'][ps])
    return mat


def build_lyapunov_data(
    words:     list[str] | None = None,
    width:     int = _DEFAULT_WIDTH,
    max_steps: int = _DEFAULT_STEPS,
) -> dict:
    """Сводный анализ Ляпунова: {words, width, max_steps, per_rule,
    most_chaotic, most_stable}."""
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
        result['rules'][rule] = {       # type: ignore[index]
            'mean_dist':  [round(v, 4) for v in p['mean_dist']],
            'max_dist':   p['max_dist'],
            'min_dist':   p['min_dist'],
            'peak_mean':  round(p['peak_mean'],  4),
            'peak_step':  p['peak_step'],
            'final_mean': round(p['final_mean'], 4),
            'converges':  p['converges'],
        }
    return result


# ── Original print helpers ─────────────────────────────────────────────────────

_BAR_CHARS = ' ▁▂▃▄▅▆▇█'
_MAX_DIST  = _N_BITS * _DEFAULT_WIDTH


def _bar_orig(val: float, max_val: float, width: int = 24) -> str:
    if max_val == 0:
        return ' ' * width
    frac   = min(1.0, val / max_val)
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
    prof      = lyapunov_profile(word, rule, width, max_steps)
    rule_col  = _RULE_COLOR.get(rule, '') if color else ''
    rule_name = _RULE_NAMES.get(rule, rule.upper())
    rst   = _RST  if color else ''
    bold  = _BOLD if color else ''
    dim   = _DIM  if color else ''

    max_d = max(prof['max_dist']) if prof['max_dist'] else 1
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
        bar_str = _bar_orig(mn, max_d)
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
            p   = lyapunov_profile(word, rule, width, max_steps)
            col = _RULE_COLOR.get(rule, '') if color else ''
            cv  = '↓' if p['converges'] else '~'
            row_parts.append(f"  {col}{p['peak_mean']:>8.2f}{cv}{rst}")
        print('  ' + ''.join(row_parts))


# ══════════════════════════════════════════════════════════════════════════════
# B. ORBIT-MODE INTERFACE — cell-level Hamming, attractor IC
# ══════════════════════════════════════════════════════════════════════════════

def _get_orbit(word: str, rule: str, width: int) -> list[tuple[int, ...]]:
    from projects.hexglyph.solan_perm import get_orbit
    return get_orbit(word.upper(), rule, width)


def _mode_step(cells: list[int], rule: str) -> list[int]:
    N   = len(cells)
    out = []
    for i in range(N):
        l, c, r = cells[(i - 1) % N], cells[i], cells[(i + 1) % N]
        if   rule == 'xor':  out.append((l ^ r)       & 63)
        elif rule == 'xor3': out.append((l ^ c ^ r)   & 63)
        elif rule == 'and':  out.append((l & r)        & 63)
        else:                out.append((l | r)        & 63)
    return out


def perturb_profile(ic: list[int], j: int, bit: int,
                    rule: str, T: int = _DEFAULT_T) -> list[int]:
    """Cell-level Hamming distance for a single-bit perturbation.

    Flips bit `bit` (0..5) of cell `j` in `ic`, evolves both states T steps.
    Returns T integers: d(t) = |{i : s_t[i] ≠ s'_t[i]}|.
    """
    ic2 = ic[:]
    ic2[j] ^= (1 << bit)
    cur, cur2 = ic[:], ic2[:]
    profile = []
    for _ in range(T):
        profile.append(sum(a != b for a, b in zip(cur, cur2)))
        cur  = _mode_step(cur,  rule)
        cur2 = _mode_step(cur2, rule)
    return profile


def perturb_all_profiles(ic: list[int], rule: str,
                          T: int = _DEFAULT_T) -> list[list[int]]:
    """All N×6 single-bit perturbation profiles (cell-level Hamming)."""
    N = len(ic)
    return [perturb_profile(ic, j, bit, rule, T)
            for j in range(N) for bit in range(6)]


def mean_d_profile(profiles: list[list[int]]) -> list[float]:
    """Average d(t) across all perturbation profiles."""
    T = max(len(p) for p in profiles)
    n = len(profiles)
    return [round(sum(p[t] if t < len(p) else 0 for p in profiles) / n, 6)
            for t in range(T)]


def detect_period(seq: list[float], min_p: int = 2,
                  max_p: int = 20) -> int | None:
    """Detect period of a numeric sequence (None if not periodic)."""
    n = len(seq)
    for p in range(min_p, max_p + 1):
        if n >= 2 * p:
            if all(abs(seq[t] - seq[t + p]) < 1e-6 for t in range(n - p)):
                return p
    return None


def classify_mode(mean_prof: list[float], T: int = _DEFAULT_T) -> str:
    """Classify Lyapunov mode: 'absorbs' | 'stabilizes' | 'plateau' | 'periodic'."""
    if len(mean_prof) > 1 and mean_prof[1] < 1e-9:
        return 'absorbs'
    t_conv = next((t for t, v in enumerate(mean_prof) if v < 1e-9), None)
    if t_conv is not None:
        return 'stabilizes'
    # Plateau: last quarter nearly constant (range < 0.5)
    last = mean_prof[3 * len(mean_prof) // 4:]
    if last and max(last) - min(last) < 0.5:
        return 'plateau'
    # True oscillating periodic
    tail = mean_prof[min(8, len(mean_prof) // 2):]
    if detect_period(tail) is not None:
        return 'periodic'
    return 'periodic'


def perturbation_cone(ic: list[int], j: int, rule: str,
                      T: int = 16) -> list[list[int]]:
    """T × N binary matrix: cone[t][i] = 1 if cell i affected at step t.
    Averaged over all 6 bit flips of cell j."""
    N  = len(ic)
    T_ = min(T, 16)
    agg = [[0.0] * N for _ in range(T_)]
    for bit in range(6):
        ic2 = ic[:]
        ic2[j] ^= (1 << bit)
        cur, cur2 = ic[:], ic2[:]
        for t in range(T_):
            for i in range(N):
                agg[t][i] += 1.0 if cur[i] != cur2[i] else 0.0
            cur  = _mode_step(cur,  rule)
            cur2 = _mode_step(cur2, rule)
    return [[round(v / 6) for v in row] for row in agg]


def lyapunov_mode_summary(word: str, rule: str,
                           width: int = _DEFAULT_W,
                           T:     int = _DEFAULT_T) -> dict:
    """Full orbit-mode Lyapunov summary (cell-level Hamming, attractor IC).

    Returns dict with keys: word, rule, period_orbit, n_cells, T,
    mean_d, max_mean_d, t_max_mean_d, t_converge, fraction_converged,
    plateau_d, mode, period_d, cone_centre,
    absorbs, stabilizes, is_plateau, is_periodic.
    """
    orbit    = _get_orbit(word, rule, width)
    ic       = list(orbit[0])
    N        = width
    profiles = perturb_all_profiles(ic, rule, T)
    mean_p   = mean_d_profile(profiles)
    mode     = classify_mode(mean_p, T)

    max_md   = max(mean_p)
    t_max_md = mean_p.index(max_md)
    t_conv   = next((t for t, v in enumerate(mean_p) if v < 1e-9), None)
    f_conv   = sum(1 for p in profiles if p[-1] == 0) / len(profiles)
    last_q   = mean_p[3 * T // 4:]
    plateau  = round(sum(last_q) / len(last_q), 4) if last_q else 0.0
    tail     = mean_p[min(8, T // 2):]
    period   = detect_period(tail) if mode == 'periodic' else None
    cone     = perturbation_cone(ic, N // 2, rule, T=min(T, 16))

    return {
        'word':               word.upper(),
        'rule':               rule,
        'period_orbit':       len(orbit),
        'n_cells':            N,
        'T':                  T,
        'mean_d':             mean_p,
        'max_mean_d':         round(max_md, 4),
        't_max_mean_d':       t_max_md,
        't_converge':         t_conv,
        'fraction_converged': round(f_conv, 4),
        'plateau_d':          plateau,
        'mode':               mode,
        'period_d':           period,
        'cone_centre':        cone,
        'absorbs':            mode == 'absorbs',
        'stabilizes':         mode == 'stabilizes',
        'is_plateau':         mode == 'plateau',
        'is_periodic':        mode == 'periodic',
    }


def all_lyapunov(word: str, width: int = _DEFAULT_W) -> dict[str, dict]:
    """lyapunov_mode_summary for all 4 rules."""
    return {rule: lyapunov_mode_summary(word, rule, width) for rule in _RULES}


def build_mode_data(words: list[str], width: int = _DEFAULT_W) -> dict:
    """Compact orbit-mode Lyapunov data for all words × rules."""
    per_rule: dict[str, dict] = {rule: {} for rule in _RULES}
    for word in words:
        for rule in _RULES:
            d = lyapunov_mode_summary(word, rule, width)
            per_rule[rule][word.upper()] = {
                k: d[k] for k in ('period_orbit', 'mean_d', 'max_mean_d',
                                   't_max_mean_d', 't_converge',
                                   'fraction_converged', 'plateau_d',
                                   'mode', 'period_d',
                                   'absorbs', 'stabilizes',
                                   'is_plateau', 'is_periodic')
            }
    return {'words': [w.upper() for w in words], 'width': width,
            'per_rule': per_rule}


# ── Orbit-mode print helpers ───────────────────────────────────────────────────

_RCOL = {'xor': '\033[96m', 'xor3': '\033[36m',
         'and': '\033[91m',  'or':   '\033[33m'}
_MCOL = {'absorbs': '\033[92m', 'stabilizes': '\033[96m',
         'plateau': '\033[93m', 'periodic':   '\033[91m'}
_BAR_W = 20


def _bar_mode(val: float, max_val: float, w: int = _BAR_W) -> str:
    frac   = val / max_val if max_val > 0 else 0.0
    filled = round(frac * w)
    return '█' * filled + '░' * (w - filled)


def print_mode(word: str = 'ТУМАН', rule: str = 'xor3',
               color: bool = True) -> None:
    """Print orbit-mode Lyapunov profile (cell-level Hamming)."""
    d   = lyapunov_mode_summary(word, rule)
    c   = _RCOL.get(rule, '') if color else ''
    mc  = _MCOL.get(d['mode'], '') if color else ''
    r   = _RST if color else ''
    lbl = {'xor': 'XOR ⊕', 'xor3': 'XOR3',
           'and': 'AND &',  'or':   'OR |'}.get(rule, rule)
    print(f'  {c}◈ Lyapunov mode  {word.upper()}  |  {lbl}  '
          f'P={d["period_orbit"]}  N={d["n_cells"]}{r}')
    print(f'  {mc}MODE: {d["mode"].upper()}{r}   '
          f'max_d={d["max_mean_d"]:.2f}@t={d["t_max_mean_d"]}  '
          f'fconv={d["fraction_converged"]:.2f}  '
          f'plateau_d={d["plateau_d"]:.2f}'
          + (f'  period_d={d["period_d"]}' if d["period_d"] else ''))
    print('  ' + '─' * 68)
    maxv = max(d['mean_d'][:20]) or 1.0
    for t in range(min(20, d['T'])):
        v   = d['mean_d'][t]
        bar = _bar_mode(v, maxv)
        cv  = ' ← converge' if d['t_converge'] == t else ''
        print(f'  t={t:>2}  mean_d={v:>5.2f}  [{bar}]{cv}')
    print()
    print(f'  Perturbation cone (centre cell={d["n_cells"]//2}):')
    for t, row in enumerate(d['cone_centre']):
        bits = ''.join('█' if v else '░' for v in row)
        h    = sum(row)
        print(f'  t={t:>2}  d={h:>2}  [{bits}]')
    print()


def print_mode_table(words: list[str] | None = None,
                     color: bool = True) -> None:
    """Print orbit-mode Lyapunov table for all words."""
    WORDS = words or list(LEXICON)
    R = _RST if color else ''
    head = '  '.join(
        (_RCOL.get(rl, '') if color else '') + f'{rl.upper():>6} mode maxd fc' + R
        for rl in _RULES)
    print(f'  {"Слово":10s}  {head}')
    print('  ' + '─' * 76)
    mode_ch = {'absorbs': 'A', 'stabilizes': 'S', 'plateau': 'P', 'periodic': '~'}
    for word in WORDS:
        parts = []
        for rule in _RULES:
            col = _RCOL.get(rule, '') if color else ''
            d   = lyapunov_mode_summary(word, rule)
            m   = mode_ch.get(d['mode'], '?')
            parts.append(f'{col}{m} {d["max_mean_d"]:>4.1f} {d["fraction_converged"]:.2f}{R}')
        print(f'  {word.upper():10s}  ' + '  '.join(parts))
    print()
    print(f'  Mode: A=absorbs  S=stabilizes  P=plateau  ~=periodic')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _main() -> None:
    parser = argparse.ArgumentParser(
        description='Lyapunov perturbation analysis for CA Q6')
    parser.add_argument('--word',      default='ГОРА')
    parser.add_argument('--rule',      default='xor3',  choices=_ALL_RULES)
    parser.add_argument('--all-rules', action='store_true')
    parser.add_argument('--stats',     action='store_true',
                        help='Original interface: bit-level Hamming stats table')
    parser.add_argument('--mode',      action='store_true',
                        help='Orbit-mode interface: cell-level Hamming + mode')
    parser.add_argument('--table',     action='store_true',
                        help='Orbit-mode table for all words')
    parser.add_argument('--steps',     type=int, default=_DEFAULT_STEPS)
    parser.add_argument('--width',     type=int, default=_DEFAULT_WIDTH)
    parser.add_argument('--json',      action='store_true', help='JSON output')
    parser.add_argument('--no-color',  action='store_true')
    args  = parser.parse_args()
    color = not args.no_color

    if args.json:
        import json as _json
        print(_json.dumps(lyapunov_dict(args.word, args.width, args.steps),
                          ensure_ascii=False, indent=2))
    elif args.stats:
        print_lyapunov_stats(color=color, max_steps=args.steps)
    elif args.table:
        print_mode_table(color=color)
    elif args.mode:
        if args.all_rules:
            for rule in _ALL_RULES:
                print_mode(args.word, rule, color)
        else:
            print_mode(args.word, args.rule, color)
    elif args.all_rules:
        for rule in _ALL_RULES:
            print_lyapunov(args.word, rule, args.width, args.steps, color)
    else:
        print_lyapunov(args.word, args.rule, args.width, args.steps, color)


if __name__ == '__main__':
    _main()
