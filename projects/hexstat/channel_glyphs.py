"""channel_glyphs — Теория информации на Q6 через систему глифов.

Каждый глиф (0..63) — точка пространства (Z₂)⁶ с вероятностью P(h).

Ключевые понятия:
  • Бинарный симметричный канал BSC(p): каждый бит переворачивается с вер. p
    P(h|x) = p^{d(h,x)} · (1−p)^{6−d(h,x)}
  • Ёмкость BSC: C = 1 − H(p) бит  (H(p) = двоичная энтропия)
  • Взаимная информация I(X;Y) = H(X) − H(X|Y)
  • KL-дивергенция: D(P‖Q) = Σ P(h) log₂(P(h)/Q(h))

Визуализация:
  • bsc   — BSC-распределение для разных p: глифы раскрашены по P(h|0)
  • kl    — KL-дивергенция между двумя распределениями
  • mutual — взаимная информация I(X;Y) при X равномерном и канале BSC(p)
  • entropy — энтропия H(X) при разных распределениях на Q6

Команды CLI:
  bsc <p>                — BSC(p) из центра 0: распределение ошибок
  kl  <p1> <p2>         — KL-дивергенция BSC(p1) ‖ BSC(p2)
  mutual                 — кривая I(X;Y) = C(p) для BSC(p), p=0..0.5
  entropy                — сравнение энтропий разных распределений
"""

from __future__ import annotations
import json
import sys
import math
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexstat.hexstat import (
    Q6Distribution,
    q6_channel_capacity_bsc,
    yang_entropy,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _popcount(x: int) -> int:
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def _h2(p: float) -> float:
    """Двоичная энтропия H(p) = −p log₂p − (1−p) log₂(1−p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _prob_color(p: float, p_max: float, p_min: float, color: bool) -> str:
    """Цвет ANSI по значению вероятности (нормированной)."""
    if not color:
        return ''
    if p_max <= p_min:
        return _YANG_ANSI[3]
    level = int(6 * (p - p_min) / (p_max - p_min))
    return _YANG_ANSI[max(0, min(6, level))]


# ---------------------------------------------------------------------------
# 1. BSC-распределение
# ---------------------------------------------------------------------------

def render_bsc(p: float, center: int = 0, color: bool = True) -> str:
    """
    8×8 сетка глифов: вероятность P(h|center) для BSC(p).

    При p=0: только сам центр получает вероятность 1.
    При p=0.5: равномерное распределение.
    При p=1: антипод получает вероятность 1.
    """
    dist = Q6Distribution.binary_symmetric_channel(center, p)
    probs = dist.probs()
    H = dist.entropy()
    cap = q6_channel_capacity_bsc(p)

    p_max = max(probs)
    p_min = min(probs)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  BSC(p={p:.3f})  центр={center}  (bits={format(center,"06b")})')
    lines.append(f'  H(выход) = {H:.4f} бит   Ёмкость C = {cap:.4f} бит/символ')
    lines.append(f'  H₂(p) = {_h2(p):.4f}   C = 6·(1 − H₂(p)) = {6*(1-_h2(p)):.4f}')
    lines.append('  Цвет: яркий=высокая P(h), тёмный=низкая P(h)')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            ph = probs[h]
            rows3 = render_glyph(h)
            if color:
                is_center = (h == center)
                is_antipode = (h == (center ^ 63))
                if is_center:
                    c_str = _YANG_BG[yang_count(h)] + _BOLD
                elif is_antipode and p > 0.4:
                    c_str = _YANG_BG[yang_count(h)] + _BOLD
                else:
                    c_str = _prob_color(ph, p_max, p_min, color)
                rows3 = [c_str + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            ph = probs[h]
            d = _popcount(h ^ center)
            if color:
                c_str = _prob_color(ph, p_max, p_min, color)
                lbl.append(f'{c_str}{ph:.3f}{_RESET}')
            else:
                lbl.append(f'{ph:.3f}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    # Группировка по расстоянию
    lines.append('  Вероятности по расстоянию d(h, center):')
    for d in range(7):
        elems = [h for h in range(64) if _popcount(h ^ center) == d]
        if not elems:
            continue
        p_d = probs[elems[0]]
        cnt = len(elems)
        total = p_d * cnt
        if color:
            c_str = _YANG_ANSI[d]
            lines.append(f'  {c_str}  d={d}: P={p_d:.5f}  ×{cnt:2d}  Σ={total:.4f}{_RESET}')
        else:
            lines.append(f'    d={d}: P={p_d:.5f}  ×{cnt:2d}  Σ={total:.4f}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. KL-дивергенция
# ---------------------------------------------------------------------------

def render_kl(p1: float, p2: float, color: bool = True) -> str:
    """
    Сравнение двух BSC-распределений через KL-дивергенцию.

    Каждый глиф раскрашен по вкладу P₁(h) log₂(P₁(h)/P₂(h)).
    """
    d1 = Q6Distribution.binary_symmetric_channel(0, p1)
    d2 = Q6Distribution.binary_symmetric_channel(0, p2)
    kl = d1.kl_divergence(d2)
    kl_rev = d2.kl_divergence(d1)

    probs1 = d1.probs()
    probs2 = d2.probs()

    # Вклад каждой вершины
    contribs = []
    for h in range(64):
        if probs1[h] > 0 and probs2[h] > 0:
            contribs.append(probs1[h] * math.log2(probs1[h] / probs2[h]))
        else:
            contribs.append(0.0)

    c_max = max(contribs)
    c_min = min(contribs)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  KL-дивергенция: BSC(p₁={p1:.3f}) ‖ BSC(p₂={p2:.3f})')
    lines.append(f'  D(P₁‖P₂) = {kl:.5f} бит')
    lines.append(f'  D(P₂‖P₁) = {kl_rev:.5f} бит')
    lines.append(f'  H₁={d1.entropy():.4f} бит   H₂={d2.entropy():.4f} бит')
    lines.append('  Цвет: вклад P₁(h)·log₂(P₁/P₂) в KL')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                cv = contribs[h]
                if c_max > c_min:
                    level = int(6 * (cv - c_min) / (c_max - c_min))
                else:
                    level = 3
                level = max(0, min(6, level))
                c_str = _YANG_ANSI[level]
                rows3 = [c_str + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            cv = contribs[h]
            if color:
                level = int(6 * (cv - c_min) / (c_max - c_min + 1e-15))
                level = max(0, min(6, level))
                c_str = _YANG_ANSI[level]
                lbl.append(f'{c_str}{cv:+.4f}{_RESET}')
            else:
                lbl.append(f'{cv:+.4f}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Кривая взаимной информации I(X;Y)
# ---------------------------------------------------------------------------

def render_mutual_info(color: bool = True) -> str:
    """
    Кривая ёмкости канала BSC(p) для p = 0 .. 0.5.

    C(p) = 6·(1 − H₂(p)) — ёмкость 6-битного BSC (бит/символ).
    I(X;Y) = H(Y) − H(Y|X) = 6 − 6·H₂(p) при равномерном входе.
    """
    steps = 26  # p = 0.00, 0.02, 0.04, ..., 0.50
    ps = [i * 0.02 for i in range(steps)]
    caps = [q6_channel_capacity_bsc(p) for p in ps]

    height = 14
    width = steps

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append('  Ёмкость C(p) = 6·(1−H₂(p)) бит  для BSC(p), 6 бит')
    lines.append('  p=0: C=6 бит (нет ошибок)   p=0.5: C=0 (полный шум)')
    lines.append('═' * 64)

    c_max = 6.0
    grid = [[' '] * width for _ in range(height)]
    for i, cap in enumerate(caps):
        row = int((height - 1) * (1 - cap / c_max))
        row = max(0, min(height - 1, row))
        grid[row][i] = '●' if color else '*'

    for ri, row_data in enumerate(grid):
        cap_val = c_max * (1 - ri / (height - 1))
        level = int(6 * (1 - ri / (height - 1)))
        if color:
            c = _YANG_ANSI[max(0, min(6, level))]
            lines.append(f'  {c}C={cap_val:.2f}{_RESET} │ ' + ''.join(row_data))
        else:
            lines.append(f'  C={cap_val:.2f} │ ' + ''.join(row_data))
    lines.append('         └' + '─' * width)
    lines.append('           p=0' + ' ' * (width - 9) + 'p=0.5')

    lines.append('')
    lines.append('  Ключевые точки:')
    key_ps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for kp in key_ps:
        cap_v = q6_channel_capacity_bsc(kp)
        h2 = _h2(kp)
        if color:
            level = int(6 * cap_v / 6.0)
            c = _YANG_ANSI[max(0, min(6, level))]
            lines.append(f'  {c}  p={kp:.2f}: H₂(p)={h2:.4f}  C={cap_v:.4f} бит{_RESET}')
        else:
            lines.append(f'    p={kp:.2f}: H₂(p)={h2:.4f}  C={cap_v:.4f} бит')

    lines.append('')
    lines.append('  Теорема Шеннона: существуют коды со скоростью < C,')
    lines.append('  исправляющие все ошибки BSC(p), при n → ∞.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Сравнение энтропий
# ---------------------------------------------------------------------------

def render_entropy_compare(color: bool = True) -> str:
    """
    Сравнить несколько ключевых распределений на Q6 по энтропии.

    Показывает: равномерное, yang-взвешенное, BSC(0.1), BSC(0.5),
    а также для каждого — 8×2 мини-сетку глифов.
    """
    dists = [
        ('Равномерное  P(h)=1/64',   Q6Distribution.uniform()),
        ('BSC(p=0.1)  центр=0',      Q6Distribution.binary_symmetric_channel(0, 0.1)),
        ('BSC(p=0.2)  центр=0',      Q6Distribution.binary_symmetric_channel(0, 0.2)),
        ('BSC(p=0.5)  центр=0',      Q6Distribution.binary_symmetric_channel(0, 0.5)),
        ('Yang-шар r=1  центр=0',    Q6Distribution.hamming_shell(0, 1)),
        ('Yang-шар r=2  центр=0',    Q6Distribution.hamming_shell(0, 2)),
    ]

    lines: list[str] = []
    lines.append('╔' + '═' * 62 + '╗')
    lines.append('║  Сравнение энтропий распределений на Q6' + ' ' * 22 + '║')
    lines.append('╚' + '═' * 62 + '╝')
    lines.append('')

    for name, dist in dists:
        H = dist.entropy()
        H_min = dist.min_entropy()
        probs = dist.probs()
        p_max = max(probs)
        p_min_val = min(probs)
        mean_yang = dist.mean_yang()

        level = int(6 * H / 6.0)
        if color:
            c = _YANG_ANSI[max(0, min(6, level))]
            lines.append(f'  {c}[{name}]{_RESET}')
            lines.append(f'  {c}  H={H:.4f} бит  H_min={H_min:.4f}  '
                         f'E[yang]={mean_yang:.3f}{_RESET}')
        else:
            lines.append(f'  [{name}]')
            lines.append(f'    H={H:.4f} бит  H_min={H_min:.4f}  '
                         f'E[yang]={mean_yang:.3f}')

        # Мини-сетка: первые 16 глифов (2 строки по 8)
        for start in [0, 8]:
            glyphs_chunk = []
            for h in range(start, start + 8):
                ph = probs[h]
                rows3 = render_glyph(h)
                if color:
                    level_h = int(6 * (ph - p_min_val) / (p_max - p_min_val + 1e-15))
                    level_h = max(0, min(6, level_h))
                    c_h = _YANG_ANSI[level_h]
                    rows3 = [c_h + r + _RESET for r in rows3]
                glyphs_chunk.append(rows3)
            for ri in range(3):
                lines.append('    ' + '  '.join(g[ri] for g in glyphs_chunk) + '  ...')
        lines.append('')

    lines.append('  Максимальная энтропия: H=6 бит (равномерное распределение)')
    lines.append('  Минимальная: H=0 бит (дираковское)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# JSON-экспорт (для пайплайнов SC-3)
# ---------------------------------------------------------------------------

def json_ca_classify(ca_data: dict) -> dict:
    """
    Классифицировать поведение КА на основе его JSON-эволюции.

    Входной формат: вывод команды 'hexca --json evolve'.
    Выходной формат: классификация по Вольфраму и энтропийная статистика.
    """
    stats = ca_data.get('evolution_stats', [])
    if not stats:
        return {'error': 'evolution_stats не найдены во входных данных'}

    initial_entropy = stats[0]['entropy']
    final_entropy = stats[-1]['entropy']
    n = len(stats)
    rule = ca_data.get('rule', 'unknown')

    # Вычислить скорость убывания энтропии
    if n > 1 and initial_entropy > 0:
        decay_per_step = (initial_entropy - final_entropy) / (n - 1)
        relative_decay = decay_per_step / initial_entropy
    else:
        decay_per_step = 0.0
        relative_decay = 0.0

    # Классификация Вольфрама по поведению энтропии
    conv_step = ca_data.get('convergence_step')
    period_1 = ca_data.get('period_1', False)
    period_2 = ca_data.get('period_2', False)

    if period_1 or (conv_step is not None and conv_step <= 3):
        wolfram_class = 'I'
        behavior = 'stable_fixed_point'
        attractor = 'fixed_point'
    elif period_2 or (conv_step is not None):
        wolfram_class = 'II'
        behavior = 'periodic'
        attractor = 'period_2' if period_2 else 'periodic'
    elif final_entropy > initial_entropy * 0.85 and final_entropy >= 2.5:
        wolfram_class = 'III_or_IV'
        behavior = 'complex_or_chaotic'
        attractor = 'orbit'
    else:
        wolfram_class = 'II'
        behavior = 'convergent'
        attractor = 'unknown'

    # Статистика по ян-балансу
    yang_initial = stats[0]['mean_yang'] if stats else 3.0
    yang_final = stats[-1]['mean_yang'] if stats else 3.0
    yang_drift = abs(yang_final - yang_initial)
    yang_balanced = yang_drift < 0.5  # Aut(Q6)-инвариант: ян-счёт примерно сохраняется

    return {
        'command': 'ca_entropy',
        'rule': rule,
        'wolfram_class': wolfram_class,
        'behavior': behavior,
        'attractor': attractor,
        'initial_entropy': initial_entropy,
        'final_entropy': final_entropy,
        'entropy_decay_per_step': round(decay_per_step, 5),
        'relative_decay': round(relative_decay, 5),
        'convergence_step': conv_step,
        'yang_initial': yang_initial,
        'yang_final': yang_final,
        'yang_balanced': yang_balanced,
        'yang_drift': round(yang_drift, 3),
        'unique_initial': stats[0]['unique'],
        'unique_final': stats[-1]['unique'],
        'steps_analyzed': n,
    }


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='channel_glyphs',
        description='Теория информации на Q6 через глифы гексаграмм',
    )
    p.add_argument('--no-color', action='store_true', help='без ANSI-цветов')
    p.add_argument('--json', action='store_true',
                   help='Машиночитаемый JSON-вывод (для пайплайнов)')
    p.add_argument('--from-ca', action='store_true',
                   help='Читать CA JSON из stdin (hexca --json evolve)')
    sub = p.add_subparsers(dest='cmd', required=True)

    # ca-entropy — классификация КА (SC-3 шаг 2)
    sub.add_parser('ca-entropy',
                   help='Классифицировать КА по энтропийной динамике → JSON')

    s = sub.add_parser('bsc', help='BSC(p)-распределение ошибок')
    s.add_argument('p', type=float, help='вероятность ошибки бита (0..1)')
    s.add_argument('--center', type=int, default=0, help='входное слово')

    s = sub.add_parser('kl', help='KL-дивергенция между BSC(p1) и BSC(p2)')
    s.add_argument('p1', type=float)
    s.add_argument('p2', type=float)

    sub.add_parser('mutual', help='кривая ёмкости C(p) для BSC(p)')
    sub.add_parser('entropy', help='сравнение энтропий распределений')

    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'ca-entropy':
        if args.from_ca:
            raw = sys.stdin.read().strip()
            ca_data = json.loads(raw)
        else:
            # Без stdin: запустить hexca внутренне для демо
            import subprocess, sys as _sys
            result = subprocess.run(
                [_sys.executable, '-m', 'projects.hexca.ca_glyphs', '--json', 'evolve'],
                capture_output=True, text=True,
            )
            ca_data = json.loads(result.stdout)
        classified = json_ca_classify(ca_data)
        if args.json:
            print(json.dumps(classified, ensure_ascii=False, indent=2))
        else:
            print(f'  Правило: {classified["rule"]}')
            print(f'  Класс Вольфрама: {classified["wolfram_class"]}')
            print(f'  Поведение: {classified["behavior"]}')
            print(f'  Энтропия: {classified["initial_entropy"]:.3f} → {classified["final_entropy"]:.3f}')
            print(f'  Аттрактор: {classified["attractor"]}')
            print(f'  Ян-баланс: {"✓ сохранён" if classified["yang_balanced"] else "✗ нарушен"}')
        return

    if args.cmd == 'bsc':
        print(render_bsc(args.p, center=args.center, color=color))
    elif args.cmd == 'kl':
        print(render_kl(args.p1, args.p2, color=color))
    elif args.cmd == 'mutual':
        print(render_mutual_info(color=color))
    elif args.cmd == 'entropy':
        print(render_entropy_compare(color=color))


if __name__ == '__main__':
    main()
