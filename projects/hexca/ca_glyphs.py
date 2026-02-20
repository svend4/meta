"""ca_glyphs — Клеточный автомат на Q6 с визуализацией через глифы.

Каждая клетка КА отображается как 3-строчный глиф.
Эволюция автомата показывается как матрица глифов — каждый шаг в новой строке.

Отличие от стандартного hexca:
  • Стандартный: одна ячейка = один символ из ' ·░▒▓█▓'  (7 видов по ян-счёту)
  • Glyph-режим:  одна ячейка = 3 строки × 3 символа = 64 уникальных состояния

Форматы отображения:
  1D history — история 1D автомата (время идёт вниз, каждый шаг = строка глифов)
  2D frame   — один шаг 2D автомата как матрица глифов
  diff       — разность двух шагов (XOR состояний) как глифы
"""

from __future__ import annotations
import json
import math
import sys
import random
from collections import Counter
from typing import Callable

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, SIZE
from projects.hexca.hexca import CA1D, CA2D
from projects.hexca.rules import get_rule, RULES
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Одна строка глифов: список h → 3 текстовые строки
# ---------------------------------------------------------------------------

def render_row_glyphs(
    row: list[int],
    highlights: set[int] | None = None,
    color: bool = True,
    width: int | None = None,
) -> list[str]:
    """
    Список гексаграмм → три строки ASCII (3 строки × 3 символа на ячейку).

    highlights: индексы ячеек (не значения), которые выделить.
    width:      если задан, обрезать/дополнить до width ячеек.
    """
    if highlights is None:
        highlights = set()
    if width is not None:
        row = row[:width]

    glyphs = [render_glyph(h) for h in row]
    result: list[str] = ['', '', '']

    for i, (h, g) in enumerate(zip(row, glyphs)):
        for ri in range(3):
            cell = g[ri]
            if color:
                yc = yang_count(h)
                c = (_YANG_BG[yc] + _BOLD) if i in highlights else _YANG_ANSI[yc]
                cell = c + cell + _RESET
            result[ri] += cell + ' '

    return result


# ---------------------------------------------------------------------------
# 1D автомат: история как поток глифов
# ---------------------------------------------------------------------------

def render_ca1d_history_glyphs(
    ca: CA1D,
    max_steps: int | None = None,
    color: bool = True,
    show_step: bool = True,
) -> str:
    """
    История 1D КА как вертикальный поток строк глифов.
    Каждый временной шаг занимает 3 строки (высота одного глифа).
    """
    history = ca.history
    if max_steps is not None:
        history = history[:max_steps + 1]

    lines: list[str] = []
    for t, row in enumerate(history):
        rows = render_row_glyphs(row, color=color)
        prefix = f'{t:3d}│' if show_step else '   │'
        pad    = '   │' if show_step else '   │'
        lines.append(prefix + rows[0])
        lines.append(pad    + rows[1])
        lines.append(pad    + rows[2])
        lines.append('   │' + '─' * (len(row) * 4))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 1D автомат: компактная история (1 строка на шаг, средняя строка глифа)
# ---------------------------------------------------------------------------

def render_ca1d_history_compact(
    ca: CA1D,
    max_steps: int | None = None,
    color: bool = True,
) -> str:
    """
    Компактная история 1D КА: каждый шаг = одна строка из средних строк глифов.
    Видно глобальную динамику, как в традиционных КА правило-90/110.
    """
    history = ca.history
    if max_steps is not None:
        history = history[:max_steps + 1]

    lines: list[str] = []
    for t, row in enumerate(history):
        parts: list[str] = []
        for h in row:
            cell = render_glyph(h)[1]  # средняя строка
            if color:
                yc = yang_count(h)
                cell = _YANG_ANSI[yc] + cell + _RESET
            parts.append(cell)
        step_str = f'{t:3d}│' if t % 5 == 0 else '   │'
        lines.append(step_str + ' '.join(parts))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 1D автомат: разность (XOR) соседних шагов
# ---------------------------------------------------------------------------

def render_ca1d_diff_glyphs(
    ca: CA1D,
    max_steps: int | None = None,
    color: bool = True,
) -> str:
    """
    XOR соседних шагов истории: diff[t] = step[t] XOR step[t-1].
    Показывает, какие ячейки изменились и как.
    """
    history = ca.history
    if max_steps is not None:
        history = history[:max_steps + 1]

    lines: list[str] = []
    lines.append('  Δ-история (XOR шагов):')
    for t in range(1, len(history)):
        diff_row = [history[t][i] ^ history[t - 1][i] for i in range(len(history[t]))]
        nonzero = sum(1 for h in diff_row if h != 0)
        rows = render_row_glyphs(diff_row, color=color)
        prefix = f'Δ{t:2d}│'
        pad    = '    │'
        lines.append(prefix + rows[0] + f'  [{nonzero} изм.]')
        lines.append(pad    + rows[1])
        lines.append(pad    + rows[2])
        lines.append('    │' + '─' * (len(diff_row) * 4))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2D автомат: один кадр
# ---------------------------------------------------------------------------

def render_ca2d_frame_glyphs(
    ca: CA2D,
    color: bool = True,
    border: bool = True,
) -> str:
    """2D кадр как матрица глифов."""
    lines: list[str] = []
    if border:
        lines.append('  ┌' + '───' * ca.width + '┐')

    for y in range(ca.height):
        row = ca.grid[y]
        rows = render_row_glyphs(row, color=color)
        pref = '  │' if border else '  '
        suff = '│' if border else ''
        for ri in range(3):
            lines.append(pref + rows[ri] + suff)
        if border:
            lines.append('  │' + '   ' * ca.width + '│')

    if border:
        lines.append('  └' + '───' * ca.width + '┘')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2D автомат: несколько кадров рядом
# ---------------------------------------------------------------------------

def render_ca2d_frames(
    ca: CA2D,
    steps: list[int],
    color: bool = True,
) -> str:
    """
    Запустить 2D КА на max(steps) шагов и показать кадры steps рядом.
    """
    max_step = max(steps) if steps else 0
    frames: dict[int, list[list[int]]] = {}

    if 0 in steps:
        frames[0] = [list(row) for row in ca.grid]

    for t in range(1, max_step + 1):
        ca.step()
        if t in steps:
            frames[t] = [list(row) for row in ca.grid]

    # Заголовки
    lines: list[str] = []
    hdr = '  ' + ''.join(
        f'  Шаг {t:<3d}  ' + ' ' * (ca.width * 4 - 9)
        for t in steps if t in frames
    )
    lines.append(hdr)

    for y in range(ca.height):
        # Три строки глифов для каждого кадра, в ряд
        combined_rows: list[str] = ['', '', '']
        for t in steps:
            if t not in frames:
                continue
            row = frames[t][y]
            rows = render_row_glyphs(row, color=color)
            for ri in range(3):
                combined_rows[ri] += '  ' + rows[ri]
        for combined in combined_rows:
            lines.append('  ' + combined)
        lines.append('')  # разделитель строк

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Статистика эволюции
# ---------------------------------------------------------------------------

def ca1d_evolution_stats(ca: CA1D) -> str:
    """Статистика эволюции: средний ян-счёт, энтропия, уникальные состояния по шагам."""
    lines: list[str] = []
    lines.append(f'  {"Шаг":>5} {"ср.Ян":>7} {"уник.":>6} {"энтропия":>10}')
    lines.append('  ' + '─' * 35)

    import math
    for t, row in enumerate(ca.history):
        mean_yang = sum(yang_count(h) for h in row) / len(row)
        unique = len(set(row))
        # Шенноновская энтропия по значениям
        from collections import Counter
        cnt = Counter(row)
        n = len(row)
        entropy = -sum(c / n * math.log2(c / n) for c in cnt.values())
        lines.append(f'  {t:>5} {mean_yang:>7.2f} {unique:>6} {entropy:>10.3f}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# JSON-экспорт (для пайплайнов)
# ---------------------------------------------------------------------------

def _step_stats(row: list[int]) -> dict:
    """Статистика одного шага: энтропия, уникальные, среднее ян."""
    cnt = Counter(row)
    n = len(row)
    entropy = -sum(c / n * math.log2(c / n) for c in cnt.values()) if n > 0 else 0.0
    mean_yang = sum(yang_count(h) for h in row) / n
    return {
        'entropy': round(entropy, 4),
        'unique': len(cnt),
        'mean_yang': round(mean_yang, 3),
    }


def json_evolve(rule_name: str = 'xor_rule',
                width: int = 20,
                steps: int = 20,
                seed: int = 42) -> dict:
    """Запустить 1D КА и вернуть JSON с историей эволюции."""
    random.seed(seed)
    rule = get_rule(rule_name)
    ca = CA1D(width, rule)
    ca.run(steps)

    evolution_stats = []
    for t, row in enumerate(ca.history):
        s = _step_stats(row)
        s['step'] = t
        evolution_stats.append(s)

    # Определить тип аттрактора
    period_1 = (len(ca.history) >= 2 and ca.history[-1] == ca.history[-2])
    period_2 = (len(ca.history) >= 3 and ca.history[-1] == ca.history[-3])
    convergence_step = None
    for t in range(1, len(ca.history)):
        if ca.history[t] == ca.history[t - 1]:
            convergence_step = t
            break

    return {
        'command': 'evolve',
        'rule': rule_name,
        'width': width,
        'steps': steps,
        'seed': seed,
        'initial_state': ca.history[0],
        'final_state': ca.history[-1],
        'period_1': period_1,
        'period_2': period_2,
        'convergence_step': convergence_step,
        'evolution_stats': evolution_stats,
        'available_rules': list(RULES.keys()),
    }


def json_all_rules(width: int = 20,
                   steps: int = 20,
                   seed: int = 42) -> dict:
    """Запустить все правила и вернуть сравнительный JSON."""
    results = []
    for rule_name in RULES:
        data = json_evolve(rule_name, width=width, steps=steps, seed=seed)
        results.append({
            'rule': rule_name,
            'convergence_step': data['convergence_step'],
            'period_1': data['period_1'],
            'period_2': data['period_2'],
            'initial_entropy': data['evolution_stats'][0]['entropy'],
            'final_entropy': data['evolution_stats'][-1]['entropy'],
            'initial_unique': data['evolution_stats'][0]['unique'],
            'final_unique': data['evolution_stats'][-1]['unique'],
            'initial_yang_mean': data['evolution_stats'][0]['mean_yang'],
            'final_yang_mean': data['evolution_stats'][-1]['mean_yang'],
        })
    return {
        'command': 'all_rules',
        'width': width,
        'steps': steps,
        'seed': seed,
        'rules': results,
    }


_CA_JSON_DISPATCH = {
    'evolve':    lambda args: json_evolve(
                     rule_name=getattr(args, 'rule', 'xor_rule'),
                     width=getattr(args, 'width', 20),
                     steps=getattr(args, 'steps', 20),
                     seed=getattr(args, 'seed', 42),
                 ),
    'all-rules': lambda args: json_all_rules(
                     width=getattr(args, 'width', 20),
                     steps=getattr(args, 'steps', 20),
                     seed=getattr(args, 'seed', 42),
                 ),
    'stats':     lambda args: json_all_rules(
                     width=getattr(args, 'width', 20),
                     steps=getattr(args, 'steps', 30),
                     seed=getattr(args, 'seed', 42),
                 ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='ca_glyphs — КА на Q6 с визуализацией через глифы'
    )
    parser.add_argument('--json', action='store_true',
                        help='Машиночитаемый JSON-вывод (для пайплайнов)')
    sub = parser.add_subparsers(dest='cmd')

    # evolve — JSON-экспорт эволюции (SC-3 шаг 1)
    p_ev = sub.add_parser('evolve', help='1D КА: JSON-экспорт эволюции → пайплайн')
    p_ev.add_argument('--rule',  default='xor_rule', choices=list(RULES))
    p_ev.add_argument('--width', type=int, default=20)
    p_ev.add_argument('--steps', type=int, default=20)
    p_ev.add_argument('--seed',  type=int, default=42)

    # all-rules — JSON-экспорт всех правил
    p_ar = sub.add_parser('all-rules', help='Сравнить все правила → JSON')
    p_ar.add_argument('--width', type=int, default=20)
    p_ar.add_argument('--steps', type=int, default=20)
    p_ar.add_argument('--seed',  type=int, default=42)

    # 1D компактная история
    p1c = sub.add_parser('compact', help='1D КА: компактная история глифов (1 строка/шаг)')
    p1c.add_argument('--rule',  default='xor_rule', choices=list(RULES))
    p1c.add_argument('--width', type=int, default=16)
    p1c.add_argument('--steps', type=int, default=20)
    p1c.add_argument('--seed',  type=int, default=42)

    # 1D полная история
    p1f = sub.add_parser('history', help='1D КА: полная история (3 строки/шаг)')
    p1f.add_argument('--rule',   default='majority_vote', choices=list(RULES))
    p1f.add_argument('--width',  type=int, default=8)
    p1f.add_argument('--steps',  type=int, default=8)
    p1f.add_argument('--seed',   type=int, default=0)

    # 1D diff история
    p1d = sub.add_parser('diff', help='1D КА: XOR-история изменений')
    p1d.add_argument('--rule',   default='xor_rule', choices=list(RULES))
    p1d.add_argument('--width',  type=int, default=12)
    p1d.add_argument('--steps',  type=int, default=10)
    p1d.add_argument('--seed',   type=int, default=7)

    # 2D кадры
    p2f = sub.add_parser('frames', help='2D КА: несколько кадров рядом')
    p2f.add_argument('--rule',   default='majority_vote', choices=list(RULES))
    p2f.add_argument('--width',  type=int, default=6)
    p2f.add_argument('--height', type=int, default=6)
    p2f.add_argument('--steps',  nargs='+', type=int, default=[0, 1, 2, 3])
    p2f.add_argument('--seed',   type=int, default=1)

    # Статистика
    p_stat = sub.add_parser('stats', help='Статистика эволюции 1D КА')
    p_stat.add_argument('--rule',  default='xor_rule', choices=list(RULES))
    p_stat.add_argument('--width', type=int, default=20)
    p_stat.add_argument('--steps', type=int, default=30)
    p_stat.add_argument('--seed',  type=int, default=42)

    for p in sub.choices.values():
        p.add_argument('--no-color', action='store_true')

    args = parser.parse_args()
    color = not getattr(args, 'no_color', False)

    # JSON-режим: evolve, all-rules, stats
    if args.json and args.cmd in _CA_JSON_DISPATCH:
        print(json.dumps(_CA_JSON_DISPATCH[args.cmd](args), ensure_ascii=False, indent=2))
        sys.exit(0)

    if args.cmd == 'evolve':
        # Human-readable evolve: run and show compact history
        random.seed(args.seed)
        rule = get_rule(args.rule)
        ca = CA1D(args.width, rule)
        ca.run(args.steps)
        print(f'  evolve: правило={args.rule}  ширина={args.width}  шагов={args.steps}')
        print()
        print(render_ca1d_history_compact(ca, color=color))
        sys.exit(0)

    if args.cmd == 'all-rules':
        # Human-readable all-rules
        for rule_name in RULES:
            random.seed(args.seed)
            rule = get_rule(rule_name)
            ca = CA1D(args.width, rule)
            ca.run(args.steps)
            s0 = _step_stats(ca.history[0])
            sN = _step_stats(ca.history[-1])
            print(f'  {rule_name:<20}  H: {s0["entropy"]:.2f}→{sN["entropy"]:.2f}'
                  f'  uniq: {s0["unique"]}→{sN["unique"]}')
        sys.exit(0)

    if args.cmd == 'compact' or args.cmd is None:
        cmd = args if args.cmd else type('Args', (), {
            'rule': 'xor_rule', 'width': 16, 'steps': 20, 'seed': 42, 'no_color': False
        })()
        random.seed(cmd.seed)
        rule = get_rule(cmd.rule)
        ca = CA1D(cmd.width, rule)
        ca.run(cmd.steps)
        print(f'  1D КА  правило={cmd.rule}  ширина={cmd.width}  шагов={cmd.steps}')
        print(f'  Каждая строка = один шаг эволюции (средние строки глифов)')
        print()
        print(render_ca1d_history_compact(ca, color=color))

    elif args.cmd == 'history':
        random.seed(args.seed)
        rule = get_rule(args.rule)
        ca = CA1D(args.width, rule)
        ca.run(args.steps)
        print(f'  1D КА  правило={args.rule}  ширина={args.width}')
        print(f'  Каждые 3 строки = один шаг (полный глиф)')
        print()
        print(render_ca1d_history_glyphs(ca, color=color))

    elif args.cmd == 'diff':
        random.seed(args.seed)
        rule = get_rule(args.rule)
        ca = CA1D(args.width, rule)
        ca.run(args.steps)
        print(f'  XOR-история 1D КА  правило={args.rule}')
        print()
        print(render_ca1d_diff_glyphs(ca, color=color))

    elif args.cmd == 'frames':
        random.seed(args.seed)
        rule = get_rule(args.rule)
        ca = CA2D(args.width, args.height, rule)
        print(f'  2D КА  правило={args.rule}  {args.width}×{args.height}')
        print()
        print(render_ca2d_frames(ca, args.steps, color=color))

    elif args.cmd == 'stats':
        random.seed(args.seed)
        rule = get_rule(args.rule)
        ca = CA1D(args.width, rule)
        ca.run(args.steps)
        print(f'  Статистика 1D КА  правило={args.rule}  ширина={args.width}')
        print()
        print(ca1d_evolution_stats(ca))
