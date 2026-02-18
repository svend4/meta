#!/usr/bin/env python3
"""
hexca — клеточный автомат на графе Q6 (64 состояния)

Каждая клетка — гексаграмма (0..63). Переход задаётся правилом:
(текущая клетка, соседи) → новая клетка.

Поддерживает:
  - 1D решётку (лента шириной N)
  - 2D тор (W×H с периодическими граничными условиями)

Использование:
    python3 hexca.py                         # демо: 1D автомат
    python3 hexca.py --mode 2d --rule majority_vote --steps 10
    python3 hexca.py --mode 1d --rule xor_rule --width 32 --steps 20
    python3 hexca.py --list-rules
"""

from __future__ import annotations
import sys
import argparse
import random

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, to_bits, SIZE, render
from projects.hexca.rules import get_rule, RULES, RuleFn

# Символы для отображения клетки по числу ян-линий (0-6)
CELL_CHARS = ' ·░▒▓█▓'   # 7 символов: 0 ян → пробел, 6 ян → полный блок


def cell_char(h: int) -> str:
    return CELL_CHARS[yang_count(h)]


# ---------------------------------------------------------------------------
# 1D клеточный автомат
# ---------------------------------------------------------------------------

class CA1D:
    """
    Одномерная решётка гексаграмм с периодическими граничными условиями.
    Каждая клетка имеет 2 соседа (левый и правый).
    """

    def __init__(self, width: int, rule: RuleFn, init: list[int] | None = None) -> None:
        self.width = width
        self.rule = rule
        if init is not None:
            assert len(init) == width
            self.grid = list(init)
        else:
            self.grid = [random.randint(0, SIZE - 1) for _ in range(width)]
        self.generation = 0
        self.history: list[list[int]] = [list(self.grid)]

    def step(self) -> None:
        new_grid = []
        for i in range(self.width):
            left = self.grid[(i - 1) % self.width]
            right = self.grid[(i + 1) % self.width]
            new_cell = self.rule(self.grid[i], [left, right])
            new_grid.append(new_cell)
        self.grid = new_grid
        self.generation += 1
        self.history.append(list(self.grid))

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def render_row(self, row: list[int] | None = None) -> str:
        g = row if row is not None else self.grid
        return ''.join(cell_char(h) for h in g)

    def render_history(self) -> str:
        lines = []
        for i, row in enumerate(self.history):
            lines.append(f"{i:4d} │{self.render_row(row)}│")
        return '\n'.join(lines)

    def stats(self) -> dict:
        yang_sum = sum(yang_count(h) for h in self.grid)
        return {
            'generation': self.generation,
            'width': self.width,
            'mean_yang': yang_sum / self.width,
            'unique_states': len(set(self.grid)),
        }


# ---------------------------------------------------------------------------
# 2D клеточный автомат (тор)
# ---------------------------------------------------------------------------

class CA2D:
    """
    Двумерная решётка на торе (ширина × высота).
    Каждая клетка имеет 4 соседей (Фон Нейман: N, E, S, W).
    """

    def __init__(self,
                 width: int,
                 height: int,
                 rule: RuleFn,
                 init: list[list[int]] | None = None) -> None:
        self.width = width
        self.height = height
        self.rule = rule
        if init is not None:
            self.grid = [list(row) for row in init]
        else:
            self.grid = [
                [random.randint(0, SIZE - 1) for _ in range(width)]
                for _ in range(height)
            ]
        self.generation = 0

    def _get(self, y: int, x: int) -> int:
        return self.grid[y % self.height][x % self.width]

    def step(self) -> None:
        new_grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                nbrs = [
                    self._get(y - 1, x),  # север
                    self._get(y + 1, x),  # юг
                    self._get(y, x - 1),  # запад
                    self._get(y, x + 1),  # восток
                ]
                row.append(self.rule(self.grid[y][x], nbrs))
            new_grid.append(row)
        self.grid = new_grid
        self.generation += 1

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def render(self) -> str:
        lines = [f"  Поколение {self.generation}  ({self.width}×{self.height})"]
        lines.append('  ┌' + '─' * self.width + '┐')
        for row in self.grid:
            lines.append('  │' + ''.join(cell_char(h) for h in row) + '│')
        lines.append('  └' + '─' * self.width + '┘')
        return '\n'.join(lines)

    def stats(self) -> dict:
        flat = [self.grid[y][x] for y in range(self.height) for x in range(self.width)]
        yang_sum = sum(yang_count(h) for h in flat)
        total = self.width * self.height
        return {
            'generation': self.generation,
            'size': f'{self.width}×{self.height}',
            'mean_yang': yang_sum / total,
            'unique_states': len(set(flat)),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def demo_1d(rule_name: str, width: int, steps: int) -> None:
    rule = get_rule(rule_name)
    # Стартовое состояние: одна «горячая» клетка в центре
    init = [0] * width
    init[width // 2] = 42  # гексаграмма 42 = 101010
    ca = CA1D(width=width, rule=rule, init=init)

    print(f"\n  hexca 1D — правило: {rule_name}")
    print(f"  Ширина: {width}, шагов: {steps}")
    print(f"  Старт: центр = гексаграмма 42 (101010)")
    print()
    print(ca.render_history())
    ca.run(steps)
    print(ca.render_history()[ca.render_history().index('\n') + 1:])  # остаток

    s = ca.stats()
    print(f"\n  Поколение {s['generation']},  "
          f"среднее ян: {s['mean_yang']:.2f},  "
          f"уникальных состояний: {s['unique_states']}")


def demo_2d(rule_name: str, width: int, height: int, steps: int) -> None:
    rule = get_rule(rule_name)
    # Случайная инициализация
    random.seed(42)
    ca = CA2D(width=width, height=height, rule=rule)

    print(f"\n  hexca 2D — правило: {rule_name}")
    print(f"  Размер: {width}×{height}, шагов: {steps}")
    print()

    for step in range(steps + 1):
        if step > 0:
            ca.step()
        if step % max(1, steps // 5) == 0 or step == steps:
            print(ca.render())
            s = ca.stats()
            print(f"    среднее ян: {s['mean_yang']:.2f}, "
                  f"уникальных: {s['unique_states']}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='hexca — клеточный автомат на Q6 (64 состояния)'
    )
    parser.add_argument('--mode', choices=['1d', '2d'], default='1d',
                        help='Режим: 1d или 2d (по умолчанию 1d)')
    parser.add_argument('--rule', default='majority_vote',
                        help='Правило перехода (по умолчанию majority_vote)')
    parser.add_argument('--width', type=int, default=40,
                        help='Ширина решётки (по умолчанию 40)')
    parser.add_argument('--height', type=int, default=20,
                        help='Высота решётки для 2D (по умолчанию 20)')
    parser.add_argument('--steps', type=int, default=15,
                        help='Число шагов (по умолчанию 15)')
    parser.add_argument('--list-rules', action='store_true',
                        help='Показать список доступных правил')
    args = parser.parse_args()

    if args.list_rules:
        print("\n  Доступные правила:")
        for name in RULES:
            print(f"    {name}")
        print()
        return

    print("╔══════════════════════════════════════╗")
    print("║      hexca — Клеточный автомат Q6    ║")
    print("╚══════════════════════════════════════╝")
    print("  Символы: ' ' = 0 ян, '·' = 1, '░' = 2, '▒' = 3, '▓' = 4, '█' = 5, '▓' = 6")

    if args.mode == '1d':
        demo_1d(args.rule, args.width, args.steps)
    else:
        demo_2d(args.rule, args.width, args.height, args.steps)


if __name__ == '__main__':
    main()
