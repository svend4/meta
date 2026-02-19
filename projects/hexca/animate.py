#!/usr/bin/env python3
"""
hexca/animate.py — текстовая анимация клеточного автомата Q6

Режимы:
  1D-анимация: прокрутка строк вниз (как осциллограф, пространство×время)
  2D-анимация: обновление кадра на месте через ANSI-escape (очистка экрана)

Использование:
    python3 animate.py                                    # 1D, xor_rule
    python3 animate.py --mode 2d --rule conway_b3s23      # 2D Conway
    python3 animate.py --mode 2d --rule majority_vote --fps 4
    python3 animate.py --mode 1d --rule xor_rule --width 60 --steps 40
    python3 animate.py --list-rules

Клавиши (только 2D): Ctrl+C для выхода.
"""

from __future__ import annotations
import sys
import time
import argparse
import random

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexca.hexca import CA1D, CA2D, cell_char, CELL_CHARS
from projects.hexca.rules import get_rule, RULES


# ANSI-escape коды
_CLEAR  = '\033[2J\033[H'    # очистить экран и перейти в (0,0)
_HIDE   = '\033[?25l'        # скрыть курсор
_SHOW   = '\033[?25h'        # показать курсор
_BOLD   = '\033[1m'
_RESET  = '\033[0m'
_DIM    = '\033[2m'


def _color_for_yang(yang: int) -> str:
    """ANSI-цвет по числу ян-линий (0-6)."""
    colors = [
        '\033[90m',    # 0: тёмно-серый (пусто)
        '\033[34m',    # 1: синий
        '\033[36m',    # 2: голубой
        '\033[32m',    # 3: зелёный
        '\033[33m',    # 4: жёлтый
        '\033[31m',    # 5: красный
        '\033[35m',    # 6: фиолетовый
    ]
    return colors[yang % len(colors)]


def _colored_cell(h: int, use_color: bool) -> str:
    from libs.hexcore.hexcore import yang_count
    ch = cell_char(h)
    if not use_color:
        return ch
    yang = yang_count(h)
    return f"{_color_for_yang(yang)}{ch}{_RESET}"


# ---------------------------------------------------------------------------
# 1D-анимация (space×time диаграмма, строки прокручиваются вниз)
# ---------------------------------------------------------------------------

def animate_1d(
    rule_name: str,
    width: int,
    steps: int,
    fps: float,
    color: bool,
    init_mode: str,
) -> None:
    rule = get_rule(rule_name)

    # Начальное состояние
    if init_mode == 'center':
        init = [0] * width
        init[width // 2] = 42
    elif init_mode == 'random':
        init = [random.randint(0, 63) for _ in range(width)]
    else:  # single
        init = [0] * width
        init[width // 2] = 1

    ca = CA1D(width=width, rule=rule, init=init)
    delay = 1.0 / fps if fps > 0 else 0

    if color:
        sys.stdout.write(_HIDE)

    try:
        print(f"{_BOLD}hexca 1D  rule={rule_name}  width={width}{_RESET}")
        print(f"{'─' * (width + 2)}")

        # Вывести начальное состояние
        row = '│' + ''.join(_colored_cell(h, color) for h in ca.grid) + '│'
        print(f"{_DIM}{0:4d}{_RESET} {row}")

        for step in range(steps):
            ca.step()
            row = '│' + ''.join(_colored_cell(h, color) for h in ca.grid) + '│'
            stats = ca.stats()
            info = f"  ян≈{stats['mean_yang']:.1f} u={stats['unique_states']}"
            print(f"{_DIM}{step+1:4d}{_RESET} {row}{_DIM}{info}{_RESET}")
            if delay > 0:
                time.sleep(delay)

        print(f"{'─' * (width + 2)}")
        print(f"Готово. {steps} шагов, правило: {rule_name}")

    except KeyboardInterrupt:
        print(f"\n{_RESET}Прервано.")
    finally:
        if color:
            sys.stdout.write(_SHOW)
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# 2D-анимация (обновление кадра на месте)
# ---------------------------------------------------------------------------

def animate_2d(
    rule_name: str,
    width: int,
    height: int,
    steps: int,
    fps: float,
    color: bool,
    init_mode: str,
) -> None:
    rule = get_rule(rule_name)
    delay = 1.0 / fps if fps > 0 else 0

    # Начальное состояние
    if init_mode == 'random':
        random.seed(None)
        init = None  # CA2D.__init__ сгенерирует случайно
    elif init_mode == 'center':
        init = [[0] * width for _ in range(height)]
        init[height // 2][width // 2] = 42
    else:
        init = [[0] * width for _ in range(height)]
        init[height // 2][width // 2] = 1

    random.seed(42)
    ca = CA2D(width=width, height=height, rule=rule, init=init)

    if color:
        sys.stdout.write(_HIDE)

    try:
        for step in range(steps + 1):
            if step > 0:
                ca.step()

            # Построить кадр
            lines: list[str] = []
            lines.append(
                f"{_BOLD}hexca 2D{_RESET}  rule={rule_name}  {width}×{height}"
                f"  поколение {ca.generation}"
            )
            lines.append('┌' + '─' * width + '┐')

            from libs.hexcore.hexcore import yang_count
            for row in ca.grid:
                cells = ''.join(_colored_cell(h, color) for h in row)
                lines.append('│' + cells + '│')

            lines.append('└' + '─' * width + '┘')

            s = ca.stats()
            lines.append(
                f"  ян≈{s['mean_yang']:.2f}  уник={s['unique_states']}  "
                f"[Ctrl+C для выхода]"
            )

            # Вывести кадр
            sys.stdout.write(_CLEAR)
            sys.stdout.write('\n'.join(lines) + '\n')
            sys.stdout.flush()

            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(_SHOW)
        sys.stdout.write('\n')
        sys.stdout.flush()

    print(f"Остановлено на поколении {ca.generation}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='hexca/animate.py — анимация клеточного автомата Q6'
    )
    parser.add_argument('--mode', choices=['1d', '2d'], default='1d',
                        help='Режим анимации (по умолчанию 1d)')
    parser.add_argument('--rule', default='xor_rule',
                        help='Правило перехода (по умолчанию xor_rule)')
    parser.add_argument('--width', type=int, default=60,
                        help='Ширина (по умолчанию 60)')
    parser.add_argument('--height', type=int, default=20,
                        help='Высота для 2D (по умолчанию 20)')
    parser.add_argument('--steps', type=int, default=30,
                        help='Число шагов (по умолчанию 30)')
    parser.add_argument('--fps', type=float, default=4.0,
                        help='Кадров в секунду (0 = максимально быстро)')
    parser.add_argument('--no-color', action='store_true',
                        help='Отключить цветной вывод')
    parser.add_argument('--init', choices=['center', 'random', 'single'],
                        default='center',
                        help='Начальное состояние (по умолчанию center)')
    parser.add_argument('--list-rules', action='store_true',
                        help='Показать доступные правила и выйти')
    args = parser.parse_args()

    if args.list_rules:
        print("\n  Доступные правила:")
        for name in RULES:
            print(f"    {name}")
        print()
        return

    color = not args.no_color

    if args.mode == '1d':
        animate_1d(
            rule_name=args.rule,
            width=args.width,
            steps=args.steps,
            fps=args.fps,
            color=color,
            init_mode=args.init,
        )
    else:
        animate_2d(
            rule_name=args.rule,
            width=args.width,
            height=args.height,
            steps=args.steps,
            fps=args.fps,
            color=color,
            init_mode=args.init,
        )


if __name__ == '__main__':
    main()
