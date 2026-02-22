#!/usr/bin/env python3
"""
hexpath CLI — игра «Стратегия на графе Q6»

Режимы:
  python3 cli.py            — человек vs AI
  python3 cli.py --pvp      — человек vs человек
  python3 cli.py --ai-vs-ai — демо AI vs AI
  python3 cli.py --no-capture — без режима захвата
"""

import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import neighbors, render, to_bits, shortest_path, hamming
from projects.hexpath.game import (
    GameState, GameResult, Player, new_game, best_move,
)


def fmt_hex(h: int) -> str:
    return f"{h:2d}({to_bits(h)})"


def draw_board(state: GameState) -> None:
    """Вывести упрощённую карту состояния."""
    print()
    a_sym = Player.A.symbol()
    b_sym = Player.B.symbol()
    print(f"  {a_sym} Игрок A: {fmt_hex(state.pos_a)}  →  цель {fmt_hex(state.target_a)}"
          f"  (расстояние {hamming(state.pos_a, state.target_a)})")
    print(f"  {b_sym} Игрок B: {fmt_hex(state.pos_b)}  →  цель {fmt_hex(state.target_b)}"
          f"  (расстояние {hamming(state.pos_b, state.target_b)})")

    if state.capture_mode:
        cap_a = sum(1 for p in state.captured.values() if p == Player.A)
        cap_b = sum(1 for p in state.captured.values() if p == Player.B)
        print(f"  Захвачено: A={cap_a}  B={cap_b}")

    print()
    # Показать текущую позицию активного игрока
    p = state.current_player
    pos = state.current_pos()
    print(f"  Ход игрока {p.symbol()} ({p.value})  —  позиция {fmt_hex(pos)}")
    r = render(pos).split('\n')
    for line in r:
        print(f"    {line}")
    print()

    moves = state.legal_moves()
    if moves:
        print("  Допустимые ходы:")
        for i, nb in enumerate(neighbors(pos)):
            if nb in moves:
                marker = '*' if nb == state.current_target() else ' '
                dist = hamming(nb, state.current_target())
                print(f"    [{(pos ^ nb).bit_length() - 1}] {fmt_hex(nb)}"
                      f"  (до цели: {dist}){marker}")
    else:
        print("  Нет допустимых ходов!")
    print()


def announce_result(state: GameState) -> None:
    res = state.result()
    print()
    print("═" * 40)
    if res == GameResult.A_WINS:
        print(f"  Победил игрок A {Player.A.symbol()}!")
    elif res == GameResult.B_WINS:
        print(f"  Победил игрок B {Player.B.symbol()}!")
    else:
        print("  Ничья!")
    print(f"  Путь A ({len(state.history_a)-1} ходов): "
          + " → ".join(fmt_hex(h) for h in state.history_a))
    print(f"  Путь B ({len(state.history_b)-1} ходов): "
          + " → ".join(fmt_hex(h) for h in state.history_b))
    print("═" * 40)


def human_move(state: GameState) -> GameState:
    """Запросить ход у человека."""
    moves = state.legal_moves()
    pos = state.current_pos()

    while True:
        raw = input(f"  Введите номер черты (0-5) или 'h' для помощи: ").strip().lower()

        if raw == 'h':
            print("  Введите цифру от 0 до 5 — номер черты для переворота.")
            print(f"  Текущая позиция: {fmt_hex(pos)}")
            for i in range(6):
                nb = pos ^ (1 << i)
                status = 'OK' if nb in moves else 'заблокировано'
                print(f"    {i}: → {fmt_hex(nb)}  [{status}]")
            continue

        if raw == 'q':
            print("  Выход из игры.")
            sys.exit(0)

        try:
            bit = int(raw)
            if not 0 <= bit <= 5:
                raise ValueError
        except ValueError:
            print("  Введите число 0-5.")
            continue

        destination = pos ^ (1 << bit)
        if destination not in moves:
            if state.capture_mode:
                owner = state.captured.get(destination)
                if owner:
                    print(f"  Узел {fmt_hex(destination)} захвачен игроком {owner.value}.")
                    continue
            print(f"  Ход недопустим.")
            continue

        return state.make_move(destination)


def ai_move(state: GameState, depth: int = 4) -> GameState:
    """Ход AI."""
    print(f"  AI думает...", end='', flush=True)
    move = best_move(state, depth=depth)
    bit = (state.current_pos() ^ move).bit_length() - 1
    print(f"\r  AI ходит: черта {bit}  →  {fmt_hex(move)}        ")
    return state.make_move(move)


def play(
    pvp: bool = False,
    ai_vs_ai: bool = False,
    capture: bool = True,
    ai_depth: int = 4,
) -> None:
    print("╔══════════════════════════════════════╗")
    print("║      hexpath — Стратегия на Q6       ║")
    print("╚══════════════════════════════════════╝")

    if capture:
        print("  Режим: захват узлов включён")
    else:
        print("  Режим: захват узлов выключен")

    if pvp:
        print("  Режим: человек vs человек")
    elif ai_vs_ai:
        print("  Режим: AI vs AI (демо)")
    else:
        print("  Режим: человек (A) vs AI (B)")

    print()
    print("  Цель: первым достичь стартовой позиции соперника")
    print(f"  A стартует с  0 (000000), идёт к 63 (111111)")
    print(f"  B стартует с 63 (111111), идёт к  0 (000000)")
    print()

    state = new_game(pos_a=0, pos_b=63, capture_mode=capture)

    while not state.is_over():
        draw_board(state)

        p = state.current_player
        is_human = (pvp) or (not ai_vs_ai and p == Player.A)

        if is_human:
            state = human_move(state)
        else:
            state = ai_move(state, depth=ai_depth)
            if ai_vs_ai:
                input("  [Enter для продолжения]")

    draw_board(state)
    announce_result(state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='hexpath — стратегическая игра на графе Q6'
    )
    parser.add_argument('--pvp', action='store_true', help='человек vs человек')
    parser.add_argument('--ai-vs-ai', action='store_true', help='AI vs AI (демо)')
    parser.add_argument('--no-capture', action='store_true', help='без режима захвата')
    parser.add_argument('--depth', type=int, default=4, help='глубина AI (по умолчанию 4)')
    args = parser.parse_args()

    play(
        pvp=args.pvp,
        ai_vs_ai=args.ai_vs_ai,
        capture=not args.no_capture,
        ai_depth=args.depth,
    )


if __name__ == '__main__':
    main()
