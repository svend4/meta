#!/usr/bin/env python3
"""
hexnav — интерактивный CLI навигатор по графу Q6

Управление:
  0-5        — перевернуть черту с номером (переход в соседа)
  g <цель>   — показать кратчайший путь до гексаграммы
  a          — показать антипод
  i          — информация о текущей гексаграмме
  h          — история пути
  e <файл>   — экспорт пути в JSON
  r          — сбросить на начало
  q          — выйти
"""

import sys
import json
import argparse

# Добавляем корень монорепо в путь
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    neighbors, hamming, flip, shortest_path,
    antipode, describe, render, to_bits, yang_count,
    upper_trigram, lower_trigram, SIZE,
)

TRIGRAM_NAMES = {
    0b000: '☷ Кунь (Земля)',
    0b001: '☶ Гэнь (Гора)',
    0b010: '☵ Кань (Вода)',
    0b011: '☴ Сунь (Ветер)',
    0b100: '☳ Чжэнь (Гром)',
    0b101: '☲ Ли (Огонь)',
    0b110: '☱ Дуй (Озеро)',
    0b111: '☰ Цянь (Небо)',
}


def fmt_hex(h: int) -> str:
    """Короткий идентификатор гексаграммы: номер и биты."""
    return f"{h:2d} ({to_bits(h)})"


def print_current(h: int) -> None:
    """Вывести текущую гексаграмму и её соседей."""
    print()
    print(f"  Гексаграмма {fmt_hex(h)}")
    print("  " + "─" * 24)
    r = render(h).split('\n')
    for line in r:
        print(f"  {line}")
    info = describe(h)
    ut = TRIGRAM_NAMES.get(info['upper_tri'], str(info['upper_tri']))
    lt = TRIGRAM_NAMES.get(info['lower_tri'], str(info['lower_tri']))
    print(f"  Ян: {info['yang']}  Инь: {info['yin']}")
    print(f"  Верх: {ut}   Низ: {lt}")
    print(f"  Антипод: {fmt_hex(info['antipode'])}")
    print()
    print("  Переходы (нажмите 0-5 для смены черты):")
    for i, nb in enumerate(neighbors(h)):
        diff_bit = i
        mark = '↑' if ((nb >> i) & 1) else '↓'
        print(f"    [{i}] черта {i} {mark}  →  гексаграмма {fmt_hex(nb)}")
    print()


def print_path(path: list[int]) -> None:
    """Вывести путь в виде цепочки."""
    if not path:
        print("  (путь пуст)")
        return
    parts = [fmt_hex(h) for h in path]
    print("  Путь: " + " → ".join(parts))
    print(f"  Шагов: {len(path) - 1}")


def run(start: int) -> None:
    current = start
    history: list[int] = [start]

    print("╔══════════════════════════════════════╗")
    print("║        hexnav — навигатор Q6         ║")
    print("╚══════════════════════════════════════╝")
    print("  Введите 0-5 для перехода, 'h' для помощи, 'q' для выхода")

    print_current(current)

    while True:
        try:
            raw = input("hexnav> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0]

        # Переход по черте
        if cmd in ('0', '1', '2', '3', '4', '5'):
            bit = int(cmd)
            current = flip(current, bit)
            history.append(current)
            print_current(current)

        elif cmd == 'g':
            if len(parts) < 2:
                print("  Использование: g <цель 0-63>")
                continue
            try:
                target = int(parts[1])
                if not 0 <= target < SIZE:
                    raise ValueError
            except ValueError:
                print(f"  Ошибка: цель должна быть числом 0-63")
                continue
            path = shortest_path(current, target)
            print()
            print(f"  Кратчайший путь {fmt_hex(current)} → {fmt_hex(target)}:")
            print_path(path)
            if len(path) > 1:
                print(f"  Черты для переворота: {[bin(path[i] ^ path[i+1]).count('1') - 1 for i in range(len(path)-1)]}")
                flips = []
                for i in range(len(path) - 1):
                    diff = path[i] ^ path[i + 1]
                    flips.append(diff.bit_length() - 1)
                print(f"  Последовательность ходов: {flips}")
            print()

        elif cmd == 'a':
            ant = antipode(current)
            print(f"\n  Антипод: {fmt_hex(ant)}")
            path = shortest_path(current, ant)
            print_path(path)
            print()

        elif cmd == 'i':
            info = describe(current)
            print()
            print(f"  Гексаграмма {fmt_hex(current)}")
            print(f"  Биты:    {info['bits']}")
            print(f"  Ян:      {info['yang']}")
            print(f"  Инь:     {info['yin']}")
            print(f"  Верх:    {TRIGRAM_NAMES.get(info['upper_tri'])}")
            print(f"  Низ:     {TRIGRAM_NAMES.get(info['lower_tri'])}")
            print(f"  Антипод: {fmt_hex(info['antipode'])}")
            print(f"  Соседи:  {[fmt_hex(n) for n in info['neighbors']]}")
            print()

        elif cmd == 'h':
            print()
            print("  0-5    — перевернуть черту (переход к соседу)")
            print("  g <N>  — кратчайший путь до гексаграммы N (0-63)")
            print("  a      — показать антипод и путь до него")
            print("  i      — полная информация о текущей гексаграмме")
            print("  hist   — показать историю пути")
            print("  e <f>  — экспорт пути в JSON-файл")
            print("  r      — сбросить путь (вернуться к старту)")
            print("  q      — выйти")
            print()

        elif cmd == 'hist':
            print()
            print(f"  История ({len(history)} узлов):")
            print_path(history)
            print()

        elif cmd == 'e':
            filename = parts[1] if len(parts) > 1 else 'path.json'
            data = {
                'start': start,
                'current': current,
                'path': history,
                'path_bits': [to_bits(h) for h in history],
                'length': len(history) - 1,
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  Путь сохранён в {filename}")

        elif cmd == 'r':
            current = start
            history = [start]
            print(f"  Сброс. Возврат к {fmt_hex(start)}")
            print_current(current)

        elif cmd == 'q':
            break

        else:
            print(f"  Неизвестная команда: '{cmd}'. Введите 'h' для помощи.")

    print(f"\n  Итоговый путь ({len(history) - 1} шагов):")
    print_path(history)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='hexnav — интерактивный навигатор по графу Q6 (64 гексаграммы)'
    )
    parser.add_argument(
        'start', nargs='?', type=int, default=0,
        help='Стартовая гексаграмма (0-63, по умолчанию 0)'
    )
    args = parser.parse_args()

    if not 0 <= args.start < SIZE:
        parser.error(f"Стартовая гексаграмма должна быть в диапазоне 0-63, получено {args.start}")

    run(args.start)


if __name__ == '__main__':
    main()
