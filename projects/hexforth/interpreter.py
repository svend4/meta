#!/usr/bin/env python3
"""
hexforth — стековый язык на графе Q6

Концепция: 6 слотов стека = 6 бит гексаграммы.
Слот стека = 0 (ложь) или 1 (истина).
Состояние стека = гексаграмма в Q6.

Каждое слово меняет ровно 1 бит → переход по ребру Q6.

Встроенные слова:
  FLIP-0 .. FLIP-5   — переворот слота 0..5
  SET-0  .. SET-5    — установить слот в 1
  CLR-0  .. CLR-5    — установить слот в 0
  NOP                — ничего не делать
  DUP                — дублировать верхний слот (FLIP-0 + SET-0? нет — это мета-операция)
  DEBUG              — вывести текущее состояние
  ASSERT-EQ <N>      — проверить, что текущая гексаграмма = N (для тестов)
  GOTO <N>           — перейти к гексаграмме N через shortest_path (макрос)
  PRINT              — вывести текущую гексаграмму

Синтаксис файла .hf:
  # комментарий
  СЛОВО  # inline комментарий
  DEFINE имя : слово1 слово2 ; # пользовательское слово

Пример программы:
  # Переход из 0 в 42
  GOTO 42
  DEBUG
"""

from __future__ import annotations
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    flip, shortest_path, to_bits, yang_count, neighbors,
    render, antipode, hamming, SIZE,
)

# ---------------------------------------------------------------------------
# Ошибки
# ---------------------------------------------------------------------------

class HexForthError(Exception):
    pass


# ---------------------------------------------------------------------------
# Интерпретатор
# ---------------------------------------------------------------------------

class HexForth:
    """Интерпретатор языка HexForth."""

    def __init__(self, start: int = 0, verbose: bool = False) -> None:
        if not 0 <= start < SIZE:
            raise HexForthError(f"Стартовая гексаграмма должна быть 0-63, получено {start}")
        self.state: int = start         # текущая гексаграмма = состояние «стека»
        self.trace: list[int] = [start] # путь по Q6
        self.verbose: bool = verbose
        self.definitions: dict[str, list[str]] = {}  # пользовательские слова
        self.output: list[str] = []      # буфер вывода

    # --- Выполнение одного токена -------------------------------------------

    def _emit(self, msg: str) -> None:
        self.output.append(msg)
        if self.verbose:
            print(msg)

    def _transition(self, new_state: int) -> None:
        """Переход в новое состояние (должен быть соседом в Q6)."""
        if new_state not in neighbors(self.state) and new_state != self.state:
            raise HexForthError(
                f"Недопустимый переход: {self.state} → {new_state} "
                f"(не соседи в Q6, расстояние = {hamming(self.state, new_state)})"
            )
        self.state = new_state
        self.trace.append(new_state)

    def execute_word(self, word: str) -> None:
        """Выполнить одно слово."""
        word = word.upper()

        # FLIP-N
        m = re.fullmatch(r'FLIP-([0-5])', word)
        if m:
            bit = int(m.group(1))
            self._transition(flip(self.state, bit))
            return

        # SET-N (установить бит N в 1)
        m = re.fullmatch(r'SET-([0-5])', word)
        if m:
            bit = int(m.group(1))
            if not (self.state >> bit) & 1:
                self._transition(flip(self.state, bit))
            return

        # CLR-N (установить бит N в 0)
        m = re.fullmatch(r'CLR-([0-5])', word)
        if m:
            bit = int(m.group(1))
            if (self.state >> bit) & 1:
                self._transition(flip(self.state, bit))
            return

        if word == 'NOP':
            return

        if word == 'DEBUG':
            info = (f"  [DEBUG] гексаграмма {self.state:2d} ({to_bits(self.state)})  "
                    f"ян={yang_count(self.state)}  "
                    f"шаг={len(self.trace)-1}")
            self._emit(info)
            return

        if word == 'PRINT':
            self._emit(f"  {self.state:2d} ({to_bits(self.state)})")
            return

        if word == 'RENDER':
            lines = render(self.state).split('\n')
            for ln in lines:
                self._emit(f"  {ln}")
            return

        if word == 'ANTIPODE':
            # Переходим к антиподу через shortest_path
            target = antipode(self.state)
            self._execute_goto(target)
            return

        # ASSERT-EQ N
        m = re.fullmatch(r'ASSERT-EQ\s+(\d+)', word)
        if m:
            expected = int(m.group(1))
            if self.state != expected:
                raise HexForthError(
                    f"ASSERT-EQ {expected} провалился: текущая гексаграмма = {self.state}"
                )
            self._emit(f"  [OK] ASSERT-EQ {expected}")
            return

        # GOTO N (макрос: shortest_path)
        m = re.fullmatch(r'GOTO\s+(\d+)', word)
        if m:
            target = int(m.group(1))
            if not 0 <= target < SIZE:
                raise HexForthError(f"GOTO: цель {target} вне диапазона 0-63")
            self._execute_goto(target)
            return

        # Пользовательское слово
        if word in self.definitions:
            body = self.definitions[word]
            ARG_WORDS = {'GOTO', 'ASSERT-EQ'}
            j = 0
            while j < len(body):
                bw = body[j].upper()
                if bw in ARG_WORDS:
                    if j + 1 >= len(body):
                        raise HexForthError(f"'{bw}' требует аргумент")
                    self.execute_word(f"{bw} {body[j + 1]}")
                    j += 2
                else:
                    self.execute_word(bw)
                    j += 1
            return

        raise HexForthError(f"Неизвестное слово: '{word}'")

    def _execute_goto(self, target: int) -> None:
        """Выполнить GOTO через BFS shortest_path."""
        path = shortest_path(self.state, target)
        for step in path[1:]:   # path[0] = текущая позиция
            self._transition(step)

    # --- Парсер -------------------------------------------------------------

    def parse(self, source: str) -> list[str]:
        """Разобрать исходный код в список токенов."""
        tokens: list[str] = []
        i = 0
        lines = source.splitlines()

        for line in lines:
            # Удалить комментарий
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if not line:
                continue

            # Обработка DEFINE
            if line.upper().lstrip().startswith('DEFINE'):
                # DEFINE имя : слово1 слово2 ... ; [остальные токены]
                parts = line.split()
                if ':' not in parts or ';' not in parts:
                    raise HexForthError(f"Синтаксис: DEFINE имя : слово1 ... ;")
                name = parts[1].upper()
                colon_idx = parts.index(':')
                semi_idx = parts.index(';')
                if semi_idx <= colon_idx:
                    raise HexForthError(f"Синтаксис: DEFINE имя : слово1 ... ;")
                body = parts[colon_idx + 1:semi_idx]
                self.definitions[name] = body
                # Токены после ';' добавляем как обычные
                rest = parts[semi_idx + 1:]
                tokens.extend(rest)
                continue

            # Обычные токены
            tokens.extend(line.split())

        return tokens

    def run(self, source: str) -> 'HexForth':
        """Выполнить программу."""
        tokens = self.parse(source)
        # Слова, потребляющие следующий токен как аргумент
        ARG_WORDS = {'GOTO', 'ASSERT-EQ'}
        i = 0
        while i < len(tokens):
            word = tokens[i].upper()
            if word in ARG_WORDS:
                if i + 1 >= len(tokens):
                    raise HexForthError(f"'{word}' требует аргумент")
                compound = f"{word} {tokens[i + 1]}"
                self.execute_word(compound)
                i += 2
            else:
                self.execute_word(word)
                i += 1
        return self

    def run_file(self, path: str) -> 'HexForth':
        with open(path, encoding='utf-8') as f:
            source = f.read()
        return self.run(source)

    def summary(self) -> str:
        lines = [
            f"  Конечная гексаграмма: {self.state:2d} ({to_bits(self.state)})",
            f"  Пройдено шагов:       {len(self.trace) - 1}",
            f"  Путь: " + ' → '.join(str(h) for h in self.trace),
        ]
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI (интерактивный REPL и запуск файла)
# ---------------------------------------------------------------------------

def repl(start: int = 0) -> None:
    print("╔══════════════════════════════════════╗")
    print("║      HexForth REPL                   ║")
    print("╚══════════════════════════════════════╝")
    print("  Слова: FLIP-0..5, SET-0..5, CLR-0..5, GOTO N, DEBUG, RENDER, ANTIPODE, NOP, quit")
    print()

    interp = HexForth(start=start, verbose=True)
    interp.verbose = False  # вывод через output

    while True:
        try:
            line = input(f"  [{interp.state:2d}|{to_bits(interp.state)}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.lower() in ('quit', 'q', 'exit'):
            break

        if not line:
            continue

        try:
            interp.run(line)
            for msg in interp.output:
                print(msg)
            interp.output.clear()
        except HexForthError as e:
            print(f"  Ошибка: {e}")

    print()
    print(interp.summary())


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description='hexforth — стековый язык на графе Q6'
    )
    parser.add_argument('file', nargs='?', help='файл .hf для выполнения')
    parser.add_argument('--start', type=int, default=0, help='стартовая гексаграмма (0-63)')
    parser.add_argument('--verbose', action='store_true', help='подробный вывод')
    parser.add_argument('--repl', action='store_true', help='интерактивный режим')
    args = parser.parse_args()

    if args.repl or not args.file:
        repl(start=args.start)
        return

    interp = HexForth(start=args.start, verbose=args.verbose)
    try:
        interp.run_file(args.file)
    except HexForthError as e:
        print(f"  Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

    for msg in interp.output:
        print(msg)
    print()
    print(interp.summary())


if __name__ == '__main__':
    main()
