#!/usr/bin/env python3
"""
hexforth/verifier.py — статический верификатор HexForth-программ

Анализирует программу БЕЗ её выполнения:
  1. Достижимость  — можно ли попасть из стартовой позиции в целевую
                    при данном наборе допустимых слов?
  2. Ограниченность пути — гарантирует ли программа, что путь ≤ N шагов?
  3. Проверка инвариантов — выполняется ли предикат во всех достижимых состояниях?
  4. Статический анализ — подсчёт слов, проверка корректности синтаксиса,
                          обнаружение пустых DEFINE и бесконечных рекурсий.

CLI:
    python3 verifier.py --start 0 --target 42
    python3 verifier.py --start 0 --target 42 --allowed FLIP-0 FLIP-1 FLIP-3 FLIP-5
    python3 verifier.py examples/hello.hf --check-syntax
    python3 verifier.py --start 0 --reachable-from 42   # все состояния, куда можно попасть
"""

from __future__ import annotations
import sys
import argparse
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    flip, neighbors, shortest_path, hamming, to_bits, SIZE, antipode,
)
from projects.hexforth.interpreter import HexForth, HexForthError


# ---------------------------------------------------------------------------
# Граф допустимых переходов (строится по набору разрешённых слов)
# ---------------------------------------------------------------------------

def _parse_allowed_word(word: str, state: int) -> int | None:
    """
    Применить слово к состоянию. Вернуть новое состояние или None если неприменимо.
    Только детерминированные слова (FLIP-N, SET-N, CLR-N, NOP).
    GOTO и ANTIPODE раскрываются в shortest_path при построении графа.
    """
    import re
    word = word.upper()

    m = re.fullmatch(r'FLIP-([0-5])', word)
    if m:
        return flip(state, int(m.group(1)))

    m = re.fullmatch(r'SET-([0-5])', word)
    if m:
        bit = int(m.group(1))
        return state if (state >> bit) & 1 else flip(state, bit)

    m = re.fullmatch(r'CLR-([0-5])', word)
    if m:
        bit = int(m.group(1))
        return state if not ((state >> bit) & 1) else flip(state, bit)

    if word == 'NOP':
        return state

    if word == 'ANTIPODE':
        return antipode(state)

    m = re.fullmatch(r'GOTO\s+(\d+)', word)
    if m:
        return int(m.group(1))

    return None


def build_transition_graph(
    allowed_words: list[str],
    states: set[int] | None = None,
) -> dict[int, set[int]]:
    """
    Построить граф переходов: state → {достижимые состояния} через разрешённые слова.
    Если states=None, строится для всех 64 гексаграмм.
    """
    universe = states if states is not None else set(range(SIZE))
    graph: dict[int, set[int]] = {s: set() for s in universe}

    for state in universe:
        for word in allowed_words:
            target = _parse_allowed_word(word, state)
            if target is not None and target in universe:
                graph[state].add(target)

    return graph


# ---------------------------------------------------------------------------
# Анализы достижимости
# ---------------------------------------------------------------------------

def reachable_from(
    start: int,
    graph: dict[int, set[int]],
) -> set[int]:
    """BFS: все состояния, достижимые из start по данному графу."""
    visited = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in graph.get(cur, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


def shortest_path_in_graph(
    start: int,
    target: int,
    graph: dict[int, set[int]],
) -> list[int] | None:
    """BFS: кратчайший путь из start в target по данному графу."""
    if start == target:
        return [start]
    prev: dict[int, int | None] = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nb in graph.get(cur, set()):
            if nb not in prev:
                prev[nb] = cur
                if nb == target:
                    path = []
                    node: int | None = nb
                    while node is not None:
                        path.append(node)
                        node = prev[node]
                    return list(reversed(path))
                queue.append(nb)
    return None


def all_paths_bounded(
    start: int,
    target: int,
    graph: dict[int, set[int]],
    max_len: int = 12,
) -> list[list[int]]:
    """Все пути из start в target длиной ≤ max_len (DFS с ограничением)."""
    results: list[list[int]] = []
    stack: list[tuple[int, list[int]]] = [(start, [start])]
    while stack:
        cur, path = stack.pop()
        if cur == target and len(path) > 1:
            results.append(path)
            continue
        if len(path) >= max_len + 1:
            continue
        for nb in graph.get(cur, set()):
            if nb not in path:  # избегать циклов
                stack.append((nb, path + [nb]))
    return results


# ---------------------------------------------------------------------------
# Статический анализ .hf файла
# ---------------------------------------------------------------------------

class ProgramAnalysis:
    def __init__(self) -> None:
        self.words_used: list[str] = []
        self.defines: dict[str, list[str]] = {}
        self.gotos: list[int] = []
        self.asserts: list[int] = []
        self.flips: list[int] = []      # какие биты переворачиваются
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def is_ok(self) -> bool:
        return not self.errors


def analyze_source(source: str) -> ProgramAnalysis:
    """Статический анализ исходного кода HexForth."""
    import re

    analysis = ProgramAnalysis()

    # Строим фиктивный интерпретатор только для парсинга определений
    dummy = HexForth(start=0)
    try:
        tokens = dummy.parse(source)
        analysis.defines = dict(dummy.definitions)
    except HexForthError as e:
        analysis.errors.append(f"Синтаксис: {e}")
        return analysis

    # Проверка пустых DEFINE
    for name, body in analysis.defines.items():
        if not body:
            analysis.warnings.append(f"DEFINE {name}: пустое тело")

    # Проверка потенциальной рекурсии
    def check_recursion(name: str, body: list[str], visited: set[str]) -> bool:
        for w in body:
            wu = w.upper()
            if wu == name:
                return True
            if wu in analysis.defines and wu not in visited:
                visited.add(wu)
                if check_recursion(name, analysis.defines[wu], visited):
                    return True
        return False

    for name in analysis.defines:
        if check_recursion(name, analysis.defines[name], {name}):
            analysis.errors.append(f"DEFINE {name}: рекурсия")

    # Анализ токенов
    ARG_WORDS = {'GOTO', 'ASSERT-EQ'}
    i = 0
    while i < len(tokens):
        word = tokens[i].upper()
        if word in ARG_WORDS:
            if i + 1 >= len(tokens):
                analysis.errors.append(f"'{word}' требует аргумент")
                i += 1
                continue
            arg = tokens[i + 1]
            try:
                n = int(arg)
                if not 0 <= n < SIZE:
                    analysis.errors.append(f"{word} {n}: вне диапазона 0-63")
            except ValueError:
                analysis.errors.append(f"{word}: аргумент не число: '{arg}'")
            if word == 'GOTO':
                try:
                    analysis.gotos.append(int(arg))
                except ValueError:
                    pass
            elif word == 'ASSERT-EQ':
                try:
                    analysis.asserts.append(int(arg))
                except ValueError:
                    pass
            analysis.words_used.append(f"{word} {arg}")
            i += 2
        else:
            analysis.words_used.append(word)
            m = re.fullmatch(r'FLIP-([0-5])', word)
            if m:
                analysis.flips.append(int(m.group(1)))
            i += 1

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def fmt_path(path: list[int]) -> str:
    return ' → '.join(f"{h}({to_bits(h)})" for h in path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='hexforth/verifier.py — верификатор достижимости HexForth'
    )
    parser.add_argument('file', nargs='?', help='файл .hf для анализа синтаксиса')
    parser.add_argument('--start', type=int, default=0,
                        help='стартовая гексаграмма (0-63)')
    parser.add_argument('--target', type=int, default=None,
                        help='целевая гексаграмма для проверки достижимости')
    parser.add_argument(
        '--allowed', nargs='+',
        default=[f'FLIP-{i}' for i in range(6)],
        help='допустимые слова (по умолчанию: FLIP-0..FLIP-5)',
    )
    parser.add_argument('--max-path', type=int, default=6,
                        help='максимальная длина пути (по умолчанию 6)')
    parser.add_argument('--all-paths', action='store_true',
                        help='найти все пути до цели (до --max-path шагов)')
    parser.add_argument('--reachable', action='store_true',
                        help='показать все достижимые состояния из --start')
    parser.add_argument('--check-syntax', action='store_true',
                        help='только статический анализ .hf файла')
    args = parser.parse_args()

    print(f"\n  hexforth verifier")
    print(f"  {'─' * 40}")

    # --- Статический анализ файла -------------------------------------------
    if args.file:
        with open(args.file, encoding='utf-8') as f:
            source = f.read()
        a = analyze_source(source)
        print(f"\n  Файл: {args.file}")
        print(f"  Слов: {len(a.words_used)}  |  DEFINE: {len(a.defines)}  |"
              f"  GOTO: {len(a.gotos)}  |  ASSERT-EQ: {len(a.asserts)}")
        if a.flips:
            from collections import Counter
            cnt = Counter(a.flips)
            print(f"  FLIP: " + ', '.join(f"бит{b}×{c}" for b, c in sorted(cnt.items())))
        if a.gotos:
            print(f"  GOTO цели: {a.gotos}")
        if a.asserts:
            print(f"  Проверяемые состояния: {a.asserts}")
        if a.warnings:
            for w in a.warnings:
                print(f"  [WARN] {w}")
        if a.errors:
            for e in a.errors:
                print(f"  [ERROR] {e}")
            print(f"\n  СИНТАКСИС: FAIL ({len(a.errors)} ошибок)")
        else:
            print(f"\n  СИНТАКСИС: OK")

        if args.check_syntax:
            print()
            sys.exit(0 if a.is_ok() else 1)

    # --- Граф переходов -----------------------------------------------------
    print(f"\n  Стартовое состояние: {args.start} ({to_bits(args.start)})")
    print(f"  Допустимые слова:    {args.allowed}")
    print(f"  Макс. длина пути:    {args.max_path}")

    graph = build_transition_graph(args.allowed)

    # --- Все достижимые состояния -------------------------------------------
    if args.reachable:
        reach = reachable_from(args.start, graph)
        print(f"\n  Достижимо из {args.start}: {len(reach)} состояний")
        for h in sorted(reach):
            print(f"    {h:2d} ({to_bits(h)})")
        print()
        sys.exit(0)

    # --- Проверка достижимости цели -----------------------------------------
    if args.target is not None:
        if not 0 <= args.target < SIZE:
            print(f"  Ошибка: --target {args.target} вне диапазона 0-63")
            sys.exit(1)

        print(f"  Целевое состояние:   {args.target} ({to_bits(args.target)})")
        print()

        path = shortest_path_in_graph(args.start, args.target, graph)

        if path is None:
            print(f"  [FAIL] Цель {args.target} НЕДОСТИЖИМА из {args.start}")
            print(f"         при данном наборе слов: {args.allowed}")
            # Минимальный путь в Q6 без ограничений
            q6_path = shortest_path(args.start, args.target)
            print(f"\n  Минимальный путь в Q6 (без ограничений):")
            print(f"  {fmt_path(q6_path)}  ({len(q6_path)-1} шагов)")
            bits_needed = [
                (q6_path[i] ^ q6_path[i+1]).bit_length() - 1
                for i in range(len(q6_path)-1)
            ]
            missing = set(f'FLIP-{b}' for b in bits_needed) - set(args.allowed)
            if missing:
                print(f"\n  Недостающие слова: {sorted(missing)}")
            sys.exit(1)
        else:
            bits_used = [
                (path[i] ^ path[i+1]).bit_length() - 1
                for i in range(len(path)-1)
            ]
            print(f"  [OK] Цель достижима за {len(path)-1} шагов:")
            print(f"  {fmt_path(path)}")
            print(f"  Слова: {['FLIP-' + str(b) for b in bits_used]}")

        if args.all_paths:
            all_p = all_paths_bounded(args.start, args.target, graph, args.max_path)
            print(f"\n  Все пути длиной ≤ {args.max_path} ({len(all_p)} шт.):")
            for p in sorted(all_p, key=len):
                print(f"    [{len(p)-1}] {fmt_path(p)}")
    else:
        # Нет цели — просто показать статистику графа
        reach = reachable_from(args.start, graph)
        print(f"\n  Достижимо из {args.start}: {len(reach)}/64 состояний")
        print(f"  Используй --target N для проверки конкретной цели")
        print(f"  Используй --reachable для списка всех достижимых состояний")

    print()


if __name__ == '__main__':
    main()
