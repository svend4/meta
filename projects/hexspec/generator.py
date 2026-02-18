#!/usr/bin/env python3
"""
hexspec/generator.py — генератор тестовых сценариев из спецификации автомата

По спецификации (.json) автоматически строит:
  1. All-states coverage    — минимальный набор путей, покрывающий все состояния
  2. All-transitions coverage — минимальный набор путей, покрывающий все переходы
  3. Boundary scenarios    — пути до запрещённых / тупиковых зон (негативные тесты)
  4. Round-trip paths      — циклы: initial → state → initial (для каждого состояния)

Формат вывода:
  - текстовые сценарии (по умолчанию)
  - JSON (--format json)
  - HexForth-программы (--format hexforth)

Использование:
    python3 generator.py examples/tcp.json
    python3 generator.py examples/tcp.json --coverage transitions --format json
    python3 generator.py examples/tcp.json --coverage all --format hexforth
"""

from __future__ import annotations
import sys
import json
import argparse
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import to_bits, hamming
from projects.hexspec.verifier import Spec, load_spec


# ---------------------------------------------------------------------------
# Вспомогательные: поиск путей в автомате
# ---------------------------------------------------------------------------

def bfs_path(spec: Spec, start: int, goal: int) -> list[int] | None:
    """BFS по разрешённым переходам спецификации."""
    if start == goal:
        return [start]
    visited = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for (a, b) in spec.transitions:
            if a == cur and b not in visited:
                visited[b] = cur
                if b == goal:
                    path = []
                    node = b
                    while node is not None:
                        path.append(node)
                        node = visited[node]
                    return list(reversed(path))
                queue.append(b)
    return None


def path_covers_transition(path: list[int], trans: tuple[int, int]) -> bool:
    a, b = trans
    for i in range(len(path) - 1):
        if path[i] == a and path[i + 1] == b:
            return True
    return False


# ---------------------------------------------------------------------------
# Стратегии покрытия
# ---------------------------------------------------------------------------

def all_states_paths(spec: Spec) -> list[list[int]]:
    """
    Минимальный набор путей из initial, покрывающий все достижимые состояния.
    Жадно: строим по одному пути, каждый раз расширяя до непосещённого состояния.
    """
    reachable = spec.reachable_states()
    covered = {spec.initial}
    paths: list[list[int]] = []

    for target in sorted(reachable - {spec.initial}):
        if target in covered:
            continue
        path = bfs_path(spec, spec.initial, target)
        if path:
            paths.append(path)
            covered.update(path)

    return paths


def all_transitions_paths(spec: Spec) -> list[list[int]]:
    """
    Минимальный набор путей из initial, покрывающий все переходы.
    Использует подход: для каждого непокрытого перехода строим путь
    initial → source(transition) → target(transition).
    """
    uncovered = set(spec.transitions)
    paths: list[list[int]] = []

    while uncovered:
        # Взять переход с наименьшим расстоянием от initial (жадно)
        trans = min(
            uncovered,
            key=lambda t: len(bfs_path(spec, spec.initial, t[0]) or [999]),
        )
        a, b = trans

        # Путь: initial → a → b
        path_to_a = bfs_path(spec, spec.initial, a)
        if path_to_a is None:
            uncovered.discard(trans)
            continue

        full_path = path_to_a + [b]

        # Отметить все покрытые этим путём переходы
        for i in range(len(full_path) - 1):
            t = (full_path[i], full_path[i + 1])
            uncovered.discard(t)

        paths.append(full_path)

    return paths


def round_trip_paths(spec: Spec) -> list[tuple[int, list[int]]]:
    """
    Для каждого состояния: цикл initial → state → initial.
    Возвращает (target_state, combined_path).
    """
    result: list[tuple[int, list[int]]] = []
    reachable = spec.reachable_states()

    for state in sorted(reachable):
        if state == spec.initial:
            continue
        there = bfs_path(spec, spec.initial, state)
        back = bfs_path(spec, state, spec.initial)
        if there and back:
            combined = there + back[1:]  # убрать дублирующий стартовый узел
            result.append((state, combined))

    return result


def negative_scenarios(spec: Spec) -> list[dict]:
    """
    Негативные сценарии: попытки достичь запрещённых / тупиковых состояний.
    """
    scenarios = []

    # Тупики
    for deadlock in sorted(spec.deadlocks()):
        path = bfs_path(spec, spec.initial, deadlock)
        if path:
            scenarios.append({
                'type': 'deadlock',
                'target': deadlock,
                'name': spec.state_name(deadlock),
                'path': path,
                'description': f'Достижение тупика {spec.state_name(deadlock)}',
            })

    # Запрещённые (если достижимы — это баг, строим путь для воспроизведения)
    for forbidden in sorted(spec.forbidden_reachable()):
        path = bfs_path(spec, spec.initial, forbidden)
        if path:
            scenarios.append({
                'type': 'forbidden_reachable',
                'target': forbidden,
                'name': spec.state_name(forbidden),
                'path': path,
                'description': f'БАГИ: достижимое запрещённое состояние {spec.state_name(forbidden)}',
            })

    return scenarios


# ---------------------------------------------------------------------------
# Форматирование вывода
# ---------------------------------------------------------------------------

def path_to_hexforth(spec: Spec, path: list[int]) -> str:
    """Преобразовать путь в HexForth-программу."""
    lines = [f"# Путь: {' → '.join(spec.state_name(h) for h in path)}"]
    lines.append(f"# Старт: {path[0]} ({to_bits(path[0])})")
    lines.append(f"GOTO {path[0]}")
    for i in range(1, len(path)):
        fr, to = path[i - 1], path[i]
        if hamming(fr, to) == 1:
            bit = (fr ^ to).bit_length() - 1
            lines.append(f"FLIP-{bit}  # {spec.state_name(fr)} → {spec.state_name(to)}")
        else:
            lines.append(f"GOTO {to}  # {spec.state_name(fr)} → {spec.state_name(to)} (макро)")
    lines.append(f"ASSERT-EQ {path[-1]}")
    return '\n'.join(lines)


def format_path(spec: Spec, path: list[int], indent: str = '  ') -> str:
    """Текстовое представление пути."""
    names = ' → '.join(spec.state_name(h) for h in path)
    bits = ' → '.join(to_bits(h) for h in path)
    return f"{indent}{names}\n{indent}({bits})\n{indent}Шагов: {len(path) - 1}"


def generate_report(spec: Spec, coverage: str = 'all') -> dict:
    """Сгенерировать полный отчёт."""
    report: dict = {
        'spec_name': spec.name,
        'initial': spec.state_name(spec.initial),
        'scenarios': [],
    }

    # All-states
    if coverage in ('states', 'all'):
        for path in all_states_paths(spec):
            report['scenarios'].append({
                'type': 'all_states',
                'target': spec.state_name(path[-1]),
                'path': [spec.state_name(h) for h in path],
                'path_bits': [to_bits(h) for h in path],
                'steps': len(path) - 1,
            })

    # All-transitions
    if coverage in ('transitions', 'all'):
        for path in all_transitions_paths(spec):
            report['scenarios'].append({
                'type': 'all_transitions',
                'covers_transition': f"{spec.state_name(path[-2])} → {spec.state_name(path[-1])}" if len(path) >= 2 else '',
                'path': [spec.state_name(h) for h in path],
                'path_bits': [to_bits(h) for h in path],
                'steps': len(path) - 1,
            })

    # Round-trips
    if coverage in ('roundtrip', 'all'):
        for state, path in round_trip_paths(spec):
            report['scenarios'].append({
                'type': 'round_trip',
                'via': spec.state_name(state),
                'path': [spec.state_name(h) for h in path],
                'path_bits': [to_bits(h) for h in path],
                'steps': len(path) - 1,
            })

    # Negative
    if coverage in ('negative', 'all'):
        for neg in negative_scenarios(spec):
            report['scenarios'].append({
                'type': neg['type'],
                'description': neg['description'],
                'path': [spec.state_name(h) for h in neg['path']],
                'path_bits': [to_bits(h) for h in neg['path']],
                'steps': len(neg['path']) - 1,
            })

    report['total_scenarios'] = len(report['scenarios'])
    return report


def print_text_report(spec: Spec, coverage: str = 'all') -> None:
    """Вывести текстовый отчёт."""
    print(f"\n{'═' * 54}")
    print(f"  hexspec generator: тесты для «{spec.name}»")
    print(f"{'═' * 54}")

    n = 1

    if coverage in ('states', 'all'):
        paths = all_states_paths(spec)
        print(f"\n  ── Покрытие состояний ({len(paths)} сценариев) ──")
        for path in paths:
            print(f"\n  [{n}] Достичь {spec.state_name(path[-1])}")
            print(format_path(spec, path))
            n += 1

    if coverage in ('transitions', 'all'):
        paths = all_transitions_paths(spec)
        print(f"\n  ── Покрытие переходов ({len(paths)} сценариев) ──")
        for path in paths:
            tr = f"{spec.state_name(path[-2])} → {spec.state_name(path[-1])}" if len(path) >= 2 else ''
            print(f"\n  [{n}] Переход {tr}")
            print(format_path(spec, path))
            n += 1

    if coverage in ('roundtrip', 'all'):
        trips = round_trip_paths(spec)
        print(f"\n  ── Round-trip тесты ({len(trips)} сценариев) ──")
        for state, path in trips:
            print(f"\n  [{n}] Цикл через {spec.state_name(state)}")
            print(format_path(spec, path))
            n += 1

    if coverage in ('negative', 'all'):
        negs = negative_scenarios(spec)
        if negs:
            print(f"\n  ── Негативные сценарии ({len(negs)}) ──")
            for neg in negs:
                print(f"\n  [{n}] {neg['description']}")
                print(format_path(spec, neg['path']))
                n += 1
        else:
            print(f"\n  ── Негативных сценариев нет ──")

    print(f"\n  Итого сценариев: {n - 1}")
    print(f"{'═' * 54}\n")


def print_hexforth_report(spec: Spec, coverage: str = 'all') -> None:
    """Вывести HexForth-программы для каждого сценария."""
    all_paths: list[tuple[str, list[int]]] = []

    if coverage in ('states', 'all'):
        for path in all_states_paths(spec):
            all_paths.append((f"Достичь {spec.state_name(path[-1])}", path))

    if coverage in ('transitions', 'all'):
        for path in all_transitions_paths(spec):
            tr = f"{spec.state_name(path[-2])} → {spec.state_name(path[-1])}" if len(path) >= 2 else ''
            all_paths.append((f"Переход {tr}", path))

    if coverage in ('roundtrip', 'all'):
        for state, path in round_trip_paths(spec):
            all_paths.append((f"Round-trip через {spec.state_name(state)}", path))

    for title, path in all_paths:
        print(f"\n# === {title} ===")
        print(path_to_hexforth(spec, path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='hexspec/generator.py — генератор тестовых сценариев'
    )
    parser.add_argument('spec', help='файл спецификации (.json)')
    parser.add_argument(
        '--coverage',
        choices=['states', 'transitions', 'roundtrip', 'negative', 'all'],
        default='all',
        help='стратегия покрытия (по умолчанию all)',
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'hexforth'],
        default='text',
        help='формат вывода (по умолчанию text)',
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)

    if args.format == 'text':
        print_text_report(spec, args.coverage)
    elif args.format == 'json':
        report = generate_report(spec, args.coverage)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.format == 'hexforth':
        print_hexforth_report(spec, args.coverage)


if __name__ == '__main__':
    main()
