#!/usr/bin/env python3
"""
hexspec — формальная верификация конечных автоматов через граф Q6

Каждое состояние автомата = гексаграмма (6 булевых свойств системы).
Каждый переход = изменение ровно одного свойства → ребро в Q6.

Верификатор проверяет:
  - достижимость (reachability): можно ли попасть из A в B?
  - тупики (deadlocks): состояния без исходящих переходов
  - недостижимые состояния (unreachable states)
  - инварианты (invariants): условия, выполняемые во всех достижимых состояниях
  - полноту (coverage): все ли переходы покрыты

Формат спецификации (Python dict или .hexspec TOML-подобный):
  bits: [имя_бита_0, имя_бита_1, ..., имя_бита_5]
  states:
    CLOSED:      000000  # имя: 6-битная строка
    ESTABLISHED: 000111
  transitions:
    CLOSED → SYN_SENT: flip_bit=1  (или flip имя бита)
  initial: CLOSED
  final:   [CLOSED]  # конечные состояния
  forbidden: [...]   # запрещённые состояния

CLI:
    python3 verifier.py examples/tcp.hexspec
    python3 verifier.py examples/traffic_light.hexspec --check reachability
"""

from __future__ import annotations
import sys
import json
import argparse
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    neighbors, hamming, shortest_path, flip, to_bits, yang_count, SIZE,
)


# ---------------------------------------------------------------------------
# Спецификация автомата
# ---------------------------------------------------------------------------

class Spec:
    """Спецификация конечного автомата на Q6."""

    def __init__(
        self,
        name: str,
        bit_names: list[str],
        states: dict[str, int],             # имя → гексаграмма
        transitions: list[tuple[int, int]], # (from, to)
        initial: int,
        final: set[int],
        forbidden: set[int],
        description: str = '',
    ) -> None:
        self.name = name
        self.bit_names = bit_names          # 6 имён битов
        self.states = states                # имя → int
        self.state_names = {v: k for k, v in states.items()}  # int → имя
        self.transitions = set(transitions)  # множество рёбер
        self.initial = initial
        self.final = final
        self.forbidden = forbidden
        self.description = description

        # Собрать предупреждения о переходах, не являющихся рёбрами Q6 (distance > 1)
        self.non_q6_transitions: list[tuple[int, int]] = [
            (a, b) for (a, b) in self.transitions if hamming(a, b) != 1
        ]

    def state_name(self, h: int) -> str:
        return self.state_names.get(h, f"#{h}({to_bits(h)})")

    def reachable_states(self) -> set[int]:
        """BFS: все состояния, достижимые из initial по заданным переходам."""
        visited = {self.initial}
        queue = deque([self.initial])
        while queue:
            cur = queue.popleft()
            for (a, b) in self.transitions:
                if a == cur and b not in visited:
                    visited.add(b)
                    queue.append(b)
        return visited

    def reverse_transitions(self) -> set[tuple[int, int]]:
        return {(b, a) for (a, b) in self.transitions}

    def backward_reachable(self, targets: set[int]) -> set[int]:
        """BFS назад: все состояния, из которых достижима хотя бы одна цель."""
        rev = self.reverse_transitions()
        visited = set(targets)
        queue = deque(targets)
        while queue:
            cur = queue.popleft()
            for (a, b) in rev:
                if a == cur and b not in visited:
                    visited.add(b)
                    queue.append(b)
        return visited

    def deadlocks(self) -> set[int]:
        """Состояния без исходящих переходов, не являющиеся конечными."""
        reachable = self.reachable_states()
        has_out = {a for (a, b) in self.transitions}
        return {s for s in reachable if s not in has_out and s not in self.final}

    def unreachable(self) -> set[int]:
        """Именованные состояния, недостижимые из initial."""
        reachable = self.reachable_states()
        return {h for h in self.states.values() if h not in reachable}

    def forbidden_reachable(self) -> set[int]:
        """Запрещённые состояния, которые всё же достижимы."""
        return self.reachable_states() & self.forbidden

    def path_to(self, target: int) -> list[int] | None:
        """Найти путь из initial в target по разрешённым переходам."""
        if target == self.initial:
            return [self.initial]
        visited = {self.initial: None}  # state → prev
        queue = deque([self.initial])
        while queue:
            cur = queue.popleft()
            for (a, b) in self.transitions:
                if a == cur and b not in visited:
                    visited[b] = cur
                    if b == target:
                        # восстановить путь
                        path = []
                        node = b
                        while node is not None:
                            path.append(node)
                            node = visited[node]
                        return list(reversed(path))
                    queue.append(b)
        return None

    def check_invariant(self, invariant_fn, name: str = 'invariant') -> list[int]:
        """Проверить инвариант: функция (гексаграмма) → bool. Вернуть нарушителей."""
        return [s for s in self.reachable_states() if not invariant_fn(s)]

    def coverage(self) -> dict:
        """Покрытие: сколько рёбер Q6 используется vs. сколько определено в Q6."""
        reachable = self.reachable_states()
        # Все возможные рёбра между достижимыми состояниями в Q6
        possible_edges = set()
        for s in reachable:
            for nb in neighbors(s):
                if nb in reachable:
                    possible_edges.add((s, nb))
        used = self.transitions
        return {
            'used': len(used),
            'possible': len(possible_edges),
            'ratio': len(used) / len(possible_edges) if possible_edges else 0.0,
        }


# ---------------------------------------------------------------------------
# Парсер спецификации (JSON-формат)
# ---------------------------------------------------------------------------

def load_spec(path: str) -> Spec:
    """
    Загрузить спецификацию из JSON-файла.

    Формат:
    {
      "name": "TCP",
      "description": "...",
      "bits": ["conn_open", "syn_sent", "syn_rcvd", "fin_sent", "fin_rcvd", "timer"],
      "states": {"CLOSED": "000000", "ESTABLISHED": "000111"},
      "transitions": [["CLOSED", "SYN_SENT"], ...],
      "initial": "CLOSED",
      "final": ["CLOSED"],
      "forbidden": []
    }
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    bit_names = data.get('bits', [f'b{i}' for i in range(6)])
    states = {
        name: int(bits, 2)
        for name, bits in data['states'].items()
    }
    name_to_h = states

    transitions = []
    for (a_name, b_name) in data.get('transitions', []):
        if a_name not in name_to_h:
            raise ValueError(f"Неизвестное состояние: '{a_name}'")
        if b_name not in name_to_h:
            raise ValueError(f"Неизвестное состояние: '{b_name}'")
        transitions.append((name_to_h[a_name], name_to_h[b_name]))

    initial_name = data['initial']
    if initial_name not in name_to_h:
        raise ValueError(f"Начальное состояние '{initial_name}' не определено")
    initial = name_to_h[initial_name]

    final = {name_to_h[n] for n in data.get('final', [])}
    forbidden = {name_to_h[n] for n in data.get('forbidden', [])}

    return Spec(
        name=data.get('name', Path(path).stem),
        bit_names=bit_names,
        states=states,
        transitions=transitions,
        initial=initial,
        final=final,
        forbidden=forbidden,
        description=data.get('description', ''),
    )


# ---------------------------------------------------------------------------
# Отчёт верификации
# ---------------------------------------------------------------------------

def verify(spec: Spec, verbose: bool = False) -> bool:
    """Запустить все проверки и вывести отчёт. Вернуть True если всё OK."""
    ok = True
    issues: list[str] = []

    print(f"\n{'═' * 50}")
    print(f"  hexspec: верификация «{spec.name}»")
    if spec.description:
        print(f"  {spec.description}")
    print(f"{'═' * 50}")

    # Состояния
    print(f"\n  Состояния ({len(spec.states)}):")
    for name, h in sorted(spec.states.items(), key=lambda x: x[1]):
        flags = []
        if h == spec.initial:
            flags.append('начальное')
        if h in spec.final:
            flags.append('конечное')
        if h in spec.forbidden:
            flags.append('запрещённое')
        flag_str = '  [' + ', '.join(flags) + ']' if flags else ''
        bits = to_bits(h)
        names_str = ', '.join(
            f'{bname}={int(b)}'
            for bname, b in zip(spec.bit_names, bits)
            if b == '1'
        ) or '∅'
        print(f"    {name:<16} {bits}  ({names_str}){flag_str}")

    # Переходы
    print(f"\n  Переходы ({len(spec.transitions)}):")
    for (a, b) in sorted(spec.transitions):
        d = hamming(a, b)
        if d == 1:
            diff_bit = (a ^ b).bit_length() - 1
            bit_name = spec.bit_names[diff_bit] if diff_bit < len(spec.bit_names) else f'b{diff_bit}'
            direction = '↑' if (b >> diff_bit) & 1 else '↓'
            edge_info = f"[{bit_name} {direction}]"
        else:
            edge_info = f"[Q6-расстояние={d}, путь={d} шагов]"
        print(f"    {spec.state_name(a):<16} → {spec.state_name(b):<16}  {edge_info}")

    # Предупреждения о non-Q6 переходах
    if spec.non_q6_transitions:
        print(f"\n  [WARN] Переходы с расстоянием Хэмминга > 1 ({len(spec.non_q6_transitions)}):")
        for (a, b) in spec.non_q6_transitions:
            path = shortest_path(a, b)
            print(f"    {spec.state_name(a)} → {spec.state_name(b)}: "
                  f"d={hamming(a, b)}, Q6-путь: {' → '.join(str(x) for x in path)}")

    # Достижимость
    reachable = spec.reachable_states()
    print(f"\n  Достижимые состояния ({len(reachable)}):")
    for h in sorted(reachable):
        print(f"    {spec.state_name(h)}")

    # Недостижимые
    unreachable = spec.unreachable()
    if unreachable:
        ok = False
        issues.append(f"Недостижимые состояния: {[spec.state_name(h) for h in unreachable]}")
        print(f"\n  [WARN] Недостижимые состояния ({len(unreachable)}):")
        for h in sorted(unreachable):
            print(f"    {spec.state_name(h)}")
    else:
        print(f"\n  [OK] Все состояния достижимы")

    # Тупики
    deadlocks = spec.deadlocks()
    if deadlocks:
        ok = False
        issues.append(f"Тупики: {[spec.state_name(h) for h in deadlocks]}")
        print(f"\n  [FAIL] Тупики ({len(deadlocks)}):")
        for h in sorted(deadlocks):
            path = spec.path_to(h)
            path_str = ' → '.join(spec.state_name(x) for x in (path or []))
            print(f"    {spec.state_name(h)}  (путь: {path_str})")
    else:
        print(f"  [OK] Тупиков нет")

    # Запрещённые состояния
    forbidden_reach = spec.forbidden_reachable()
    if forbidden_reach:
        ok = False
        issues.append(f"Достижимые запрещённые: {[spec.state_name(h) for h in forbidden_reach]}")
        print(f"\n  [FAIL] Запрещённые достижимые состояния ({len(forbidden_reach)}):")
        for h in sorted(forbidden_reach):
            path = spec.path_to(h)
            path_str = ' → '.join(spec.state_name(x) for x in (path or []))
            print(f"    {spec.state_name(h)}  (путь: {path_str})")
    else:
        if spec.forbidden:
            print(f"  [OK] Запрещённые состояния недостижимы")

    # Достижимость конечных состояний
    if spec.final:
        unreachable_final = spec.final - reachable
        if unreachable_final:
            ok = False
            issues.append(f"Недостижимые конечные состояния: "
                          f"{[spec.state_name(h) for h in unreachable_final]}")
            print(f"\n  [FAIL] Конечные состояния недостижимы: "
                  f"{[spec.state_name(h) for h in unreachable_final]}")
        else:
            print(f"  [OK] Все конечные состояния достижимы")

    # Покрытие рёбер
    cov = spec.coverage()
    print(f"\n  Покрытие рёбер Q6: {cov['used']}/{cov['possible']} "
          f"({cov['ratio']:.0%})")

    # Итог
    print(f"\n{'─' * 50}")
    if ok:
        print(f"  РЕЗУЛЬТАТ: OK — нарушений не найдено")
    else:
        print(f"  РЕЗУЛЬТАТ: FAIL — найдено {len(issues)} проблем(а/ы):")
        for issue in issues:
            print(f"    • {issue}")
    print(f"{'═' * 50}\n")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='hexspec — верификатор конечных автоматов на Q6'
    )
    parser.add_argument('spec', help='файл спецификации (.json)')
    parser.add_argument('--verbose', action='store_true', help='подробный вывод')
    parser.add_argument(
        '--path', nargs=2, metavar=('FROM', 'TO'),
        help='найти путь между двумя состояниями'
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    ok = verify(spec, verbose=args.verbose)

    if args.path:
        from_name, to_name = args.path
        from_h = spec.states.get(from_name)
        to_h = spec.states.get(to_name)
        if from_h is None:
            print(f"  Неизвестное состояние: '{from_name}'")
            sys.exit(1)
        if to_h is None:
            print(f"  Неизвестное состояние: '{to_name}'")
            sys.exit(1)
        path = spec.path_to(to_h) if from_h == spec.initial else None
        if path:
            print(f"  Путь {from_name} → {to_name}:")
            print('  ' + ' → '.join(spec.state_name(h) for h in path))
        else:
            print(f"  Путь {from_name} → {to_name}: недостижимо по заданным переходам")

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
