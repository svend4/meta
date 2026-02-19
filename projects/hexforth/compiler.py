#!/usr/bin/env python3
"""
hexforth/compiler.py — компилятор HexForth → Python / JSON-байткод

Компилятор транслирует программу HexForth в:
  1. Python-функцию: замкнутая функция, меняющая int → int (гексаграмма)
  2. JSON-байткод: список операций для легковесного VM
  3. Dot-граф: визуализация пути через Q6

Использование:
    python3 compiler.py examples/hello.hf --target python
    python3 compiler.py examples/hello.hf --target json
    python3 compiler.py examples/hello.hf --target dot | dot -Tpng > path.png
    python3 compiler.py --start 0 --program "GOTO 42 ANTIPODE DEBUG"
"""

from __future__ import annotations
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    flip, shortest_path, antipode, to_bits, SIZE,
)
from projects.hexforth.interpreter import HexForth, HexForthError


# ---------------------------------------------------------------------------
# Байткод (IR — промежуточное представление)
# ---------------------------------------------------------------------------

# Каждая инструкция — словарь {"op": ..., ...}
Instruction = dict[str, Any]


def compile_to_ir(source: str, start: int = 0) -> list[Instruction]:
    """
    Скомпилировать HexForth-программу в список инструкций IR.
    Также разворачивает GOTO в последовательность FLIP.
    """
    interp = HexForth(start=start)
    # Трассируем выполнение: записываем каждый переход
    instructions: list[Instruction] = []

    class TracingHexForth(HexForth):
        def _transition(self, new_state: int) -> None:
            bit = (self.state ^ new_state).bit_length() - 1
            instructions.append({
                'op': 'FLIP',
                'bit': bit,
                'from': self.state,
                'to': new_state,
            })
            super()._transition(new_state)

        def _emit(self, msg: str) -> None:
            instructions.append({'op': 'PRINT', 'msg': msg})
            super()._emit(msg)

    tr = TracingHexForth(start=start)
    tr.run(source)

    # Добавить финальный assert
    instructions.append({'op': 'FINAL', 'state': tr.state})
    return instructions


# ---------------------------------------------------------------------------
# Цель 1: Python-функция
# ---------------------------------------------------------------------------

def _start_state(ir: list[Instruction]) -> int:
    for i in ir:
        if i['op'] == 'FLIP':
            return i['from']
    return 0


def to_python(ir: list[Instruction], func_name: str = 'hexforth_program') -> str:
    """Сгенерировать Python-код: функция (start: int) -> int."""
    lines = [
        f"def {func_name}(state: int = {_start_state(ir)}) -> int:",
        f'    """Сгенерировано hexforth/compiler.py"""',
    ]

    for instr in ir:
        op = instr['op']
        if op == 'FLIP':
            bit = instr['bit']
            fr = instr['from']
            to = instr['to']
            lines.append(
                f"    state = state ^ (1 << {bit})  "
                f"# {to_bits(fr)} → {to_bits(to)}  (bit {bit})"
            )
        elif op == 'PRINT':
            msg = instr['msg'].replace("'", "\\'")
            lines.append(f"    print('{msg}')")
        elif op == 'FINAL':
            lines.append(f"    assert state == {instr['state']}, "
                         f"f'Ожидалось {instr['state']}, получено {{state}}'")
            lines.append(f"    return state")

    if not any(i['op'] == 'FINAL' for i in ir):
        lines.append("    return state")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Цель 2: JSON-байткод
# ---------------------------------------------------------------------------

def to_json_bytecode(ir: list[Instruction]) -> str:
    """Сгенерировать JSON-байткод."""
    # Упрощённый формат: только FLIP операции + метаданные
    bytecode = {
        'version': 1,
        'start': _start_state(ir),
        'end': next((i['state'] for i in ir if i['op'] == 'FINAL'), None),
        'instructions': [
            {'op': i['op'], **{k: v for k, v in i.items() if k != 'op'}}
            for i in ir
            if i['op'] in ('FLIP', 'PRINT', 'FINAL')
        ],
    }
    return json.dumps(bytecode, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Цель 3: DOT-граф (Graphviz)
# ---------------------------------------------------------------------------

def to_dot(ir: list[Instruction], title: str = 'HexForth Path') -> str:
    """Сгенерировать DOT-граф для Graphviz."""
    lines = [
        f'digraph hexforth {{',
        f'  label="{title}";',
        f'  rankdir=LR;',
        f'  node [shape=box, fontname="Courier"];',
        f'  edge [fontname="Courier", fontsize=10];',
        '',
    ]

    nodes: set[int] = set()
    edges: list[tuple[int, int, int]] = []  # from, to, bit

    for instr in ir:
        if instr['op'] == 'FLIP':
            fr, to, bit = instr['from'], instr['to'], instr['bit']
            nodes.add(fr)
            nodes.add(to)
            edges.append((fr, to, bit))

    final_state = next((i['state'] for i in ir if i['op'] == 'FINAL'), None)
    start_state = _start_state(ir)

    # Узлы
    for h in sorted(nodes):
        label = f"{h}\\n{to_bits(h)}"
        style = 'style=filled, fillcolor="#aaffaa"' if h == final_state else ''
        if h == start_state:
            style = 'style=filled, fillcolor="#ffaaaa"'
        lines.append(f'  h{h} [label="{label}" {style}];')

    lines.append('')

    # Рёбра
    for (fr, to, bit) in edges:
        lines.append(f'  h{fr} -> h{to} [label="bit{bit}"];')

    lines.append('}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Цель 4: Статистика пути
# ---------------------------------------------------------------------------

def path_stats(ir: list[Instruction]) -> dict:
    flips = [i for i in ir if i['op'] == 'FLIP']
    bit_counts = {}
    for f in flips:
        b = f['bit']
        bit_counts[b] = bit_counts.get(b, 0) + 1

    return {
        'total_steps': len(flips),
        'start': _start_state(ir) if flips else None,
        'end': next((i['state'] for i in ir if i['op'] == 'FINAL'), None),
        'bit_usage': bit_counts,
        'unique_states': len({i['from'] for i in flips} | {i['to'] for i in flips}),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='hexforth/compiler.py — компилятор HexForth в Python/JSON/DOT'
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('file', nargs='?', help='файл .hf')
    src.add_argument('--program', help='программа из командной строки')
    parser.add_argument('--start', type=int, default=0,
                        help='стартовая гексаграмма (0-63)')
    parser.add_argument('--target', choices=['python', 'json', 'dot', 'stats'],
                        default='python', help='цель компиляции')
    parser.add_argument('--func-name', default='hexforth_program',
                        help='имя функции для --target python')
    parser.add_argument('--out', help='файл вывода (по умолчанию stdout)')
    args = parser.parse_args()

    # Загрузить исходный код
    if args.file:
        path = args.file
        with open(path, encoding='utf-8') as f:
            source = f.read()
        title = Path(path).stem
    else:
        source = args.program
        title = 'inline'

    # Компилировать
    try:
        ir = compile_to_ir(source, start=args.start)
    except HexForthError as e:
        print(f"Ошибка компиляции: {e}", file=sys.stderr)
        sys.exit(1)

    # Генерировать вывод
    if args.target == 'python':
        result = to_python(ir, func_name=args.func_name)
    elif args.target == 'json':
        result = to_json_bytecode(ir)
    elif args.target == 'dot':
        result = to_dot(ir, title=title)
    elif args.target == 'stats':
        stats = path_stats(ir)
        lines = [
            f"  Шагов:          {stats['total_steps']}",
            f"  Старт:          {stats['start']} ({to_bits(stats['start']) if stats['start'] is not None else '?'})",
            f"  Конец:          {stats['end']} ({to_bits(stats['end']) if stats['end'] is not None else '?'})",
            f"  Уник. состояний:{stats['unique_states']}",
            f"  Использование бит:",
        ]
        for b, cnt in sorted(stats['bit_usage'].items()):
            lines.append(f"    бит {b}: {cnt} раз")
        result = '\n'.join(lines)
    else:
        result = ''

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Записано в {args.out}")
    else:
        print(result)


if __name__ == '__main__':
    main()
