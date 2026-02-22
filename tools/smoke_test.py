#!/usr/bin/env python3
"""
Smoke-тест: запускает каждый из 24 проектов с простыми аргументами
и проверяет, что код завершается с кодом 0 (без ошибок).

Использование:
    python3 tools/smoke_test.py
    python3 tools/smoke_test.py --verbose
    python3 tools/smoke_test.py --project hexgraph
"""
import subprocess
import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# ── Команды для каждого проекта ───────────────────────────────────────────
# Формат: ('project_name', [аргументы], stdin_bytes | None)
COMMANDS: list[tuple[str, list[str], bytes | None]] = [
    # hexnav: навигатор (интерактивный — передаём 'q' для немедленного выхода)
    ('hexnav',    ['projects/hexnav/hexnav.py', '42'],                    b'q\n'),

    # hexca: клеточный автомат (1D, 5 шагов)
    ('hexca',     ['projects/hexca/hexca.py', '--rule', 'xor_rule', '--steps', '5'], None),

    # hexpath: игра (AI vs AI, 1 ход)
    ('hexpath',   ['projects/hexpath/puzzle.py', 'list'],                            None),

    # hexforth: интерпретатор
    ('hexforth',  ['projects/hexforth/interpreter.py',
                   'projects/hexforth/examples/hello.hf'],                           None),

    # karnaugh6: минимизатор
    ('karnaugh6', ['projects/karnaugh6/minimize.py', '0', '1', '2', '3'],            None),

    # hexspec: верификатор
    ('hexspec',   ['projects/hexspec/verifier.py',
                   'projects/hexspec/examples/tcp.json'],                            None),

    # hexgraph: граф-анализ
    ('hexgraph',  ['projects/hexgraph/hexgraph.py', 'layer', '3'],                  None),

    # hexvis: визуализация
    ('hexvis',    ['projects/hexvis/hexvis.py', 'hexagram', '42'],                  None),

    # hexcode: коды
    ('hexcode',   ['projects/hexcode/hexcode.py', 'standard'],                      None),

    # hexlearn: ML
    ('hexlearn',  ['projects/hexlearn/hexlearn.py', 'kmeans', '--k', '4'],          None),

    # hexopt: оптимизация
    ('hexopt',    ['projects/hexopt/hexopt.py', 'hexagram', 'all'],                 None),

    # hexring: булевы функции
    ('hexring',   ['projects/hexring/hexring.py', 'info', 'bent'],                  None),

    # hexsym: группа автоморфизмов
    ('hexsym',    ['projects/hexsym/hexsym.py', 'info'],                            None),

    # hexnet: сеть
    ('hexnet',    ['projects/hexnet/hexnet.py', 'stats'],                           None),

    # hexcrypt: криптография
    ('hexcrypt',  ['projects/hexcrypt/hexcrypt.py', 'info', 'random'],              None),

    # hexstat: статистика
    ('hexstat',   ['projects/hexstat/hexstat.py', 'info'],                          None),

    # hexgeom: геометрия
    ('hexgeom',   ['projects/hexgeom/hexgeom.py', 'ball', '0', '2'],               None),

    # hexdim: размерности
    ('hexdim',    ['projects/hexdim/hexdim.py', 'info'],                            None),

    # hexalg: гармонический анализ
    ('hexalg',    ['projects/hexalg/hexalg.py', 'characters'],                     None),

    # hexphys: физика
    ('hexphys',   ['projects/hexphys/hexphys.py', 'ising', '1.0'],                 None),

    # hexgf: поле Галуа
    ('hexgf',     ['projects/hexgf/hexgf.py', 'info'],                             None),

    # hexmat: линейная алгебра
    ('hexmat',    ['projects/hexmat/hexmat.py', 'info'],                            None),

    # hexbio: биоинформатика
    ('hexbio',    ['projects/hexbio/hexbio.py', 'info'],                           None),

    # hexlat: решётка
    ('hexlat',    ['projects/hexlat/hexlat.py', 'info'],                           None),
]


def run_command(
    project: str, args: list[str], verbose: bool,
    stdin_input: bytes | None = None,
    timeout: int = 30,
) -> bool:
    """Запустить команду и вернуть True если код завершения == 0."""
    cmd = [PYTHON] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            input=stdin_input,
            timeout=timeout,
            cwd=ROOT,
        )
        if result.returncode == 0:
            return True
        else:
            if not verbose and result.stderr:
                print(f"    stderr: {result.stderr.decode()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Smoke-тест: проверяет запуск всех 24 CLI-команд'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='показывать вывод программ')
    parser.add_argument('--project', '-p', default=None,
                        help='запустить только один проект')
    args = parser.parse_args()

    commands = COMMANDS
    if args.project:
        commands = [(p, c, s) for (p, c, s) in COMMANDS if p == args.project]
        if not commands:
            print(f"Проект '{args.project}' не найден. Доступные: "
                  f"{[p for p, _, _ in COMMANDS]}")
            return 1

    print(f"\n{'═' * 52}")
    print(f"  Smoke-тест: {len(commands)} проект(ов)")
    print(f"{'═' * 52}\n")

    passed = failed = 0
    results: list[tuple[str, bool, float]] = []

    for project, cmd_args, stdin_input in commands:
        t0 = time.time()
        ok = run_command(project, cmd_args, args.verbose, stdin_input=stdin_input)
        elapsed = time.time() - t0
        results.append((project, ok, elapsed))

        status = 'OK  ' if ok else 'FAIL'
        print(f"  [{status}] {project:<16} ({elapsed:.2f}s)")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'─' * 52}")
    print(f"  Итог: {passed} OK, {failed} FAIL из {len(commands)}")
    print(f"{'═' * 52}\n")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
