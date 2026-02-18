"""
puzzle.py — однопользовательский режим hexpath

Головоломка: достичь целевой гексаграммы за минимальное число ходов,
обходя заблокированные узлы.

Структура головоломки:
  - start: начальная гексаграмма
  - goal:  целевая гексаграмма
  - blocked: набор недоступных узлов
  - par:   «нормативное» число ходов (BFS-расстояние без блоков)

Решатель (BFS) находит минимальный путь с обходом блоков.
Генератор создаёт случайные или тематические головоломки.
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    neighbors, hamming, shortest_path, yang_count, sphere, SIZE,
)


# ---------------------------------------------------------------------------
# Головоломка
# ---------------------------------------------------------------------------

@dataclass
class Puzzle:
    """Одиночная головоломка: пройти из start в goal, обходя blocked."""
    start: int
    goal: int
    blocked: frozenset[int] = field(default_factory=frozenset)
    title: str = ''
    hint: str = ''

    def __post_init__(self) -> None:
        self.blocked = frozenset(self.blocked)
        # Проверки корректности
        if self.start in self.blocked:
            raise ValueError(f"start {self.start} заблокирован")
        if self.goal in self.blocked:
            raise ValueError(f"goal {self.goal} заблокирован")

    @property
    def par(self) -> int:
        """Оптимальное число ходов BFS с учётом блоков."""
        path = solve(self)
        return len(path) - 1 if path else -1

    @property
    def direct_distance(self) -> int:
        """Расстояние Хэмминга (без учёта блоков)."""
        return hamming(self.start, self.goal)

    def is_solvable(self) -> bool:
        return solve(self) is not None

    def summary(self) -> str:
        path = solve(self)
        if path is None:
            status = "НЕРАЗРЕШИМА"
            par_str = "—"
        else:
            status = "разрешима"
            par_str = str(len(path) - 1)
        return (
            f"Головоломка: {self.title or 'без названия'}\n"
            f"  Старт → Цель: {self.start} → {self.goal}\n"
            f"  Заблокировано узлов: {len(self.blocked)}\n"
            f"  Расстояние Хэмминга: {self.direct_distance}\n"
            f"  Статус: {status}, par={par_str}"
        )


# ---------------------------------------------------------------------------
# Решатель (BFS)
# ---------------------------------------------------------------------------

def solve(puzzle: Puzzle) -> list[int] | None:
    """
    BFS: найти кратчайший путь из start в goal с обходом blocked.
    Возвращает список узлов (включая start и goal) или None если нет пути.
    """
    if puzzle.start == puzzle.goal:
        return [puzzle.start]

    visited = {puzzle.start}
    queue: deque[list[int]] = deque([[puzzle.start]])

    while queue:
        path = queue.popleft()
        current = path[-1]
        for nb in neighbors(current):
            if nb in puzzle.blocked or nb in visited:
                continue
            new_path = path + [nb]
            if nb == puzzle.goal:
                return new_path
            visited.add(nb)
            queue.append(new_path)

    return None


def solve_all(puzzle: Puzzle, max_paths: int = 20) -> list[list[int]]:
    """
    Найти все кратчайшие пути (до max_paths). DFS с отсечением по длине.
    """
    optimal = solve(puzzle)
    if optimal is None:
        return []
    target_len = len(optimal)

    results: list[list[int]] = []
    stack: list[list[int]] = [[puzzle.start]]

    while stack and len(results) < max_paths:
        path = stack.pop()
        if len(path) > target_len:
            continue
        current = path[-1]
        if current == puzzle.goal:
            results.append(path)
            continue
        for nb in neighbors(current):
            if nb in puzzle.blocked or nb in path:
                continue
            stack.append(path + [nb])

    return results


# ---------------------------------------------------------------------------
# Состояние игрока во время решения головоломки
# ---------------------------------------------------------------------------

@dataclass
class PuzzleState:
    """Текущее состояние игрока, решающего головоломку."""
    puzzle: Puzzle
    path: list[int] = field(default_factory=list)
    moves: int = 0

    def __post_init__(self) -> None:
        if not self.path:
            self.path = [self.puzzle.start]

    @property
    def current(self) -> int:
        return self.path[-1]

    @property
    def is_solved(self) -> bool:
        return self.current == self.puzzle.goal

    @property
    def is_stuck(self) -> bool:
        return not self.available_moves()

    def available_moves(self) -> list[int]:
        """Соседи, не заблокированные и ещё не посещённые в текущем пути."""
        return [
            nb for nb in neighbors(self.current)
            if nb not in self.puzzle.blocked and nb not in self.path
        ]

    def move(self, destination: int) -> 'PuzzleState':
        """Сделать ход. Возвращает новое состояние."""
        if destination in self.puzzle.blocked:
            raise ValueError(f"Узел {destination} заблокирован")
        if destination not in neighbors(self.current):
            raise ValueError(f"{destination} не является соседом {self.current}")
        return PuzzleState(
            puzzle=self.puzzle,
            path=self.path + [destination],
            moves=self.moves + 1,
        )

    def undo(self) -> 'PuzzleState':
        """Откатить последний ход."""
        if len(self.path) <= 1:
            return self
        return PuzzleState(
            puzzle=self.puzzle,
            path=self.path[:-1],
            moves=self.moves,   # откат не считается ходом
        )

    def rating(self) -> str:
        """Оценка прохождения относительно par."""
        if not self.is_solved:
            return '?'
        par = self.puzzle.par
        if par < 0:
            return '?'
        diff = self.moves - par
        if diff == 0:
            return '★★★ (идеально)'
        elif diff <= 2:
            return '★★☆ (хорошо)'
        elif diff <= 5:
            return '★☆☆ (нормально)'
        else:
            return '☆☆☆ (можно лучше)'


# ---------------------------------------------------------------------------
# Генератор головоломок
# ---------------------------------------------------------------------------

def generate_puzzle(
    difficulty: str = 'medium',
    seed: int | None = None,
    start: int | None = None,
    goal: int | None = None,
) -> Puzzle:
    """
    Сгенерировать случайную головоломку заданной сложности.

    difficulty:
      'easy'   — 1-2 заблокированных узла, расстояние 3-4
      'medium' — 3-5 заблокированных узлов, расстояние 4-5
      'hard'   — 6-10 заблокированных узлов, расстояние 5-6
    """
    rng = random.Random(seed)

    params = {
        'easy':   {'n_blocked': (1, 2), 'dist': (3, 4)},
        'medium': {'n_blocked': (3, 5), 'dist': (4, 5)},
        'hard':   {'n_blocked': (6, 10), 'dist': (5, 6)},
    }
    if difficulty not in params:
        raise ValueError(f"difficulty должен быть one of {list(params)}")

    p = params[difficulty]
    min_b, max_b = p['n_blocked']
    min_d, max_d = p['dist']

    for _ in range(1000):   # ограниченное число попыток
        # Выбрать start и goal
        if start is None:
            s = rng.randrange(SIZE)
        else:
            s = start
        if goal is None:
            dist = rng.randint(min_d, max_d)
            candidates = sphere(s, dist)
            if not candidates:
                continue
            g = rng.choice(candidates)
        else:
            g = goal

        if s == g:
            continue

        # Выбрать заблокированные узлы (не start и не goal)
        n_blocked = rng.randint(min_b, max_b)
        pool = [h for h in range(SIZE) if h != s and h != g]
        rng.shuffle(pool)
        blocked = frozenset(pool[:n_blocked])

        puzzle = Puzzle(start=s, goal=g, blocked=blocked,
                        title=f'{difficulty.capitalize()} puzzle (seed={seed})')
        if puzzle.is_solvable():
            return puzzle

    raise RuntimeError(f"Не удалось сгенерировать головоломку difficulty={difficulty!r}")


# ---------------------------------------------------------------------------
# Встроенные головоломки
# ---------------------------------------------------------------------------

BUILTIN_PUZZLES: list[Puzzle] = [
    Puzzle(
        start=0,
        goal=63,
        blocked=frozenset({21, 42}),
        title='Через середину',
        hint='Антиподы заблокированы — ищи обход.',
    ),
    Puzzle(
        start=0,
        goal=7,
        blocked=frozenset({1, 2, 4}),
        title='Нижняя триграмма',
        hint='Все прямые пути перекрыты.',
    ),
    Puzzle(
        start=63,
        goal=56,
        blocked=frozenset({57, 59, 61}),
        title='Верхняя триграмма',
        hint='Перевернуть один бит, но какой?',
    ),
    Puzzle(
        start=0,
        goal=42,
        blocked=frozenset({8, 10, 32, 34}),
        title='Диагональный прыжок',
        hint='Расстояние Хэмминга=3, но средина закрыта.',
    ),
    Puzzle(
        start=21,
        goal=42,
        blocked=frozenset({0, 63, 15, 48}),
        title='Антиподные барьеры',
        hint='Угловые узлы заблокированы.',
    ),
]


def get_builtin(index: int) -> Puzzle:
    """Получить встроенную головоломку по номеру (0-based)."""
    if not 0 <= index < len(BUILTIN_PUZZLES):
        raise IndexError(f"Нет головоломки #{index}. Всего: {len(BUILTIN_PUZZLES)}")
    return BUILTIN_PUZZLES[index]


# ---------------------------------------------------------------------------
# Быстрый CLI
# ---------------------------------------------------------------------------

def _print_puzzle(puzzle: Puzzle) -> None:
    """Вывести описание головоломки."""
    path = solve(puzzle)
    print(f"\n{'='*50}")
    print(puzzle.summary())
    if puzzle.hint:
        print(f"  Подсказка: {puzzle.hint}")
    if path:
        print(f"  Решение ({len(path)-1} шагов): {' → '.join(map(str, path))}")
    print('='*50)


def _interactive(puzzle: Puzzle) -> None:
    """Интерактивное прохождение головоломки в терминале."""
    state = PuzzleState(puzzle=puzzle)
    print(f"\n=== {puzzle.title or 'Головоломка'} ===")
    print(f"Начало: {puzzle.start}  Цель: {puzzle.goal}")
    print(f"Заблокировано: {sorted(puzzle.blocked)}")
    print("Команды: число=ход, u=undo, h=подсказка, s=показать решение, q=выход\n")

    while not state.is_solved:
        pos = state.current
        avail = state.available_moves()
        dist = hamming(pos, puzzle.goal)
        print(f"Позиция: {pos}  (до цели: {dist})  Ходов: {state.moves}")
        print(f"Доступно: {avail}")
        if state.is_stuck:
            print("Нет ходов! Введите u для отмены.")

        try:
            cmd = input(">>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return

        if cmd == 'q':
            return
        elif cmd == 'u':
            state = state.undo()
        elif cmd == 'h':
            if puzzle.hint:
                print(f"Подсказка: {puzzle.hint}")
            else:
                # Показать следующий оптимальный ход
                opt = solve(Puzzle(
                    start=pos, goal=puzzle.goal,
                    blocked=puzzle.blocked | frozenset(state.path[:-1]),
                ))
                if opt and len(opt) > 1:
                    print(f"Подсказка: следующий оптимальный ход = {opt[1]}")
        elif cmd == 's':
            opt = solve(puzzle)
            if opt:
                print(f"Решение: {' → '.join(map(str, opt))}")
        else:
            try:
                dest = int(cmd)
                state = state.move(dest)
            except (ValueError, KeyError) as e:
                print(f"Ошибка: {e}")

    print(f"\n★ Решено за {state.moves} ходов! {state.rating()}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HexPath Puzzle — однопользовательский режим')
    sub = parser.add_subparsers(dest='cmd')

    p_list = sub.add_parser('list', help='Показать встроенные головоломки')

    p_play = sub.add_parser('play', help='Играть')
    p_play.add_argument('--id', type=int, default=0, help='Номер встроенной головоломки')
    p_play.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                        help='Сгенерировать случайную головоломку')
    p_play.add_argument('--seed', type=int, default=None)
    p_play.add_argument('--solve', action='store_true', help='Только показать решение')

    p_gen = sub.add_parser('generate', help='Сгенерировать и показать головоломку')
    p_gen.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium')
    p_gen.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.cmd == 'list':
        for i, puz in enumerate(BUILTIN_PUZZLES):
            print(f"  [{i}] {puz.title}: {puz.start}→{puz.goal}, "
                  f"blocked={len(puz.blocked)}, par={puz.par}")

    elif args.cmd == 'generate':
        puz = generate_puzzle(difficulty=args.difficulty, seed=args.seed)
        _print_puzzle(puz)

    elif args.cmd == 'play':
        if args.difficulty:
            puz = generate_puzzle(difficulty=args.difficulty, seed=args.seed)
        else:
            puz = get_builtin(args.id)

        if args.solve:
            _print_puzzle(puz)
        else:
            _interactive(puz)

    else:
        # По умолчанию — показать первую встроенную
        parser.print_help()
        print()
        for i, puz in enumerate(BUILTIN_PUZZLES):
            _print_puzzle(puz)
