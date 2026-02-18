"""
hexcore — ядро системы гексаграмм (граф Q6)

Граф Q6: 64 узла (гексаграммы), каждый с 6 рёбрами (переворот одной черты).
Это граф Кэли группы (Z₂)⁶ — 6-мерный гиперкуб.

Соглашения:
  - Гексаграмма = целое число 0..63
  - Бит 0 = нижняя черта, бит 5 = верхняя черта
  - 0 = прерывистая (инь), 1 = сплошная (ян)
  - Расстояние Хэмминга между двумя гексаграммами = число отличающихся черт
"""

from __future__ import annotations
from collections import deque
from typing import Iterator

LINES = 6          # число черт в гексаграмме
SIZE = 1 << LINES  # 64 гексаграммы


# ---------------------------------------------------------------------------
# Базовые операции
# ---------------------------------------------------------------------------

def to_bits(h: int) -> str:
    """Гексаграмма → 6-битная строка (бит 5 слева, бит 0 справа)."""
    return format(h, '06b')


def from_bits(bits: str) -> int:
    """6-битная строка → гексаграмма."""
    return int(bits, 2)


def neighbors(h: int) -> list[int]:
    """Все 6 гексаграмм, достижимых за один шаг (переворот одной черты)."""
    return [h ^ (1 << i) for i in range(LINES)]


def hamming(a: int, b: int) -> int:
    """Расстояние Хэмминга — число черт, в которых гексаграммы отличаются."""
    return bin(a ^ b).count('1')


def flip(h: int, line: int) -> int:
    """Перевернуть черту с номером line (0 = нижняя, 5 = верхняя)."""
    if not 0 <= line < LINES:
        raise ValueError(f"line must be 0..{LINES - 1}, got {line}")
    return h ^ (1 << line)


# ---------------------------------------------------------------------------
# Поиск пути
# ---------------------------------------------------------------------------

def shortest_path(start: int, end: int) -> list[int]:
    """
    Кратчайший путь между двумя гексаграммами (BFS).
    Длина пути = расстояние Хэмминга (гарантируется структурой Q6).
    """
    if start == end:
        return [start]
    queue: deque[list[int]] = deque([[start]])
    visited: set[int] = {start}
    while queue:
        path = queue.popleft()
        for nb in neighbors(path[-1]):
            if nb == end:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                queue.append(path + [nb])
    return []  # недостижимо (в Q6 не случается)


def all_paths(start: int, end: int, max_length: int | None = None) -> list[list[int]]:
    """
    Все пути между start и end без повторного посещения узлов.
    max_length ограничивает длину поиска (по умолчанию = 2*LINES).
    """
    if max_length is None:
        max_length = 2 * LINES
    result: list[list[int]] = []

    def dfs(current: int, path: list[int]) -> None:
        if current == end:
            result.append(path[:])
            return
        if len(path) > max_length:
            return
        for nb in neighbors(current):
            if nb not in path:
                path.append(nb)
                dfs(nb, path)
                path.pop()

    dfs(start, [start])
    return result


# ---------------------------------------------------------------------------
# Обход графа
# ---------------------------------------------------------------------------

def gray_code() -> list[int]:
    """
    Код Грея — Гамильтонов путь по Q6:
    посещает все 64 гексаграммы, каждый раз меняя ровно 1 черту.
    """
    n = SIZE
    result = [0] * n
    for i in range(n):
        result[i] = i ^ (i >> 1)
    return result


def bfs_from(start: int) -> dict[int, int]:
    """BFS-расстояния от start до всех остальных гексаграмм."""
    dist: dict[int, int] = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        for nb in neighbors(current):
            if nb not in dist:
                dist[nb] = dist[current] + 1
                queue.append(nb)
    return dist


def antipode(h: int) -> int:
    """Антипод гексаграммы — максимально удалённая (переворот всех 6 черт)."""
    return h ^ (SIZE - 1)


# ---------------------------------------------------------------------------
# Характеристики гексаграммы
# ---------------------------------------------------------------------------

def yang_count(h: int) -> int:
    """Число сплошных черт (ян)."""
    return bin(h).count('1')


def yin_count(h: int) -> int:
    """Число прерывистых черт (инь)."""
    return LINES - yang_count(h)


def upper_trigram(h: int) -> int:
    """Верхняя триграмма (биты 3..5)."""
    return (h >> 3) & 0b111


def lower_trigram(h: int) -> int:
    """Нижняя триграмма (биты 0..2)."""
    return h & 0b111


def render(h: int) -> str:
    """Текстовое отображение гексаграммы (верхняя черта сверху)."""
    bits = to_bits(h)  # бит 5..0
    lines = []
    for i in range(LINES - 1, -1, -1):
        b = (h >> i) & 1
        lines.append('━━━━━━' if b else '━━━  ━━━')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Итераторы
# ---------------------------------------------------------------------------

def all_hexagrams() -> Iterator[int]:
    """Все 64 гексаграммы в порядке 0..63."""
    return iter(range(SIZE))


def hexagrams_by_yang(count: int) -> list[int]:
    """Все гексаграммы с заданным числом сплошных черт (ян)."""
    return [h for h in range(SIZE) if yang_count(h) == count]


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------

def describe(h: int) -> dict:
    """Словарь с основными характеристиками гексаграммы."""
    return {
        'index':        h,
        'bits':         to_bits(h),
        'yang':         yang_count(h),
        'yin':          yin_count(h),
        'upper_tri':    upper_trigram(h),
        'lower_tri':    lower_trigram(h),
        'antipode':     antipode(h),
        'neighbors':    neighbors(h),
    }


if __name__ == '__main__':
    # Быстрая демонстрация
    print("=== hexcore demo ===\n")

    h = 0b101010  # 42
    print(f"Гексаграмма {h} ({to_bits(h)}):")
    print(render(h))
    print()

    info = describe(h)
    print(f"  Ян: {info['yang']}, Инь: {info['yin']}")
    print(f"  Антипод: {info['antipode']} ({to_bits(info['antipode'])})")
    print(f"  Соседи: {info['neighbors']}")
    print()

    path = shortest_path(0, 63)
    print(f"Кратчайший путь 0→63: {path}")
    print(f"  Длина: {len(path)-1} шагов (= расстояние Хэмминга {hamming(0, 63)})")
    print()

    gc = gray_code()
    print(f"Код Грея (первые 8): {gc[:8]}")
    print(f"  Все переходы по 1 биту: {all(hamming(gc[i], gc[i+1]) == 1 for i in range(len(gc)-1))}")
