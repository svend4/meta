"""
hexgraph — граф-теоретический анализ Q6 и его подграфов

Предоставляет инструменты для изучения структурных свойств графа Q6
(6-мерный гиперкуб, 64 вершины, 192 рёбра) и его произвольных подграфов.

Ключевые возможности:
  - Подграфы Q6: создание, проверка свойств
  - Независимые множества, вершинное покрытие, доминирующие множества
  - Хроматическое число, раскраска
  - Поиск гамильтоновых путей/циклов
  - Спектральный анализ (собственные значения матрицы смежности)
  - Изоморфизм малых подграфов
  - Граф пересечений / граф Кнезера на Q6
"""

from __future__ import annotations
import sys
from collections import deque
from itertools import combinations
from typing import Callable, Iterator

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import (
    neighbors, hamming, flip, yang_count, antipode, SIZE,
    gray_code, all_hexagrams, orbit, all_orbits, subcubes,
    apply_permutation,
)


# ---------------------------------------------------------------------------
# Подграф Q6
# ---------------------------------------------------------------------------

class Subgraph:
    """
    Подграф Q6: произвольное подмножество вершин и все рёбра между ними,
    унаследованные из Q6.

    Vertices — frozenset[int], edges — набор пар (u, v) с u < v.
    """

    def __init__(self, vertices: set[int] | frozenset[int]) -> None:
        self.vertices: frozenset[int] = frozenset(vertices)
        self.edges: frozenset[tuple[int, int]] = frozenset(
            (min(u, v), max(u, v))
            for u in self.vertices
            for v in neighbors(u)
            if v in self.vertices
        )

    # ---- базовые свойства -----------------------------------------------

    def __len__(self) -> int:
        return len(self.vertices)

    def __contains__(self, v: int) -> bool:
        return v in self.vertices

    def __repr__(self) -> str:
        return f"Subgraph(n={len(self.vertices)}, m={len(self.edges)})"

    def order(self) -> int:
        """Число вершин."""
        return len(self.vertices)

    def size(self) -> int:
        """Число рёбер."""
        return len(self.edges)

    def degree(self, v: int) -> int:
        """Степень вершины v."""
        return sum(1 for nb in neighbors(v) if nb in self.vertices)

    def degree_sequence(self) -> list[int]:
        """Отсортированная последовательность степеней."""
        return sorted(self.degree(v) for v in self.vertices)

    def adjacency_matrix(self) -> list[list[int]]:
        """
        Матрица смежности (список списков, по алфавитному порядку вершин).
        Возвращает также упорядоченный список вершин.
        """
        verts = sorted(self.vertices)
        n = len(verts)
        idx = {v: i for i, v in enumerate(verts)}
        mat = [[0] * n for _ in range(n)]
        for u, v in self.edges:
            mat[idx[u]][idx[v]] = 1
            mat[idx[v]][idx[u]] = 1
        return mat

    # ---- связность -------------------------------------------------------

    def is_connected(self) -> bool:
        """Проверить связность подграфа."""
        if not self.vertices:
            return True
        start = next(iter(self.vertices))
        visited = {start}
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for nb in neighbors(v):
                if nb in self.vertices and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited == self.vertices

    def connected_components(self) -> list[frozenset[int]]:
        """Список компонент связности."""
        remaining = set(self.vertices)
        components = []
        while remaining:
            start = next(iter(remaining))
            visited = {start}
            queue = deque([start])
            while queue:
                v = queue.popleft()
                for nb in neighbors(v):
                    if nb in remaining and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            components.append(frozenset(visited))
            remaining -= visited
        return components

    def diameter(self) -> int:
        """
        Диаметр подграфа (максимум кратчайших путей).
        Возвращает -1 если несвязный.
        """
        if not self.is_connected():
            return -1
        diam = 0
        for start in self.vertices:
            dist = self._bfs_distances(start)
            diam = max(diam, max(dist.values()))
        return diam

    def _bfs_distances(self, start: int) -> dict[int, int]:
        """BFS расстояния от start до всех вершин подграфа."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for nb in neighbors(v):
                if nb in self.vertices and nb not in dist:
                    dist[nb] = dist[v] + 1
                    queue.append(nb)
        return dist

    def girth(self) -> int:
        """
        Обхват (длина наименьшего цикла).
        Возвращает inf (float) если циклов нет (дерево или лес).
        """
        g = float('inf')
        for start in self.vertices:
            # BFS с отслеживанием родителя
            parent = {start: -1}
            dist_from = {start: 0}
            queue = deque([start])
            while queue:
                v = queue.popleft()
                for nb in neighbors(v):
                    if nb not in self.vertices:
                        continue
                    if nb not in dist_from:
                        dist_from[nb] = dist_from[v] + 1
                        parent[nb] = v
                        queue.append(nb)
                    elif parent[v] != nb:
                        # Найден цикл
                        cycle_len = dist_from[v] + dist_from[nb] + 1
                        g = min(g, cycle_len)
        return g

    # ---- раскраска -------------------------------------------------------

    def chromatic_number_upper(self) -> int:
        """
        Верхняя оценка хроматического числа (жадная раскраска).
        Для Q6 и его подграфов всегда ≤ 2 (двудольный).
        """
        coloring = self.two_coloring()
        if coloring is not None:
            return 1 if len(set(coloring.values())) == 1 else 2
        return 3   # не двудольный (теоретически, но Q6-подграфы всегда двудольны)

    def two_coloring(self) -> dict[int, int] | None:
        """
        Попытка 2-раскраски (BFS). Возвращает dict {vertex: 0|1} или None.
        Все подграфы Q6 двудольны (биты чётности ян-счёта), поэтому
        раскраска всегда существует.
        """
        coloring: dict[int, int] = {}
        for start in self.vertices:
            if start in coloring:
                continue
            coloring[start] = yang_count(start) % 2
            queue = deque([start])
            while queue:
                v = queue.popleft()
                for nb in neighbors(v):
                    if nb not in self.vertices:
                        continue
                    expected = 1 - coloring[v]
                    if nb in coloring:
                        if coloring[nb] != expected:
                            return None
                    else:
                        coloring[nb] = expected
                        queue.append(nb)
        return coloring

    def is_bipartite(self) -> bool:
        return self.two_coloring() is not None

    # ---- независимые множества / доминирование --------------------------

    def max_independent_set(self) -> frozenset[int]:
        """
        Максимальное независимое множество (жадный алгоритм + класс
        двудольности: берём большую долю).
        Для Q6 точный ответ — 32 (все вершины одного класса двудольности).
        """
        col = self.two_coloring()
        if col is None:
            return frozenset()
        part0 = frozenset(v for v, c in col.items() if c == 0)
        part1 = frozenset(v for v, c in col.items() if c == 1)
        return part0 if len(part0) >= len(part1) else part1

    def independence_number(self) -> int:
        """Размер максимального независимого множества."""
        return len(self.max_independent_set())

    def min_dominating_set(self) -> frozenset[int]:
        """
        Минимальное доминирующее множество (жадный покрывающий алгоритм).
        Вершина v доминирует над v и всеми её соседями в подграфе.

        Даёт не всегда оптимальный результат, но хорошую аппроксимацию.
        """
        dominated = set()
        chosen = set()
        uncovered = set(self.vertices)

        while uncovered - dominated:
            # Выбрать вершину, максимизирующую число вновь покрытых
            best = max(
                uncovered,
                key=lambda v: len(
                    ({v} | {nb for nb in neighbors(v) if nb in self.vertices})
                    - dominated
                )
            )
            chosen.add(best)
            dominated.add(best)
            dominated.update(nb for nb in neighbors(best) if nb in self.vertices)

        return frozenset(chosen)

    def domination_number(self) -> int:
        return len(self.min_dominating_set())

    def min_vertex_cover(self) -> frozenset[int]:
        """
        Минимальное вершинное покрытие.
        Для двудольных графов: complement of max independent set (теорема Кёнига).
        """
        mis = self.max_independent_set()
        return frozenset(self.vertices - mis)

    def vertex_cover_number(self) -> int:
        return len(self.min_vertex_cover())

    # ---- гамильтоновость -------------------------------------------------

    def find_hamiltonian_path(
        self,
        start: int | None = None,
    ) -> list[int] | None:
        """
        Поиск гамильтонова пути (DFS с обратным отслеживанием).
        Работает разумно быстро для подграфов Q6 размером до ~20 вершин.
        Для полного Q6 используйте gray_code().
        """
        if not self.vertices:
            return []
        if start is None:
            start = next(iter(self.vertices))

        path = [start]
        visited = {start}

        def dfs() -> bool:
            if len(path) == len(self.vertices):
                return True
            v = path[-1]
            for nb in neighbors(v):
                if nb in self.vertices and nb not in visited:
                    path.append(nb)
                    visited.add(nb)
                    if dfs():
                        return True
                    path.pop()
                    visited.remove(nb)
            return False

        return path if dfs() else None

    def find_hamiltonian_cycle(self) -> list[int] | None:
        """
        Поиск гамильтонова цикла. Возвращает путь где path[0] == path[-1].
        """
        if len(self.vertices) < 3:
            return None
        start = min(self.vertices)
        path = [start]
        visited = {start}

        def dfs() -> bool:
            if len(path) == len(self.vertices):
                # Проверить замыкание
                return path[-1] in [nb for nb in neighbors(start) if nb in self.vertices]
            v = path[-1]
            for nb in sorted(neighbors(v)):
                if nb in self.vertices and nb not in visited:
                    path.append(nb)
                    visited.add(nb)
                    if dfs():
                        return True
                    path.pop()
                    visited.remove(nb)
            return False

        if dfs():
            return path + [path[0]]
        return None

    # ---- клики -----------------------------------------------------------

    def max_clique(self) -> frozenset[int]:
        """
        Максимальная клика. В гиперкубе максимальная клика — ребро (размер 2).
        Возвращает frozenset вершин.
        """
        best: frozenset[int] = frozenset()
        # Bron-Kerbosch
        def bk(R: set, P: set, X: set) -> None:
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = frozenset(R)
                return
            pivot = max(P | X, key=lambda v: len(
                {nb for nb in neighbors(v) if nb in P}
            ))
            pivot_nbrs = {nb for nb in neighbors(pivot) if nb in self.vertices}
            for v in list(P - pivot_nbrs):
                nbrs_v = {nb for nb in neighbors(v) if nb in self.vertices}
                bk(R | {v}, P & nbrs_v, X & nbrs_v)
                P.remove(v)
                X.add(v)

        bk(set(), set(self.vertices), set())
        return best

    def clique_number(self) -> int:
        return len(self.max_clique())

    # ---- изоморфизм (малые графы) ----------------------------------------

    def is_isomorphic_to(self, other: 'Subgraph') -> bool:
        """
        Проверка изоморфизма (полный перебор, только для малых графов n ≤ 12).
        Использует перебор биекций с обрезкой по степеням.
        """
        if len(self.vertices) != len(other.vertices):
            return False
        if len(self.edges) != len(other.edges):
            return False
        if self.degree_sequence() != other.degree_sequence():
            return False
        if len(self.vertices) > 12:
            raise ValueError("is_isomorphic_to поддерживает только графы с n ≤ 12")

        verts_s = sorted(self.vertices)
        verts_o = sorted(other.vertices)
        adj_s = {v: {nb for nb in neighbors(v) if nb in self.vertices} for v in verts_s}
        adj_o = {v: {nb for nb in neighbors(v) if nb in other.vertices} for v in verts_o}

        from itertools import permutations
        # Группировка по степеням для ускорения
        deg_s = {v: len(adj_s[v]) for v in verts_s}
        deg_o = {v: len(adj_o[v]) for v in verts_o}

        def backtrack(mapping: dict, remaining_s: list, remaining_o: list) -> bool:
            if not remaining_s:
                return True
            v_s = remaining_s[0]
            dv = deg_s[v_s]
            for v_o in remaining_o:
                if deg_o[v_o] != dv:
                    continue
                # Проверить совместимость
                ok = all(
                    (mapping[nb_s] in adj_o[v_o]) == (nb_s in adj_s[v_s])
                    for nb_s in mapping
                )
                if ok:
                    mapping[v_s] = v_o
                    new_rem_o = [x for x in remaining_o if x != v_o]
                    if backtrack(mapping, remaining_s[1:], new_rem_o):
                        return True
                    del mapping[v_s]
            return False

        return backtrack({}, verts_s, verts_o)


# ---------------------------------------------------------------------------
# Глобальные свойства Q6
# ---------------------------------------------------------------------------

def q6_full() -> Subgraph:
    """Полный граф Q6."""
    return Subgraph(set(range(SIZE)))


def q6_layer(yang: int) -> Subgraph:
    """
    Слой гексаграмм с ровно yang ян-чертами.
    Слои Q6: L0..L6, |Lk| = C(6,k).
    """
    from libs.hexcore.hexcore import hexagrams_by_yang
    verts = hexagrams_by_yang(yang)
    return Subgraph(set(verts))


def q6_ball(center: int, radius: int) -> Subgraph:
    """Шар радиуса radius вокруг center."""
    from libs.hexcore.hexcore import ball
    return Subgraph(set(ball(center, radius)))


def induced_subgraph(vertices: set[int]) -> Subgraph:
    """Индуцированный подграф на заданном наборе вершин."""
    return Subgraph(vertices)


# ---------------------------------------------------------------------------
# Спектральный анализ (без numpy — методом степенного итерирования)
# ---------------------------------------------------------------------------

def _mat_vec(mat: list[list[int]], vec: list[float]) -> list[float]:
    n = len(vec)
    return [sum(mat[i][j] * vec[j] for j in range(n)) for i in range(n)]


def _norm(vec: list[float]) -> float:
    import math
    return math.sqrt(sum(x * x for x in vec))


def largest_eigenvalue(subgraph: Subgraph, max_iter: int = 200) -> float:
    """
    Наибольшее собственное значение матрицы смежности (метод степеней).
    Для Q6: λ_max = 6 (степень регулярного графа).
    """
    mat = subgraph.adjacency_matrix()
    n = len(mat)
    if n == 0:
        return 0.0
    # Начальный вектор
    vec = [1.0 / n] * n
    for _ in range(max_iter):
        new_vec = _mat_vec(mat, vec)
        norm = _norm(new_vec)
        if norm < 1e-12:
            break
        vec = [x / norm for x in new_vec]
    # Rayleigh quotient
    av = _mat_vec(mat, vec)
    return sum(av[i] * vec[i] for i in range(n))


def spectrum_approx(subgraph: Subgraph, k: int = 6) -> list[float]:
    """
    Аппроксимация k наибольших собственных значений (deflation).
    Работает точно для малых подграфов.
    """
    mat = [row[:] for row in subgraph.adjacency_matrix()]
    n = len(mat)
    k = min(k, n)
    eigenvalues = []

    current_mat = [row[:] for row in mat]

    for _ in range(k):
        vec = [1.0 / n] * n
        lam = 0.0
        for _ in range(200):
            new_vec = _mat_vec(current_mat, vec)
            norm = _norm(new_vec)
            if norm < 1e-12:
                break
            vec = [x / norm for x in new_vec]
            lam_new = sum(_mat_vec(current_mat, vec)[i] * vec[i] for i in range(n))
            if abs(lam_new - lam) < 1e-9:
                lam = lam_new
                break
            lam = lam_new
        eigenvalues.append(round(lam, 6))
        # Deflate: A := A - λ * v * vᵀ
        for i in range(n):
            for j in range(n):
                current_mat[i][j] -= lam * vec[i] * vec[j]

    return eigenvalues


# ---------------------------------------------------------------------------
# Автоморфизмы
# ---------------------------------------------------------------------------

def bit_permutation_automorphism(perm: list[int]) -> Callable[[int], int]:
    """
    Создать автоморфизм Q6 из перестановки битов.
    Существует 6! = 720 таких автоморфизмов.
    """
    return lambda h: apply_permutation(h, perm)


def xor_automorphism(mask: int) -> Callable[[int], int]:
    """
    Создать автоморфизм Q6 из XOR-сдвига (трансляция на mask).
    Существует 2^6 = 64 таких автоморфизмов.
    """
    return lambda h: h ^ mask


def count_automorphisms() -> int:
    """
    Число автоморфизмов Q6 = 2^6 × 6! = 64 × 720 = 46 080.
    (Порядок группы автоморфизмов гиперкуба B6.)
    """
    import math
    return (2 ** 6) * math.factorial(6)


# ---------------------------------------------------------------------------
# Граф расстояний (distance-k graph)
# ---------------------------------------------------------------------------

def distance_graph(k: int) -> Subgraph:
    """
    Граф расстояний D_k(Q6): вершины = гексаграммы,
    рёбра = пары на расстоянии Хэмминга ровно k.
    Возвращает Subgraph с дополнительным набором рёбер (не только Q6-рёбра).
    """
    # Мы не можем использовать Subgraph (он берёт рёбра из Q6),
    # поэтому возвращаем структуру напрямую
    edges = set()
    for h in range(SIZE):
        for other in range(h + 1, SIZE):
            if hamming(h, other) == k:
                edges.add((h, other))
    return _DistanceSubgraph(set(range(SIZE)), edges)


class _DistanceSubgraph(Subgraph):
    """Subgraph с произвольным набором рёбер (для distance-k графов)."""

    def __init__(self, vertices: set[int], edges: set[tuple[int, int]]) -> None:
        self.vertices = frozenset(vertices)
        self.edges = frozenset(edges)


# ---------------------------------------------------------------------------
# Утилиты анализа
# ---------------------------------------------------------------------------

def analyze(subgraph: Subgraph) -> dict:
    """
    Полный анализ подграфа Q6.
    Возвращает словарь с основными характеристиками.
    """
    connected = subgraph.is_connected()
    result: dict = {
        'order': subgraph.order(),
        'size': subgraph.size(),
        'connected': connected,
        'bipartite': subgraph.is_bipartite(),
        'degree_sequence': subgraph.degree_sequence(),
        'independence_number': subgraph.independence_number(),
        'vertex_cover_number': subgraph.vertex_cover_number(),
        'domination_number': subgraph.domination_number(),
        'clique_number': subgraph.clique_number(),
    }
    if connected:
        result['diameter'] = subgraph.diameter()
        result['girth'] = subgraph.girth()
    else:
        n_comp = len(subgraph.connected_components())
        result['components'] = n_comp
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, json

    parser = argparse.ArgumentParser(description='hexgraph — граф Q6 анализатор')
    sub = parser.add_subparsers(dest='cmd')

    p_full = sub.add_parser('q6', help='Анализ полного Q6')
    p_full.add_argument('--spectrum', action='store_true', help='Спектр матрицы смежности')

    p_layer = sub.add_parser('layer', help='Анализ слоя (yang=k)')
    p_layer.add_argument('yang', type=int, help='Число ян-черт (0..6)')

    p_ball = sub.add_parser('ball', help='Анализ шара вокруг h')
    p_ball.add_argument('center', type=int)
    p_ball.add_argument('radius', type=int)

    p_sub = sub.add_parser('subgraph', help='Анализ подграфа на заданных вершинах')
    p_sub.add_argument('vertices', nargs='+', type=int)

    p_iso = sub.add_parser('isocheck', help='Проверка изоморфизма двух подграфов')
    p_iso.add_argument('--g1', nargs='+', type=int, required=True)
    p_iso.add_argument('--g2', nargs='+', type=int, required=True)

    p_ham = sub.add_parser('hamilton', help='Поиск гамильтонова пути')
    p_ham.add_argument('vertices', nargs='+', type=int)
    p_ham.add_argument('--cycle', action='store_true')

    p_dist = sub.add_parser('distgraph', help='Граф расстояний D_k(Q6)')
    p_dist.add_argument('k', type=int)

    args = parser.parse_args()

    def _print_analysis(g: Subgraph, title: str) -> None:
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        info = analyze(g)
        for k, v in info.items():
            print(f"  {k:25s}: {v}")

    if args.cmd == 'q6':
        g = q6_full()
        _print_analysis(g, 'Полный граф Q6')
        print(f"  {'automorphisms':25s}: {count_automorphisms():,}")
        if args.spectrum:
            print(f"\n  Спектр (6 largest eigenvalues): {spectrum_approx(g, k=6)}")

    elif args.cmd == 'layer':
        g = q6_layer(args.yang)
        _print_analysis(g, f'Слой Q6 (yang={args.yang}), C(6,{args.yang})={len(g)} вершин')

    elif args.cmd == 'ball':
        g = q6_ball(args.center, args.radius)
        _print_analysis(g, f'Шар B({args.center}, r={args.radius})')

    elif args.cmd == 'subgraph':
        g = induced_subgraph(set(args.vertices))
        _print_analysis(g, f'Подграф на {args.vertices}')

    elif args.cmd == 'isocheck':
        g1 = induced_subgraph(set(args.g1))
        g2 = induced_subgraph(set(args.g2))
        if max(len(g1), len(g2)) > 12:
            print("Ошибка: изоморфизм поддерживается для n ≤ 12")
        else:
            result = g1.is_isomorphic_to(g2)
            print(f"Изоморфны: {result}")

    elif args.cmd == 'hamilton':
        g = induced_subgraph(set(args.vertices))
        if args.cycle:
            path = g.find_hamiltonian_cycle()
            label = 'Гамильтонов цикл'
        else:
            path = g.find_hamiltonian_path()
            label = 'Гамильтонов путь'
        if path:
            print(f"{label}: {' → '.join(map(str, path))}")
        else:
            print(f"{label} не найден")

    elif args.cmd == 'distgraph':
        g = distance_graph(args.k)
        print(f"\nГраф расстояний D_{args.k}(Q6):")
        print(f"  Вершин: {g.order()}")
        print(f"  Рёбер:  {g.size()}")

    else:
        parser.print_help()
