"""
hexnet — Q6 как коммуникационная сеть (гиперкубическая сеть)

Гиперкубические сети Q_n используются в параллельных вычислениях:
каждый узел — процессор, каждое ребро — канал связи.
Q6 (n=6) = 64 узла, 192 ребра, степень каждого узла = 6.

Ключевые характеристики Q6:
  Диаметр              : 6 (максимальный путь)
  Степень вершин       : 6 (число прямых соседей)
  Ширина бисекции      : 32 (Хэмминг: половинное разбиение по биту i)
  Узловая связность    : 6 (потребует 6 отказов для изоляции вершины)
  Рёберная связность   : 6 (= степени вершин)
  Гамильтоновый        : да (содержит Гамильтонов цикл)

Маршрутизация:
  E-cube (dimension-ordered): детерминированная, длина = hamming(src, dst)
  Адаптивная: объезжает отказавшие узлы

Широковещание:
  Алгоритм BFS: одновременная рассылка от корня за 6 шагов
  Gossip (all-to-all): O(log n) = 6 шагов
"""

from __future__ import annotations
import sys
import random
from collections import deque, defaultdict

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import neighbors, hamming, yang_count, SIZE


# ---------------------------------------------------------------------------
# Маршрутизация
# ---------------------------------------------------------------------------

def ecube_route(src: int, dst: int) -> list[int]:
    """
    E-cube (dimension-ordered) маршрутизация: исправлять биты по очереди 0..5.
    Длина маршрута = hamming(src, dst).
    Детерминированная, минимальная.
    """
    path = [src]
    current = src
    diff = current ^ dst
    for bit in range(6):
        if diff & (1 << bit):
            current ^= (1 << bit)
            path.append(current)
    return path


def all_shortest_paths(src: int, dst: int) -> list[list[int]]:
    """
    Все кратчайшие пути от src до dst.
    Только «монотонные» пути (без повторных посещений, только уменьшая расстояние).
    Число таких путей = hamming(src,dst)! (факториал расстояния).
    """
    if src == dst:
        return [[src]]
    d = hamming(src, dst)
    result = []

    def dfs(current: int, path: list[int]) -> None:
        if current == dst:
            result.append(path[:])
            return
        for nb in neighbors(current):
            if hamming(nb, dst) < hamming(current, dst):
                path.append(nb)
                dfs(nb, path)
                path.pop()

    dfs(src, [src])
    return result


def count_shortest_paths(src: int, dst: int) -> int:
    """
    Число кратчайших путей от src до dst = hamming(src,dst)!
    (Переставить d шагов-измерений всеми способами.)
    """
    import math
    return math.factorial(hamming(src, dst))


def fault_tolerant_route(
    src: int,
    dst: int,
    failed_nodes: set[int] | None = None,
    failed_edges: set[frozenset] | None = None,
) -> list[int] | None:
    """
    Маршрутизация с обходом отказавших узлов/рёбер.
    Использует BFS (Дейкстра по числу хопов).
    Возвращает путь или None если dst недостижим.
    """
    bad_nodes = failed_nodes or set()
    bad_edges = failed_edges or set()

    if src in bad_nodes or dst in bad_nodes:
        return None

    visited = {src: None}  # vertex → predecessor
    queue = deque([src])

    while queue:
        current = queue.popleft()
        if current == dst:
            # Восстановить путь
            path = []
            v = dst
            while v is not None:
                path.append(v)
                v = visited[v]
            return path[::-1]

        for nb in neighbors(current):
            if nb in bad_nodes:
                continue
            edge = frozenset({current, nb})
            if edge in bad_edges:
                continue
            if nb not in visited:
                visited[nb] = current
                queue.append(nb)

    return None  # dst не достижим


def adaptive_route(
    src: int,
    dst: int,
    failed_nodes: set[int] | None = None,
) -> list[int] | None:
    """
    Адаптивная маршрутизация: предпочитать соседей, уменьшающих расстояние до dst.
    Если все «хорошие» соседи заняты — использовать BFS-обход.
    """
    bad = failed_nodes or set()
    if src in bad or dst in bad:
        return None

    # Сначала пробуем жадный подход
    path = [src]
    current = src
    visited_greedy = {src}

    while current != dst:
        # Предпочтительные соседи: уменьшают расстояние до dst
        preferred = [nb for nb in neighbors(current)
                     if nb not in bad and nb not in visited_greedy
                     and hamming(nb, dst) < hamming(current, dst)]

        if preferred:
            # Выбрать сосед с минимальным расстоянием до dst
            nxt = min(preferred, key=lambda h: hamming(h, dst))
        else:
            # Сбой жадного — откат к BFS
            return fault_tolerant_route(src, dst, failed_nodes)

        path.append(nxt)
        visited_greedy.add(nxt)
        current = nxt

    return path


# ---------------------------------------------------------------------------
# Широковещание
# ---------------------------------------------------------------------------

def broadcast_tree(root: int) -> dict[int, int | None]:
    """
    Дерево широковещания BFS от корня root.
    Возвращает parent[v] = родитель v в BFS-дереве (parent[root] = None).
    """
    parent: dict[int, int | None] = {root: None}
    queue = deque([root])
    while queue:
        v = queue.popleft()
        for nb in neighbors(v):
            if nb not in parent:
                parent[nb] = v
                queue.append(nb)
    return parent


def broadcast_schedule(root: int) -> list[list[tuple[int, int]]]:
    """
    Расписание широковещания: список шагов, на каждом шаге — список (отправитель, получатель).
    BFS-вариант: узел отправляет всем своим детям одновременно.
    Завершается за diameter(Q6) = 6 шагов.
    """
    parent = broadcast_tree(root)
    children: dict[int, list[int]] = defaultdict(list)
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)

    schedule = []
    level = [root]
    while level:
        step = []
        next_level = []
        for v in level:
            for c in children[v]:
                step.append((v, c))
                next_level.append(c)
        if step:
            schedule.append(step)
        level = next_level
    return schedule


def gossip(seed: int | None = None) -> int:
    """
    Gossip (all-to-all информирование): минимальное число раундов до
    того, как каждый узел узнает информацию от всех остальных.
    Для Q_n: 2n - 2 = 10 раундов (оценка).
    Возвращает теоретическую нижнюю оценку.
    """
    # Для Q_n gossip-оптимальное число раундов = 2n - 2 (известный результат)
    return 2 * 6 - 2   # = 10


def broadcast_steps(root: int = 0) -> int:
    """Число шагов для завершения BFS-широковещания = диаметр = 6."""
    return len(broadcast_schedule(root))


# ---------------------------------------------------------------------------
# Характеристики сети
# ---------------------------------------------------------------------------

def network_diameter(failed_nodes: set[int] | None = None) -> int:
    """
    Диаметр сети (макс кратчайший путь) с учётом отказавших узлов.
    Без отказов: diameter(Q6) = 6.
    """
    if not failed_nodes:
        return 6  # точное значение для Q6

    active = [h for h in range(SIZE) if h not in failed_nodes]
    if len(active) < 2:
        return 0

    max_d = 0
    for i, u in enumerate(active):
        for v in active[i + 1:]:
            path = fault_tolerant_route(u, v, failed_nodes)
            if path is None:
                return float('inf')  # несвязная сеть
            max_d = max(max_d, len(path) - 1)
    return max_d


def bisection_width() -> int:
    """
    Ширина бисекции Q6: минимальное число рёбер между двумя равными полуразделами.
    Оптимальное разбиение: фиксировать бит i (32 узла с битом i=0, 32 с битом i=1).
    Число пересекающих рёбер = 2^5 = 32.
    """
    return 32  # 2^{n-1} для Q_n


def node_connectivity() -> int:
    """
    Узловая связность Q6: минимальное число узлов, удаление которых
    разрывает сеть. Для Q_n = n = 6 (максимально возможная, т.к. min_degree = n).
    """
    return 6


def edge_connectivity() -> int:
    """
    Рёберная связность Q6: минимальное число рёбер, удаление которых
    разрывает сеть. Для Q_n = n = 6 (= min_degree по теореме Уитни).
    """
    return 6


def average_path_length() -> float:
    """
    Среднее кратчайшее расстояние между всеми парами узлов Q6.
    E[d] = Σ_{k=0}^{6} k × C(6,k) / 63 = ... = 3.0 (половина диаметра).
    """
    total = sum(hamming(u, v) for u in range(SIZE) for v in range(u + 1, SIZE))
    pairs = SIZE * (SIZE - 1) // 2
    return total / pairs


def ecube_path_length_distribution() -> dict[int, int]:
    """
    Распределение длин путей e-cube по всем парам (src, dst):
    {длина: число пар} = {d: C(6,d) × C(63,1)} — нет, проще:
    {d: число пар на расстоянии d} = {d: C(6,d) × C(64,1)} нет...
    Правильно: число пар (u,v) с hamming(u,v)=d = C(6,d) × 64 / 2 нет.

    Для Q6: число упорядоченных пар на расстоянии d = 64 × C(6,d).
    """
    dist: dict[int, int] = defaultdict(int)
    for u in range(SIZE):
        for v in range(SIZE):
            if u != v:
                dist[hamming(u, v)] += 1
    return dict(sorted(dist.items()))


# ---------------------------------------------------------------------------
# Надёжность (перколяция)
# ---------------------------------------------------------------------------

def bond_percolation(
    src: int,
    dst: int,
    p_fail: float,
    n_trials: int = 2000,
    seed: int | None = None,
) -> float:
    """
    Оценка надёжности сети методом Монте-Карло (bond percolation):
    вероятность того, что src достигает dst при независимом отказе
    каждого ребра с вероятностью p_fail.
    """
    rng = random.Random(seed)
    # Список рёбер Q6
    edges = [(u, v) for u in range(SIZE) for v in neighbors(u) if u < v]
    success = 0

    for _ in range(n_trials):
        # Случайно удалить рёбра
        bad_edges = frozenset(
            frozenset({u, v}) for u, v in edges if rng.random() < p_fail
        )
        path = fault_tolerant_route(src, dst, failed_edges=bad_edges)
        if path is not None:
            success += 1

    return success / n_trials


def site_percolation(
    p_fail: float,
    n_trials: int = 1000,
    seed: int | None = None,
) -> float:
    """
    Оценка доли гигантской компоненты (site percolation):
    при независимом отказе каждого узла с вероятностью p_fail,
    какова средняя доля узлов в наибольшей связной компоненте?
    """
    rng = random.Random(seed)
    fractions = []

    for _ in range(n_trials):
        bad = frozenset(h for h in range(SIZE) if rng.random() < p_fail)
        active = set(range(SIZE)) - bad
        if not active:
            fractions.append(0.0)
            continue

        # BFS для нахождения наибольшей компоненты
        max_comp = 0
        visited = set()
        for start in active:
            if start in visited:
                continue
            comp = {start}
            q = deque([start])
            while q:
                v = q.popleft()
                for nb in neighbors(v):
                    if nb in active and nb not in comp:
                        comp.add(nb)
                        q.append(nb)
            visited |= comp
            max_comp = max(max_comp, len(comp))

        fractions.append(max_comp / SIZE)

    return sum(fractions) / len(fractions)


def k_fault_diameter(k: int, n_trials: int = 100, seed: int | None = None) -> int:
    """
    Эмпирический диаметр Q6 при k случайных отказавших узлах.
    Возвращает максимальное наблюдённое расстояние по n_trials реализациям.
    """
    rng = random.Random(seed)
    max_diam = 0

    for _ in range(n_trials):
        bad = set(rng.sample(range(SIZE), k))
        active = [h for h in range(SIZE) if h not in bad]
        if len(active) < 2:
            continue
        # Выборочно проверить пары
        sample_pairs = [(active[i], active[j])
                        for i in range(len(active))
                        for j in range(i + 1, min(i + 5, len(active)))]
        for u, v in sample_pairs:
            path = fault_tolerant_route(u, v, bad)
            if path is not None:
                max_diam = max(max_diam, len(path) - 1)

    return max_diam


# ---------------------------------------------------------------------------
# Трафик и нагрузка
# ---------------------------------------------------------------------------

def ecube_traffic_load() -> dict[frozenset, int]:
    """
    Нагрузка на рёбра при e-cube маршрутизации (всё-ко-всему).
    load[edge] = число маршрутов, проходящих через edge.
    """
    load: dict[frozenset, int] = defaultdict(int)
    for src in range(SIZE):
        for dst in range(SIZE):
            if src == dst:
                continue
            path = ecube_route(src, dst)
            for i in range(len(path) - 1):
                e = frozenset({path[i], path[i + 1]})
                load[e] += 1
    return dict(load)


def traffic_statistics() -> dict[str, float]:
    """
    Статистика нагрузки при e-cube маршрутизации.
    """
    load = ecube_traffic_load()
    values = list(load.values())
    avg = sum(values) / len(values)
    mx = max(values)
    mn = min(values)
    return {
        'total_routes': SIZE * (SIZE - 1),
        'edges': len(load),
        'avg_load': round(avg, 2),
        'max_load': mx,
        'min_load': mn,
        'load_imbalance': round(mx / avg, 3),
    }


def hot_spot_edges(top_k: int = 5) -> list[tuple[frozenset, int]]:
    """Топ-k наиболее нагруженных рёбер."""
    load = ecube_traffic_load()
    return sorted(load.items(), key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# Сетевые задачи
# ---------------------------------------------------------------------------

def find_hamiltonian_path() -> list[int]:
    """
    Найти Гамильтонов путь в Q6.
    Q6 = Gray code: стандартный обход по коду Грея.
    Gray(i) = i XOR (i >> 1).
    """
    return [i ^ (i >> 1) for i in range(SIZE)]


def gray_code_cycle() -> list[int]:
    """
    Гамильтонов цикл в Q6: код Грея длиной 64 (замкнутый).
    Переход между соседями: XOR ровно одного бита.
    """
    path = find_hamiltonian_path()
    # Проверить замыкаемость: путь[63] и путь[0] должны отличаться на 1 бит
    return path


def is_hamiltonian_path(path: list[int]) -> bool:
    """Проверить, является ли path Гамильтоновым путём в Q6."""
    if len(path) != SIZE or len(set(path)) != SIZE:
        return False
    for i in range(len(path) - 1):
        if hamming(path[i], path[i + 1]) != 1:
            return False
    return True


def is_hamiltonian_cycle(path: list[int]) -> bool:
    """Проверить, является ли path Гамильтоновым циклом в Q6."""
    return is_hamiltonian_path(path) and hamming(path[-1], path[0]) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='hexnet — Q6 как сеть')
    sub = parser.add_subparsers(dest='cmd')

    p_route = sub.add_parser('route', help='Маршрутизация src → dst')
    p_route.add_argument('src', type=int)
    p_route.add_argument('dst', type=int)
    p_route.add_argument('--algo', choices=['ecube', 'all', 'adaptive'], default='ecube')
    p_route.add_argument('--failed', nargs='*', type=int, default=[],
                          help='Отказавшие узлы')

    p_bcast = sub.add_parser('broadcast', help='Дерево и расписание широковещания')
    p_bcast.add_argument('root', type=int, nargs='?', default=0)

    p_stats = sub.add_parser('stats', help='Статистика сети Q6')

    p_perc = sub.add_parser('percolation', help='Монте-Карло перколяция')
    p_perc.add_argument('--src', type=int, default=0)
    p_perc.add_argument('--dst', type=int, default=63)
    p_perc.add_argument('--trials', type=int, default=2000)
    p_perc.add_argument('--seed', type=int, default=42)

    p_traf = sub.add_parser('traffic', help='Нагрузка на рёбра (e-cube)')

    p_ham = sub.add_parser('hamilton', help='Гамильтонов путь (код Грея)')

    args = parser.parse_args()

    if args.cmd == 'route':
        bad = set(args.failed)
        if args.algo == 'ecube':
            path = ecube_route(args.src, args.dst)
            print(f"E-cube {args.src}→{args.dst}: длина={len(path)-1}")
            print(f"  Путь: {path}")
        elif args.algo == 'all':
            paths = all_shortest_paths(args.src, args.dst)
            d = hamming(args.src, args.dst)
            print(f"Все кратчайшие пути {args.src}→{args.dst}: "
                  f"{len(paths)} путей длиной {d}")
            for p in paths[:5]:
                print(f"  {p}")
            if len(paths) > 5:
                print(f"  ... и ещё {len(paths)-5}")
        elif args.algo == 'adaptive':
            path = adaptive_route(args.src, args.dst, bad or None)
            if path:
                print(f"Адаптивный {args.src}→{args.dst} (отказы={list(bad)}): "
                      f"длина={len(path)-1}")
                print(f"  Путь: {path}")
            else:
                print(f"Путь не найден: {args.dst} недостижим из {args.src}")

    elif args.cmd == 'broadcast':
        sched = broadcast_schedule(args.root)
        print(f"Широковещание от узла {args.root}:")
        print(f"  Шагов: {len(sched)} (= диаметр Q6)")
        for step, msgs in enumerate(sched):
            print(f"  Шаг {step+1}: {len(msgs)} сообщений  {msgs[:3]}{'...' if len(msgs) > 3 else ''}")

    elif args.cmd == 'stats':
        apl = average_path_length()
        stats = traffic_statistics()
        print("Характеристики сети Q6:")
        print(f"  Узлов               : {SIZE}")
        total_edges = sum(len(neighbors(h)) for h in range(SIZE)) // 2
        print(f"  Рёбер               : {total_edges}")
        print(f"  Степень вершин      : 6")
        print(f"  Диаметр             : 6")
        print(f"  Среднее расстояние  : {apl:.4f}")
        print(f"  Ширина бисекции     : {bisection_width()}")
        print(f"  Узловая связность   : {node_connectivity()}")
        print(f"  Рёберная связность  : {edge_connectivity()}")
        print(f"  Гамильтонов цикл    : да (код Грея)")
        print(f"\nE-cube трафик:")
        for k, v in stats.items():
            print(f"  {k:<20}: {v}")

    elif args.cmd == 'percolation':
        print(f"Bond percolation: {args.src}→{args.dst} ({args.trials} пробы)")
        print(f"{'p_fail':>8}  {'P(связь)':>10}")
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            prob = bond_percolation(args.src, args.dst, p, args.trials, args.seed)
            bar = '█' * int(prob * 20)
            print(f"{p:8.1f}  {prob:10.4f}  {bar}")

    elif args.cmd == 'traffic':
        stats = traffic_statistics()
        hot = hot_spot_edges()
        print("Нагрузка рёбер (e-cube, все пары):")
        for k, v in stats.items():
            print(f"  {k:<20}: {v}")
        print("\nТоп-5 нагруженных рёбер:")
        for edge, load in hot:
            u, v = tuple(edge)
            print(f"  {{{u:2d},{v:2d}}}: {load} маршрутов")

    elif args.cmd == 'hamilton':
        path = find_hamiltonian_path()
        print(f"Гамильтонов путь Q6 (код Грея, {len(path)} узлов):")
        for i, h in enumerate(path):
            b = format(h, '06b')
            change = '' if i == 0 else f'← бит {(path[i-1]^h).bit_length()-1}'
            print(f"  [{i:2d}] {b} ({h:2d}) {change}")
        is_hp = is_hamiltonian_path(path)
        is_hc = is_hamiltonian_cycle(path)
        print(f"\nГамильтонов путь : {is_hp}")
        print(f"Гамильтонов цикл : {is_hc}")

    else:
        parser.print_help()
