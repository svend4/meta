"""
hexsym — группа автоморфизмов Q6 и теория симметрий

Граф Q6 — 6-мерный гиперкуб. Его группа автоморфизмов:

    Aut(Q6) = B₆ = S₆ ⋉ (Z₂)⁶   (гиперoctaэдральная группа)

  |Aut(Q6)| = 6! × 2⁶ = 720 × 64 = 46 080

Каждый автоморфизм задаётся парой (π, m):
  - π ∈ S₆ — перестановка 6 позиций битов
  - m ∈ {0..63} — маска XOR (глобальный «переворот» битов)
  Действие: h → π(h) ⊕ m

Умножение: (π₁,m₁) · (π₂,m₂) = (π₁∘π₂, π₁(m₂) ⊕ m₁)

Ключевые результаты:
  - Q6 вершинно-транзитивен: одна орбита на вершинах (под Aut(Q6))
  - Под S₆ (перестановки битов): 7 орбит по весу Хэмминга (= yang_count)
  - Лемма Бёрнсайда: число различных раскрасок = среднее число неподвижных
  - Порядок Aut(Q6) = 46080 = |S₆| × |Z₂|⁶

Стандартные генераторы Aut(Q6):
  - τ_i = транспозиция (i, i+1) битов, i=0..4 (5 генераторов S₆)
  - σ = переворот бита 0 (генератор (Z₂)⁶ часть)
  → 6 генераторов порождают всю Aut(Q6)
"""

from __future__ import annotations
import sys
from itertools import permutations, combinations
from collections import defaultdict
from functools import lru_cache

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import neighbors, hamming, yang_count, SIZE


# ---------------------------------------------------------------------------
# Класс автоморфизма
# ---------------------------------------------------------------------------

class Automorphism:
    """
    Автоморфизм Q6: (π, m) где π ∈ S₆ и m ∈ (Z₂)⁶.
    Действие на h: π(h) ⊕ m.

    π кодируется кортежем из 6 элементов: perm[i] = позиция, куда идёт бит i.
    """

    __slots__ = ('perm', 'mask')

    def __init__(self, perm: tuple[int, ...], mask: int) -> None:
        self.perm = tuple(perm)
        self.mask = mask & 63

    def __call__(self, h: int) -> int:
        """Применить автоморфизм к гексаграмме h."""
        result = 0
        for i in range(6):
            if (h >> i) & 1:
                result |= 1 << self.perm[i]
        return result ^ self.mask

    def __mul__(self, other: 'Automorphism') -> 'Automorphism':
        """
        Композиция self ∘ other: сначала other, потом self.
        (π₁,m₁) · (π₂,m₂) = (π₁∘π₂, π₁(m₂) ⊕ m₁)
        """
        new_perm = tuple(self.perm[other.perm[i]] for i in range(6))
        # Применить перестановку π₁ к маске m₂
        pm2 = 0
        for i in range(6):
            if (other.mask >> i) & 1:
                pm2 |= 1 << self.perm[i]
        return Automorphism(new_perm, pm2 ^ self.mask)

    def inverse(self) -> 'Automorphism':
        """
        Обратный автоморфизм: (π,m)⁻¹ = (π⁻¹, π⁻¹(m)).
        """
        inv_perm = [0] * 6
        for i in range(6):
            inv_perm[self.perm[i]] = i
        inv_perm = tuple(inv_perm)
        inv_m = 0
        for i in range(6):
            if (self.mask >> i) & 1:
                inv_m |= 1 << inv_perm[i]
        return Automorphism(inv_perm, inv_m)

    def order(self) -> int:
        """Порядок элемента в группе: наименьшее k > 0 с self^k = id."""
        current = self
        identity = identity_aut()
        for k in range(1, 50):
            if current.perm == identity.perm and current.mask == identity.mask:
                return k
            current = self * current
        return -1  # не должно случиться

    def is_identity(self) -> bool:
        return self.perm == (0, 1, 2, 3, 4, 5) and self.mask == 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Automorphism):
            return self.perm == other.perm and self.mask == other.mask
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.perm, self.mask))

    def __repr__(self) -> str:
        return f"Aut(π={list(self.perm)}, m={self.mask:06b})"


# ---------------------------------------------------------------------------
# Стандартные автоморфизмы
# ---------------------------------------------------------------------------

def identity_aut() -> Automorphism:
    """Тождественный автоморфизм."""
    return Automorphism((0, 1, 2, 3, 4, 5), 0)


def bit_transposition(i: int, j: int) -> Automorphism:
    """Транспозиция: поменять биты i и j местами."""
    perm = list(range(6))
    perm[i], perm[j] = perm[j], perm[i]
    return Automorphism(perm, 0)


def bit_flip_single(i: int) -> Automorphism:
    """Переворот бита i: m = 1 << i."""
    return Automorphism((0, 1, 2, 3, 4, 5), 1 << i)


def bit_flip_mask(mask: int) -> Automorphism:
    """XOR с произвольной маской mask."""
    return Automorphism((0, 1, 2, 3, 4, 5), mask)


def bit_permutation(perm: list[int] | tuple[int, ...]) -> Automorphism:
    """Автоморфизм из перестановки с нулевой маской."""
    return Automorphism(tuple(perm), 0)


def complement_aut() -> Automorphism:
    """Дополнение: h → 63 - h = ~h (переворот всех бит). Порядок 2."""
    return Automorphism((0, 1, 2, 3, 4, 5), 63)


# ---------------------------------------------------------------------------
# Стандартные порождающие множества
# ---------------------------------------------------------------------------

def s6_generators() -> list[Automorphism]:
    """
    Смежные транспозиции (0 1), (1 2), ..., (4 5) — порождают S₆.
    """
    return [bit_transposition(i, i + 1) for i in range(5)]


def aut_generators() -> list[Automorphism]:
    """
    6 генераторов Aut(Q6) = B₆:
    - 5 смежных транспозиций (порождают S₆)
    - переворот бита 0 (совместно с S₆ порождает (Z₂)⁶ часть)
    """
    return s6_generators() + [bit_flip_single(0)]


# ---------------------------------------------------------------------------
# Орбиты
# ---------------------------------------------------------------------------

def orbit(h: int, generators: list[Automorphism]) -> frozenset[int]:
    """
    Орбита вершины h под группой, порождённой generators.
    BFS по действию генераторов и их обратных.
    """
    result: set[int] = {h}
    queue = [h]
    all_gens = generators + [g.inverse() for g in generators]
    while queue:
        current = queue.pop()
        for g in all_gens:
            image = g(current)
            if image not in result:
                result.add(image)
                queue.append(image)
    return frozenset(result)


def all_orbits(generators: list[Automorphism]) -> list[frozenset[int]]:
    """
    Разбиение Q6 на орбиты под группой, порождённой generators.
    """
    remaining = set(range(SIZE))
    orbits_list = []
    while remaining:
        h = min(remaining)
        orb = orbit(h, generators)
        orbits_list.append(orb)
        remaining -= orb
    return sorted(orbits_list, key=min)


def canonical_form(h: int, generators: list[Automorphism]) -> int:
    """
    Канонический представитель орбиты h: минимальный элемент.
    """
    return min(orbit(h, generators))


def canonical_map(generators: list[Automorphism]) -> list[int]:
    """
    canonical[h] = канонический представитель орбиты h для всех h ∈ Q6.
    """
    result = [0] * SIZE
    for h in range(SIZE):
        result[h] = canonical_form(h, generators)
    return result


# ---------------------------------------------------------------------------
# Стабилизатор (вычислительно дорого для больших групп)
# ---------------------------------------------------------------------------

def count_stabilizer(h: int, generators: list[Automorphism]) -> int:
    """
    Мощность стабилизатора Stab(h) по теореме об орбите–стабилизаторе:
    |Stab(h)| = |G| / |Orb(h)|.
    Требует знания |G| — переданного пользователем.

    Для Aut(Q6): |G| = 46080.
    """
    orb_size = len(orbit(h, generators))
    # Размер группы вычислим по BFS от identity (дорого для больших групп)
    # Используем теорему: |Stab| = |G| / |Orb|
    # |G| = 46080 для полной Aut(Q6)
    group_order = 46080  # предполагаем полную Aut(Q6)
    return group_order // orb_size


# ---------------------------------------------------------------------------
# Циклы перестановки на Q6
# ---------------------------------------------------------------------------

def cycle_decomposition(aut: Automorphism) -> list[list[int]]:
    """
    Разложение на циклы действия aut на Q6.
    Возвращает список циклов (каждый цикл — список вершин).
    """
    visited = [False] * SIZE
    cycles = []
    for start in range(SIZE):
        if visited[start]:
            continue
        cycle = []
        current = start
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = aut(current)
        cycles.append(cycle)
    return cycles


def cycle_count(aut: Automorphism) -> int:
    """Число циклов в разложении aut на Q6 (включая неподвижные точки)."""
    return len(cycle_decomposition(aut))


def fixed_points(aut: Automorphism) -> list[int]:
    """Неподвижные точки автоморфизма: {h : aut(h) = h}."""
    return [h for h in range(SIZE) if aut(h) == h]


# ---------------------------------------------------------------------------
# Лемма Бёрнсайда
# ---------------------------------------------------------------------------

def generate_group(generators: list[Automorphism]) -> list[Automorphism]:
    """
    Сгенерировать группу по BFS из генераторов.
    Для Aut(Q6) возвращает 46080 элементов (медленно).
    Для малых подгрупп — быстро.
    """
    group: set[tuple] = set()
    all_gens = generators + [g.inverse() for g in generators]
    queue = [identity_aut()]
    group.add((queue[0].perm, queue[0].mask))

    while queue:
        current = queue.pop()
        for g in all_gens:
            nxt = g * current
            key = (nxt.perm, nxt.mask)
            if key not in group:
                group.add(key)
                queue.append(nxt)

    return [Automorphism(perm, mask) for perm, mask in group]


def burnside_count(n_colors: int, group: list[Automorphism]) -> int:
    """
    Число различных n_colors-раскрасок вершин Q6 по лемме Бёрнсайда:

    |Orbits| = (1/|G|) Σ_{g ∈ G} |Fix_colorings(g)|

    Раскраска c: Q6 → {0,...,n_colors-1} неподвижна под g ↔
    c постоянна на циклах g.

    Fix_colorings(g) = n_colors^{число циклов g на Q6}.
    """
    total_fixed = sum(n_colors ** cycle_count(g) for g in group)
    return total_fixed // len(group)


def burnside_subset(k: int, group: list[Automorphism]) -> int:
    """
    Число различных k-подмножеств Q6 под действием group (теорема Бёрнсайда).
    Подмножество S неподвижно под g ↔ g переставляет циклы целиком
    (каждый цикл g либо целиком в S, либо целиком вне S).

    Число неподвижных k-подмножеств = число способов выбрать циклы g
    суммарного размера k (задача о сумме подмножеств, решается DP за O(c×k)).
    """
    def fixed_subsets_dp(cycle_sizes: list[int], target: int) -> int:
        """Число подмножеств cycle_sizes с суммой = target (DP)."""
        dp = [0] * (target + 1)
        dp[0] = 1
        for size in cycle_sizes:
            if size <= target:
                for j in range(target, size - 1, -1):
                    dp[j] += dp[j - size]
        return dp[target]

    total = 0
    for g in group:
        cycles = cycle_decomposition(g)
        sizes = [len(c) for c in cycles]
        total += fixed_subsets_dp(sizes, k)
    return total // len(group)


# ---------------------------------------------------------------------------
# Анализ орбит при конкретных группах
# ---------------------------------------------------------------------------

def yang_orbits() -> list[frozenset[int]]:
    """
    Орбиты под S₆ (перестановки битов): 7 орбит по весу Хэмминга.
    Орбита k = {h ∈ Q6 : yang_count(h) = k}, |орбита k| = C(6,k).
    """
    return all_orbits(s6_generators())


def antipodal_orbits() -> list[frozenset[int]]:
    """
    Орбиты под {переворот всех бит}: пары {h, 63-h}.
    32 орбиты размера 2 (кроме разве что фиксированных, но у complement нет f.p.).
    """
    compl = complement_aut()
    return all_orbits([compl])


def full_aut_orbits() -> list[frozenset[int]]:
    """
    Орбиты под полной Aut(Q6) = B₆.
    Q6 вершинно-транзитивен → одна орбита размера 64.
    """
    return all_orbits(aut_generators())


def edge_orbits(generators: list[Automorphism]) -> list[frozenset[frozenset]]:
    """
    Орбиты рёбер Q6 под действием group(generators).
    Ребро = неупорядоченная пара {u, v}.
    """
    all_edges = [frozenset({u, v})
                 for u in range(SIZE) for v in neighbors(u) if u < v]
    remaining = set(range(len(all_edges)))
    orbits_list = []

    def edge_orbit(idx: int) -> set[int]:
        u, v = tuple(all_edges[idx])
        orb: set[int] = set()
        queue = [idx]
        all_gens = generators + [g.inverse() for g in generators]
        seen = {idx}
        while queue:
            cur_idx = queue.pop()
            u_cur, v_cur = tuple(all_edges[cur_idx])
            orb.add(cur_idx)
            for g in all_gens:
                gu, gv = g(u_cur), g(v_cur)
                ne = frozenset({gu, gv})
                try:
                    ne_idx = all_edges.index(ne)
                    if ne_idx not in seen:
                        seen.add(ne_idx)
                        queue.append(ne_idx)
                except ValueError:
                    pass
        return orb

    while remaining:
        idx = min(remaining)
        orb = edge_orbit(idx)
        orbits_list.append(frozenset(all_edges[i] for i in orb))
        remaining -= orb

    return orbits_list


# ---------------------------------------------------------------------------
# Индекс циклов (Полья)
# ---------------------------------------------------------------------------

def cycle_index_s6_on_q6() -> dict[tuple[int, ...], int]:
    """
    Индекс циклов группы S₆, действующей на Q6 (64 вершины).
    Возвращает словарь {cycle_type: multiplicity} где cycle_type =
    кортеж (a₁, a₂, ..., a₆₄) с aᵢ = число циклов длины i.

    Используется в теореме Полья для подсчёта различных раскрасок.
    """
    index: dict[tuple[int, ...], int] = defaultdict(int)
    for perm in permutations(range(6)):
        aut = bit_permutation(perm)
        cycles = cycle_decomposition(aut)
        lengths = [len(c) for c in cycles]
        # Тип цикла: (число циклов длины 1, длины 2, ...)
        max_len = max(lengths) if lengths else 1
        cycle_type = tuple(lengths.count(k) for k in range(1, max_len + 1))
        index[cycle_type] += 1
    return dict(index)


def polya_count(n_colors: int, generators: list[Automorphism] | None = None) -> int:
    """
    Число различных n_colors-раскрасок вершин Q6 под S₆.
    Использует индекс циклов S₆ (Полья).
    """
    if generators is None:
        generators = s6_generators()
    group = generate_group(generators)
    return burnside_count(n_colors, group)


# ---------------------------------------------------------------------------
# Полезные статистики
# ---------------------------------------------------------------------------

def orbit_size_distribution(generators: list[Automorphism]) -> dict[int, int]:
    """Распределение размеров орбит {размер: количество орбит}."""
    orbits = all_orbits(generators)
    dist: dict[int, int] = defaultdict(int)
    for orb in orbits:
        dist[len(orb)] += 1
    return dict(dist)


def group_order_estimate(generators: list[Automorphism], sample: int = 1000) -> int:
    """
    Оценить порядок группы по числу орбит.
    Для вершинно-транзитивной группы: |G| ≥ |V| = 64.
    """
    orbits = all_orbits(generators)
    return -1  # Placeholder; точный порядок через BFS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import math

    parser = argparse.ArgumentParser(description='hexsym — симметрии Q6')
    sub = parser.add_subparsers(dest='cmd')

    p_info = sub.add_parser('info', help='Сводка о группе Aut(Q6)')

    p_orbits = sub.add_parser('orbits', help='Орбиты вершин')
    p_orbits.add_argument('group', choices=['s6', 'full', 'flip', 'complement'])

    p_burnside = sub.add_parser('burnside', help='Лемма Бёрнсайда (раскраски)')
    p_burnside.add_argument('n_colors', type=int, help='Число цветов')
    p_burnside.add_argument('group', choices=['s6', 'full'],
                             help='Действующая группа')

    p_subset = sub.add_parser('subsets', help='Различные k-подмножества Q6')
    p_subset.add_argument('k', type=int)
    p_subset.add_argument('group', choices=['s6', 'full'])

    p_aut = sub.add_parser('aut', help='Анализ конкретного автоморфизма')
    p_aut.add_argument('kind',
                        choices=['complement', 'swap01', 'flip0', 'cycle123'])
    p_aut.add_argument('hex', type=int, nargs='?', default=42,
                        help='Гексаграмма для применения')

    p_edges = sub.add_parser('edge-orbits', help='Орбиты рёбер Q6')
    p_edges.add_argument('group', choices=['s6', 'full'])

    args = parser.parse_args()

    def make_group(name: str) -> list[Automorphism]:
        if name == 's6':
            return generate_group(s6_generators())
        elif name == 'full':
            return generate_group(aut_generators())
        elif name == 'flip':
            return generate_group([bit_flip_single(i) for i in range(6)])
        elif name == 'complement':
            return generate_group([complement_aut()])
        return []

    if args.cmd == 'info':
        print("Группа автоморфизмов Q6:")
        print(f"  Тип         : Aut(Q6) = B₆ = S₆ ⋉ (Z₂)⁶")
        print(f"  |Aut(Q6)|   : 6! × 2⁶ = 720 × 64 = {720*64}")
        print(f"  Генераторов : 6 (5 транспозиций + 1 переворот бита)")
        print()
        print("Орбиты на вершинах Q6:")
        print("  Под S₆ (перестановки битов):")
        for orb in yang_orbits():
            k = yang_count(min(orb))
            print(f"    weight={k}: {len(orb)} вершин  (C(6,{k})={math.comb(6,k)})")
        print(f"  Под Aut(Q6): 1 орбита (Q6 вершинно-транзитивен)")
        print()
        print("Рёбра Q6:")
        total_edges = sum(len(neighbors(h)) for h in range(SIZE)) // 2
        print(f"  Всего рёбер: {total_edges}")
        print(f"  Под S₆: рёбра = {total_edges} (все эквивалентны под полной Aut)")

    elif args.cmd == 'orbits':
        gens_map = {
            's6': s6_generators(),
            'full': aut_generators(),
            'flip': [bit_flip_single(i) for i in range(6)],
            'complement': [complement_aut()],
        }
        gens = gens_map[args.group]
        orbits = all_orbits(gens)
        print(f"Орбиты Q6 под группой '{args.group}' ({len(orbits)} орбит):")
        for i, orb in enumerate(orbits):
            sorted_orb = sorted(orb)
            first = sorted_orb[0]
            print(f"  [{i}] |орбита|={len(orb)}, min={first}, "
                  f"yang={yang_count(first)}  {sorted_orb[:8]}{'...' if len(orb) > 8 else ''}")

    elif args.cmd == 'burnside':
        print(f"Подсчёт различных {args.n_colors}-раскрасок под группой '{args.group}':")
        if args.group == 's6':
            group = generate_group(s6_generators())
            print(f"  |S₆| = {len(group)}")
        else:
            print("  Генерация Aut(Q6) (46080 элементов, ~30с)...")
            group = generate_group(aut_generators())
            print(f"  |Aut(Q6)| = {len(group)}")
        count = burnside_count(args.n_colors, group)
        print(f"  Различных раскрасок: {count}")

    elif args.cmd == 'subsets':
        import math as _m
        print(f"Различных {args.k}-подмножеств Q6 под группой '{args.group}':")
        if args.group == 's6':
            group = generate_group(s6_generators())
        else:
            print("  Генерация Aut(Q6)...")
            group = generate_group(aut_generators())
        count = burnside_subset(args.k, group)
        naive = _m.comb(SIZE, args.k)
        print(f"  Наивно (без симметрий): C(64,{args.k}) = {naive}")
        print(f"  С учётом симметрии:     {count}")
        print(f"  Коэффициент сжатия:     {naive / count:.1f}×")

    elif args.cmd == 'aut':
        auts = {
            'complement': complement_aut(),
            'swap01': bit_transposition(0, 1),
            'flip0': bit_flip_single(0),
            'cycle123': bit_permutation([1, 2, 3, 0, 4, 5]),
        }
        a = auts[args.kind]
        h = args.hex
        cycles = cycle_decomposition(a)
        fps = fixed_points(a)
        print(f"Автоморфизм '{args.kind}': {a}")
        print(f"  Порядок            : {a.order()}")
        print(f"  Число циклов на Q6 : {len(cycles)}")
        print(f"  Размеры циклов     : {sorted(len(c) for c in cycles)}")
        print(f"  Неподвижных точек  : {len(fps)}")
        print(f"  a({h}) = {a(h)}")
        print(f"  Цикл, содержащий {h}: {next(c for c in cycles if h in c)}")

    elif args.cmd == 'edge-orbits':
        gens_map = {
            's6': s6_generators(),
            'full': aut_generators(),
        }
        gens = gens_map[args.group]
        print(f"Орбиты рёбер Q6 под группой '{args.group}':")
        try:
            orbits = edge_orbits(gens)
            print(f"  Всего орбит: {len(orbits)}")
            for i, orb in enumerate(orbits[:5]):
                sample = list(orb)[0]
                u, v = tuple(sample)
                print(f"  [{i}] |орбита|={len(orb)}, пример: {{{u},{v}}}")
            if len(orbits) > 5:
                print(f"  ... и ещё {len(orbits)-5}")
        except Exception as e:
            print(f"  Ошибка: {e}")

    else:
        parser.print_help()
