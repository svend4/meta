"""hexbio — биоинформатика: генетический код как граф мутаций на Q6.

64 кодона = 64 гексаграммы.
Каждый нуклеотид кодируется 2 битами: A=00, C=01, G=10, U=11.
Кодон XYZ → 6-битное число: биты [5:4]=X, [3:2]=Y, [1:0]=Z.
Точечная мутация = замена одного нуклеотида = XOR двух соседних битов.
Граф мутаций ≠ Q6, т.к. мутация меняет оба бита позиции сразу.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------------------------
# Кодирование нуклеотидов
# ---------------------------------------------------------------------------

_NUC = {'A': 0b00, 'C': 0b01, 'G': 0b10, 'U': 0b11}
_NUC_REV = {0b00: 'A', 0b01: 'C', 0b10: 'G', 0b11: 'U'}
_NUC_LIST = ['A', 'C', 'G', 'U']

# Все возможные замены нуклеотида (3 замены на позицию)
_ALTERNATIVES = {n: [m for m in _NUC_LIST if m != n] for n in _NUC_LIST}


def codon_to_int(s: str) -> int:
    """Кодон (3 нуклеотида) → 6-битное число.

    Кодон 'XYZ': биты [5:4]=X, [3:2]=Y, [1:0]=Z.
    Принимает буквы A, C, G, U (RNA) или T→U (DNA).
    """
    s = s.upper().replace('T', 'U')
    if len(s) != 3:
        raise ValueError(f"Кодон должен быть длиной 3, получено: {s!r}")
    x, y, z = s[0], s[1], s[2]
    if x not in _NUC or y not in _NUC or z not in _NUC:
        raise ValueError(f"Неизвестный нуклеотид в кодоне: {s!r}")
    return (_NUC[x] << 4) | (_NUC[y] << 2) | _NUC[z]


def int_to_codon(n: int) -> str:
    """6-битное число → строка кодона (RNA, 3 символа)."""
    if not (0 <= n <= 63):
        raise ValueError(f"Значение вне диапазона: {n}")
    x = _NUC_REV[(n >> 4) & 0b11]
    y = _NUC_REV[(n >> 2) & 0b11]
    z = _NUC_REV[n & 0b11]
    return x + y + z


def codon_nucleotides(n: int) -> tuple:
    """Вернуть кортеж (X, Y, Z) нуклеотидов кодона."""
    return (
        _NUC_REV[(n >> 4) & 0b11],
        _NUC_REV[(n >> 2) & 0b11],
        _NUC_REV[n & 0b11],
    )


# ---------------------------------------------------------------------------
# Граф мутаций
# ---------------------------------------------------------------------------

def point_mutations(n: int) -> list:
    """Список кодонов, отличающихся ровно одним нуклеотидом (9 соседей).

    Это НЕ Q6-соседи (Q6 меняет 1 бит), а мутации нуклеотида:
    каждая замена меняет сразу 2 бита (позиция 1, 2 или 3).
    """
    result = []
    nuc = list(codon_nucleotides(n))
    for pos in range(3):
        original = nuc[pos]
        for alt in _ALTERNATIVES[original]:
            mutated = nuc[:]
            mutated[pos] = alt
            m_int = (_NUC[mutated[0]] << 4) | (_NUC[mutated[1]] << 2) | _NUC[mutated[2]]
            result.append(m_int)
    return result


def mutation_distance(a: int, b: int) -> int:
    """Число позиций нуклеотидов, в которых кодоны a и b различаются (0..3)."""
    dist = 0
    for pos in (4, 2, 0):
        if ((a >> pos) & 0b11) != ((b >> pos) & 0b11):
            dist += 1
    return dist


def hamming_distance_bits(a: int, b: int) -> int:
    """Расстояние Хэмминга в битах между двумя кодонами (0..6)."""
    diff = a ^ b
    return bin(diff).count('1')


def synonymous_path(a: int, b: int) -> list | None:
    """BFS по графу синонимичных мутаций от a до b (или None)."""
    if a == b:
        return [a]
    from collections import deque
    queue = deque([[a]])
    visited = {a}
    aa_a = translate(a)
    aa_b = translate(b)
    if aa_a != aa_b:
        return None  # разные аминокислоты — синонимичного пути нет
    while queue:
        path = queue.popleft()
        curr = path[-1]
        for nb in point_mutations(curr):
            if nb in visited:
                continue
            if translate(nb) != aa_a:
                continue
            new_path = path + [nb]
            if nb == b:
                return new_path
            visited.add(nb)
            queue.append(new_path)
    return None


# ---------------------------------------------------------------------------
# Стандартный генетический код (RNA-кодоны)
# ---------------------------------------------------------------------------

_GENETIC_CODE_TABLE = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Числовой индекс → аминокислота
_INT_TO_AA: dict[int, str] = {}
for _codon_str, _aa in _GENETIC_CODE_TABLE.items():
    _INT_TO_AA[codon_to_int(_codon_str)] = _aa


def translate(n: int) -> str:
    """6-битный кодон → аминокислота (1 буква) или '*' (стоп-кодон)."""
    return _INT_TO_AA[n]


def stop_codons() -> list:
    """Список всех стоп-кодонов (целые числа)."""
    return [n for n in range(64) if translate(n) == '*']


def synonymous_codons(aa: str) -> list:
    """Список всех кодонов (целые числа), кодирующих аминокислоту aa."""
    return [n for n in range(64) if translate(n) == aa]


def degeneracy_class(n: int) -> int:
    """Кратность вырождения: сколько кодонов кодируют ту же аминокислоту."""
    return len(synonymous_codons(translate(n)))


def is_synonymous_mutation(a: int, b: int) -> bool:
    """True, если a и b — однонуклеотидные мутации с одинаковой аминокислотой."""
    if mutation_distance(a, b) != 1:
        return False
    return translate(a) == translate(b)


def synonymous_neighbors(n: int) -> list:
    """Список соседей по точечным мутациям с той же аминокислотой."""
    aa = translate(n)
    return [m for m in point_mutations(n) if translate(m) == aa]


def nonsynonymous_neighbors(n: int) -> list:
    """Список соседей по точечным мутациям с другой аминокислотой."""
    aa = translate(n)
    return [m for m in point_mutations(n) if translate(m) != aa]


# ---------------------------------------------------------------------------
# Структура генетического кода
# ---------------------------------------------------------------------------

def third_position_structure() -> dict:
    """Анализ структуры 3-й позиции: wobble, 4-кратное вырождение.

    Возвращает словарь {codon_prefix (XY): {
        'codons': list[int],
        'amino_acids': list[str],
        'fourfold': bool (все 4 кодона XY* → одна AA),
        'twofold': bool (2 кодона → одна AA, 2 → другая),
    }}
    """
    result = {}
    for x in range(4):
        for y in range(4):
            prefix = (x << 4) | (y << 2)
            codons = [prefix | z for z in range(4)]
            aas = [translate(c) for c in codons]
            fourfold = len(set(aas)) == 1
            twofold = (len(set(aas)) == 2 and
                       aas.count(aas[0]) == 2 and aas.count(aas[2]) == 2)
            prefix_str = _NUC_REV[x] + _NUC_REV[y]
            result[prefix_str] = {
                'codons': codons,
                'amino_acids': aas,
                'fourfold': fourfold,
                'twofold': twofold,
            }
    return result


def wobble_pairs() -> list:
    """Список пар кодонов (a, b), отличающихся только 3-й позицией, с одинаковой AA.

    Wobble-позиция — 3-й нуклеотид кодона (биты [1:0]).
    """
    result = []
    for n in range(64):
        aa = translate(n)
        if aa == '*':
            continue
        wobble_base = n & ~0b11  # обнулить биты 3-й позиции
        for z in range(4):
            m = wobble_base | z
            if m != n and translate(m) == aa:
                pair = tuple(sorted([n, m]))
                if pair not in result:
                    result.append(pair)
    return result


def gc_content(n: int) -> float:
    """GC-содержание кодона (доля G+C нуклеотидов, 0.0..1.0)."""
    count_gc = 0
    for pos in (4, 2, 0):
        nuc = (n >> pos) & 0b11
        if nuc in (0b01, 0b10):  # C=01, G=10
            count_gc += 1
    return count_gc / 3.0


def purine_count(n: int) -> int:
    """Число пуринов (A=00, G=10) в кодоне (0..3)."""
    count = 0
    for pos in (4, 2, 0):
        nuc = (n >> pos) & 0b11
        if nuc in (0b00, 0b10):  # A=00, G=10
            count += 1
    return count


# ---------------------------------------------------------------------------
# Статистика использования кодонов (RSCU, CAI)
# ---------------------------------------------------------------------------

def rscu(usage_counts: dict) -> dict:
    """Relative Synonymous Codon Usage (RSCU).

    Параметры:
        usage_counts: {codon_int: count}

    RSCU(c) = observed(c) / expected(c)
             = observed(c) / (total_for_AA / degeneracy)
    Возвращает {codon_int: rscu_value}.
    """
    result = {}
    # Сгруппировать по аминокислотам
    aa_codons: dict[str, list] = {}
    for n in range(64):
        aa = translate(n)
        aa_codons.setdefault(aa, []).append(n)

    for aa, codons in aa_codons.items():
        deg = len(codons)
        total = sum(usage_counts.get(c, 0) for c in codons)
        expected = total / deg if total > 0 else 0
        for c in codons:
            obs = usage_counts.get(c, 0)
            result[c] = obs / expected if expected > 0 else 0.0
    return result


def codon_adaptation_index_weights(reference_counts: dict) -> dict:
    """Вычислить w-веса для CAI (Codon Adaptation Index).

    w(c) = RSCU(c) / max_{synonymous c'} RSCU(c')
    Возвращает {codon_int: w_value}.
    """
    rscu_vals = rscu(reference_counts)
    weights = {}
    aa_codons: dict[str, list] = {}
    for n in range(64):
        aa = translate(n)
        aa_codons.setdefault(aa, []).append(n)

    for aa, codons in aa_codons.items():
        max_rscu = max(rscu_vals.get(c, 0) for c in codons)
        for c in codons:
            r = rscu_vals.get(c, 0)
            weights[c] = r / max_rscu if max_rscu > 0 else 0.0
    return weights


# ---------------------------------------------------------------------------
# Эволюционные расстояния
# ---------------------------------------------------------------------------

# Матрица физико-химических расстояний между аминокислотами (Grantham, 1974).
# Значения приближённые (0 = одинаковые, 215 = максимальное).
_GRANTHAM: dict[tuple, int] = {
    ('A', 'R'): 112, ('A', 'N'): 111, ('A', 'D'): 126, ('A', 'C'): 195,
    ('A', 'Q'): 91,  ('A', 'E'): 107, ('A', 'G'): 60,  ('A', 'H'): 86,
    ('A', 'I'): 94,  ('A', 'L'): 96,  ('A', 'K'): 106, ('A', 'M'): 84,
    ('A', 'F'): 113, ('A', 'P'): 27,  ('A', 'S'): 99,  ('A', 'T'): 58,
    ('A', 'W'): 148, ('A', 'Y'): 112, ('A', 'V'): 64,
    ('R', 'N'): 86,  ('R', 'D'): 96,  ('R', 'C'): 180, ('R', 'Q'): 43,
    ('R', 'E'): 54,  ('R', 'G'): 125, ('R', 'H'): 29,  ('R', 'I'): 97,
    ('R', 'L'): 102, ('R', 'K'): 26,  ('R', 'M'): 91,  ('R', 'F'): 97,
    ('R', 'P'): 103, ('R', 'S'): 110, ('R', 'T'): 71,  ('R', 'W'): 101,
    ('R', 'Y'): 77,  ('R', 'V'): 96,
    ('N', 'D'): 23,  ('N', 'C'): 139, ('N', 'Q'): 46,  ('N', 'E'): 42,
    ('N', 'G'): 80,  ('N', 'H'): 68,  ('N', 'I'): 149, ('N', 'L'): 153,
    ('N', 'K'): 94,  ('N', 'M'): 142, ('N', 'F'): 158, ('N', 'P'): 91,
    ('N', 'S'): 46,  ('N', 'T'): 65,  ('N', 'W'): 174, ('N', 'Y'): 143,
    ('N', 'V'): 133,
    ('D', 'C'): 154, ('D', 'Q'): 61,  ('D', 'E'): 45,  ('D', 'G'): 94,
    ('D', 'H'): 81,  ('D', 'I'): 168, ('D', 'L'): 172, ('D', 'K'): 101,
    ('D', 'M'): 160, ('D', 'F'): 177, ('D', 'P'): 108, ('D', 'S'): 65,
    ('D', 'T'): 85,  ('D', 'W'): 181, ('D', 'Y'): 160, ('D', 'V'): 152,
    ('C', 'Q'): 154, ('C', 'E'): 170, ('C', 'G'): 159, ('C', 'H'): 174,
    ('C', 'I'): 198, ('C', 'L'): 198, ('C', 'K'): 202, ('C', 'M'): 196,
    ('C', 'F'): 205, ('C', 'P'): 169, ('C', 'S'): 112, ('C', 'T'): 149,
    ('C', 'W'): 215, ('C', 'Y'): 194, ('C', 'V'): 192,
    ('Q', 'E'): 29,  ('Q', 'G'): 87,  ('Q', 'H'): 24,  ('Q', 'I'): 109,
    ('Q', 'L'): 113, ('Q', 'K'): 53,  ('Q', 'M'): 101, ('Q', 'F'): 116,
    ('Q', 'P'): 76,  ('Q', 'S'): 68,  ('Q', 'T'): 42,  ('Q', 'W'): 130,
    ('Q', 'Y'): 99,  ('Q', 'V'): 96,
    ('E', 'G'): 98,  ('E', 'H'): 40,  ('E', 'I'): 134, ('E', 'L'): 138,
    ('E', 'K'): 56,  ('E', 'M'): 126, ('E', 'F'): 140, ('E', 'P'): 93,
    ('E', 'S'): 80,  ('E', 'T'): 65,  ('E', 'W'): 152, ('E', 'Y'): 122,
    ('E', 'V'): 121,
    ('G', 'H'): 98,  ('G', 'I'): 135, ('G', 'L'): 138, ('G', 'K'): 127,
    ('G', 'M'): 127, ('G', 'F'): 153, ('G', 'P'): 42,  ('G', 'S'): 56,
    ('G', 'T'): 59,  ('G', 'W'): 184, ('G', 'Y'): 147, ('G', 'V'): 109,
    ('H', 'I'): 94,  ('H', 'L'): 99,  ('H', 'K'): 32,  ('H', 'M'): 87,
    ('H', 'F'): 100, ('H', 'P'): 77,  ('H', 'S'): 89,  ('H', 'T'): 47,
    ('H', 'W'): 115, ('H', 'Y'): 83,  ('H', 'V'): 84,
    ('I', 'L'): 5,   ('I', 'K'): 102, ('I', 'M'): 10,  ('I', 'F'): 21,
    ('I', 'P'): 95,  ('I', 'S'): 142, ('I', 'T'): 89,  ('I', 'W'): 61,
    ('I', 'Y'): 33,  ('I', 'V'): 29,
    ('L', 'K'): 107, ('L', 'M'): 15,  ('L', 'F'): 22,  ('L', 'P'): 98,
    ('L', 'S'): 145, ('L', 'T'): 92,  ('L', 'W'): 61,  ('L', 'Y'): 36,
    ('L', 'V'): 32,
    ('K', 'M'): 95,  ('K', 'F'): 102, ('K', 'P'): 103, ('K', 'S'): 121,
    ('K', 'T'): 78,  ('K', 'W'): 110, ('K', 'Y'): 85,  ('K', 'V'): 97,
    ('M', 'F'): 28,  ('M', 'P'): 87,  ('M', 'S'): 135, ('M', 'T'): 81,
    ('M', 'W'): 67,  ('M', 'Y'): 36,  ('M', 'V'): 21,
    ('F', 'P'): 114, ('F', 'S'): 155, ('F', 'T'): 103, ('F', 'W'): 40,
    ('F', 'Y'): 22,  ('F', 'V'): 50,
    ('P', 'S'): 74,  ('P', 'T'): 38,  ('P', 'W'): 147, ('P', 'Y'): 110,
    ('P', 'V'): 68,
    ('S', 'T'): 58,  ('S', 'W'): 177, ('S', 'Y'): 144, ('S', 'V'): 124,
    ('T', 'W'): 128, ('T', 'Y'): 92,  ('T', 'V'): 69,
    ('W', 'Y'): 37,  ('W', 'V'): 88,
    ('Y', 'V'): 55,
}


def amino_acid_distance(aa1: str, aa2: str) -> int:
    """Физико-химическое расстояние Гранхэма между аминокислотами (0..215)."""
    if aa1 == aa2:
        return 0
    key = (min(aa1, aa2), max(aa1, aa2))
    return _GRANTHAM.get(key, -1)


def evolutionary_distance_matrix() -> dict:
    """Матрица эволюционных расстояний для всех пар кодонов.

    Возвращает {(a, b): distance} для 0 ≤ a < b ≤ 63.
    Расстояние = число точечных мутаций (BFS) в графе мутаций.
    """
    from collections import deque
    dist_matrix = {}
    for start in range(64):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for nb in point_mutations(curr):
                if nb not in dist:
                    dist[nb] = dist[curr] + 1
                    queue.append(nb)
        for end in range(start + 1, 64):
            dist_matrix[(start, end)] = dist.get(end, -1)
    return dist_matrix


# ---------------------------------------------------------------------------
# Анализ генетического кода
# ---------------------------------------------------------------------------

def amino_acids() -> list:
    """Список всех 20 стандартных аминокислот (1-буквенный код)."""
    return sorted(set(v for v in _GENETIC_CODE_TABLE.values() if v != '*'))


def codon_table_summary() -> dict:
    """Сводная таблица генетического кода.

    Возвращает {aa: {'codons': [...], 'degeneracy': int}}.
    """
    result = {}
    for aa in amino_acids():
        codons = synonymous_codons(aa)
        result[aa] = {'codons': codons, 'degeneracy': len(codons)}
    result['*'] = {'codons': stop_codons(), 'degeneracy': len(stop_codons())}
    return result


def synonymous_mutation_fraction() -> float:
    """Доля синонимичных точечных мутаций среди всех возможных."""
    total = 0
    synonymous = 0
    for n in range(64):
        for m in point_mutations(n):
            total += 1
            if translate(n) == translate(m):
                synonymous += 1
    return synonymous / total if total > 0 else 0.0


def mutation_graph_edges() -> list:
    """Все рёбра графа мутаций (неориентированного), 64×9/2 = 288 рёбер.

    Возвращает список пар (a, b) с a < b.
    """
    edges = set()
    for n in range(64):
        for m in point_mutations(n):
            edge = (min(n, m), max(n, m))
            edges.add(edge)
    return sorted(edges)


def q6_vs_mutation_comparison() -> dict:
    """Сравнение Q6 (1-битные изменения) и графа мутаций (1 нуклеотид).

    Q6: 64 вершины, 6 соседей, 192 ребра.
    Граф мутаций: 64 вершины, 9 соседей, 288 рёбер.
    """
    q6_neighbors_count = 6
    mutation_neighbors_count = 9
    q6_edges = 64 * 6 // 2
    mutation_edges = len(mutation_graph_edges())
    # Общие рёбра
    q6_edge_set = set()
    for n in range(64):
        for bit in range(6):
            m = n ^ (1 << bit)
            q6_edge_set.add((min(n, m), max(n, m)))
    mutation_edge_set = set(mutation_graph_edges())
    shared = q6_edge_set & mutation_edge_set
    return {
        'q6_neighbors': q6_neighbors_count,
        'mutation_neighbors': mutation_neighbors_count,
        'q6_edges': q6_edges,
        'mutation_edges': mutation_edges,
        'shared_edges': len(shared),
        'mutation_only_edges': mutation_edges - len(shared),
        'q6_only_edges': q6_edges - len(shared),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_info(_args):
    """Общая информация о генетическом коде на Q6."""
    print("Генетический код как граф мутаций на Q6")
    print("=" * 45)
    print(f"Всего кодонов: 64")
    print(f"Аминокислот:   20")
    print(f"Стоп-кодонов:  {len(stop_codons())}")
    print(f"Стоп-кодоны:   {[int_to_codon(c) for c in stop_codons()]}")
    print()
    print("Вырождение генетического кода:")
    by_deg: dict[int, list] = {}
    for aa in amino_acids():
        deg = len(synonymous_codons(aa))
        by_deg.setdefault(deg, []).append(aa)
    for deg in sorted(by_deg):
        aas = ', '.join(sorted(by_deg[deg]))
        print(f"  {deg}-кратное: {aas}")
    print()
    print(f"Доля синонимичных мутаций: {synonymous_mutation_fraction():.3f}")
    cmp = q6_vs_mutation_comparison()
    print()
    print("Q6 vs граф мутаций:")
    print(f"  Q6: {cmp['q6_neighbors']} соседей, {cmp['q6_edges']} рёбер")
    print(f"  Мутации: {cmp['mutation_neighbors']} соседей, {cmp['mutation_edges']} рёбер")
    print(f"  Общих рёбер: {cmp['shared_edges']}")


def _cmd_codon(args):
    """Информация о конкретном кодоне."""
    if not args:
        print("Использование: codon <XYZ>")
        return
    codon_str = args[0].upper().replace('T', 'U')
    n = codon_to_int(codon_str)
    aa = translate(n)
    print(f"Кодон {codon_str} = {n} (0b{n:06b})")
    print(f"Аминокислота: {aa}")
    print(f"Вырождение: {degeneracy_class(n)}-кратное")
    print(f"GC-содержание: {gc_content(n):.2f}")
    print(f"Синонимичные кодоны: {[int_to_codon(c) for c in synonymous_codons(aa) if c != n]}")
    print(f"Точечные мутации (9):")
    for m in point_mutations(n):
        syn = "синоним" if is_synonymous_mutation(n, m) else f"→{translate(m)}"
        print(f"  {int_to_codon(m)} ({syn})")


def _cmd_mutation(args):
    """Найти путь мутаций между двумя кодонами."""
    if len(args) < 2:
        print("Использование: mutation <XYZ> <ABC>")
        return
    a = codon_to_int(args[0].upper().replace('T', 'U'))
    b = codon_to_int(args[1].upper().replace('T', 'U'))
    d = mutation_distance(a, b)
    print(f"Кодон A: {int_to_codon(a)} ({translate(a)})")
    print(f"Кодон B: {int_to_codon(b)} ({translate(b)})")
    print(f"Расстояние мутаций: {d}")
    print(f"Расстояние Хэмминга (биты): {hamming_distance_bits(a, b)}")
    if d == 1:
        print(f"Синонимичная мутация: {is_synonymous_mutation(a, b)}")


def _cmd_graph(_args):
    """Статистика графа мутаций."""
    cmp = q6_vs_mutation_comparison()
    print("Граф мутаций кодонов:")
    print(f"  Вершин: 64, соседей на вершину: {cmp['mutation_neighbors']}")
    print(f"  Рёбер всего: {cmp['mutation_edges']}")
    print()
    print("Сравнение с Q6 (граф Хэмминга):")
    for k, v in cmp.items():
        print(f"  {k}: {v}")


def _cmd_wobble(_args):
    """Анализ wobble-позиции (3-я позиция кодона)."""
    pairs = wobble_pairs()
    print(f"Wobble-пары (одинаковая AA, отличается 3-я позиция): {len(pairs)}")
    struct = third_position_structure()
    fourfold = sum(1 for v in struct.values() if v['fourfold'])
    twofold = sum(1 for v in struct.values() if v['twofold'])
    print(f"4-кратно вырожденных боксов (XY*→один AA): {fourfold}")
    print(f"2-кратно вырожденных боксов: {twofold}")


def main():
    import sys
    if len(sys.argv) < 2:
        print("Использование: hexbio.py <команда> [аргументы]")
        print("Команды: info, codon <XYZ>, mutation <A> <B>, graph, wobble")
        return
    cmd = sys.argv[1]
    args = sys.argv[2:]
    commands = {
        'info': _cmd_info,
        'codon': _cmd_codon,
        'mutation': _cmd_mutation,
        'graph': _cmd_graph,
        'wobble': _cmd_wobble,
    }
    if cmd not in commands:
        print(f"Неизвестная команда: {cmd}")
        print(f"Доступные: {list(commands)}")
        return
    commands[cmd](args)


if __name__ == '__main__':
    main()
