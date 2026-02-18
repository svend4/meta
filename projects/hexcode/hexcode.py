"""
hexcode — двоичные линейные коды в пространстве Q6

Пространство Q6 = (Z₂)⁶ является естественным пространством для двоичных
линейных кодов длины n=6. Каждый код — подпространство (Z₂)⁶.

Ключевые понятия:
  - BinaryCode(G)        : код, заданный порождающей матрицей G (k×6)
  - Расстояние Хэмминга  : min weight ненулевого кодового слова
  - Синдромное декодирование: ближайший сосед через таблицу синдромов
  - Покрывающий радиус   : max расстояние до ближайшего кодового слова
  - Совершенный код      : ball-упаковка покрывает всё (Z₂)⁶

Стандартные коды длины 6:
  repetition_code()          [6,1,6]  — один бит, повторённый 6 раз
  parity_check_code()        [6,5,2]  — 5 информ. бит + бит чётности
  dual_parity_code()         [6,2,4]  — минимальный дистанционный код
  shortened_hamming()        [6,3,4]  — укороченный код Хэмминга
  even_weight_code()         [6,5,2]  — слова чётного веса
  all_ones_code()            [6,1,6]  — {000000, 111111}
"""

from __future__ import annotations
import sys
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import hamming, yang_count, to_bits, flip, SIZE


# ---------------------------------------------------------------------------
# Вспомогательные операции над GF(2)
# ---------------------------------------------------------------------------

def _gf2_dot(row: list[int], col: list[int]) -> int:
    """Скалярное произведение в GF(2)."""
    return sum(a * b for a, b in zip(row, col)) % 2


def _gf2_matmul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Умножение матриц в GF(2)."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B
    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A)) % 2
    return result


def _int_to_bits(v: int, n: int) -> list[int]:
    """Целое число → список бит (MSB первый) длиной n."""
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def _bits_to_int(bits: list[int]) -> int:
    """Список бит (MSB первый) → целое число."""
    result = 0
    for b in bits:
        result = (result << 1) | b
    return result


def _row_reduce_gf2(matrix: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    """
    Приведение матрицы над GF(2) к ступенчатому виду.
    Возвращает (reduced_matrix, pivot_cols).
    """
    mat = [row[:] for row in matrix]
    n_rows = len(mat)
    if n_rows == 0:
        return mat, []
    n_cols = len(mat[0])
    pivots = []
    row_ptr = 0
    for col in range(n_cols):
        # Найти ненулевую строку
        pivot = None
        for r in range(row_ptr, n_rows):
            if mat[r][col]:
                pivot = r
                break
        if pivot is None:
            continue
        mat[row_ptr], mat[pivot] = mat[pivot], mat[row_ptr]
        pivots.append(col)
        # Обнулить остальные строки в этом столбце
        for r in range(n_rows):
            if r != row_ptr and mat[r][col]:
                mat[r] = [(mat[r][c] ^ mat[row_ptr][c]) for c in range(n_cols)]
        row_ptr += 1
    return mat, pivots


# ---------------------------------------------------------------------------
# Основной класс: двоичный линейный код
# ---------------------------------------------------------------------------

class BinaryCode:
    """
    Двоичный линейный [n=6, k, d] код.

    Параметры:
        G : порождающая матрица (k строк, 6 столбцов), элементы 0/1
            Каждая строка — кодовое слово-генератор.
    """

    N = 6   # длина кода = размерность Q6

    def __init__(self, G: list[list[int]]) -> None:
        if not G:
            raise ValueError("Порождающая матрица не может быть пустой")
        if any(len(row) != self.N for row in G):
            raise ValueError(f"Каждая строка должна иметь длину {self.N}")
        # Приведение к ступенчатому виду для нахождения реального k
        reduced, pivots = _row_reduce_gf2(G)
        self.k = len(pivots)
        # Оставить только линейно независимые строки
        self.G: list[list[int]] = [reduced[i] for i in range(self.k)]
        self._codewords_cache: list[int] | None = None
        self._syndrome_table: dict[int, int] | None = None

    @property
    def n(self) -> int:
        return self.N

    def __repr__(self) -> str:
        return f"BinaryCode[n={self.n}, k={self.k}, d={self.min_distance()}]"

    # ---- кодовые слова --------------------------------------------------

    def codewords(self) -> list[int]:
        """Все 2^k кодовых слова как целые числа (MSB = бит 5)."""
        if self._codewords_cache is not None:
            return self._codewords_cache
        cws: set[int] = {0}
        gen_ints = [_bits_to_int(row) for row in self.G]
        for mask in range(1, 1 << self.k):
            c = 0
            for i in range(self.k):
                if (mask >> i) & 1:
                    c ^= gen_ints[i]
            cws.add(c)
        self._codewords_cache = sorted(cws)
        return self._codewords_cache

    def is_codeword(self, h: int) -> bool:
        return h in self.codewords()

    # ---- параметры кода -------------------------------------------------

    def min_distance(self) -> int:
        """Минимальное расстояние = min weight ненулевого кодового слова."""
        cws = self.codewords()
        return min((bin(c).count('1') for c in cws if c != 0), default=self.N)

    def weight_distribution(self) -> dict[int, int]:
        """Распределение весов {weight: count}."""
        dist: dict[int, int] = defaultdict(int)
        for c in self.codewords():
            dist[bin(c).count('1')] += 1
        return dict(dist)

    def covering_radius(self) -> int:
        """
        Покрывающий радиус: max расстояние от произвольной точки до кода.
        rho(C) = max_{h ∈ (Z₂)⁶} min_{c ∈ C} d(h, c)
        """
        cws = set(self.codewords())
        return max(
            min(hamming(h, c) for c in cws)
            for h in range(SIZE)
        )

    def is_perfect(self) -> bool:
        """
        Совершенный код: шары радиуса t = (d-1)//2 вокруг кодовых слов
        покрывают всё (Z₂)⁶ без пересечений.
        Критерий: 2^k × Σ_{i=0}^t C(n,i) = 2^n.
        """
        import math
        d = self.min_distance()
        t = (d - 1) // 2
        sphere_vol = sum(math.comb(self.N, i) for i in range(t + 1))
        return (len(self.codewords()) * sphere_vol) == SIZE

    def is_mds(self) -> bool:
        """MDS-код (Maximum Distance Separable): d = n - k + 1."""
        return self.min_distance() == self.N - self.k + 1

    def rate(self) -> float:
        """Скорость кода R = k/n."""
        return self.k / self.N

    # ---- кодирование и декодирование ------------------------------------

    def encode(self, message: list[int]) -> int:
        """
        Кодирование: message (k бит) → кодовое слово (int).
        message[0] = MSB сообщения.
        """
        if len(message) != self.k:
            raise ValueError(f"Длина сообщения должна быть {self.k}, получено {len(message)}")
        c = 0
        for i, bit in enumerate(message):
            if bit:
                c ^= _bits_to_int(self.G[i])
        return c

    def _build_syndrome_table(self) -> dict[int, int]:
        """
        Таблица синдромов для синдромного декодирования.
        Синдром s = H·r (mod 2), где H — проверочная матрица.
        Таблица: {syndrome: error_pattern}.
        """
        H = self.parity_check_matrix()
        if not H:
            return {}

        table: dict[int, int] = {}
        r = len(H)           # число строк H = n - k

        def syndrome(e: int) -> int:
            e_bits = _int_to_bits(e, self.N)
            s_bits = [_gf2_dot(H[i], e_bits) for i in range(r)]
            return _bits_to_int(s_bits)

        # Сначала нулевая ошибка (синдром = 0)
        table[0] = 0

        # Ошибки веса 1 (однократные)
        for bit in range(self.N):
            e = 1 << bit
            s = syndrome(e)
            if s not in table:
                table[s] = e

        # Ошибки веса 2 (двукратные) — если t ≥ 2
        d = self.min_distance()
        if d >= 5:
            for i, j in combinations(range(self.N), 2):
                e = (1 << i) | (1 << j)
                s = syndrome(e)
                if s not in table:
                    table[s] = e

        return table

    def decode(self, received: int) -> int | None:
        """
        Синдромное декодирование.
        Исправляет ошибки весом ≤ t = (d-1)//2.
        Возвращает ближайшее кодовое слово или None если не удаётся.
        """
        if self._syndrome_table is None:
            self._syndrome_table = self._build_syndrome_table()

        H = self.parity_check_matrix()
        if not H:
            return received if received in self.codewords() else None

        r_bits = _int_to_bits(received, self.N)
        r = len(H)
        s_bits = [_gf2_dot(H[i], r_bits) for i in range(r)]
        s = _bits_to_int(s_bits)

        error = self._syndrome_table.get(s)
        if error is None:
            # Не можем исправить — ошибка слишком велика
            return None
        return received ^ error

    def nearest_codeword(self, h: int) -> int:
        """Ближайшее кодовое слово (декодирование полным перебором)."""
        cws = self.codewords()
        return min(cws, key=lambda c: hamming(h, c))

    # ---- проверочная матрица --------------------------------------------

    def parity_check_matrix(self) -> list[list[int]]:
        """
        Проверочная матрица H (r × n, r = n - k):
        H · c = 0 для всех c ∈ C.
        Строится из нулевого пространства G.
        """
        # Приведём G к форме [I_k | P]
        G_ext = [row[:] for row in self.G]
        _, pivots = _row_reduce_gf2(G_ext)
        if len(pivots) < self.k:
            return []

        # Системная форма: H = [P^T | I_{n-k}]
        n, k = self.N, self.k
        r = n - k
        if r == 0:
            return []

        # Полная RREF G
        G_rref, pivots = _row_reduce_gf2([row[:] for row in self.G])
        G_rref = G_rref[:k]

        non_pivots = [c for c in range(n) if c not in pivots]

        # H строки: для каждого не-pivot столбца j —
        # строка H[j] имеет 1 в позиции j и G[i][j] в позициях pivot[i]
        H = []
        for j in non_pivots:
            h_row = [0] * n
            h_row[j] = 1
            for i, p in enumerate(pivots):
                h_row[p] = G_rref[i][j]
            H.append(h_row)

        return H

    # ---- специальные коды -----------------------------------------------

    @classmethod
    def dual(cls, code: 'BinaryCode') -> 'BinaryCode':
        """Двойственный код: H кода C как G двойственного C⊥."""
        H = code.parity_check_matrix()
        if not H:
            # Код = всё пространство, двойственный = {0}
            # Но нулевой код не имеет смысла; вернём тривиальный
            raise ValueError("Нет нетривиального двойственного кода")
        return cls(H)

    def cosets(self) -> list[list[int]]:
        """
        Разложение (Z₂)⁶ по смежным классам кода C.
        Возвращает список классов [[cw1, cw2, ...], ...].
        """
        cws = set(self.codewords())
        remaining = set(range(SIZE))
        result = []
        while remaining:
            leader = min(remaining)
            coset = sorted(leader ^ c for c in cws)
            result.append(coset)
            remaining -= set(coset)
        return result

    def info(self) -> dict:
        """Сводная информация о коде."""
        d = self.min_distance()
        t = (d - 1) // 2
        return {
            'n': self.N,
            'k': self.k,
            'd': d,
            't': t,
            'rate': round(self.rate(), 4),
            'size': len(self.codewords()),
            'covering_radius': self.covering_radius(),
            'is_perfect': self.is_perfect(),
            'is_mds': self.is_mds(),
            'weight_distribution': self.weight_distribution(),
        }

    def display(self) -> str:
        """Текстовое описание кода."""
        info = self.info()
        lines = [
            f"[{self.N}, {self.k}, {info['d']}]-код",
            f"  Скорость R = {info['rate']}",
            f"  Размер: {info['size']} кодовых слов",
            f"  Исправляет ошибок: t = {info['t']}",
            f"  Покрывающий радиус: {info['covering_radius']}",
            f"  Совершенный: {'да' if info['is_perfect'] else 'нет'}",
            f"  MDS: {'да' if info['is_mds'] else 'нет'}",
            "  Распределение весов: " + str(dict(sorted(info['weight_distribution'].items()))),
            "  Кодовые слова:",
        ]
        for c in self.codewords():
            lines.append(f"    {to_bits(c)}  ({c})")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Стандартные коды длины 6
# ---------------------------------------------------------------------------

def repetition_code() -> BinaryCode:
    """
    [6,1,6]-код: одно информационное слово, повторённое 6 раз.
    G = [[1,1,1,1,1,1]], кодовые слова: {000000, 111111}.
    Совершенный нет (нечётный MDS). Покрывает с радиусом 3.
    """
    return BinaryCode([[1, 1, 1, 1, 1, 1]])


def parity_check_code() -> BinaryCode:
    """
    [6,5,2]-код: один бит чётности.
    G = I_5 дополненная столбцом чётности.
    Обнаруживает однократные ошибки, не исправляет.
    """
    G = []
    for i in range(5):
        row = [0] * 6
        row[i] = 1
        row[5] = 1   # бит чётности = XOR всех информационных
        G.append(row)
    return BinaryCode(G)


def dual_repetition_code() -> BinaryCode:
    """
    [6,5,2]-код (= parity check code).
    Двойственный к repetition_code.
    """
    return BinaryCode.dual(repetition_code())


def shortened_hamming_code() -> BinaryCode:
    """
    Укороченный [6,3,3]-код Хэмминга.
    Получен укорочением стандартного [7,4,3]-кода Хэмминга
    (фиксация бита 0 = 0, удаление этой координаты).
    Минимальное расстояние 3: исправляет 1 ошибку.
    G генерирует 8 кодовых слов.
    """
    G = [
        [1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1],
    ]
    return BinaryCode(G)


def even_weight_code() -> BinaryCode:
    """
    [6,5,2]-код чётного веса.
    Содержит все слова с чётным числом единиц (32 слова).
    """
    G = []
    for i in range(5):
        row = [0] * 6
        row[i] = 1
        row[5] = 1
        G.append(row)
    return BinaryCode(G)


def hexcode_312() -> BinaryCode:
    """
    Специальный [6,3,3]-код: расстояние 3, исправляет 1 ошибку.
    8 кодовых слов.
    """
    G = [
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1],
    ]
    return BinaryCode(G)


def full_space_code() -> BinaryCode:
    """[6,6,1]-код: всё пространство (Z₂)⁶, 64 слова."""
    return BinaryCode([[int(i == j) for j in range(6)] for i in range(6)])


# ---------------------------------------------------------------------------
# Поиск кодов
# ---------------------------------------------------------------------------

def find_codes(d_min: int, k: int | None = None) -> list[BinaryCode]:
    """
    Найти все двоичные линейные [6, k, ≥d_min] коды.
    Перебирает все возможные порождающие матрицы (до k строк).

    Предупреждение: может работать медленно для малых d_min.
    """
    found: list[BinaryCode] = []
    seen: set[frozenset[int]] = set()

    max_k = k or 5

    for dim in range(1, max_k + 1):
        # Попробовать все наборы из dim ненулевых векторов
        nonzero = list(range(1, SIZE))
        for rows_ints in combinations(nonzero, dim):
            G = [_int_to_bits(v, 6) for v in rows_ints]
            try:
                code = BinaryCode(G)
            except ValueError:
                continue
            if code.k != dim:
                continue
            cws = frozenset(code.codewords())
            if cws in seen:
                continue
            seen.add(cws)
            if code.min_distance() >= d_min:
                if k is None or code.k == k:
                    found.append(code)

    return found


# ---------------------------------------------------------------------------
# Покрывающие коды
# ---------------------------------------------------------------------------

def min_covering_code(radius: int) -> BinaryCode | None:
    """
    Найти наименьший линейный код с покрывающим радиусом ≤ radius.
    Возвращает код с минимальным k.
    """
    for k in range(1, 7):
        nonzero = list(range(1, SIZE))
        for rows_ints in combinations(nonzero, k):
            G = [_int_to_bits(v, 6) for v in rows_ints]
            try:
                code = BinaryCode(G)
            except ValueError:
                continue
            if code.k != k:
                continue
            if code.covering_radius() <= radius:
                return code
    return None


# ---------------------------------------------------------------------------
# Шаровая упаковка (невозможность)
# ---------------------------------------------------------------------------

def singleton_bound(k: int, d: int, n: int = 6) -> bool:
    """Граница Синглтона: d ≤ n - k + 1."""
    return d <= n - k + 1


def hamming_bound(k: int, d: int, n: int = 6) -> bool:
    """
    Граница Хэмминга (sphere packing bound):
    2^k × Σ C(n,i) ≤ 2^n, i=0..(d-1)//2.
    """
    import math
    t = (d - 1) // 2
    vol = sum(math.comb(n, i) for i in range(t + 1))
    return (2 ** k) * vol <= (2 ** n)


def plotkin_bound(k: int, d: int, n: int = 6) -> bool:
    """
    Граница Плоткина: если d > n/2, то 2^k ≤ 2d/(2d-n).
    """
    if 2 * d <= n:
        return True  # граница не применима
    import math
    return (2 ** k) <= 2 * d // (2 * d - n)


def feasible(k: int, d: int, n: int = 6) -> bool:
    """Проверить, допустимы ли параметры [n,k,d] по всем границам."""
    return (singleton_bound(k, d, n) and
            hamming_bound(k, d, n) and
            d >= 1 and k >= 1 and k <= n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='hexcode — коды в Q6')
    sub = parser.add_subparsers(dest='cmd')

    p_std = sub.add_parser('standard', help='Показать стандартные коды')

    p_info = sub.add_parser('info', help='Анализ кода по G-матрице (строки через запятую)')
    p_info.add_argument('matrix', nargs='+', help='Строки G-матрицы в виде "101010"')

    p_encode = sub.add_parser('encode', help='Кодирование сообщения')
    p_encode.add_argument('code', choices=['rep', 'parity', 'hamming', 'hex312'])
    p_encode.add_argument('message', help='Бит-строка сообщения, напр. "101"')

    p_decode = sub.add_parser('decode', help='Декодирование принятого слова')
    p_decode.add_argument('code', choices=['rep', 'parity', 'hamming', 'hex312'])
    p_decode.add_argument('received', help='Принятое слово (6 бит), напр. "101010"')

    p_find = sub.add_parser('find', help='Найти коды с d ≥ d_min')
    p_find.add_argument('d_min', type=int)
    p_find.add_argument('--k', type=int, default=None)

    p_bounds = sub.add_parser('bounds', help='Таблица кодовых границ')

    args = parser.parse_args()

    _STD_CODES = {
        'rep': repetition_code,
        'parity': parity_check_code,
        'hamming': shortened_hamming_code,
        'hex312': hexcode_312,
    }

    if args.cmd == 'standard':
        for name, factory in [
            ('Repetition [6,1,6]', repetition_code),
            ('Parity check [6,5,2]', parity_check_code),
            ('Shortened Hamming [6,?,4]', shortened_hamming_code),
            ('HexCode [6,3,3]', hexcode_312),
            ('Full space [6,6,1]', full_space_code),
        ]:
            code = factory()
            print(f"\n{'='*50}")
            print(f"  {name}")
            print('='*50)
            print(code.display())

    elif args.cmd == 'info':
        G = []
        for row_str in args.matrix:
            G.append([int(b) for b in row_str if b in '01'])
        code = BinaryCode(G)
        print(code.display())

    elif args.cmd == 'encode':
        code = _STD_CODES[args.code]()
        msg = [int(b) for b in args.message if b in '01']
        cw = code.encode(msg)
        print(f"Сообщение:     {''.join(map(str, msg))}")
        print(f"Кодовое слово: {to_bits(cw)}  ({cw})")

    elif args.cmd == 'decode':
        code = _STD_CODES[args.code]()
        recv = int(args.received, 2)
        result = code.decode(recv)
        print(f"Принято:    {to_bits(recv)}  ({recv})")
        if result is not None:
            print(f"Декодировано: {to_bits(result)}  ({result})")
        else:
            print("Ошибка: не удалось декодировать")

    elif args.cmd == 'find':
        codes = find_codes(args.d_min, k=args.k)
        print(f"Найдено {len(codes)} кодов с d ≥ {args.d_min}"
              + (f", k={args.k}" if args.k else ''))
        for c in codes[:10]:
            print(f"  {c}")

    elif args.cmd == 'bounds':
        print(f"{'k':>3} {'d':>3} | {'Singleton':>10} {'Hamming':>10} {'Plotkin':>10} {'feasible':>10}")
        print('-' * 55)
        for k_ in range(1, 7):
            for d_ in range(1, 7):
                row = (f"{k_:>3} {d_:>3} | "
                       f"{'OK' if singleton_bound(k_,d_) else 'FAIL':>10} "
                       f"{'OK' if hamming_bound(k_,d_) else 'FAIL':>10} "
                       f"{'OK' if plotkin_bound(k_,d_) else 'FAIL':>10} "
                       f"{'YES' if feasible(k_,d_) else 'no':>10}")
                print(row)

    else:
        parser.print_help()
