"""
hexring — булевы функции и алгебраические структуры на Q6

Q6 = (Z₂)⁶ — 6-мерное векторное пространство над GF(2), содержащее 64 точки.
Каждая булева функция f: (Z₂)⁶ → {0,1} задаётся таблицей истинности из 64 бит.

Кольцевая структура:
  ((Z₂)⁶, XOR, AND) — булево кольцо; XOR = сложение, AND = умножение.

Ключевые понятия:
  Таблица истинности (TT) : вектор (f(0), f(1), ..., f(63)) ∈ {0,1}⁶⁴
  АНФ (ANF)               : алгебраическая нормальная форма (мономы x^I)
  WHT                     : преобразование Уолша–Адамара; Ŵ(u) = Σ_x (-1)^{f(x)+u·x}
  Нелинейность            : nl(f) = мин расстояние от f до аффинных функций
                            = (64 - max_u |Ŵ(u)|) / 2
  Bent-функция            : nl(f) = 28 (максимум для n=6), |Ŵ(u)| = 8 ∀u
  Корреляционный иммунитет: CI(t) ↔ Ŵ(u) = 0 для всех 0 < weight(u) ≤ t
  Алгебраическая степень  : max weight(I) для мономов x^I с ненулевым коэффициентом

Коды Рида–Маллера RM(r, 6):
  Кодовые слова = таблицы истинности функций степени ≤ r.
  Длина 64, размерность Σ_{i≤r} C(6,i), min расстояние 2^{6-r}.

Примечание: здесь «кодовые слова» имеют длину 64 (= 2^6), в отличие от hexcode,
где длина 6. RM-код — «внешний» код, индексированный гексаграммами Q6.
"""

from __future__ import annotations
import sys
from itertools import combinations
from typing import Callable

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import hamming, yang_count, SIZE


# ---------------------------------------------------------------------------
# Быстрые преобразования (без numpy)
# ---------------------------------------------------------------------------

def _wht_inplace(W: list[int]) -> None:
    """Быстрое преобразование Уолша–Адамара (butterfly) на месте."""
    n = len(W)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(i, i + step):
                a, b = W[j], W[j + step]
                W[j] = a + b
                W[j + step] = a - b
        step <<= 1


def _mobius_inplace(a: list[int]) -> None:
    """Быстрое преобразование Мёбиуса (XOR butterfly) для ANF."""
    n = len(a)
    step = 1
    while step < n:
        for i in range(0, n, 2 * step):
            for j in range(i, i + step):
                a[j + step] ^= a[j]
        step <<= 1


# ---------------------------------------------------------------------------
# Основной класс: булева функция
# ---------------------------------------------------------------------------

class BoolFunc:
    """
    Булева функция f: (Z₂)⁶ → {0,1}.

    Внутреннее представление: таблица истинности `_tt` — список из 64 значений {0,1},
    где _tt[x] = f(x) для x ∈ {0, 1, ..., 63}.

    Создание:
        BoolFunc(list_of_64_bits)   — из таблицы истинности
        BoolFunc(64_bit_int)        — из целочисленной маски (бит i = f(i))
        BoolFunc(callable)          — из функции f: int → int
    """

    N = 6   # число переменных
    SIZE = 64  # 2^N

    def __init__(self, table: list[int] | int | Callable) -> None:
        if callable(table):
            self._tt = [int(table(x)) & 1 for x in range(self.SIZE)]
        elif isinstance(table, int):
            self._tt = [(table >> x) & 1 for x in range(self.SIZE)]
        else:
            if len(table) != self.SIZE:
                raise ValueError(f"Таблица истинности должна иметь {self.SIZE} элементов")
            self._tt = [int(b) & 1 for b in table]

    def __call__(self, x: int) -> int:
        """Вычислить f(x)."""
        return self._tt[x & (self.SIZE - 1)]

    def __repr__(self) -> str:
        return (f"BoolFunc(degree={self.algebraic_degree()}, "
                f"nl={self.nonlinearity()}, balanced={self.is_balanced()})")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BoolFunc):
            return self._tt == other._tt
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self._tt))

    # ---- представления ---------------------------------------------------

    def truth_table(self) -> list[int]:
        """Таблица истинности: список из 64 бит."""
        return self._tt[:]

    def as_int(self) -> int:
        """64-битная маска: бит i = f(i)."""
        result = 0
        for i, b in enumerate(self._tt):
            if b:
                result |= (1 << i)
        return result

    def as_hex(self) -> str:
        """Шестнадцатеричная запись маски (16 символов)."""
        return f'{self.as_int():016x}'

    # ---- спектральные преобразования ------------------------------------

    def wht(self) -> list[int]:
        """
        Преобразование Уолша–Адамара.
        Ŵ(u) = Σ_{x ∈ (Z₂)⁶} (-1)^{f(x) + u·x}
        Результат: список из 64 целых чисел (сумма ±1 по 64 слагаемым).
        """
        W = [1 - 2 * b for b in self._tt]
        _wht_inplace(W)
        return W

    def anf_coeffs(self) -> list[int]:
        """
        ANF-коэффициенты через преобразование Мёбиуса.
        result[mask] = коэффициент монома x^mask в алгебраической нормальной форме.
        x^mask = произведение x_i для всех i: (mask >> i) & 1 = 1.
        """
        a = self._tt[:]
        _mobius_inplace(a)
        return a

    def anf_monomials(self) -> list[tuple[int, ...]]:
        """
        Список мономов ANF с ненулевым коэффициентом.
        Каждый моном задаётся кортежем индексов задействованных переменных.
        """
        coeffs = self.anf_coeffs()
        monomials = []
        for mask, c in enumerate(coeffs):
            if c:
                indices = tuple(i for i in range(self.N) if (mask >> i) & 1)
                monomials.append(indices)
        return monomials

    # ---- криптографические свойства ------------------------------------

    def nonlinearity(self) -> int:
        """
        Нелинейность: расстояние Хэмминга до ближайшей аффинной функции.
        nl(f) = (64 - max_u |Ŵ(u)|) / 2.
        Диапазон: 0 (аффинная) ≤ nl ≤ 28 (bent).
        """
        W = self.wht()
        return (self.SIZE - max(abs(w) for w in W)) // 2

    def algebraic_degree(self) -> int:
        """
        Алгебраическая степень: максимальный вес ненулевого монома ANF.
        Степень аффинной функции ≤ 1, квадратичной = 2 и т.д.
        """
        coeffs = self.anf_coeffs()
        return max(
            (bin(mask).count('1') for mask, c in enumerate(coeffs) if c),
            default=0
        )

    def correlation_immunity(self) -> int:
        """
        Корреляционный иммунитет (CI):
        максимальное t: Ŵ(u) = 0 для всех u с 0 < weight(u) ≤ t.
        CI = 0 означает, что некоторый Ŵ(u) ≠ 0 при weight(u) = 1.
        """
        W = self.wht()
        for t in range(1, self.N + 1):
            for u in range(self.SIZE):
                if bin(u).count('1') == t and W[u] != 0:
                    return t - 1
        return self.N

    def resilience(self) -> int:
        """
        Стойкость: CI(t) для сбалансированной функции.
        -1 если функция не сбалансирована.
        """
        if not self.is_balanced():
            return -1
        return self.correlation_immunity()

    def is_bent(self) -> bool:
        """
        Bent-функция: все |Ŵ(u)| = 2^{n/2} = 8.
        Максимально нелинейная функция: nl = 28.
        Bent-функции не сбалансированы.
        """
        W = self.wht()
        return all(abs(w) == 8 for w in W)

    def is_balanced(self) -> bool:
        """Сбалансированная функция: ровно 32 нуля и 32 единицы."""
        return sum(self._tt) == 32

    def is_affine(self) -> bool:
        """Аффинная функция: f(x) = a·x + b. Степень ≤ 1."""
        coeffs = self.anf_coeffs()
        return all(bin(mask).count('1') <= 1 for mask, c in enumerate(coeffs) if c)

    def is_linear(self) -> bool:
        """Линейная функция: f(x) = a·x (f(0) = 0). Степень ≤ 1, f(0) = 0."""
        return self.is_affine() and self._tt[0] == 0

    def is_symmetric(self) -> bool:
        """
        Симметричная функция: f(x) зависит только от yang_count(x).
        f(x) = f(y) если weight(x) = weight(y).
        """
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if bin(x).count('1') == bin(y).count('1') and self._tt[x] != self._tt[y]:
                    return False
        return True

    # ---- арифметика над GF(2) ------------------------------------------

    def __add__(self, other: 'BoolFunc') -> 'BoolFunc':
        """XOR двух функций: (f + g)(x) = f(x) ⊕ g(x)."""
        if isinstance(other, BoolFunc):
            return BoolFunc([a ^ b for a, b in zip(self._tt, other._tt)])
        return NotImplemented

    def __mul__(self, other: 'BoolFunc') -> 'BoolFunc':
        """AND двух функций: (f · g)(x) = f(x) ∧ g(x)."""
        if isinstance(other, BoolFunc):
            return BoolFunc([a & b for a, b in zip(self._tt, other._tt)])
        return NotImplemented

    def __neg__(self) -> 'BoolFunc':
        """Дополнение: (¬f)(x) = 1 - f(x)."""
        return BoolFunc([1 - b for b in self._tt])

    def __xor__(self, other: 'BoolFunc') -> 'BoolFunc':
        return self.__add__(other)

    # ---- дисплей --------------------------------------------------------

    def display(self, compact: bool = False) -> str:
        """Текстовое описание функции."""
        lines = [
            f"BoolFunc:",
            f"  ANF: {self._anf_str()}",
            f"  Степень      : {self.algebraic_degree()}",
            f"  Нелинейность : {self.nonlinearity()}",
            f"  Сбаланс.     : {self.is_balanced()}",
            f"  Bent         : {self.is_bent()}",
            f"  CI           : {self.correlation_immunity()}",
        ]
        if not compact:
            lines.append(f"  Hex          : {self.as_hex()}")
            tt = ''.join(map(str, self._tt))
            lines.append(f"  TT           : {tt}")
        return '\n'.join(lines)

    def _anf_str(self) -> str:
        """ANF в виде строки: x0*x3 + x1*x2*x4 + 1."""
        monomials = []
        for mask, c in enumerate(self.anf_coeffs()):
            if not c:
                continue
            if mask == 0:
                monomials.append('1')
            else:
                term = '*'.join(f'x{i}' for i in range(self.N) if (mask >> i) & 1)
                monomials.append(term)
        return ' + '.join(monomials) if monomials else '0'


# ---------------------------------------------------------------------------
# Стандартные булевы функции на Q6
# ---------------------------------------------------------------------------

def zero_func() -> BoolFunc:
    """Нулевая функция f(x) = 0."""
    return BoolFunc([0] * SIZE)


def one_func() -> BoolFunc:
    """Единичная функция f(x) = 1."""
    return BoolFunc([1] * SIZE)


def coordinate(i: int) -> BoolFunc:
    """Проекция: f(x) = x_i (i-й бит x), i ∈ {0..5}."""
    if not 0 <= i < 6:
        raise ValueError("i должен быть в {0..5}")
    return BoolFunc(lambda x: (x >> i) & 1)


def inner_product(a: int) -> BoolFunc:
    """Линейная функция f(x) = a·x = Σ a_i x_i (mod 2)."""
    def f(x: int) -> int:
        return bin(a & x).count('1') % 2
    return BoolFunc(f)


def inner_product_bent() -> BoolFunc:
    """
    Квадратичная bent-функция: f(x) = x0x1 + x2x3 + x4x5.
    Максимальная нелинейность 28.
    Это Maiorana-McFarland конструкция для n=6.
    """
    def f(x: int) -> int:
        b = [(x >> i) & 1 for i in range(6)]
        return (b[0] * b[1] + b[2] * b[3] + b[4] * b[5]) % 2
    return BoolFunc(f)


def maiorana_mcfarland(h_func: Callable[[int], int] | None = None) -> BoolFunc:
    """
    Конструкция Майораны–Макфарланда для n=6 (n=2m, m=3).
    f(x, y) = x·π(y) + g(y), x,y ∈ (Z₂)³.
    π: (Z₂)³ → (Z₂)³ — биекция (по умолчанию тождественная).
    g: (Z₂)³ → {0,1} — произвольная функция (по умолчанию 0).
    """
    # x = нижние 3 бита, y = верхние 3 бита
    def pi_default(y: int) -> int:
        return y  # тождественная биекция

    pi = pi_default

    def f(z: int) -> int:
        x = z & 7          # биты 0,1,2
        y = (z >> 3) & 7   # биты 3,4,5
        xdp = bin(x & pi(y)).count('1') % 2  # x · π(y)
        g_y = (h_func(y) if h_func else 0)   # g(y)
        return (xdp + g_y) % 2

    return BoolFunc(f)


def yang_parity() -> BoolFunc:
    """Чётность веса: f(x) = yang_count(x) mod 2 = XOR всех битов."""
    return BoolFunc(lambda x: bin(x).count('1') % 2)


def threshold_func(t: int) -> BoolFunc:
    """Пороговая функция: f(x) = 1 iff yang_count(x) ≥ t."""
    return BoolFunc(lambda x: int(bin(x).count('1') >= t))


def all_linear_functions() -> list[BoolFunc]:
    """Все 64 линейные функции f_a(x) = a·x, a ∈ (Z₂)⁶."""
    return [inner_product(a) for a in range(SIZE)]


def all_affine_functions() -> list[BoolFunc]:
    """Все 128 аффинных функций f_{a,b}(x) = a·x + b, a ∈ (Z₂)⁶, b ∈ {0,1}."""
    result = []
    for a in range(SIZE):
        f = inner_product(a)
        result.append(f)
        result.append(-f)
    return result


# ---------------------------------------------------------------------------
# Код Рида–Маллера RM(r, 6)
# ---------------------------------------------------------------------------

class ReedMullerCode:
    """
    Код Рида–Маллера RM(r, m=6).

    Кодовые слова — таблицы истинности булевых функций степени ≤ r.
    Длина: n = 2^m = 64 (каждый бит — значение функции на гексаграмме Q6).
    Размерность: k = Σ_{i=0}^{r} C(m, i).
    Минимальное расстояние: d = 2^{m-r}.

    Это «внешний» код: координаты кодового слова — гексаграммы Q6.
    """

    M = 6

    def __init__(self, r: int) -> None:
        if not 0 <= r <= self.M:
            raise ValueError(f"r должно быть в [0, {self.M}]")
        self.r = r

    @property
    def n(self) -> int:
        return 1 << self.M   # 64

    @property
    def k(self) -> int:
        import math
        return sum(math.comb(self.M, i) for i in range(self.r + 1))

    @property
    def d(self) -> int:
        return 1 << (self.M - self.r)

    def _monomials(self) -> list[int]:
        """Все маски мономов степени ≤ r (в порядке возрастания степени)."""
        result = [0]  # константа (степень 0)
        for deg in range(1, self.r + 1):
            for bits in combinations(range(self.M), deg):
                mask = sum(1 << b for b in bits)
                result.append(mask)
        return result

    def generator_functions(self) -> list[BoolFunc]:
        """
        Порождающие функции кода: BoolFunc для каждого монома степени ≤ r.
        Порядок: 1, x0, x1, ..., x5, x0x1, x0x2, ..., x4x5, ...
        """
        funcs = []
        for mask in self._monomials():
            if mask == 0:
                funcs.append(one_func())
            else:
                # Произведение проекций
                f = one_func()
                for i in range(self.M):
                    if (mask >> i) & 1:
                        f = f * coordinate(i)
                funcs.append(f)
        return funcs

    def generator_matrix(self) -> list[list[int]]:
        """
        Порождающая матрица G (k × n).
        Строка i = таблица истинности i-го монома.
        """
        return [f.truth_table() for f in self.generator_functions()]

    def encode(self, message: list[int]) -> BoolFunc:
        """
        Кодирование: message (k бит) → кодовое слово BoolFunc.
        message[i] = 1 iff i-й моном включён в сумму.
        """
        if len(message) != self.k:
            raise ValueError(f"Длина сообщения должна быть {self.k}, получено {len(message)}")
        gens = self.generator_functions()
        result = zero_func()
        for i, bit in enumerate(message):
            if bit:
                result = result + gens[i]
        return result

    def contains(self, f: BoolFunc) -> bool:
        """Проверить, является ли f кодовым словом RM(r, 6)."""
        return f.algebraic_degree() <= self.r

    def decode(self, received: BoolFunc) -> BoolFunc:
        """
        Мягкое декодирование: найти кодовое слово на минимальном расстоянии.
        Для RM(1, m): алгоритм Хэдамара (по WHT).
        Для r ≥ 2: поэтапное декодирование Рида (приближённое).
        """
        if self.r == 0:
            # Ближайшая константа
            ones = sum(received.truth_table())
            return one_func() if ones >= 32 else zero_func()

        if self.r == 1:
            # RM(1,6): WHT → найти u с max |Ŵ(u)|
            W = received.wht()
            best_u = max(range(SIZE), key=lambda u: abs(W[u]))
            # Знак определяет константный член
            f = inner_product(best_u)
            if W[best_u] < 0:
                f = -f + one_func()
            elif W[best_u] > 0:
                pass  # f(x) = best_u · x
            return f

        # Для r ≥ 2: проектировать на RM(1,6) рекурсивно (упрощённо)
        # Принимаем ближайшую функцию степени ≤ r по ANF
        anf = received.anf_coeffs()
        # Обнулить все мономы степени > r
        new_anf = [
            c if bin(mask).count('1') <= self.r else 0
            for mask, c in enumerate(anf)
        ]
        # Обратное преобразование Мёбиуса = то же самое (над GF(2))
        _mobius_inplace(new_anf)
        return BoolFunc(new_anf)

    def __repr__(self) -> str:
        return f"RM({self.r}, {self.M}): [{self.n}, {self.k}, {self.d}]"

    def info(self) -> str:
        lines = [
            f"RM({self.r}, {self.M})-код:",
            f"  Длина n          : {self.n}",
            f"  Размерность k    : {self.k}",
            f"  Мин. расстояние d: {self.d}",
            f"  Скорость R       : {self.k/self.n:.4f}",
            f"  Исправляет ошибок: t = {(self.d - 1) // 2}",
        ]
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Аппроксимация и анализ
# ---------------------------------------------------------------------------

def best_affine_approximation(f: BoolFunc) -> BoolFunc:
    """
    Ближайшая аффинная функция к f.
    Находится через WHT: g(x) = (sign Ŵ(u*) > 0) * (u*·x) + const.
    """
    W = f.wht()
    best_u = max(range(SIZE), key=lambda u: abs(W[u]))
    g = inner_product(best_u)
    if W[best_u] < 0:
        # Знак < 0 означает, что f ближе к 1 + u·x
        g = -g + one_func()
    return g


def auto_correlation(f: BoolFunc, a: int) -> int:
    """
    Автокорреляция функции f при сдвиге a:
    C_f(a) = Σ_x (-1)^{f(x) + f(x + a)}
    Связь с WHT: C_f(a) = (1/64) Σ_u Ŵ(u)² * (-1)^{u·a}.
    """
    return sum(
        (1 - 2 * f(x)) * (1 - 2 * f(x ^ a))
        for x in range(SIZE)
    )


def auto_correlation_table(f: BoolFunc) -> list[int]:
    """ACF(a) для всех a ∈ Q6."""
    return [auto_correlation(f, a) for a in range(SIZE)]


def power_moment(f: BoolFunc, k: int) -> int:
    """k-й момент WHT: Σ_u Ŵ(u)^k."""
    W = f.wht()
    return sum(w ** k for w in W)


def count_bent_in_rm2() -> int:
    """
    Подсчитать bent-функции среди кодовых слов RM(2,6).
    (Вычислительно интенсивно: 2^22 ≈ 4 млн слов.)
    Используется только для справочных целей.
    """
    # Это слишком медленно для реального использования
    raise NotImplementedError("Слишком медленно (2^22 кодовых слов)")


# ---------------------------------------------------------------------------
# Поиск функций с заданными свойствами
# ---------------------------------------------------------------------------

def find_bent_examples(n_max: int = 10) -> list[BoolFunc]:
    """
    Найти несколько bent-функций на Q6 методом случайного поиска.
    Гарантированный bent: inner_product_bent() и его производные.
    """
    result = []

    # Известные конструкции
    base = inner_product_bent()
    result.append(base)
    if len(result) >= n_max:
        return result

    # Константный сдвиг: f' = f + 1 (дополнение) тоже bent
    result.append(base + one_func())
    if len(result) >= n_max:
        return result

    # Аффинный эквивалент: f' = f + linear (тоже bent)
    for a in range(1, SIZE):
        shifted = base + inner_product(a)
        if shifted.is_bent() and shifted not in result:
            result.append(shifted)
            if len(result) >= n_max:
                return result

    return result


def find_resilient(ci_order: int = 1, n_max: int = 5) -> list[BoolFunc]:
    """
    Найти несколько сбалансированных функций с CI ≥ ci_order.
    Стойкие функции (resilient): сбалансированные + CI.
    """
    # Конструкция Сигерталера: функция, не зависящая от части переменных
    # и сбалансированная по оставшимся, является CI по этим переменным
    result = []
    # Простая конструкция: f(x) = g(x_{k+1}, ..., x_5) для сбаланс. g
    # — функция, не зависящая от x_0..x_{k-1}, CI(k-1) относительно них
    for mask_free in range(1, SIZE):
        if bin(mask_free).count('1') < ci_order:
            continue
        # Функция, зависящая только от битов вне mask_free
        dep_bits = [i for i in range(6) if not (mask_free >> i) & 1]
        if not dep_bits:
            continue
        # Случайная сбалансированная функция от dep_bits
        dep_size = 1 << len(dep_bits)
        tt = [0] * SIZE
        # Заполнить half-half по dep_bits
        half = dep_size // 2
        vals = [1] * half + [0] * (dep_size - half)
        for x in range(SIZE):
            dep_val = sum(((x >> b) & 1) << i for i, b in enumerate(dep_bits))
            tt[x] = vals[dep_val % dep_size]
        f = BoolFunc(tt)
        if f.is_balanced() and f.correlation_immunity() >= ci_order:
            if f not in result:
                result.append(f)
                if len(result) >= n_max:
                    return result
    return result


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def hamming_distance_func(f: BoolFunc, g: BoolFunc) -> int:
    """Расстояние Хэмминга между f и g как двоичными векторами длины 64."""
    return sum(a != b for a, b in zip(f.truth_table(), g.truth_table()))


def nonlinearity_profile(f: BoolFunc) -> dict[int, int]:
    """
    Профиль нелинейности по уровням RM:
    {r: расстояние от f до RM(r,6)}.
    """
    profile = {}
    for r in range(7):
        rm = ReedMullerCode(r)
        # Расстояние до RM(r,6) = расстояние до ближайшего кодового слова
        # = расстояние до ближайшей функции степени ≤ r
        decoded = rm.decode(f)
        profile[r] = hamming_distance_func(f, decoded)
    return profile


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='hexring — булевы функции на Q6')
    sub = parser.add_subparsers(dest='cmd')

    p_info = sub.add_parser('info', help='Анализ булевой функции')
    p_info.add_argument('func', help='Функция: "bent", "yang_parity", "coord:i", hex-маска')

    p_rm = sub.add_parser('rm', help='Информация о коде Рида–Маллера')
    p_rm.add_argument('r', type=int, help='Порядок RM(r, 6)')
    p_rm.add_argument('--encode', help='Сообщение (строка битов k бит) → кодовое слово')

    p_bent = sub.add_parser('bent', help='Найти и показать bent-функции')
    p_bent.add_argument('--n', type=int, default=5)

    p_wht = sub.add_parser('wht', help='WHT-спектр функции')
    p_wht.add_argument('func', help='Функция: "bent", hex-маска, "coord:i"')

    p_acf = sub.add_parser('acf', help='Таблица автокорреляции')
    p_acf.add_argument('func', help='Функция')

    p_table = sub.add_parser('table', help='Таблица RM-кодов')

    args = parser.parse_args()

    def parse_func(s: str) -> BoolFunc:
        if s == 'bent':
            return inner_product_bent()
        elif s == 'yang_parity':
            return yang_parity()
        elif s == 'zero':
            return zero_func()
        elif s == 'one':
            return one_func()
        elif s.startswith('coord:'):
            i = int(s.split(':')[1])
            return coordinate(i)
        elif s.startswith('linear:'):
            a = int(s.split(':')[1])
            return inner_product(a)
        elif s.startswith('threshold:'):
            t = int(s.split(':')[1])
            return threshold_func(t)
        else:
            # Шестнадцатеричная маска (64-битная = 16 hex-символов)
            return BoolFunc(int(s, 16))

    if args.cmd == 'info':
        f = parse_func(args.func)
        print(f.display())

    elif args.cmd == 'rm':
        rm = ReedMullerCode(args.r)
        print(rm.info())
        if args.encode:
            msg = [int(b) for b in args.encode if b in '01']
            if len(msg) != rm.k:
                print(f"Ошибка: нужно {rm.k} бит, получено {len(msg)}")
            else:
                cw = rm.encode(msg)
                print(f"\nКодовое слово:")
                print(f"  Степень: {cw.algebraic_degree()}")
                print(f"  ANF: {cw._anf_str()}")
                print(f"  TT: {''.join(map(str, cw.truth_table()))}")

    elif args.cmd == 'bent':
        bents = find_bent_examples(args.n)
        print(f"Bent-функции на Q6 ({len(bents)} примеров):")
        for i, f in enumerate(bents):
            print(f"\n[{i+1}] {f._anf_str()}")
            print(f"     nl={f.nonlinearity()}, degree={f.algebraic_degree()}, hex={f.as_hex()}")

    elif args.cmd == 'wht':
        f = parse_func(args.func)
        W = f.wht()
        print(f"WHT-спектр функции '{args.func}':")
        print(f"  max|Ŵ|  = {max(abs(w) for w in W)}")
        print(f"  nl      = {f.nonlinearity()}")
        print(f"  Ŵ(0)    = {W[0]}  (сумма f = {(64 - W[0]) // 2} единиц)")
        print(f"\n  u → Ŵ(u) [только ненулевые]:")
        for u, w in enumerate(W):
            if w != 0:
                yang_u = bin(u).count('1')
                print(f"  {u:2d} (yang={yang_u}): {w:+4d}")

    elif args.cmd == 'acf':
        f = parse_func(args.func)
        acf = auto_correlation_table(f)
        print(f"Автокорреляция '{args.func}':")
        for a, c in enumerate(acf):
            if c != 64:   # 64 = самокорреляция при a=0
                yang_a = bin(a).count('1')
                print(f"  a={a:2d} (yang={yang_a}): ACF={c:+4d}")

    elif args.cmd == 'table':
        import math
        print(f"{'r':>3} | {'k':>4} {'d':>4} {'R':>7} | Описание")
        print('-' * 50)
        descs = [
            'Repetition',
            'Affine (linear + const)',
            'Quadratic + lower',
            'Cubic + lower',
            'Quartic + lower',
            'Quintic + lower',
            'All functions',
        ]
        for r in range(7):
            rm = ReedMullerCode(r)
            print(f"  {r:1d} | {rm.k:4d} {rm.d:4d} {rm.k/rm.n:7.4f} | {descs[r]}")

    else:
        parser.print_help()
