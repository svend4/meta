"""hexmat.py — Линейная алгебра над GF(2) на пространстве Q6 = (GF(2))^6.

(GF(2))^6 как 6-мерное векторное пространство над полем GF(2) = {0, 1}:
  Вектор: 6-битное число 0..63,  bit k = компонента k
  Матрица: список 6 строк [r₀,...,r₅], каждая rᵢ — 6-битное число
           M[i][j] = (M[i] >> j) & 1

Операции:
  • Умножение матриц:   C[i][j] = Σ_k M[i][k]·N[k][j] mod 2
  • Умножение на вектор: (Mv)[i] = Σ_j M[i][j]·v[j] mod 2 = popcount(M[i] & v) mod 2
  • Транспонирование, сложение (XOR по строкам)
  • Ранг через элиминацию Гаусса по GF(2)
  • RREF (reduced row echelon form)
  • Ядро (null space) и образ (column space)
  • Обратная матрица (для элементов GL(6,2))
  • Определитель: 0 или 1 (ненулевой ↔ матрица обратима)

Специальные матрицы:
  • GL(6,2): группа обратимых матриц,
    |GL(6,2)| = (2^6−1)·(2^6−2)·(2^6−4)·(2^6−8)·(2^6−16)·(2^6−32) = 20 158 709 760
  • Матрица перестановки, симплектическая матрица J
  • Матрица Адамара над GF(2): H[u][v] = ⟨u,v⟩ mod 2 = popcount(u&v) mod 2

Связь с другими проектами:
  • hexsym:  Aut(Q6) содержит GL(6,2) как подгруппу (перестановки координат ≤ GL)
  • hexalg:  линейные отображения на Q6 → операторы свёртки через матрицы Кэли
  • hexcode: порождающие и проверочные матрицы линейных кодов
  • hexcrypt: линейный слой блочных шифров = обратимая матрица над GF(2)
"""

# ── helpers ────────────────────────────────────────────────────────────────────

def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def _dot_gf2(a, b):
    """Скалярное произведение двух 6-битных векторов над GF(2)."""
    return _popcount(a & b) & 1


# ── конструкторы матриц ──────────────────────────────────────────────────────

def mat_zero():
    """Нулевая матрица 6×6 над GF(2)."""
    return [0] * 6


def mat_identity():
    """Единичная матрица 6×6 над GF(2). I[i][j] = δᵢⱼ."""
    return [1 << i for i in range(6)]


def mat_from_cols(cols):
    """
    Построить матрицу из 6 столбцов (каждый столбец — 6-битный вектор).
    cols[j] = j-й столбец, cols[j][i] = M[i][j].
    """
    M = [0] * 6
    for j, col in enumerate(cols):
        for i in range(6):
            if (col >> i) & 1:
                M[i] |= (1 << j)
    return M


def mat_from_rows(rows):
    """Построить матрицу из 6 строк (список 6-битных чисел)."""
    return list(rows)


def mat_permutation(perm):
    """
    Матрица перестановки для перестановки perm: [0..5] → [0..5].
    (P·v)[i] = v[perm[i]], т.е. строка i = e_{perm[i]}.
    """
    return [1 << perm[i] for i in range(6)]


# ── базовые операции ──────────────────────────────────────────────────────────

def mat_add(A, B):
    """Сложение матриц над GF(2): A + B (XOR по строкам)."""
    return [a ^ b for a, b in zip(A, B)]


def mat_neg(A):
    """Отрицание над GF(2): −A = A (характеристика 2)."""
    return list(A)


def mat_transpose(M):
    """Транспонирование 6×6 матрицы над GF(2)."""
    T = [0] * 6
    for i in range(6):
        for j in range(6):
            if (M[i] >> j) & 1:
                T[j] |= (1 << i)
    return T


def mat_vec_mul(M, v):
    """
    Умножение матрицы на вектор: (Mv)[i] = popcount(M[i] & v) mod 2.
    M — матрица 6×6, v — 6-битный вектор (столбец).
    Результат — 6-битный вектор.
    """
    result = 0
    for i in range(6):
        bit = _dot_gf2(M[i], v)
        result |= bit << i
    return result


def mat_mul(A, B):
    """
    Умножение матриц A·B над GF(2).
    C[i][j] = Σ_k A[i][k]·B[k][j] mod 2 = popcount(A[i] & BT[j]) mod 2.
    """
    BT = mat_transpose(B)
    C = []
    for i in range(6):
        row = 0
        for j in range(6):
            row |= _dot_gf2(A[i], BT[j]) << j
        C.append(row)
    return C


def mat_pow(M, n):
    """Возведение матрицы в степень n ≥ 0 над GF(2)."""
    if n == 0:
        return mat_identity()
    result = mat_identity()
    base = list(M)
    while n:
        if n & 1:
            result = mat_mul(result, base)
        base = mat_mul(base, base)
        n >>= 1
    return result


# ── след и определитель ───────────────────────────────────────────────────────

def mat_trace(M):
    """След матрицы: Tr(M) = Σ M[i][i] mod 2."""
    t = 0
    for i in range(6):
        t ^= (M[i] >> i) & 1
    return t


def mat_det(M):
    """
    Определитель матрицы над GF(2): 0 или 1.
    det(M) = 1 ↔ M обратима ↔ rank(M) = 6.
    """
    _, rank, _ = row_reduce(list(M))
    return 1 if rank == 6 else 0


def is_invertible(M):
    """Является ли матрица M обратимой (det ≠ 0, rank = 6)?"""
    return mat_det(M) == 1


# ── элиминация Гаусса ────────────────────────────────────────────────────────

def row_reduce(M, augmented=None):
    """
    Приведение матрицы M к RREF над GF(2) (Гаусс–Жордан).
    Если augmented — матрица (список строк), применяются те же операции к ней.
    Возвращает (rref, rank, pivot_cols).
    """
    rows = list(M)
    aug = [list(a) for a in augmented] if augmented else None
    pivot_cols = []
    pivot_row = 0

    for col in range(6):
        # Найти ненулевой элемент в столбце col, начиная со строки pivot_row
        found = -1
        for r in range(pivot_row, 6):
            if (rows[r] >> col) & 1:
                found = r
                break
        if found == -1:
            continue
        # Переставить строки
        rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
        if aug:
            aug[pivot_row], aug[found] = aug[found], aug[pivot_row]
        # Обнулить остальные строки в столбце col
        for r in range(6):
            if r != pivot_row and (rows[r] >> col) & 1:
                rows[r] ^= rows[pivot_row]
                if aug:
                    for j in range(len(aug[r])):
                        aug[r][j] ^= aug[pivot_row][j]
        pivot_cols.append(col)
        pivot_row += 1

    rank = pivot_row
    if aug:
        return rows, rank, pivot_cols, aug
    return rows, rank, pivot_cols


def mat_rank(M):
    """Ранг матрицы M над GF(2)."""
    _, rank, _ = row_reduce(list(M))
    return rank


# ── ядро и образ ─────────────────────────────────────────────────────────────

def mat_kernel(M):
    """
    Ядро (null space) матрицы M: {v : Mv = 0}.
    Возвращает базис ядра как список 6-битных векторов.
    dim(ker) + dim(im) = 6 (теорема о ранге).
    """
    rref, rank, pivot_cols = row_reduce(list(M))
    free_cols = [j for j in range(6) if j not in pivot_cols]
    kernel = []
    for fc in free_cols:
        # Построить базисный вектор ядра: free_col = 1, остальные свободные = 0
        v = 1 << fc
        for r, pc in enumerate(pivot_cols):
            if (rref[r] >> fc) & 1:
                v |= 1 << pc
        kernel.append(v)
    return kernel


def mat_image(M):
    """
    Образ матрицы M (column space): {Mv : v ∈ GF(2)^6}.
    Возвращает базис образа (список 6-битных столбцов образа).
    """
    # Образ = пространство, порождённое столбцами M
    cols = [sum(((M[i] >> j) & 1) << i for i in range(6)) for j in range(6)]
    # Редуцировать систему столбцов
    basis = []
    seen = set()
    for col in cols:
        if col == 0:
            continue
        v = col
        for b in basis:
            v = min(v, v ^ b)
        if v != 0 and v not in seen:
            basis.append(v)
            seen.add(v)
    return basis


def mat_column_space(M):
    """Базис пространства столбцов матрицы M (= образ M как линейного оператора)."""
    MT = mat_transpose(M)
    rref, rank, pivot_cols = row_reduce(MT)
    return [rref[r] for r in range(rank)]


# ── обратная матрица ──────────────────────────────────────────────────────────

def mat_inv(M):
    """
    Обратная матрица M^{−1} над GF(2).
    Метод: Гаусс–Жордан на расширенной матрице [M | I] → [I | M^{−1}].
    Raises ValueError если M вырождена.
    """
    I = [[1 if i == j else 0 for j in range(6)] for i in range(6)]
    rows = list(M)
    aug = I
    rref, rank, pivot_cols, aug_rref = row_reduce(rows, aug)
    if rank < 6:
        raise ValueError("Матрица вырождена (необратима)")
    # Восстановить результат из aug_rref
    result = [0] * 6
    for i in range(6):
        row_val = 0
        for j in range(6):
            row_val |= aug_rref[i][j] << j
        result[i] = row_val
    return result


# ── специальные матрицы ───────────────────────────────────────────────────────

def mat_hadamard_gf2():
    """
    Матрица Адамара над GF(2): H[u][v] = ⟨u, v⟩ = popcount(u & v) mod 2.
    H[i][j] = (i & j) != 0 (точнее: число единичных битов в i & j) mod 2.
    Связана с WHT через (−1)^{H[u][v]}.
    """
    return [sum(_dot_gf2(i, j) << j for j in range(6)) for i in range(6)]


def symplectic_matrix():
    """
    Стандартная симплектическая матрица J на (GF(2))^6 = (GF(2))^{2+2+2}:
    J = diag(J₂, J₂, J₂),  J₂ = [[0,1],[1,0]] (в GF(2), антисимметрия = симметрия).
    J^2 = I,  J^T = J.
    """
    # J[2i][2i+1] = 1, J[2i+1][2i] = 1 для i=0,1,2
    J = mat_zero()
    for i in range(3):
        J[2 * i] |= 1 << (2 * i + 1)
        J[2 * i + 1] |= 1 << (2 * i)
    return J


# ── GL(6,2) ───────────────────────────────────────────────────────────────────

def gl6_order():
    """
    Порядок GL(6,2) = Π_{k=0}^{5} (2^6 − 2^k).
    = (64-1)(64-2)(64-4)(64-8)(64-16)(64-32)
    = 63 × 62 × 60 × 56 × 48 × 32 = 20 158 709 760.
    """
    result = 1
    for k in range(6):
        result *= (64 - (1 << k))
    return result


def random_invertible(seed=42):
    """
    Случайная обратимая матрица из GL(6,2).
    Метод: генерировать случайные строки пока ранг < 6 (QR-подобный подход).
    """
    import random
    rng = random.Random(seed)
    while True:
        M = [rng.randint(0, 63) for _ in range(6)]
        if mat_rank(M) == 6:
            return M


def count_invertible_6x6():
    """
    Число обратимых матриц 6×6 над GF(2) = |GL(6,2)|.
    Вычисляется аналитически (не перебором).
    """
    return gl6_order()


# ── проекции и разложения ─────────────────────────────────────────────────────

def mat_projection(subspace_basis):
    """
    Ортогональный проектор на линейное подпространство (над GF(2)).
    subspace_basis — список базисных векторов (6-битных чисел).
    Проектор P: v ↦ ближайший вектор в подпространстве (Хэммингов смысл).
    В GF(2): P = G·(G^T·G)^{-1}·G^T где G — матрица с базисными столбцами.
    """
    k = len(subspace_basis)
    if k == 0:
        return mat_zero()
    if k == 6:
        return mat_identity()
    # G: 6 × k матрица (k столбцов = базисные векторы)
    # (G^T G)_{ij} = ⟨b_i, b_j⟩ mod 2
    B = subspace_basis
    # Построить G^T G (k×k матрица)
    GtG = [[_dot_gf2(B[i], B[j]) for j in range(k)] for i in range(k)]
    # Инвертировать G^T G
    try:
        # Используем row_reduce для малой матрицы
        def small_inv(mat_k):
            n = len(mat_k)
            aug = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            rows = [list(r) for r in mat_k]
            pivot_row = 0
            for col in range(n):
                found = -1
                for r in range(pivot_row, n):
                    if rows[r][col]:
                        found = r
                        break
                if found == -1:
                    return None
                rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
                aug[pivot_row], aug[found] = aug[found], aug[pivot_row]
                for r in range(n):
                    if r != pivot_row and rows[r][col]:
                        rows[r] = [a ^ b for a, b in zip(rows[r], rows[pivot_row])]
                        aug[r] = [a ^ b for a, b in zip(aug[r], aug[pivot_row])]
                pivot_row += 1
            return aug
        inv_GtG = small_inv(GtG)
        if inv_GtG is None:
            return None
    except Exception:
        return None
    # P = G (G^T G)^{-1} G^T
    # Вычислить P как 6×6 матрица: P[i][j] = Σ_{a,b} G[i][a] (G^T G)^{-1}[a][b] G[j][b]
    P = mat_zero()
    for i in range(6):
        for j in range(6):
            val = 0
            for a in range(k):
                for b in range(k):
                    val ^= ((B[a] >> i) & 1) * inv_GtG[a][b] * ((B[b] >> j) & 1)
            if val & 1:
                P[i] |= 1 << j
    return P


def orthogonal_complement(subspace_basis):
    """
    Ортогональное дополнение V⊥ = {w : ⟨w,v⟩=0 для всех v∈V}.
    Совпадает с ядром матрицы, составленной из строк subspace_basis.
    """
    if not subspace_basis:
        return [1 << i for i in range(6)]  # весь V
    M = subspace_basis + [0] * (6 - len(subspace_basis))
    return mat_kernel(M)


# ── линейные коды (порождающая матрица) ────────────────────────────────────────

def linear_code_from_generator(G):
    """
    Линейный код из порождающей матрицы G (k строк по 6 бит).
    Кодовые слова = {Σ uᵢ·G[i] : u ∈ GF(2)^k} (линейные комбинации строк G).
    Возвращает frozenset кодовых слов.
    """
    k = len(G)
    codewords = set()
    for u in range(1 << k):
        word = 0
        for i in range(k):
            if (u >> i) & 1:
                word ^= G[i]
        codewords.add(word)
    return frozenset(codewords)


def parity_check_matrix(G):
    """
    Проверочная матрица H для кода с порождающей матрицей G.
    H·G^T = 0,  образ G = ядро H.
    """
    # H — матрица из базиса ортогонального дополнения образа G
    k = len(G)
    n = 6  # фиксированная длина Q6
    # Ортогональное дополнение к пространству строк G
    # = ядро матрицы, составленной из строк G
    rows = list(G) + [0] * (n - k)
    H_basis = mat_kernel(rows[:n] if len(rows) >= n else rows)
    return H_basis


def minimum_distance(codewords):
    """Минимальное расстояние Хэмминга кода (минимальный вес ненулевых слов)."""
    nonzero = [w for w in codewords if w != 0]
    if not nonzero:
        return 0
    return min(_popcount(w) for w in nonzero)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if cmd == 'info':
        I = mat_identity()
        print(f"Линейная алгебра над GF(2) на Q6 = (GF(2))^6:")
        print(f"  |GL(6,2)| = {gl6_order():,}")
        print(f"  Единичная матрица: {I}")
        M = random_invertible(seed=42)
        print(f"  Случайная обратимая матрица (seed=42): {M}")
        print(f"  Ранг: {mat_rank(M)}, det: {mat_det(M)}")

    elif cmd == 'mul':
        A = [int(x) for x in sys.argv[2:8]] if len(sys.argv) >= 8 else mat_identity()
        B = [int(x) for x in sys.argv[8:14]] if len(sys.argv) >= 14 else A
        C = mat_mul(A, B)
        print(f"A·B =")
        for row in C:
            print(f"  {row:06b}")

    elif cmd == 'rank':
        A = [int(x, 0) for x in sys.argv[2:8]] if len(sys.argv) >= 8 else [63]*6
        print(f"Ранг: {mat_rank(A)}")
        print(f"Ядро: {mat_kernel(A)}")

    elif cmd == 'code':
        # Пример: [3,6]-код с порождающей матрицей
        G = [0b100110, 0b010101, 0b001011]  # 3 строки × 6 бит
        words = linear_code_from_generator(G)
        d = minimum_distance(words)
        print(f"Линейный [6,3,{d}]-код:")
        print(f"  Кодовых слов: {len(words)}")
        print(f"  Минимальное расстояние: {d}")

    else:
        print("hexmat.py — Линейная алгебра над GF(2) на Q6")
        print("Команды: info  mul [строки_A строки_B]  rank [строки]  code")


if __name__ == '__main__':
    main()
