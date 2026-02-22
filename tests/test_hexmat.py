"""Тесты для hexmat — линейная алгебра над GF(2) на Q6."""
import unittest

from projects.hexmat import (
    mat_zero, mat_identity, mat_from_cols, mat_from_rows, mat_permutation,
    mat_add, mat_neg, mat_transpose, mat_vec_mul, mat_mul, mat_pow,
    mat_trace, mat_det, is_invertible,
    row_reduce, mat_rank,
    mat_kernel, mat_image, mat_column_space,
    mat_inv,
    mat_hadamard_gf2, symplectic_matrix,
    gl6_order, random_invertible, count_invertible_6x6,
    orthogonal_complement,
    linear_code_from_generator, parity_check_matrix, minimum_distance,
)
from projects.hexmat.hexmat import mat_projection


# ─────────────────────────────────────────────────────────────────────────────
def _popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


# ─────────────────────────────────────────────────────────────────────────────
class TestMatConstructors(unittest.TestCase):

    def test_zero_matrix(self):
        """Нулевая матрица — 6 нулей."""
        Z = mat_zero()
        self.assertEqual(len(Z), 6)
        self.assertTrue(all(r == 0 for r in Z))

    def test_identity_matrix(self):
        """Единичная матрица: I[i][i]=1, I[i][j]=0 для i≠j."""
        I = mat_identity()
        self.assertEqual(len(I), 6)
        for i in range(6):
            self.assertEqual((I[i] >> i) & 1, 1)
            for j in range(6):
                if j != i:
                    self.assertEqual((I[i] >> j) & 1, 0)

    def test_from_cols_inverse_of_transpose(self):
        """mat_from_cols(columns) создаёт матрицу правильно."""
        cols = [1 << i for i in range(6)]  # единичные векторы
        M = mat_from_cols(cols)
        self.assertEqual(M, mat_identity())

    def test_permutation_matrix(self):
        """Матрица перестановки: (P·v)[i] = v[perm[i]] (перестановка компонент)."""
        perm = [1, 2, 3, 4, 5, 0]  # циклический сдвиг
        P = mat_permutation(perm)
        for v in [1, 7, 42, 63]:
            Pv = mat_vec_mul(P, v)
            for i in range(6):
                expected = (v >> perm[i]) & 1
                actual = (Pv >> i) & 1
                self.assertEqual(actual, expected, f"v={v}, i={i}")


# ─────────────────────────────────────────────────────────────────────────────
class TestMatBasicOps(unittest.TestCase):

    def test_add_zero(self):
        """A + 0 = A."""
        I = mat_identity()
        Z = mat_zero()
        self.assertEqual(mat_add(I, Z), I)

    def test_add_self_zero(self):
        """A + A = 0 (характеристика 2)."""
        I = mat_identity()
        self.assertEqual(mat_add(I, I), mat_zero())

    def test_add_commutativity(self):
        """A + B = B + A."""
        A = random_invertible(42)
        B = random_invertible(7)
        self.assertEqual(mat_add(A, B), mat_add(B, A))

    def test_neg_equals_self(self):
        """−A = A в GF(2)."""
        M = random_invertible(42)
        self.assertEqual(mat_neg(M), M)

    def test_transpose_twice(self):
        """(A^T)^T = A."""
        M = random_invertible(42)
        self.assertEqual(mat_transpose(mat_transpose(M)), M)

    def test_transpose_sum(self):
        """(A+B)^T = A^T + B^T."""
        A = random_invertible(42)
        B = random_invertible(7)
        self.assertEqual(mat_transpose(mat_add(A, B)),
                         mat_add(mat_transpose(A), mat_transpose(B)))

    def test_trace_identity(self):
        """Tr(I) = 6 mod 2 = 0."""
        self.assertEqual(mat_trace(mat_identity()), 0)

    def test_trace_zero(self):
        """Tr(0) = 0."""
        self.assertEqual(mat_trace(mat_zero()), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestMatVecMul(unittest.TestCase):

    def test_identity_vec_mul(self):
        """I · v = v для любого v."""
        I = mat_identity()
        for v in range(64):
            self.assertEqual(mat_vec_mul(I, v), v)

    def test_zero_vec_mul(self):
        """0 · v = 0."""
        Z = mat_zero()
        for v in [0, 1, 42, 63]:
            self.assertEqual(mat_vec_mul(Z, v), 0)

    def test_vec_mul_linearity(self):
        """M·(u⊕v) = M·u ⊕ M·v (линейность над GF(2))."""
        M = random_invertible(42)
        for u in [7, 13, 42]:
            for v in [3, 21, 56]:
                lhs = mat_vec_mul(M, u ^ v)
                rhs = mat_vec_mul(M, u) ^ mat_vec_mul(M, v)
                self.assertEqual(lhs, rhs)


# ─────────────────────────────────────────────────────────────────────────────
class TestMatMul(unittest.TestCase):

    def test_mul_by_identity_left(self):
        """I · A = A."""
        I = mat_identity()
        M = random_invertible(42)
        self.assertEqual(mat_mul(I, M), M)

    def test_mul_by_identity_right(self):
        """A · I = A."""
        I = mat_identity()
        M = random_invertible(42)
        self.assertEqual(mat_mul(M, I), M)

    def test_mul_by_zero(self):
        """A · 0 = 0."""
        Z = mat_zero()
        M = random_invertible(42)
        self.assertEqual(mat_mul(M, Z), mat_zero())

    def test_mul_associativity(self):
        """(A·B)·C = A·(B·C)."""
        A = random_invertible(1)
        B = random_invertible(2)
        C = random_invertible(3)
        self.assertEqual(mat_mul(mat_mul(A, B), C),
                         mat_mul(A, mat_mul(B, C)))

    def test_mul_consistency_with_vec(self):
        """(A·B)·v = A·(B·v)."""
        A = random_invertible(42)
        B = random_invertible(7)
        for v in [1, 7, 42, 63]:
            lhs = mat_vec_mul(mat_mul(A, B), v)
            rhs = mat_vec_mul(A, mat_vec_mul(B, v))
            self.assertEqual(lhs, rhs)

    def test_mat_pow_zero(self):
        """M^0 = I."""
        M = random_invertible(42)
        self.assertEqual(mat_pow(M, 0), mat_identity())

    def test_mat_pow_one(self):
        """M^1 = M."""
        M = random_invertible(42)
        self.assertEqual(mat_pow(M, 1), M)

    def test_mat_pow_two(self):
        """M^2 = M·M."""
        M = random_invertible(42)
        self.assertEqual(mat_pow(M, 2), mat_mul(M, M))


# ─────────────────────────────────────────────────────────────────────────────
class TestMatRankDet(unittest.TestCase):

    def test_rank_identity(self):
        """rank(I) = 6."""
        self.assertEqual(mat_rank(mat_identity()), 6)

    def test_rank_zero(self):
        """rank(0) = 0."""
        self.assertEqual(mat_rank(mat_zero()), 0)

    def test_rank_full(self):
        """Случайная обратимая матрица имеет ранг 6."""
        M = random_invertible(42)
        self.assertEqual(mat_rank(M), 6)

    def test_rank_singular(self):
        """Вырожденная матрица имеет ранг < 6."""
        M = mat_zero()
        M[0] = 7  # только одна ненулевая строка
        self.assertLess(mat_rank(M), 6)

    def test_det_identity(self):
        """det(I) = 1."""
        self.assertEqual(mat_det(mat_identity()), 1)

    def test_det_zero(self):
        """det(0) = 0."""
        self.assertEqual(mat_det(mat_zero()), 0)

    def test_det_product(self):
        """det(A·B) = det(A)·det(B) mod 2."""
        A = random_invertible(42)
        B = random_invertible(7)
        dAB = mat_det(mat_mul(A, B))
        dA_dB = (mat_det(A) * mat_det(B)) % 2
        self.assertEqual(dAB, dA_dB)

    def test_is_invertible_full_rank(self):
        """Полноранговая матрица обратима."""
        M = random_invertible(42)
        self.assertTrue(is_invertible(M))

    def test_not_invertible_zero(self):
        """Нулевая матрица не обратима."""
        self.assertFalse(is_invertible(mat_zero()))


# ─────────────────────────────────────────────────────────────────────────────
class TestRowReduce(unittest.TestCase):

    def test_rref_identity(self):
        """RREF(I) = I."""
        rref, rank, pivots = row_reduce(mat_identity())
        self.assertEqual(rank, 6)
        self.assertEqual(len(pivots), 6)

    def test_rref_rank_consistency(self):
        """Ранг = число опорных столбцов."""
        M = random_invertible(42)
        rref, rank, pivots = row_reduce(M)
        self.assertEqual(rank, len(pivots))

    def test_rref_is_rref(self):
        """После приведения: в каждом опорном столбце ровно одна 1."""
        M = random_invertible(42)
        rref, rank, pivots = row_reduce(M)
        for j in pivots:
            col_vals = [(rref[i] >> j) & 1 for i in range(6)]
            self.assertEqual(sum(col_vals), 1)


# ─────────────────────────────────────────────────────────────────────────────
class TestMatInverse(unittest.TestCase):

    def test_inv_identity(self):
        """I^{−1} = I."""
        I = mat_identity()
        self.assertEqual(mat_inv(I), I)

    def test_inv_times_original(self):
        """M · M^{−1} = I."""
        M = random_invertible(42)
        Minv = mat_inv(M)
        self.assertEqual(mat_mul(M, Minv), mat_identity())

    def test_original_times_inv(self):
        """M^{−1} · M = I."""
        M = random_invertible(42)
        Minv = mat_inv(M)
        self.assertEqual(mat_mul(Minv, M), mat_identity())

    def test_inv_singular_raises(self):
        """Обратная к вырожденной матрице бросает исключение."""
        M = mat_zero()
        with self.assertRaises((ValueError, Exception)):
            mat_inv(M)

    def test_inv_involution(self):
        """(M^{−1})^{−1} = M."""
        M = random_invertible(42)
        self.assertEqual(mat_inv(mat_inv(M)), M)


# ─────────────────────────────────────────────────────────────────────────────
class TestMatKernelImage(unittest.TestCase):

    def test_kernel_identity_empty(self):
        """Ядро единичной матрицы пустое (нулевое пространство = {0})."""
        ker = mat_kernel(mat_identity())
        self.assertEqual(ker, [])

    def test_kernel_zero_is_full(self):
        """Ядро нулевой матрицы = весь Q6 (6 базисных векторов)."""
        ker = mat_kernel(mat_zero())
        self.assertEqual(len(ker), 6)

    def test_kernel_rank_nullity(self):
        """dim(ker) + rank = 6."""
        M = random_invertible(42)
        ker = mat_kernel(M)
        self.assertEqual(len(ker) + mat_rank(M), 6)

    def test_kernel_vectors_in_kernel(self):
        """Все базисные векторы ядра удовлетворяют M·v = 0."""
        M = mat_zero()
        M[0] = 0b101010
        M[1] = 0b010101
        ker = mat_kernel(M)
        for v in ker:
            self.assertEqual(mat_vec_mul(M, v), 0)

    def test_image_rank_dimension(self):
        """dim(image) = rank(M)."""
        M = random_invertible(42)
        im = mat_column_space(M)
        self.assertEqual(len(im), mat_rank(M))


# ─────────────────────────────────────────────────────────────────────────────
class TestSpecialMatrices(unittest.TestCase):

    def test_hadamard_gf2_shape(self):
        """Матрица Адамара GF(2) имеет размер 6."""
        H = mat_hadamard_gf2()
        self.assertEqual(len(H), 6)

    def test_symplectic_matrix_skew_symmetric(self):
        """Симплектическая матрица J = J^T (в GF(2): J является симметричной)."""
        J = symplectic_matrix()
        self.assertEqual(J, mat_transpose(J))

    def test_symplectic_matrix_squared(self):
        """J^2 = I."""
        J = symplectic_matrix()
        self.assertEqual(mat_mul(J, J), mat_identity())

    def test_symplectic_nondegenerate(self):
        """J обратима."""
        J = symplectic_matrix()
        self.assertTrue(is_invertible(J))


# ─────────────────────────────────────────────────────────────────────────────
class TestGL6(unittest.TestCase):

    def test_gl6_order(self):
        """|GL(6,2)| = 20 158 709 760."""
        self.assertEqual(gl6_order(), 20_158_709_760)

    def test_random_invertible_is_invertible(self):
        """random_invertible возвращает обратимую матрицу."""
        for seed in [1, 42, 100, 999]:
            M = random_invertible(seed)
            self.assertTrue(is_invertible(M))

    def test_random_invertible_6_rows(self):
        """random_invertible возвращает матрицу из 6 строк."""
        M = random_invertible(42)
        self.assertEqual(len(M), 6)


# ─────────────────────────────────────────────────────────────────────────────
class TestLinearCodes(unittest.TestCase):

    def test_code_contains_zero(self):
        """Линейный код всегда содержит нулевое слово."""
        G = [0b100110, 0b010101, 0b001011]
        words = linear_code_from_generator(G)
        self.assertIn(0, words)

    def test_code_size(self):
        """Линейный [6,k]-код имеет 2^k кодовых слов."""
        G = [0b100110, 0b010101, 0b001011]   # k=3 независимых строк
        words = linear_code_from_generator(G)
        self.assertEqual(len(words), 8)   # 2^3 = 8

    def test_code_linear_combination(self):
        """Сумма двух кодовых слов — кодовое слово."""
        G = [0b111000, 0b000111, 0b101010]
        words = linear_code_from_generator(G)
        words_list = list(words)
        for i in range(min(4, len(words_list))):
            for j in range(min(4, len(words_list))):
                self.assertIn(words_list[i] ^ words_list[j], words)

    def test_parity_check_orthogonality(self):
        """H · G^T = 0 (проверочная матрица ортогональна порождающей)."""
        G = [0b100110, 0b010101, 0b001011]
        H = parity_check_matrix(G)
        for h_row in H:
            for g_row in G:
                # h_row · g_row (как 6-битные векторы) = 0
                from projects.hexmat.hexmat import _dot_gf2
                self.assertEqual(_dot_gf2(h_row, g_row), 0)

    def test_minimum_distance_positive(self):
        """Минимальное расстояние кода > 0."""
        G = [0b100110, 0b010101, 0b001011]
        words = linear_code_from_generator(G)
        d = minimum_distance(words)
        self.assertGreater(d, 0)

    def test_repetition_code_distance(self):
        """Repetition code [6,1]: генератор = [0b111111], min distance = 6."""
        G = [0b111111]  # все 6 битов = 1
        words = linear_code_from_generator(G)
        d = minimum_distance(words)
        self.assertEqual(d, 6)


# ─────────────────────────────────────────────────────────────────────────────
class TestOrthogonalComplement(unittest.TestCase):

    def test_complement_full_space(self):
        """Ортогональное дополнение V к V = {0} (orth comp — ядро)."""
        # Полное пространство → дополнение = {0}
        basis = [1 << i for i in range(6)]  # все 6 базисных векторов
        comp = orthogonal_complement(basis)
        self.assertEqual(comp, [])  # только нулевой вектор (пустой базис)

    def test_complement_of_empty(self):
        """Ортогональное дополнение {0} = весь Q6."""
        comp = orthogonal_complement([])
        self.assertEqual(len(comp), 6)


class TestMatProjection(unittest.TestCase):
    """Тесты mat_projection — ортогональный проектор (над GF(2))."""

    def test_empty_basis_is_zero(self):
        """Проекция на {0} = нулевая матрица."""
        self.assertEqual(mat_projection([]), mat_zero())

    def test_full_basis_is_identity(self):
        """Проекция на весь GF(2)^6 = единичная матрица."""
        full_basis = [1 << i for i in range(6)]
        self.assertEqual(mat_projection(full_basis), mat_identity())

    def test_single_basis_vector(self):
        """Проекция на ⟨e0⟩ = матрица с единственной ненулевой строкой."""
        p = mat_projection([1])  # e0 = 000001
        self.assertIsNotNone(p)
        self.assertIsInstance(p, list)
        self.assertEqual(len(p), 6)

    def test_single_basis_first_row(self):
        """Проекция на ⟨e0⟩: нулевая строка 0 = 1 (P[0]=1 → бит 0)."""
        p = mat_projection([1])
        self.assertEqual(p[0], 1)  # строка 0: только бит 0

    def test_two_independent_basis(self):
        """Проекция на ⟨e0, e1⟩: ненулевая матрица."""
        p = mat_projection([1, 2])  # e0=000001, e1=000010
        self.assertIsNotNone(p)
        # Первые две строки ненулевые
        self.assertNotEqual(p[0], 0)
        self.assertNotEqual(p[1], 0)

    def test_linearly_dependent_returns_none(self):
        """Линейно зависимый базис → None (необратимая матрица)."""
        result = mat_projection([1, 1])  # оба вектора одинаковы
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
