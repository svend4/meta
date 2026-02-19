"""Tests for hexmatroot — Analytical roots of 2×2 matrices."""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexmatroot.hexmatroot import (
    MatrixAlgebra,
    _det,
    _tr,
    _mul_mat,
    _mat_eq,
    _inv,
    _eye,
    _scale,
)


class TestMatrixHelpers(unittest.TestCase):
    def test_det(self):
        A = [[3.0, 2.0], [4.0, 3.0]]
        self.assertAlmostEqual(_det(A), 1.0)

    def test_tr(self):
        A = [[3.0, 2.0], [4.0, 3.0]]
        self.assertAlmostEqual(_tr(A), 6.0)

    def test_eye(self):
        E = _eye()
        self.assertEqual(E, [[1.0, 0.0], [0.0, 1.0]])

    def test_inv(self):
        A = [[3.0, 2.0], [4.0, 3.0]]
        inv = _inv(A)
        product = _mul_mat(A, inv)
        self.assertTrue(_mat_eq(product, _eye()))

    def test_singular_raises(self):
        A = [[1.0, 2.0], [2.0, 4.0]]
        with self.assertRaises(ValueError):
            _inv(A)

    def test_mul_mat_identity(self):
        A = [[2.0, 1.0], [3.0, 4.0]]
        result = _mul_mat(A, _eye())
        self.assertTrue(_mat_eq(result, A))

    def test_scale(self):
        A = [[1.0, 0.0], [0.0, 1.0]]
        result = _scale(3.0, A)
        self.assertAlmostEqual(result[0][0], 3.0)
        self.assertAlmostEqual(result[1][1], 3.0)


class TestInversionOperator(unittest.TestCase):
    def setUp(self):
        self.ma = MatrixAlgebra()
        self.A = [[3.0, 2.0], [4.0, 3.0]]

    def test_inversion_op_property(self):
        # A · Aₒ = A⁻¹
        res = self.ma.verify_inversion_operator(self.A)
        self.assertTrue(res["ok"])

    def test_inversion_op_identity(self):
        # For E: Eₒ = E⁻² = E
        res = self.ma.verify_inversion_operator(_eye())
        self.assertTrue(res["ok"])

    def test_inversion_op_2x2(self):
        # Verify Aₒ = (A⁻¹)²
        Ao = self.ma.inversion_operator(self.A)
        inv_A = _inv(self.A)
        expected = _mul_mat(inv_A, inv_A)
        self.assertTrue(_mat_eq(Ao, expected))


class TestSqrtMatrix(unittest.TestCase):
    def setUp(self):
        self.ma = MatrixAlgebra()

    def test_sqrt_identity(self):
        # √E = E
        roots = self.ma.sqrt_matrix(_eye())
        self.assertGreater(len(roots), 0)
        found = any(_mat_eq(r, _eye()) for r in roots)
        self.assertTrue(found, "E should be a square root of E")

    def test_sqrt_squared_gives_original(self):
        A = [[3.0, 2.0], [4.0, 3.0]]  # det=1 > 0
        roots = self.ma.sqrt_matrix(A)
        self.assertGreater(len(roots), 0)
        for r in roots:
            r2 = _mul_mat(r, r)
            self.assertTrue(_mat_eq(r2, A, tol=1e-5),
                            f"root² ≠ A: {r2} vs {A}")

    def test_sqrt_negative_det_raises(self):
        # det < 0 → ValueError
        A = [[1.0, 2.0], [3.0, 1.0]]  # det = 1 - 6 = -5
        with self.assertRaises(ValueError):
            self.ma.sqrt_matrix(A)

    def test_has_sqrt_true(self):
        A = [[3.0, 2.0], [4.0, 3.0]]
        self.assertTrue(self.ma.has_sqrt(A))

    def test_has_sqrt_false_negative_det(self):
        A = [[0.0, 1.0], [-1.0, 0.0]]  # det = 1, but tr = 0, tr - 2√1 = -2 < 0
        # Actually det([[0,1],[-1,0]]) = 0*0 - 1*(-1) = 1 > 0
        # tr = 0; tr ± 2√det = 0 ± 2; one branch exists
        A2 = [[1.0, 2.0], [3.0, 1.0]]  # det=-5
        self.assertFalse(self.ma.has_sqrt(A2))


class TestIdempotent(unittest.TestCase):
    def setUp(self):
        self.ma = MatrixAlgebra()

    def test_idempotent_true(self):
        # A = [[1,0],[0,0]]: det=0, tr=1 → idempotent
        A = [[1.0, 0.0], [0.0, 0.0]]
        self.assertTrue(self.ma.is_idempotent(A))

    def test_idempotent_projection(self):
        # A = [[0.5, 0.5],[0.5, 0.5]]: det=0.25-0.25=0, tr=1 → idempotent
        A = [[0.5, 0.5], [0.5, 0.5]]
        self.assertTrue(self.ma.is_idempotent(A))

    def test_idempotent_false(self):
        A = [[2.0, 0.0], [0.0, 2.0]]
        self.assertFalse(self.ma.is_idempotent(A))

    def test_idempotent_verify(self):
        A = [[1.0, 0.0], [0.0, 0.0]]
        res = self.ma.is_idempotent_verify(A)
        self.assertTrue(res["A2_eq_A"])
        self.assertTrue(res["condition_det_zero"])
        self.assertTrue(res["condition_tr_one"])


class TestPauli(unittest.TestCase):
    def test_pauli_1(self):
        S1 = MatrixAlgebra.pauli(1)
        self.assertEqual(S1, [[0.0, 1.0], [1.0, 0.0]])

    def test_pauli_2(self):
        S2 = MatrixAlgebra.pauli(2)
        self.assertEqual(S2, [[0.0, -1.0], [1.0, 0.0]])

    def test_pauli_3(self):
        S3 = MatrixAlgebra.pauli(3)
        self.assertEqual(S3, [[1.0, 0.0], [0.0, -1.0]])

    def test_pauli_invalid(self):
        with self.assertRaises(ValueError):
            MatrixAlgebra.pauli(0)

    def test_pauli_det_1_and_3(self):
        # S1 det = 0*0-1*1 = -1; S3 det = 1*(-1)-0*0 = -1
        self.assertAlmostEqual(_det(MatrixAlgebra.pauli(1)), -1.0)
        self.assertAlmostEqual(_det(MatrixAlgebra.pauli(3)), -1.0)


class TestSolveQuadratic(unittest.TestCase):
    def setUp(self):
        self.ma = MatrixAlgebra()

    def test_solve_real_case(self):
        # X² + 0X - E = 0 → X² = E → roots ±E
        C = _scale(-1.0, _eye())  # -E
        solutions = self.ma.solve_quadratic(0.0, C)
        self.assertGreater(len(solutions), 0)
        for X in solutions:
            X2 = _mul_mat(X, X)
            # X² - E = 0 → X² = E
            self.assertTrue(_mat_eq(X2, _eye(), tol=1e-5),
                            f"X² ≠ E: {X2}")


if __name__ == "__main__":
    unittest.main()
