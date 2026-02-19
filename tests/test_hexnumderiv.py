"""Тесты для hexnumderiv — производная числа (теория Германа)."""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexnumderiv.hexnumderiv import NumberDerivative, _proper_divisors


class TestProperDivisors(unittest.TestCase):
    def test_divisors_1(self):
        self.assertEqual(_proper_divisors(1), [])

    def test_divisors_prime(self):
        self.assertEqual(_proper_divisors(7), [1])

    def test_divisors_6(self):
        self.assertEqual(sorted(_proper_divisors(6)), [1, 2, 3])

    def test_divisors_12(self):
        self.assertEqual(sorted(_proper_divisors(12)), [1, 2, 3, 4, 6])


class TestDerivative(unittest.TestCase):
    def setUp(self):
        self.nd = NumberDerivative()

    def test_derivative_1(self):
        self.assertEqual(self.nd.derivative(1), 1)

    def test_derivative_prime(self):
        """∂p = 2 для любого простого p."""
        for p in [2, 3, 5, 7, 11, 13]:
            self.assertEqual(self.nd.derivative(p), 2)

    def test_derivative_6(self):
        """∂6 = 1 + 1 + 2 + 3 = 7."""
        self.assertEqual(self.nd.derivative(6), 7)

    def test_derivative_12(self):
        """∂12 = 1 + 1 + 2 + 3 + 4 + 6 = 17."""
        self.assertEqual(self.nd.derivative(12), 17)

    def test_derivative_28(self):
        """В стандартной теории σ(28)-28 = 28 (совершенное), но здесь ∂28 = 1+(σ(28)-28) = 29."""
        self.assertEqual(self.nd.derivative(28), 29)

    def test_derivative_invalid(self):
        with self.assertRaises(ValueError):
            self.nd.derivative(0)


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.nd = NumberDerivative()

    def test_perfect_2(self):
        """∂2 = 2: 2 — совершенное в этой теории."""
        self.assertEqual(self.nd.classify(2), "perfect")

    def test_perfect_4(self):
        """∂4 = 4: 4 — совершенное."""
        self.assertEqual(self.nd.classify(4), "perfect")

    def test_ordinary_prime(self):
        """Простые числа — обычные (цепочка: p → 2 → 2 → ..., не превышает p)."""
        self.assertEqual(self.nd.classify(5), "ordinary")

    def test_ordinary_10(self):
        """Цепочка 10 не превышает 10."""
        self.assertEqual(self.nd.classify(10), "ordinary")

    def test_super_138(self):
        """138 — супер-число (цепочка выходит за 138)."""
        self.assertEqual(self.nd.classify(138), "super")

    def test_6_is_not_perfect_here(self):
        """∂6 = 7 ≠ 6, так что 6 — не совершенное."""
        self.assertNotEqual(self.nd.classify(6), "perfect")


class TestChain(unittest.TestCase):
    def setUp(self):
        self.nd = NumberDerivative()

    def test_chain_starts_with_n(self):
        ch = self.nd.chain(10)
        self.assertEqual(ch[0], 10)

    def test_chain_finite_for_ordinary(self):
        """Цепочка конечна (завершается циклом или 1)."""
        ch = self.nd.chain(10)
        self.assertIsInstance(ch, list)
        self.assertGreater(len(ch), 1)

    def test_chain_perfect_is_fixed(self):
        """∂2 = 2, поэтому 2 — неподвижная точка."""
        ch = self.nd.chain(2, max_steps=5)
        self.assertEqual(ch[0], 2)
        self.assertEqual(ch[1], 2)   # ∂2 = 2

    def test_chain_prime(self):
        """5 → 2 → 2 (∂5=2, ∂2=2 — фиксированная точка)."""
        ch = self.nd.chain(5)
        self.assertEqual(ch[:3], [5, 2, 2])

    def test_chain_10(self):
        """∂10 = 1+1+2+5 = 9, ∂9 = 1+1+3 = 5, ∂5 = 2, ∂2 = 2."""
        ch = self.nd.chain(10)
        self.assertEqual(ch[0], 10)
        self.assertEqual(ch[1], 9)   # ∂10 = 9
        self.assertEqual(ch[2], 5)   # ∂9  = 5


class TestPerfectNumbers(unittest.TestCase):
    def setUp(self):
        self.nd = NumberDerivative()

    def test_perfect_up_to_1000(self):
        """В этой теории 'совершенные' = степени двойки: ∂(2^k) = 2^k."""
        perfects = self.nd.perfect_numbers(1000)
        for k in range(1, 10):
            p = 2 ** k
            if p <= 1000:
                self.assertIn(p, perfects, f"2^{k}={p} должно быть совершенным")

    def test_6_is_not_perfect_here(self):
        """В этой теории ∂6 = 7 ≠ 6, поэтому 6 не является совершенным."""
        nd = NumberDerivative()
        self.assertEqual(nd.derivative(6), 7)
        self.assertNotEqual(nd.classify(6), "perfect")

    def test_2_is_perfect(self):
        """∂2 = 2 — совершенное."""
        perfects = self.nd.perfect_numbers(10)
        self.assertIn(2, perfects)

    def test_4_is_perfect(self):
        """∂4 = 1+1+2 = 4 — совершенное."""
        self.assertEqual(self.nd.derivative(4), 4)


class TestLeibnizRule(unittest.TestCase):
    def setUp(self):
        self.nd = NumberDerivative()

    def test_leibniz_returns_dict(self):
        """leibniz_rule возвращает словарь с нужными ключами."""
        res = self.nd.leibniz_rule(2, 3)
        self.assertIn("km", res)
        self.assertIn("lhs", res)
        self.assertIn("rhs", res)
        self.assertIn("holds", res)

    def test_leibniz_components_2_3(self):
        """Для k=2, m=3: ∂2=2, ∂3=2, ∂6=7."""
        res = self.nd.leibniz_rule(2, 3)
        self.assertEqual(res["dk"], 2)
        self.assertEqual(res["dm"], 2)
        self.assertEqual(res["lhs"], 7)   # ∂6 = 7
        # rhs = 2·2 + 3·2 + 2·2 = 14 (Лейбниц не выполняется с этой ∂)
        self.assertEqual(res["rhs"], 14)

    def test_leibniz_not_coprime(self):
        with self.assertRaises(ValueError):
            self.nd.leibniz_rule(2, 4)  # gcd(2,4)=2


class TestEnvelopes(unittest.TestCase):
    def test_upper_envelope(self):
        self.assertAlmostEqual(NumberDerivative.upper_envelope(4), 5.0)

    def test_lower_envelope(self):
        self.assertAlmostEqual(NumberDerivative.lower_envelope(9), 4.0)


if __name__ == "__main__":
    unittest.main()
