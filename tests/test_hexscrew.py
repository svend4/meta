"""Tests for hexscrew — Screw Group Bₙ."""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexscrew.hexscrew import (
    ScrewGroup,
    _cycle_decomposition,
    _sign,
    _compose,
    _inverse,
    _order_of_element,
)


class TestHelpers(unittest.TestCase):
    def test_compose(self):
        # p∘q: apply q first then p
        p = [2, 1, 3]
        q = [1, 3, 2]
        result = _compose(p, q)
        # q: 1→1,2→3,3→2; p: 1→2,2→1,3→3
        # (p∘q)(1)=p(q(1))=p(1)=2, (p∘q)(2)=p(q(2))=p(3)=3, (p∘q)(3)=p(q(3))=p(2)=1
        self.assertEqual(result, [2, 3, 1])

    def test_inverse(self):
        perm = [2, 3, 1]
        inv = _inverse(perm)
        # p(1)=2, p(2)=3, p(3)=1 → inv(2)=1, inv(3)=2, inv(1)=3 → [3,1,2]
        self.assertEqual(inv, [3, 1, 2])
        # p∘inv = identity
        identity = _compose(perm, inv)
        self.assertEqual(identity, [1, 2, 3])

    def test_sign_even(self):
        # Cyclic permutation (1 2 3) — even (2 transpositions)
        self.assertEqual(_sign([2, 3, 1]), 1)

    def test_sign_odd(self):
        # Transposition (1 2) — odd
        self.assertEqual(_sign([2, 1, 3]), -1)

    def test_sign_identity(self):
        self.assertEqual(_sign([1, 2, 3]), 1)

    def test_cycle_decomposition(self):
        # Permutation [2, 3, 1] = cycle (1 2 3)
        cycles = _cycle_decomposition([2, 3, 1])
        self.assertEqual(len(cycles), 1)
        self.assertEqual(sorted(cycles[0]), [1, 2, 3])

    def test_order_of_element_identity(self):
        self.assertEqual(_order_of_element([1, 2, 3]), 1)

    def test_order_of_element_transposition(self):
        self.assertEqual(_order_of_element([2, 1, 3]), 2)

    def test_order_of_element_3cycle(self):
        self.assertEqual(_order_of_element([2, 3, 1]), 3)


class TestScrewGroupOrder(unittest.TestCase):
    def test_order_b2(self):
        self.assertEqual(ScrewGroup(2).order(), 1)

    def test_order_b3(self):
        self.assertEqual(ScrewGroup(3).order(), 2)

    def test_order_b4(self):
        self.assertEqual(ScrewGroup(4).order(), 6)

    def test_order_b5(self):
        self.assertEqual(ScrewGroup(5).order(), 24)

    def test_elements_count_b4(self):
        bg = ScrewGroup(4)
        elems = bg.elements()
        self.assertEqual(len(elems), 6)

    def test_elements_all_fix_1(self):
        bg = ScrewGroup(4)
        for e in bg.elements():
            self.assertEqual(e[0], 1, f"Element {e} does not fix 1")

    def test_is_member_true(self):
        bg = ScrewGroup(4)
        self.assertTrue(bg.is_member([1, 3, 2, 4]))

    def test_is_member_false(self):
        bg = ScrewGroup(4)
        self.assertFalse(bg.is_member([2, 1, 3, 4]))

    def test_invalid_n(self):
        with self.assertRaises(ValueError):
            ScrewGroup(1)


class TestScrewGroupOperations(unittest.TestCase):
    def setUp(self):
        self.bg = ScrewGroup(4)

    def test_multiply_closed(self):
        p = [1, 3, 2, 4]
        q = [1, 2, 4, 3]
        result = self.bg.multiply(p, q)
        self.assertTrue(self.bg.is_member(result))

    def test_multiply_identity(self):
        identity = [1, 2, 3, 4]
        p = [1, 3, 4, 2]
        self.assertEqual(self.bg.multiply(p, identity), p)
        self.assertEqual(self.bg.multiply(identity, p), p)

    def test_inverse_gives_identity(self):
        p = [1, 3, 4, 2]
        inv = self.bg.inverse(p)
        result = self.bg.multiply(p, inv)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_order_of_transposition(self):
        p = [1, 3, 2, 4]  # swaps 2 and 3 (keeps 1 fixed)
        self.assertEqual(self.bg.order_of(p), 2)

    def test_order_of_3cycle(self):
        p = [1, 3, 4, 2]  # 3-cycle (2 3 4)
        self.assertEqual(self.bg.order_of(p), 3)


class TestScrewGroupStructure(unittest.TestCase):
    def test_generating_transpositions_b3(self):
        bg = ScrewGroup(3)
        trans = bg.generating_transpositions()
        self.assertIn((2, 3), trans)
        self.assertEqual(len(trans), 1)

    def test_generating_transpositions_b4(self):
        bg = ScrewGroup(4)
        trans = bg.generating_transpositions()
        # C(3,2) = 3: (2,3), (2,4), (3,4)
        self.assertEqual(len(trans), 3)

    def test_left_spin_identity(self):
        bg = ScrewGroup(4)
        identity = [1, 2, 3, 4]
        self.assertEqual(bg.left_spin(identity), 1)

    def test_left_spin_transposition(self):
        bg = ScrewGroup(4)
        p = [1, 3, 2, 4]  # swap 2↔3
        self.assertEqual(bg.left_spin(p), -1)

    def test_cycle_type(self):
        bg = ScrewGroup(4)
        # [1,3,4,2] has cycles: {1} and (2 3 4)
        ct = bg.cycle_type([1, 3, 4, 2])
        self.assertIn(3, ct)
        self.assertIn(1, ct)

    def test_conjugacy_classes_b3(self):
        bg = ScrewGroup(3)
        classes = bg.conjugacy_classes()
        # B3 ≅ S2: identity + one transposition → 2 conjugacy classes
        self.assertEqual(len(classes), 2)

    def test_isomorphism_b3(self):
        bg = ScrewGroup(3)
        phi = bg.isomorphism_to_symmetric()
        # B3 has 2 elements, S2 has 2 elements
        self.assertEqual(len(phi), 2)


if __name__ == "__main__":
    unittest.main()
