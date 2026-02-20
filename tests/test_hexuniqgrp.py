"""Tests for hexuniqgrp — Unique Groups (Herman)."""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexuniqgrp.hexuniqgrp import (
    UniqueGroups,
    FiniteGroup,
    classify_uvw,
    uvw_product,
)


class TestClassifyUVW(unittest.TestCase):
    def test_u_class(self):
        # n ≡ 1 (mod 6)
        for n in [1, 7, 13, 19, 25]:
            self.assertEqual(classify_uvw(n), "U", f"Failed for n={n}")

    def test_v_class(self):
        # n ≡ 3 (mod 6)
        for n in [3, 9, 15, 21]:
            self.assertEqual(classify_uvw(n), "V", f"Failed for n={n}")

    def test_w_class(self):
        # n ≡ 5 (mod 6)
        for n in [5, 11, 17, 23, 29]:
            self.assertEqual(classify_uvw(n), "W", f"Failed for n={n}")

    def test_even_class(self):
        for n in [2, 4, 6, 8]:
            self.assertEqual(classify_uvw(n), "E")


class TestUVWProduct(unittest.TestCase):
    def test_uu(self):
        self.assertEqual(uvw_product("U", "U"), "U")

    def test_ww(self):
        self.assertEqual(uvw_product("W", "W"), "U")

    def test_uw(self):
        self.assertEqual(uvw_product("U", "W"), "W")

    def test_wu(self):
        self.assertEqual(uvw_product("W", "U"), "W")

    def test_uv(self):
        self.assertEqual(uvw_product("U", "V"), "V")

    def test_wv(self):
        self.assertEqual(uvw_product("W", "V"), "V")

    def test_vv(self):
        self.assertEqual(uvw_product("V", "V"), "V")


class TestIsUniqueOrder(unittest.TestCase):
    def setUp(self):
        self.ug = UniqueGroups()

    def test_15_is_unique(self):
        self.assertTrue(self.ug.is_unique_order(15))

    def test_35_is_unique(self):
        self.assertTrue(self.ug.is_unique_order(35))

    def test_prime_not_unique(self):
        # Primes have exactly one group (trivial) but not "interesting"
        for p in [2, 3, 5, 7, 11]:
            self.assertFalse(self.ug.is_unique_order(p))

    def test_even_not_unique(self):
        for n in [2, 4, 6, 10, 15 * 2]:
            self.assertFalse(self.ug.is_unique_order(n))

    def test_square_not_unique(self):
        # 9 = 3² has a square factor
        self.assertFalse(self.ug.is_unique_order(9))
        self.assertFalse(self.ug.is_unique_order(25))

    def test_not_unique_q_cong_1_mod_p(self):
        # 21 = 3×7; 7 ≡ 1 (mod 3) → NOT unique
        self.assertFalse(self.ug.is_unique_order(21))

    def test_known_list_subset_of_computed(self):
        # Herman's PDF lists 35 unique orders ≤ 300; the algorithm may find more.
        # The known list must be a subset of the computed list.
        result = self.ug.verify_known_list()
        self.assertEqual(result["only_known"], [],
                         f"Known orders missing from computed: {result['only_known']}")

    def test_list_up_to_50(self):
        orders = self.ug.unique_orders_up_to(50)
        self.assertIn(15, orders)
        self.assertIn(33, orders)
        self.assertIn(35, orders)
        self.assertNotIn(21, orders)


class TestChordCount(unittest.TestCase):
    def setUp(self):
        self.ug = UniqueGroups()

    def test_chord_n4_k1(self):
        # C(4,2) = 6; / 1! = 6
        self.assertEqual(self.ug.chord_count(4, 1), 6)

    def test_chord_n4_k2(self):
        # i=1: C(4,2)=6; i=2: C(2,2)=1; product=6; /2! = 3
        self.assertEqual(self.ug.chord_count(4, 2), 3)

    def test_chord_zero_k(self):
        self.assertEqual(self.ug.chord_count(6, 0), 0)

    def test_chord_too_large_k(self):
        # 2k > n
        self.assertEqual(self.ug.chord_count(4, 3), 0)

    def test_chord_n6_k1(self):
        # C(6,2) = 15
        self.assertEqual(self.ug.chord_count(6, 1), 15)


class TestFiniteGroup(unittest.TestCase):
    def test_group_15(self):
        g = FiniteGroup(15)  # 15 = 3×5
        self.assertEqual(g.order(), 15)

    def test_group_repr(self):
        g = FiniteGroup(15)
        s = repr(g)
        self.assertIn("Z3", s)
        self.assertIn("Z5", s)

    def test_subgroups_15(self):
        g = FiniteGroup(15)
        subs = g.subgroups()
        self.assertIn("{e}", subs)
        self.assertIn("Z3", subs)
        self.assertIn("Z5", subs)

    def test_element_orders_15(self):
        g = FiniteGroup(15)
        ords = g.element_orders()
        # Should have elements of order 1, 3, 5, 15
        self.assertIn(1, ords)
        self.assertIn(15, ords)

    def test_cayley_table_nonempty(self):
        g = FiniteGroup(15)
        table = g.cayley_table()
        self.assertIsInstance(table, list)
        self.assertGreater(len(table), 0)


class TestClassifyOdd(unittest.TestCase):
    def test_classify_odd_u_class(self):
        ug = UniqueGroups()
        for n in [1, 7, 13, 19]:
            self.assertEqual(ug.classify_odd(n), "U", f"Failed for n={n}")

    def test_classify_odd_v_class(self):
        ug = UniqueGroups()
        for n in [3, 9, 15, 21]:
            self.assertEqual(ug.classify_odd(n), "V", f"Failed for n={n}")


class TestBuildGroup(unittest.TestCase):
    def test_build_group_15(self):
        ug = UniqueGroups()
        g = ug.build_group(15)
        self.assertIsInstance(g, FiniteGroup)
        self.assertEqual(g.order(), 15)

    def test_build_group_non_unique_raises(self):
        ug = UniqueGroups()
        with self.assertRaises(ValueError):
            ug.build_group(6)  # 6 = 2×3, even → not unique


class TestKnownUniqueOrders300(unittest.TestCase):
    def test_known_list_nonempty(self):
        ug = UniqueGroups()
        lst = ug.known_unique_orders_300()
        self.assertIsInstance(lst, list)
        self.assertGreater(len(lst), 0)
        self.assertIn(15, lst)


if __name__ == "__main__":
    unittest.main()
