"""Tests for hexhept — Heptahedron (RP² model)."""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexhept.hexhept import (
    Heptahedron,
    RP2Checker,
)


class TestHeptahedron(unittest.TestCase):
    def setUp(self):
        self.h = Heptahedron(1.0)

    def test_vertices(self):
        self.assertEqual(self.h.vertices(), 6)

    def test_faces(self):
        self.assertEqual(self.h.faces(), 7)

    def test_edges(self):
        self.assertEqual(self.h.edges(), 12)

    def test_euler_characteristic(self):
        self.assertEqual(self.h.euler_characteristic(), 1)

    def test_face_types(self):
        ft = self.h.face_types()
        self.assertEqual(ft["triangle"], 4)
        self.assertEqual(ft["square"], 3)
        self.assertEqual(ft["triangle"] + ft["square"], 7)

    def test_surface_area_unit(self):
        # S = √3·(1+√3) for a=1
        expected = math.sqrt(3) * (1 + math.sqrt(3))
        self.assertAlmostEqual(self.h.surface_area(), expected)

    def test_surface_area_scaled(self):
        h2 = Heptahedron(2.0)
        # S scales as a²
        self.assertAlmostEqual(h2.surface_area(), 4 * self.h.surface_area())

    def test_pseudo_volume_unit(self):
        # V = √2/6 for a=1
        expected = math.sqrt(2) / 6
        self.assertAlmostEqual(self.h.pseudo_volume(), expected)

    def test_pseudo_volume_scaled(self):
        h3 = Heptahedron(3.0)
        self.assertAlmostEqual(h3.pseudo_volume(), 27 * self.h.pseudo_volume())

    def test_is_rp2_model(self):
        self.assertTrue(self.h.is_rp2_model())

    def test_summary_contains_chi(self):
        s = self.h.summary()
        self.assertIn("1", s)  # χ=1

    def test_net_description(self):
        net = self.h.net_description()
        self.assertIn("12", net)
        self.assertIn("χ", net)


class TestRP2Checker(unittest.TestCase):
    def setUp(self):
        self.checker = RP2Checker()

    def test_heptahedron_is_rp2(self):
        result = self.checker.check_heptahedron()
        self.assertTrue(result["ok"])
        self.assertEqual(result["chi"], 1)

    def test_icosahedron_not_rp2(self):
        result = self.checker.check_icosahedron()
        self.assertFalse(result["ok"])
        self.assertEqual(result["chi"], 2)

    def test_cuboctahedron_not_rp2(self):
        result = self.checker.check_cuboctahedron()
        self.assertFalse(result["ok"])
        self.assertEqual(result["chi"], 2)

    def test_check_wrong_chi(self):
        result = self.checker.check(4, 4, 6)  # tetrahedron χ=2
        self.assertFalse(result["ok"])

    def test_check_chi_1(self):
        result = self.checker.check(6, 7, 12)  # heptahedron
        self.assertTrue(result["ok"])
        self.assertEqual(result["chi"], 1)

    def test_enumerate_candidates_contains_heptahedron(self):
        candidates = self.checker.enumerate_rp2_candidates(8)
        found = any(c["vertices"] == 6 and c["faces"] == 7 for c in candidates)
        self.assertTrue(found, "Heptahedron (6,7,12) should appear in candidates")

    def test_enumerate_candidates_all_chi1(self):
        candidates = self.checker.enumerate_rp2_candidates(8)
        for c in candidates:
            self.assertEqual(c["chi"], 1)


if __name__ == "__main__":
    unittest.main()
