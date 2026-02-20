"""Tests for hexhept — Heptahedron (RP² model)."""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexhept.hexhept import (
    Heptahedron,
    RP2Checker,
    _euler_characteristic,
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


class TestEulerCharacteristic(unittest.TestCase):
    def test_heptahedron(self):
        self.assertEqual(_euler_characteristic(6, 7, 12), 1)

    def test_tetrahedron(self):
        self.assertEqual(_euler_characteristic(4, 4, 6), 2)

    def test_icosahedron(self):
        self.assertEqual(_euler_characteristic(12, 20, 30), 2)


class TestSphereIsomorphism(unittest.TestCase):
    def test_contains_rp2_and_sphere(self):
        h = Heptahedron(1.0)
        s = h.sphere_isomorphism()
        self.assertIn('RP²', s)
        self.assertIn('S²', s)

    def test_returns_string(self):
        h = Heptahedron(1.0)
        self.assertIsInstance(h.sphere_isomorphism(), str)


class TestHeptahedronScaleIndependence(unittest.TestCase):
    def test_vertices_independent_of_scale(self):
        """Топология не зависит от длины ребра."""
        for a in [0.5, 2.0, 10.0]:
            h = Heptahedron(a)
            self.assertEqual(h.vertices(), 6)
            self.assertEqual(h.faces(), 7)
            self.assertEqual(h.edges(), 12)
            self.assertEqual(h.euler_characteristic(), 1)

    def test_summary_contains_face_count(self):
        """summary() содержит число граней (7) и рёбер (12)."""
        h = Heptahedron(1.0)
        s = h.summary()
        self.assertIn("7", s)
        self.assertIn("12", s)
        self.assertIn("χ", s)


class TestRP2CheckerExtended(unittest.TestCase):
    def setUp(self):
        self.checker = RP2Checker()

    def test_check_reason_key_when_chi_ne_1(self):
        """check() возвращает 'reason' для χ ≠ 1."""
        result = self.checker.check(4, 4, 6)   # tetrahedron χ=2
        self.assertIn("reason", result)
        self.assertFalse(result["ok"])

    def test_enumerate_candidates_dict_structure(self):
        """Каждый кандидат имеет ключи vertices, faces, edges, chi, ok."""
        candidates = self.checker.enumerate_rp2_candidates(8)
        for c in candidates:
            for key in ("vertices", "faces", "edges", "chi", "ok"):
                self.assertIn(key, c, f"Ключ '{key}' отсутствует в {c}")

    def test_check_with_face_types(self):
        """check() с face_types для гептаэдра → ok=True."""
        result = self.checker.check(
            6, 7, 12,
            face_types={"triangle": 4, "square": 3}
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["chi"], 1)


if __name__ == "__main__":
    unittest.main()
