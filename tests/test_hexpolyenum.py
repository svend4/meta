"""Tests for hexpolyenum — Polyhedron enumeration + ExDodecahedron."""
import sys
import os
import math
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexpolyenum.hexpolyenum import (
    PolyhedronEnumerator,
    PolyhedronRecord,
    ExDodecahedron,
    _platonic_name,
)

_PHI = (1 + math.sqrt(5)) / 2


class TestPolyhedronRecord(unittest.TestCase):
    def test_euler_tetrahedron(self):
        # Tetrahedron: B=4, Γ=4, P=6
        rec = PolyhedronRecord("Tetra", 3, 3, 4, 4, 6)
        self.assertEqual(rec.euler(), 2)

    def test_euler_cube(self):
        rec = PolyhedronRecord("Cube", 3, 4, 8, 6, 12)
        self.assertEqual(rec.euler(), 2)

    def test_diagonal_count(self):
        # Cube: B=8 → B(B-1)/2 - P = 28 - 12 = 16
        rec = PolyhedronRecord("Cube", 3, 4, 8, 6, 12)
        self.assertEqual(rec.diagonal_count(), 16)

    def test_repr(self):
        rec = PolyhedronRecord("Cube", 3, 4, 8, 6, 12)
        s = repr(rec)
        self.assertIn("Cube", s)
        self.assertIn("χ=2", s)


class TestPlatonicName(unittest.TestCase):
    def test_tetrahedron(self):
        self.assertEqual(_platonic_name(3, 3), "Тетраэдр")

    def test_cube(self):
        self.assertEqual(_platonic_name(3, 4), "Куб")

    def test_octahedron(self):
        self.assertEqual(_platonic_name(4, 3), "Октаэдр")

    def test_dodecahedron(self):
        self.assertEqual(_platonic_name(3, 5), "Додекаэдр")

    def test_icosahedron(self):
        self.assertEqual(_platonic_name(5, 3), "Икосаэдр")

    def test_unknown(self):
        name = _platonic_name(6, 6)
        self.assertIn("(6,6)", name)


class TestPolyhedronEnumerator(unittest.TestCase):
    def setUp(self):
        self.pe = PolyhedronEnumerator()

    def test_spherical_contains_platonic_5(self):
        recs = self.pe.enumerate_spherical()
        names = [r.name for r in recs]
        for expected in ["Тетраэдр", "Куб", "Октаэдр", "Додекаэдр", "Икосаэдр"]:
            self.assertIn(expected, names)

    def test_spherical_all_chi_2(self):
        for r in self.pe.enumerate_spherical():
            self.assertEqual(r.euler(), 2)

    def test_spherical_tetrahedron(self):
        rec = self.pe.from_degrees(face_degree=3, vertex_degree=3)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.vertices, 4)
        self.assertEqual(rec.faces, 4)
        self.assertEqual(rec.edges, 6)

    def test_spherical_cube(self):
        rec = self.pe.from_degrees(face_degree=4, vertex_degree=3)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.vertices, 8)
        self.assertEqual(rec.faces, 6)
        self.assertEqual(rec.edges, 12)

    def test_toroidal(self):
        toroidal = self.pe.enumerate_toroidal()
        cbs = [t["cb"] for t in toroidal]
        cgs = [t["cg"] for t in toroidal]
        # Should include (3,6), (4,4), (6,3)
        self.assertIn(3, cbs)
        self.assertIn(4, cbs)
        self.assertIn(6, cbs)

    def test_toroidal_satisfy_equation(self):
        toroidal = self.pe.enumerate_toroidal()
        for t in toroidal:
            cb, cg = t["cb"], t["cg"]
            self.assertEqual(2 * (cb + cg), cb * cg,
                             f"Toroidal equation fails for ({cb},{cg})")

    def test_check_euler(self):
        e = self.pe.check_euler(4, 4, 6)
        self.assertEqual(e["chi"], 2)
        self.assertTrue(e["spherical"])
        self.assertFalse(e["toroidal"])

    def test_diagonal_count_tetrahedron(self):
        # B=4: 4*3/2 - 6 = 6 - 6 = 0
        self.assertEqual(self.pe.diagonal_count(4, 4, 6), 0)

    def test_diagonal_count_cube(self):
        # B=8: 8*7/2 - 12 = 28 - 12 = 16
        self.assertEqual(self.pe.diagonal_count(8, 6, 12), 16)

    def test_compare_table_str(self):
        table = self.pe.compare_table()
        self.assertIn("Тетраэдр", table)
        self.assertIn("Куб", table)


class TestExDodecahedron(unittest.TestCase):
    def setUp(self):
        self.ed = ExDodecahedron(1.0)

    def test_vertices(self):
        self.assertEqual(self.ed.vertices(), 32)

    def test_faces(self):
        self.assertEqual(self.ed.faces(), 24)

    def test_edges(self):
        self.assertEqual(self.ed.edges(), 54)

    def test_euler(self):
        self.assertEqual(self.ed.euler(), 2)

    def test_volume_unit(self):
        # V = (1/2)(4 + 3φ)
        expected = 0.5 * (4 + 3 * _PHI)
        self.assertAlmostEqual(self.ed.volume(), expected)

    def test_volume_scaled(self):
        ed2 = ExDodecahedron(2.0)
        self.assertAlmostEqual(ed2.volume(), 8.0 * self.ed.volume())

    def test_diagonal_count(self):
        # B=32: 32*31/2 - 54 = 496 - 54 = 442
        self.assertEqual(self.ed.diagonal_count(), 442)

    def test_construction_steps(self):
        steps = self.ed.construction_steps()
        self.assertEqual(len(steps), 5)
        self.assertTrue(any("32" in s for s in steps))

    def test_face_types(self):
        ft = self.ed.face_types()
        total = sum(ft.values())
        self.assertEqual(total, self.ed.faces())

    def test_summary(self):
        s = self.ed.summary()
        self.assertIn("32", s)
        self.assertIn("54", s)
        self.assertIn("χ", s)

    def test_to_obj_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            fname = f.name
        try:
            self.ed.to_obj(fname)
            with open(fname) as fh:
                content = fh.read()
            self.assertIn("ExDodecahedron", content)
        finally:
            os.unlink(fname)


class TestFromDegreesExtra(unittest.TestCase):
    def setUp(self):
        self.pe = PolyhedronEnumerator()

    def test_icosahedron(self):
        """face_degree=3, vertex_degree=5 → Icosahedron: 12V, 20F, 30E."""
        rec = self.pe.from_degrees(face_degree=3, vertex_degree=5)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.vertices, 12)
        self.assertEqual(rec.faces, 20)
        self.assertEqual(rec.edges, 30)

    def test_octahedron(self):
        """face_degree=3, vertex_degree=4 → Octahedron: 6V, 8F, 12E."""
        rec = self.pe.from_degrees(face_degree=3, vertex_degree=4)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.vertices, 6)
        self.assertEqual(rec.faces, 8)
        self.assertEqual(rec.edges, 12)

    def test_from_degrees_unknown_returns_none(self):
        rec = self.pe.from_degrees(face_degree=7, vertex_degree=7)
        self.assertIsNone(rec)


class TestCheckEulerVariants(unittest.TestCase):
    def test_toroidal_torus(self):
        """Тор: B=0, Gamma=0, P=0 → chi=0 (toroidal)."""
        pe = PolyhedronEnumerator()
        e = pe.check_euler(4, 4, 8)  # 4+4-8=0 → torus
        self.assertEqual(e["chi"], 0)
        self.assertTrue(e["toroidal"])
        self.assertFalse(e["spherical"])

    def test_rp2_case(self):
        """RP²: chi=1."""
        pe = PolyhedronEnumerator()
        e = pe.check_euler(4, 4, 7)  # 4+4-7=1 → RP²
        self.assertEqual(e["chi"], 1)
        self.assertTrue(e["rp2"])


if __name__ == "__main__":
    unittest.main()
