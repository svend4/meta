"""Tests for hextile — Aperiodic tilings (Herman)."""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hextile.hextile import (
    Rhombus,
    HermanArrow,
    AperiodicTiling,
    AperiodicTiler,
    QuasicrystalMosaic,
    _ALPHA, _BETA, _GOLD, _DEG,
)


class TestRhombus(unittest.TestCase):
    def test_thin_vertices_count(self):
        r = Rhombus(0, 0, 0, 1.0, _ALPHA)
        self.assertEqual(len(r.vertices()), 4)

    def test_thick_vertices_count(self):
        r = Rhombus(0, 0, 0, 1.0, _BETA)
        self.assertEqual(len(r.vertices()), 4)

    def test_tile_type_thin(self):
        r = Rhombus(0, 0, 0, 1.0, _ALPHA)
        self.assertEqual(r.tile_type(), Rhombus.THIN)

    def test_tile_type_thick(self):
        r = Rhombus(0, 0, 0, 1.0, _BETA)
        self.assertEqual(r.tile_type(), Rhombus.THICK)

    def test_area_thin(self):
        # area = side² · sin(36°)
        r = Rhombus(0, 0, 0, 1.0, _ALPHA)
        expected = math.sin(_ALPHA)
        self.assertAlmostEqual(r.area(), expected)

    def test_area_thick(self):
        # area = side² · sin(72°)
        r = Rhombus(0, 0, 0, 1.0, _BETA)
        expected = math.sin(_BETA)
        self.assertAlmostEqual(r.area(), expected)

    def test_area_scaling(self):
        r1 = Rhombus(0, 0, 0, 1.0, _ALPHA)
        r2 = Rhombus(0, 0, 0, 2.0, _ALPHA)
        self.assertAlmostEqual(r2.area(), 4.0 * r1.area())

    def test_center_is_centroid(self):
        # Center (x, y) should be average of vertices
        r = Rhombus(3.0, 5.0, 0, 1.0, _ALPHA)
        verts = r.vertices()
        cx = sum(v[0] for v in verts) / 4
        cy = sum(v[1] for v in verts) / 4
        self.assertAlmostEqual(cx, r.x, places=5)
        self.assertAlmostEqual(cy, r.y, places=5)

    def test_vertices_form_rhombus_diagonals(self):
        # Opposite vertices should be centrosymmetric
        r = Rhombus(0, 0, 0, 1.0, _ALPHA)
        verts = r.vertices()
        # v[0] + v[2] ≈ (0,0)*2
        self.assertAlmostEqual(verts[0][0] + verts[2][0], 0.0, places=5)
        self.assertAlmostEqual(verts[1][0] + verts[3][0], 0.0, places=5)

    def test_repr(self):
        r = Rhombus(0, 0, 0, 1.0, _ALPHA)
        s = repr(r)
        self.assertIn("thin", s)


class TestHermanArrow(unittest.TestCase):
    def setUp(self):
        self.arrow = HermanArrow(0.0, 0.0, 0.0, 1.0)

    def test_has_thin_rhombus(self):
        self.assertEqual(self.arrow.thin.tile_type(), Rhombus.THIN)

    def test_has_thick_rhombus(self):
        self.assertEqual(self.arrow.thick.tile_type(), Rhombus.THICK)

    def test_vertices_count(self):
        # thin has 4 + thick has 4 = 8 total
        self.assertEqual(len(self.arrow.vertices()), 8)

    def test_bounding_box_finite(self):
        bb = self.arrow.bounding_box()
        self.assertEqual(len(bb), 4)
        xmin, ymin, xmax, ymax = bb
        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)

    def test_center(self):
        cx, cy = self.arrow.center()
        self.assertAlmostEqual(cx, 0.0)
        self.assertAlmostEqual(cy, 0.0)

    def test_area_positive(self):
        self.assertGreater(self.arrow.area(), 0.0)

    def test_area_gt_single_rhombus(self):
        # Arrow area = thin + thick > either alone
        thin_area = self.arrow.thin.area()
        thick_area = self.arrow.thick.area()
        self.assertAlmostEqual(self.arrow.area(), thin_area + thick_area)

    def test_next_positions_count(self):
        positions = self.arrow.next_positions()
        # 5-fold symmetry → 5 positions
        self.assertEqual(len(positions), 5)

    def test_next_positions_structure(self):
        for pos in self.arrow.next_positions():
            self.assertEqual(len(pos), 3)  # (x, y, angle)

    def test_scaling(self):
        a2 = HermanArrow(0.0, 0.0, 0.0, 2.0)
        # Bounding box should scale by 2
        bb1 = self.arrow.bounding_box()
        bb2 = a2.bounding_box()
        self.assertAlmostEqual((bb2[2] - bb2[0]) / (bb1[2] - bb1[0]), 2.0,
                               places=5)

    def test_repr(self):
        s = repr(self.arrow)
        self.assertIn("HermanArrow", s)


class TestAperiodicTiling(unittest.TestCase):
    def setUp(self):
        tiles = [HermanArrow(0.0, 0.0, 0.0, 1.0),
                 HermanArrow(2.0, 0.0, 36 * _DEG, 1.0),
                 HermanArrow(0.0, 2.0, 72 * _DEG, 1.0),
                 HermanArrow(-2.0, 0.0, 108 * _DEG, 1.0),
                 HermanArrow(0.0, -2.0, 144 * _DEG, 1.0)]
        self.tiling = AperiodicTiling(tiles)

    def test_len(self):
        self.assertEqual(len(self.tiling), 5)

    def test_bounding_box_not_degenerate(self):
        xmin, ymin, xmax, ymax = self.tiling.bounding_box()
        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)

    def test_not_translational_symmetric(self):
        # Multiple different angles → not periodic
        self.assertFalse(self.tiling.has_translational_symmetry())

    def test_symmetry_group(self):
        self.assertEqual(self.tiling.symmetry_group(), "5-fold")

    def test_tile_types(self):
        types = self.tiling.tile_types()
        self.assertEqual(types["arrows"], 5)
        self.assertEqual(types["thin"], 5)
        self.assertEqual(types["thick"], 5)

    def test_to_ascii_returns_string(self):
        s = self.tiling.to_ascii(40)
        self.assertIsInstance(s, str)
        self.assertIn("│", s)

    def test_empty_tiling_bbox(self):
        empty = AperiodicTiling([])
        bb = empty.bounding_box()
        self.assertEqual(bb, (0.0, 0.0, 0.0, 0.0))


class TestAperiodicTiler(unittest.TestCase):
    def test_generate_returns_tiling(self):
        tiler = AperiodicTiler()
        tiling = tiler.generate(n_tiles=10, seed=1)
        self.assertIsInstance(tiling, AperiodicTiling)

    def test_generate_n_tiles(self):
        tiler = AperiodicTiler()
        tiling = tiler.generate(n_tiles=20, seed=42)
        self.assertGreater(len(tiling), 0)
        self.assertLessEqual(len(tiling), 20)

    def test_reproducible_with_same_seed(self):
        tiler = AperiodicTiler()
        t1 = tiler.generate(n_tiles=15, seed=7)
        t2 = tiler.generate(n_tiles=15, seed=7)
        self.assertEqual(len(t1), len(t2))

    def test_different_seeds_may_differ(self):
        tiler = AperiodicTiler()
        t1 = tiler.generate(n_tiles=30, seed=1)
        t2 = tiler.generate(n_tiles=30, seed=999)
        # Not guaranteed to differ, but usually will
        # Just check both work
        self.assertGreater(len(t1), 0)
        self.assertGreater(len(t2), 0)

    def test_aperiodic(self):
        tiler = AperiodicTiler()
        tiler.generate(n_tiles=30, seed=42)
        self.assertFalse(tiler.has_translational_symmetry())

    def test_symmetry_group(self):
        tiler = AperiodicTiler()
        tiler.generate(n_tiles=20, seed=42)
        self.assertEqual(tiler.symmetry_group(), "5-fold")

    def test_to_ascii(self):
        tiler = AperiodicTiler()
        tiler.generate(n_tiles=15, seed=42)
        s = tiler.to_ascii(40)
        self.assertIn("│", s)

    def test_invalid_tile_type(self):
        with self.assertRaises(ValueError):
            AperiodicTiler(tile_type="unknown")

    def test_no_generate_raises(self):
        tiler = AperiodicTiler()
        with self.assertRaises(RuntimeError):
            tiler.has_translational_symmetry()

    def test_penrose_type(self):
        tiler = AperiodicTiler(tile_type="penrose")
        tiling = tiler.generate(n_tiles=10, seed=1)
        self.assertGreater(len(tiling), 0)


class TestQuasicrystalMosaic(unittest.TestCase):
    def test_generate_returns_self(self):
        qm = QuasicrystalMosaic(8, 5)
        result = qm.generate()
        self.assertIs(result, qm)

    def test_segment_count_positive(self):
        qm = QuasicrystalMosaic(6, 4)
        qm.generate()
        self.assertGreater(qm.segment_count(), 0)

    def test_vertex_count_equals_n_times_k(self):
        n, k = 6, 4
        qm = QuasicrystalMosaic(n, k)
        qm.generate()
        # n sides × k points per side = n*k points
        self.assertEqual(qm.vertex_count(), n * k)

    def test_symmetry_order_even_k(self):
        # k=4 (even) → 2n
        qm = QuasicrystalMosaic(8, 4)
        self.assertEqual(qm.symmetry_order(), 16)

    def test_symmetry_order_odd_k(self):
        # k=5 (odd) → n
        qm = QuasicrystalMosaic(8, 5)
        self.assertEqual(qm.symmetry_order(), 8)

    def test_to_ascii_returns_string(self):
        qm = QuasicrystalMosaic(6, 3)
        qm.generate()
        s = qm.to_ascii(40)
        self.assertIsInstance(s, str)
        self.assertIn("+", s)

    def test_auto_generate_on_vertex_count(self):
        qm = QuasicrystalMosaic(5, 3)
        # Should auto-generate when vertex_count is called
        n = qm.vertex_count()
        self.assertEqual(n, 5 * 3)

    def test_invalid_polygon(self):
        with self.assertRaises(ValueError):
            QuasicrystalMosaic(2, 5)
        with self.assertRaises(ValueError):
            QuasicrystalMosaic(13, 5)

    def test_invalid_divisions(self):
        with self.assertRaises(ValueError):
            QuasicrystalMosaic(6, 1)

    def test_different_polygons(self):
        for n in [3, 5, 6, 8, 10, 12]:
            qm = QuasicrystalMosaic(n, 3)
            qm.generate()
            self.assertGreater(qm.segment_count(), 0)


if __name__ == "__main__":
    unittest.main()
