"""Tests for hexmobius — Möbius surfaces (Herman)."""
import sys
import os
import math
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projects.hexmobius.hexmobius import MobiusSurface, SurfaceTopology

_TAU = 2 * math.pi


class TestMobiusSurfaceConstruction(unittest.TestCase):
    def test_default_creation(self):
        mb = MobiusSurface()
        self.assertEqual(mb.R, 3.0)
        self.assertEqual(mb.width, 1.0)
        self.assertEqual(mb.twists, 1)

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            MobiusSurface(R=0.0)
        with self.assertRaises(ValueError):
            MobiusSurface(R=-1.0)

    def test_invalid_width(self):
        with self.assertRaises(ValueError):
            MobiusSurface(width=0.0)
        with self.assertRaises(ValueError):
            MobiusSurface(width=-0.5)

    def test_invalid_twists(self):
        with self.assertRaises(ValueError):
            MobiusSurface(twists=-1)

    def test_repr(self):
        mb = MobiusSurface(R=2.0, width=0.5, twists=1)
        s = repr(mb)
        self.assertIn("MobiusSurface", s)
        self.assertIn("twists=1", s)


class TestParametrization(unittest.TestCase):
    def test_point_at_v0_lies_on_axis_circle(self):
        # At v=0, point lies on circle of radius R
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        for u in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            x, y, z = mb.point(u, 0.0)
            r = math.sqrt(x ** 2 + y ** 2)
            self.assertAlmostEqual(r, mb.R, places=5)

    def test_point_at_v0_z_is_zero_for_odd_twist(self):
        # At v=0, z=0 since z = v·sin(...)
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        for u in [0.1, 0.5, 1.0, 2.0]:
            x, y, z = mb.point(u, 0.0)
            self.assertAlmostEqual(z, 0.0, places=10)

    def test_cylinder_z_always_zero(self):
        # N=0 (cylinder): z = v·sin(0) = 0
        mb = MobiusSurface(R=3.0, width=1.0, twists=0)
        for u in [0.0, 1.0, 2.0, 3.0]:
            for v in [-0.5, 0.0, 0.5]:
                x, y, z = mb.point(u, v)
                self.assertAlmostEqual(z, 0.0, places=10)

    def test_cylinder_r_constant(self):
        # N=0: x² + y² = (R + v·cos(0))² = (R+v)² — independent of u
        mb = MobiusSurface(R=3.0, width=1.0, twists=0)
        v = 0.5
        expected_r = mb.R + v
        for u in [0.1, 1.0, 2.5]:
            x, y, z = mb.point(u, v)
            r = math.sqrt(x ** 2 + y ** 2)
            self.assertAlmostEqual(r, expected_r, places=5)

    def test_points_grid_shape(self):
        mb = MobiusSurface()
        grid = mb.points(u_steps=10, v_steps=5)
        self.assertEqual(len(grid), 11)
        self.assertEqual(len(grid[0]), 6)

    def test_normal_is_unit(self):
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        for u in [0.5, 1.5, 3.0]:
            for v in [-0.5, 0.0, 0.5]:
                n = mb.normal(u, v)
                length = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
                self.assertAlmostEqual(length, 1.0, places=3)

    def test_normal_changes_with_position(self):
        # Normal should vary across the surface (not constant)
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        n1 = mb.normal(0.0, 0.5)
        n2 = mb.normal(math.pi, 0.5)
        # They should differ
        diff = sum((n1[i] - n2[i]) ** 2 for i in range(3))
        self.assertGreater(diff, 1e-6)


class TestTopologicalInvariants(unittest.TestCase):
    def test_euler_characteristic_all_zero(self):
        for N in range(4):
            mb = MobiusSurface(twists=N)
            self.assertEqual(mb.euler_characteristic(), 0)

    def test_cylinder_orientable(self):
        mb = MobiusSurface(twists=0)
        self.assertTrue(mb.is_orientable())

    def test_mobius_not_orientable(self):
        mb = MobiusSurface(twists=1)
        self.assertFalse(mb.is_orientable())

    def test_klein_not_orientable(self):
        mb = MobiusSurface(twists=2)
        self.assertFalse(mb.is_orientable())

    def test_cylinder_two_boundaries(self):
        mb = MobiusSurface(twists=0)
        self.assertEqual(mb.num_boundary_components(), 2)

    def test_mobius_one_boundary(self):
        mb = MobiusSurface(twists=1)
        self.assertEqual(mb.num_boundary_components(), 1)

    def test_klein_zero_boundaries(self):
        mb = MobiusSurface(twists=2)
        self.assertEqual(mb.num_boundary_components(), 0)

    def test_odd_twists_one_boundary(self):
        for N in [1, 3, 5]:
            mb = MobiusSurface(twists=N)
            self.assertEqual(mb.num_boundary_components(), 1,
                             f"N={N} should have 1 boundary")

    def test_even_twists_zero_boundaries(self):
        for N in [2, 4, 6]:
            mb = MobiusSurface(twists=N)
            self.assertEqual(mb.num_boundary_components(), 0,
                             f"N={N} should have 0 boundaries")

    def test_writhing_number(self):
        for N in range(5):
            mb = MobiusSurface(twists=N)
            self.assertAlmostEqual(mb.writhing_number(), N / 2.0)

    def test_surface_class_cylinder(self):
        mb = MobiusSurface(twists=0)
        self.assertEqual(mb.surface_class(), "Cylinder")

    def test_surface_class_mobius(self):
        mb = MobiusSurface(twists=1)
        self.assertEqual(mb.surface_class(), "Möbius band")

    def test_surface_class_klein(self):
        mb = MobiusSurface(twists=2)
        self.assertEqual(mb.surface_class(), "Klein bottle")

    def test_surface_class_generalized(self):
        mb = MobiusSurface(twists=3)
        cls = mb.surface_class()
        self.assertIn("Generalized", cls)
        self.assertIn("3", cls)


class TestGeometricProperties(unittest.TestCase):
    def test_axial_length(self):
        mb = MobiusSurface(R=3.0)
        self.assertAlmostEqual(mb.axial_length(), 2 * math.pi * 3.0)

    def test_axial_length_scales_with_R(self):
        mb1 = MobiusSurface(R=1.0)
        mb2 = MobiusSurface(R=5.0)
        self.assertAlmostEqual(mb2.axial_length() / mb1.axial_length(), 5.0)

    def test_surface_area_positive(self):
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        area = mb.surface_area(u_steps=20, v_steps=5)
        self.assertGreater(area, 0.0)

    def test_cylinder_area_approx(self):
        # Cylinder area ≈ 2 * (2πR) * width = 4πRw (inner+outer)
        # But our parametrization gives one sheet, area ≈ 2πR * 2w = 4πRw
        R, w = 5.0, 0.5
        mb = MobiusSurface(R=R, width=w, twists=0)
        area = mb.surface_area(u_steps=50, v_steps=10)
        expected = _TAU * R * 2 * w
        # Allow 5% error from numerical integration
        self.assertAlmostEqual(area, expected, delta=expected * 0.05)

    def test_summary_contains_invariants(self):
        mb = MobiusSurface(R=2.0, width=0.5, twists=1)
        s = mb.summary()
        self.assertIn("χ", s)
        self.assertIn("0", s)
        self.assertIn("Möbius", s)


class TestAsciiProjection(unittest.TestCase):
    def test_ascii_returns_string(self):
        mb = MobiusSurface()
        s = mb.ascii_project(width=40)
        self.assertIsInstance(s, str)

    def test_ascii_contains_dots(self):
        mb = MobiusSurface()
        s = mb.ascii_project(width=40)
        self.assertIn("•", s)

    def test_ascii_views(self):
        mb = MobiusSurface()
        for view in ["xy", "xz", "yz"]:
            s = mb.ascii_project(width=40, view=view)
            self.assertIn(view.upper(), s)

    def test_summary_string(self):
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        s = mb.summary()
        self.assertIn("Möbius", s)
        self.assertIn("нет", s)    # not orientable


class TestExport(unittest.TestCase):
    def test_to_obj_creates_file(self):
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            fname = f.name
        try:
            mb.to_obj(fname, u_steps=10, v_steps=5)
            with open(fname) as f:
                content = f.read()
            self.assertIn("v ", content)
            self.assertIn("f ", content)
        finally:
            os.unlink(fname)

    def test_to_stl_creates_file(self):
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            fname = f.name
        try:
            mb.to_stl(fname, u_steps=10, v_steps=5)
            with open(fname) as f:
                content = f.read()
            self.assertIn("solid", content)
            self.assertIn("facet normal", content)
            self.assertIn("vertex", content)
        finally:
            os.unlink(fname)

    def test_obj_vertex_count(self):
        mb = MobiusSurface(R=3.0, width=1.0, twists=1)
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            fname = f.name
        try:
            u_steps, v_steps = 6, 4
            mb.to_obj(fname, u_steps=u_steps, v_steps=v_steps)
            with open(fname) as f:
                lines = f.readlines()
            vertex_lines = [l for l in lines if l.startswith("v ")]
            # (u_steps+1) * (v_steps+1) vertices
            expected = (u_steps + 1) * (v_steps + 1)
            self.assertEqual(len(vertex_lines), expected)
        finally:
            os.unlink(fname)


class TestSurfaceTopology(unittest.TestCase):
    def test_compare_returns_string(self):
        surfaces = [MobiusSurface(twists=N) for N in range(3)]
        topo = SurfaceTopology()
        s = topo.compare(surfaces)
        self.assertIsInstance(s, str)
        self.assertIn("Cylinder", s)
        self.assertIn("Möbius", s)

    def test_standard_classification(self):
        s = SurfaceTopology.standard_classification()
        self.assertIn("Cylinder", s)
        self.assertIn("Möbius", s)
        self.assertIn("Klein", s)

    def test_topological_invariants_dict(self):
        mb = MobiusSurface(twists=1)
        inv = SurfaceTopology.topological_invariants(mb)
        self.assertEqual(inv["twists"], 1)
        self.assertEqual(inv["chi"], 0)
        self.assertFalse(inv["orientable"])
        self.assertEqual(inv["boundaries"], 1)
        self.assertAlmostEqual(inv["writhe"], 0.5)

    def test_invariants_cylinder(self):
        mb = MobiusSurface(twists=0)
        inv = SurfaceTopology.topological_invariants(mb)
        self.assertTrue(inv["orientable"])
        self.assertEqual(inv["boundaries"], 2)
        self.assertAlmostEqual(inv["writhe"], 0.0)

    def test_invariants_klein(self):
        mb = MobiusSurface(twists=2)
        inv = SurfaceTopology.topological_invariants(mb)
        self.assertFalse(inv["orientable"])
        self.assertEqual(inv["boundaries"], 0)
        self.assertAlmostEqual(inv["writhe"], 1.0)


if __name__ == "__main__":
    unittest.main()
