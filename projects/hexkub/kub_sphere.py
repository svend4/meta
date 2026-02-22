"""
kub_sphere.py — Кубатура шара / Шаратура куба

Геометрические отношения между кубом и его тремя сферами:
  1. Вписанная сфера   (касается центров 6 граней)
  2. Средняя сфера     (проходит через 12 середин рёбер)
  3. Описанная сфера   (проходит через 8 вершин)

«Кубатура шара»: какому кубу эквивалентен данный шар (по объёму)?
«Шаратура куба»: какому шару эквивалентен данный куб (по объёму)?

Связь с Q6:
  Три сферы куба ↔ МВС/СВС/БВС Крюкова ↔ шары Хэмминга в Q6
  Вписанная → r=1, средняя → r=2, описанная → r=3 (в дискретном смысле)
"""

import math
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, ball, sphere


# ---------------------------------------------------------------------------
# CubeSphereGeometry — куб и его три сферы
# ---------------------------------------------------------------------------

class CubeSphereGeometry:
    """Геометрические отношения куба и вписанных/описанных сфер."""

    def __init__(self, a: float = 1.0):
        """a — длина ребра куба."""
        self.a = a

    # --- три сферы ----------------------------------------------------------

    def inscribed(self) -> dict:
        """Вписанная сфера (касается центров 6 граней), r = a/2."""
        r = self.a / 2
        return {
            "name": "вписанная",
            "r": r,
            "V_sphere": self._V_sphere(r),
            "V_cube": self._V_cube(),
            "ratio_cube_over_sphere": self._V_cube() / self._V_sphere(r),
            "contact": "6 центров граней",
        }

    def midedge(self) -> dict:
        """Средняя сфера (проходит через середины 12 рёбер), ρ = a√2/2."""
        rho = self.a * math.sqrt(2) / 2
        return {
            "name": "средняя (рёбра)",
            "r": rho,
            "V_sphere": self._V_sphere(rho),
            "V_cube": self._V_cube(),
            "ratio_cube_over_sphere": self._V_cube() / self._V_sphere(rho),
            "contact": "12 середин рёбер",
        }

    def circumscribed(self) -> dict:
        """Описанная сфера (проходит через 8 вершин), R = a√3/2."""
        R = self.a * math.sqrt(3) / 2
        return {
            "name": "описанная",
            "r": R,
            "V_sphere": self._V_sphere(R),
            "V_cube": self._V_cube(),
            "ratio_cube_over_sphere": self._V_cube() / self._V_sphere(R),
            "contact": "8 вершин",
        }

    def all_three(self) -> list:
        return [self.inscribed(), self.midedge(), self.circumscribed()]

    def radius_ratios(self) -> dict:
        """Отношения радиусов трёх сфер."""
        r = self.inscribed()["r"]
        rho = self.midedge()["r"]
        R = self.circumscribed()["r"]
        return {
            "r":   r,
            "rho": rho,
            "R":   R,
            "rho_over_r": rho / r,        # = √2
            "R_over_r":   R / r,          # = √3
            "R_over_rho": R / rho,        # = √(3/2)
            "r_rho_R": f"1 : √2 : √3 = 1 : {math.sqrt(2):.4f} : {math.sqrt(3):.4f}",
        }

    # --- кубатура и шаратура -----------------------------------------------

    def cubature(self, R: float = None) -> dict:
        """
        Кубатура шара: куб с тем же объёмом, что шар радиуса R.
        «Шар объёмом V → куб со стороной a = (4πR³/3)^(1/3)»
        """
        if R is None:
            R = self.circumscribed()["r"]
        V_sphere = self._V_sphere(R)
        a_equiv = V_sphere ** (1/3)
        return {
            "sphere_radius": R,
            "V_sphere": V_sphere,
            "cube_side_equiv": a_equiv,
            "ratio_a_over_R": a_equiv / R,
            "formula": "a = (4π/3)^(1/3) · R ≈ 1.612 · R",
        }

    def spherature(self, a: float = None) -> dict:
        """
        Шаратура куба: шар с тем же объёмом, что куб со стороной a.
        «Куб объёмом V → шар радиуса R = (3a³/4π)^(1/3)»
        """
        if a is None:
            a = self.a
        V_cube = a ** 3
        R_equiv = (3 * V_cube / (4 * math.pi)) ** (1/3)
        return {
            "cube_side": a,
            "V_cube": V_cube,
            "sphere_radius_equiv": R_equiv,
            "ratio_R_over_a": R_equiv / a,
            "formula": "R = (3/4π)^(1/3) · a ≈ 0.620 · a",
        }

    # --- квадратура круга (2D аналог) --------------------------------------

    def squaring_of_circle(self, r2d: float = 1.0) -> dict:
        """
        2D: «Квадратура круга» — квадрат той же площади, что круг.
        Площадь круга = πr², сторона квадрата = r√π.
        """
        S_circle = math.pi * r2d ** 2
        a_sq = math.sqrt(S_circle)
        return {
            "circle_radius": r2d,
            "S_circle": S_circle,
            "square_side": a_sq,
            "ratio_a_over_r": a_sq / r2d,     # = √π ≈ 1.772
            "formula": "a = √π · r ≈ 1.772 · r",
        }

    # --- вспомогательные ----------------------------------------------------

    def _V_sphere(self, r: float) -> float:
        return (4/3) * math.pi * r**3

    def _V_cube(self) -> float:
        return self.a ** 3

    # --- ASCII --------------------------------------------------------------

    def ascii_three_spheres(self) -> str:
        """ASCII-схема куба с тремя сферами."""
        r_d = self.inscribed()
        m_d = self.midedge()
        c_d = self.circumscribed()
        ratios = self.radius_ratios()

        lines = [
            f"Куб (ребро a={self.a}) и его три сферы:",
            "",
            "     ░░░░░░░░░░░░  ← описанная сфера R=a√3/2",
            "    ░ ┌────────┐ ░",
            "   ░ /|  ·····  |░",
            "  ░ / | ·     · |░  ← средняя сфера ρ=a√2/2",
            " ░ *──┼──*   * ·|░",
            " ░ |  * ─── ─ * |░",
            " ░ | /   ·   / |░   ← вписанная сфера r=a/2",
            "  ░ *────────* ░",
            "   ░░░░░░░░░░░░",
            "",
            f"  Вписанная (6 граней):  r  = a/2      = {r_d['r']:.6f}",
            f"  Средняя   (12 рёбер):  ρ  = a√2/2    = {m_d['r']:.6f}",
            f"  Описанная (8 вершин):  R  = a√3/2    = {c_d['r']:.6f}",
            "",
            f"  Отношения: r : ρ : R = {ratios['r_rho_R']}",
            "",
            f"  V_куб/V_вписанной   = {r_d['ratio_cube_over_sphere']:.4f}  (= 6/π)",
            f"  V_куб/V_средней     = {m_d['ratio_cube_over_sphere']:.4f}  (= 3/π√2)",
            f"  V_куб/V_описанной   = {c_d['ratio_cube_over_sphere']:.4f}  (= 2/π√3)",
            "",
            f"  Кубатура шара описанного:  куб со стороной {self.cubature()['cube_side_equiv']:.4f}",
            f"  Шаратура куба:             шар радиусом    {self.spherature()['sphere_radius_equiv']:.4f}",
        ]
        return "\n".join(lines)

    def ascii_q6_spheres(self) -> str:
        """Соответствие трёх сфер куба и сфер Хэмминга в Q6."""
        lines = [
            "Три сферы куба ↔ шары Хэмминга в Q6 ↔ сферы Крюкова:",
            "",
            "  Куб (геометрия)    Q6 (дискретная)      Крюков (бой)",
            "  ─────────────────  ───────────────────   ────────────",
            f"  Вписанная r=a/2   ball(h,1) = 7 вершин  МВС (контакт)",
            f"  Средняя  ρ=a√2/2  ball(h,2) = 22 вершины СВС (ближний)",
            f"  Описанная R=a√3/2 ball(h,3) = 42 вершины БВС (дальний)",
            "",
            "  Отношения радиусов:",
            "    Геометрия: r : ρ : R = 1 : √2 : √3 = 1 : 1.414 : 1.732",
            "    Q6:        1 : 2  : 3 (шаги Хэмминга)",
            "",
            "  Объёмы (число вершин Q6):",
        ]
        for r in range(1, 7):
            b = ball(0, r)
            s = sphere(0, r)
            lines.append(
                f"    ball(h,{r}): {len(b):2d} вершин  "
                f"(+{len(s):2d} на оболочке r={r})"
            )
        lines.extend([
            "",
            "  Полный Q6 = ball(h,6) = 64 вершины  (= V_куб 4×4×4)",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="kub_sphere — Кубатура шара / Шаратура куба",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python kub_sphere.py                 Три сферы единичного куба
  python kub_sphere.py --edge 2.5      Куб со стороной 2.5
  python kub_sphere.py --cubature 1.0  Кубатура шара R=1
  python kub_sphere.py --spherature 1  Шаратура куба a=1
  python kub_sphere.py --q6            Соответствие с Q6 и Крюковым
  python kub_sphere.py --circle 1.0    Квадратура круга r=1
        """,
    )
    parser.add_argument("--edge", type=float, default=1.0, metavar="A",
                        help="Длина ребра куба (по умолчанию 1.0)")
    parser.add_argument("--cubature", type=float, metavar="R",
                        help="Кубатура шара радиуса R")
    parser.add_argument("--spherature", type=float, metavar="A",
                        help="Шаратура куба со стороной A")
    parser.add_argument("--q6", action="store_true",
                        help="Соответствие трёх сфер и Q6")
    parser.add_argument("--circle", type=float, metavar="R",
                        help="Квадратура круга радиуса R (2D аналог)")
    args = parser.parse_args()

    csg = CubeSphereGeometry(args.edge)

    if args.cubature is not None:
        res = csg.cubature(args.cubature)
        print(f"\nКубатура шара R={args.cubature}:")
        print(f"  V_шара = {res['V_sphere']:.6f}")
        print(f"  Куб с тем же объёмом: a = {res['cube_side_equiv']:.6f}")
        print(f"  a/R = {res['ratio_a_over_R']:.6f}  (= ∛(4π/3) ≈ 1.6120)")
        print(f"  {res['formula']}")
    elif args.spherature is not None:
        res = csg.spherature(args.spherature)
        print(f"\nШаратура куба a={args.spherature}:")
        print(f"  V_куба = {res['V_cube']:.6f}")
        print(f"  Шар с тем же объёмом: R = {res['sphere_radius_equiv']:.6f}")
        print(f"  R/a = {res['ratio_R_over_a']:.6f}  (= ∛(3/4π) ≈ 0.6204)")
        print(f"  {res['formula']}")
    elif args.q6:
        print("\n" + csg.ascii_q6_spheres())
    elif args.circle is not None:
        res = csg.squaring_of_circle(args.circle)
        print(f"\nКвадратура круга r={args.circle}:")
        print(f"  S_круга  = πr² = {res['S_circle']:.6f}")
        print(f"  Квадрат той же площади: a = {res['square_side']:.6f}")
        print(f"  a/r = √π = {res['ratio_a_over_r']:.6f}")
        print(f"  {res['formula']}")
        print()
        print(f"  3D аналог — шаратура/кубатура:")
        sp = csg.spherature(args.circle)
        cu = csg.cubature(args.circle)
        print(f"  Шар R={args.circle} → куб a={cu['cube_side_equiv']:.6f}  (a/R=∛(4π/3)≈1.612)")
        print(f"  Куб a={args.circle} → шар R={sp['sphere_radius_equiv']:.6f}  (R/a=∛(3/4π)≈0.620)")
    else:
        print("\n" + csg.ascii_three_spheres())


if __name__ == "__main__":
    main()
