"""hexmobius.py — Аналитическая теория поверхностей Мёбиуса (Герман).

Источник: PDF «Аналитическая теория мёбиусовых поверхностей» (65b)
          Франца Германа.

Классификация поверхностей по числу скручиваний N:
  N=0  → Цилиндр (ориентируемый, 2 компоненты края, χ=0)
  N=1  → Лента Мёбиуса (неориентируемая, 1 компонента края, χ=0)
  N=2  → Бутылка Клейна (неориентируемая, 0 компонент края, χ=0)
  N≥3  → Обобщённая поверхность Мёбиуса

Параметризация:
  x(u,v) = (R + v·cos(N·u/2))·cos(u)
  y(u,v) = (R + v·cos(N·u/2))·sin(u)
  z(u,v) = v·sin(N·u/2)
  u ∈ [0, 2π],  v ∈ [-w, w]
"""

import math
import sys
import argparse

_TAU = 2 * math.pi


class MobiusSurface:
    """
    Обобщённая поверхность Мёбиуса с N скручиваниями.

    N=0 → Цилиндр (ориентируемый, 2 края)
    N=1 → Стандартная лента Мёбиуса (1 край)
    N=2 → Бутылка Клейна (0 краёв)
    N≥3 → Обобщённая поверхность
    """

    def __init__(self, R: float = 3.0, width: float = 1.0, twists: int = 1):
        """
        Args:
            R:      радиус центральной окружности (> 0)
            width:  полуширина ленты (v ∈ [-width, width])
            twists: число скручиваний N (≥ 0)
        """
        if R <= 0:
            raise ValueError(f"R должен быть > 0, получено {R}")
        if width <= 0:
            raise ValueError(f"width должна быть > 0, получено {width}")
        if twists < 0:
            raise ValueError(f"twists должно быть ≥ 0, получено {twists}")
        self.R = R
        self.width = width
        self.twists = twists

    # ── параметризация ────────────────────────────────────────────────────────

    def point(self, u: float, v: float) -> tuple:
        """
        Точка поверхности при параметрах (u, v).

        Args:
            u: ∈ [0, 2π]
            v: ∈ [-width, width]

        Returns:
            (x, y, z) — трёхмерная точка
        """
        N = self.twists
        r = self.R + v * math.cos(N * u / 2)
        x = r * math.cos(u)
        y = r * math.sin(u)
        z = v * math.sin(N * u / 2)
        return (x, y, z)

    def points(self, u_steps: int = 100,
               v_steps: int = 20) -> list:
        """
        Сетка точек u_steps × v_steps.

        Returns:
            2D список pts[i][j] = (x, y, z)
            i = 0..u_steps, j = 0..v_steps
        """
        u_vals = [_TAU * i / u_steps for i in range(u_steps + 1)]
        v_vals = [self.width * (2 * j / v_steps - 1)
                  for j in range(v_steps + 1)]
        return [[self.point(u, v) for v in v_vals] for u in u_vals]

    def normal(self, u: float, v: float, eps: float = 1e-5) -> tuple:
        """
        Нормальный вектор в точке (u, v) (метод конечных разностей).

        Returns:
            (nx, ny, nz) — единичный нормальный вектор
        """
        p0 = self.point(u, v)
        pu = self.point(u + eps, v)
        pv = self.point(u, v + eps)

        tu = ((pu[0] - p0[0]) / eps,
              (pu[1] - p0[1]) / eps,
              (pu[2] - p0[2]) / eps)
        tv = ((pv[0] - p0[0]) / eps,
              (pv[1] - p0[1]) / eps,
              (pv[2] - p0[2]) / eps)

        # Нормаль = tu × tv
        nx = tu[1] * tv[2] - tu[2] * tv[1]
        ny = tu[2] * tv[0] - tu[0] * tv[2]
        nz = tu[0] * tv[1] - tu[1] * tv[0]
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length < 1e-12:
            return (0.0, 0.0, 1.0)
        return (nx / length, ny / length, nz / length)

    # ── топологические инварианты ─────────────────────────────────────────────

    def euler_characteristic(self) -> int:
        """
        Характеристика Эйлера.

        Для ленты Мёбиуса и её обобщений: χ = 0.
        (Лента Мёбиуса ≅ замкнутое кольцо с N скручиваниями.)
        """
        return 0

    def is_orientable(self) -> bool:
        """
        Ориентируемость поверхности.

        N=0 (цилиндр): ориентируема.
        N≥1 нечётное (Мёбиус, тройной и т.д.): неориентируема.
        N≥2 чётное (Клейн, ...): неориентируема.
        """
        return self.twists == 0

    def num_boundary_components(self) -> int:
        """
        Число компонент границы (краёв).

        N=0 (цилиндр):        2 края
        N нечётное:            1 край (лента замкнута в одну кривую)
        N чётное (≥2 Клейн):   0 краёв (замкнутая поверхность)
        """
        if self.twists == 0:
            return 2
        elif self.twists % 2 == 1:
            return 1
        else:
            return 0

    def writhing_number(self) -> float:
        """
        Число кручения осевой кривой: writhe = N / 2.
        """
        return self.twists / 2.0

    def surface_class(self) -> str:
        """Название класса поверхности."""
        N = self.twists
        if N == 0:
            return "Cylinder"
        elif N == 1:
            return "Möbius band"
        elif N == 2:
            return "Klein bottle"
        else:
            return f"Generalized Möbius (N={N})"

    # ── геометрические свойства ───────────────────────────────────────────────

    def surface_area(self, u_steps: int = 200, v_steps: int = 40) -> float:
        """
        Численная площадь поверхности методом суммирования |tu × tv|·du·dv.

        Для стандартной ленты Мёбиуса (R≫w): S ≈ 4πR·w.
        """
        total = 0.0
        du = _TAU / u_steps
        dv = 2 * self.width / v_steps
        eps = du * 0.1

        for i in range(u_steps):
            u = _TAU * i / u_steps
            for j in range(v_steps):
                v = self.width * (2 * j / v_steps - 1)
                p0 = self.point(u, v)
                pu = self.point(u + eps, v)
                pv = self.point(u, v + eps)

                tu = ((pu[0]-p0[0])/eps, (pu[1]-p0[1])/eps, (pu[2]-p0[2])/eps)
                tv = ((pv[0]-p0[0])/eps, (pv[1]-p0[1])/eps, (pv[2]-p0[2])/eps)

                cx = tu[1]*tv[2] - tu[2]*tv[1]
                cy = tu[2]*tv[0] - tu[0]*tv[2]
                cz = tu[0]*tv[1] - tu[1]*tv[0]
                total += math.sqrt(cx*cx + cy*cy + cz*cz) * du * dv

        return total

    def axial_length(self) -> float:
        """Длина центральной окружности = 2πR."""
        return _TAU * self.R

    # ── экспорт ───────────────────────────────────────────────────────────────

    def to_obj(self, filename: str, u_steps: int = 60,
               v_steps: int = 15) -> None:
        """Экспорт в формат Wavefront OBJ."""
        grid = self.points(u_steps, v_steps)
        with open(filename, "w") as f:
            f.write(f"# {self.surface_class()}\n")
            f.write(f"# R={self.R}, width={self.width}, twists={self.twists}\n")
            f.write("# Generated by hexmobius\n\n")
            f.write(f"o {self.surface_class().replace(' ', '_')}\n\n")

            # Вершины
            index: dict = {}
            idx = 1
            for i, row in enumerate(grid):
                for j, (x, y, z) in enumerate(row):
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                    index[(i, j)] = idx
                    idx += 1
            f.write("\n")

            # Грани (треугольники)
            for i in range(u_steps):
                for j in range(v_steps):
                    i2 = (i + 1) % (u_steps + 1)
                    v00 = index[(i,  j)]
                    v10 = index[(i2, j)]
                    v01 = index[(i,  j + 1)]
                    v11 = index[(i2, j + 1)]
                    f.write(f"f {v00} {v10} {v11}\n")
                    f.write(f"f {v00} {v11} {v01}\n")

    def to_stl(self, filename: str, u_steps: int = 60,
               v_steps: int = 15) -> None:
        """Экспорт в ASCII STL (для 3D-печати)."""
        grid = self.points(u_steps, v_steps)
        with open(filename, "w") as f:
            name = self.surface_class().replace(" ", "_")
            f.write(f"solid {name}\n")
            for i in range(u_steps):
                i2 = (i + 1) % (u_steps + 1)
                for j in range(v_steps):
                    p00 = grid[i][j]
                    p10 = grid[i2][j]
                    p01 = grid[i][j + 1]
                    p11 = grid[i2][j + 1]
                    for tri in [(p00, p10, p11), (p00, p11, p01)]:
                        a, b, c = tri
                        ux = b[0]-a[0]; uy = b[1]-a[1]; uz = b[2]-a[2]
                        vx = c[0]-a[0]; vy = c[1]-a[1]; vz = c[2]-a[2]
                        nx = uy*vz - uz*vy
                        ny = uz*vx - ux*vz
                        nz = ux*vy - uy*vx
                        L = math.sqrt(nx*nx + ny*ny + nz*nz)
                        if L > 1e-12:
                            nx /= L; ny /= L; nz /= L
                        f.write(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}\n")
                        f.write("    outer loop\n")
                        for pt in tri:
                            f.write(f"      vertex "
                                    f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
                        f.write("    endloop\n")
                        f.write("  endfacet\n")
            f.write(f"endsolid {name}\n")

    # ── визуализация ──────────────────────────────────────────────────────────

    def ascii_project(self, width: int = 60, view: str = "xz") -> str:
        """
        ASCII-проекция поверхности.

        Args:
            width: ширина символьного поля
            view:  "xy", "xz" или "yz"
        """
        pts_3d = self.points(80, 16)
        flat = [pt for row in pts_3d for pt in row]

        if view == "xz":
            proj = [(pt[0], pt[2]) for pt in flat]
        elif view == "yz":
            proj = [(pt[1], pt[2]) for pt in flat]
        else:  # xy
            proj = [(pt[0], pt[1]) for pt in flat]

        xs = [p[0] for p in proj]
        ys = [p[1] for p in proj]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx = max(xmax - xmin, 1e-3)
        dy = max(ymax - ymin, 1e-3)
        height = max(12, width // 3)

        grid = [[" "] * width for _ in range(height)]
        for x, y in proj:
            col = int((x - xmin) / dx * (width - 1))
            row = int((1 - (y - ymin) / dy) * (height - 1))
            row = max(0, min(height - 1, row))
            col = max(0, min(width - 1, col))
            grid[row][col] = "•"

        top = "┌" + "─" * width + "┐"
        bot = "└" + "─" * width + "┘"
        lines = [
            f"Проекция {view.upper()}: {self.surface_class()}",
            f"R={self.R}, w={self.width}, N={self.twists}",
            top
        ]
        for row in grid:
            lines.append("│" + "".join(row) + "│")
        lines.append(bot)
        return "\n".join(lines)

    def summary(self) -> str:
        """Текстовое описание топологических инвариантов."""
        lines = [
            f"Поверхность:          {self.surface_class()}",
            f"R={self.R}, ширина={self.width}, скручиваний N={self.twists}",
            f"Характеристика Эйлера χ = {self.euler_characteristic()}",
            f"Ориентируема:         {'да' if self.is_orientable() else 'нет'}",
            f"Компонент края:       {self.num_boundary_components()}",
            f"Число кручения:       {self.writhing_number():.1f}",
            f"Длина оси:            {self.axial_length():.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MobiusSurface(R={self.R}, width={self.width}, "
                f"twists={self.twists})")


# ── сравнение поверхностей ────────────────────────────────────────────────────

class SurfaceTopology:
    """Сравнение топологических инвариантов набора поверхностей."""

    def compare(self, surfaces: list) -> str:
        """
        Возвращает ASCII-таблицу сравнения инвариантов.

        Args:
            surfaces: список MobiusSurface

        Returns:
            Отформатированная текстовая таблица
        """
        header = (f"{'Поверхность':<24} {'N':>3} {'χ':>3} "
                  f"{'Ориент.':>8} {'Края':>5} {'Кручение':>9}")
        sep = "─" * len(header)
        lines = [sep, header, sep]
        for s in surfaces:
            cls = s.surface_class()
            chi = s.euler_characteristic()
            ori = "да" if s.is_orientable() else "нет"
            bnd = str(s.num_boundary_components())
            wr = f"{s.writhing_number():.1f}"
            lines.append(
                f"{cls:<24} {s.twists:>3} {chi:>3} {ori:>8} {bnd:>5} {wr:>9}"
            )
        lines.append(sep)
        return "\n".join(lines)

    @staticmethod
    def standard_classification() -> str:
        """Таблица N=0..3 (цилиндр, лента, Клейн, тройной)."""
        surfaces = [MobiusSurface(R=3.0, width=1.0, twists=N) for N in range(4)]
        return SurfaceTopology().compare(surfaces)

    @staticmethod
    def topological_invariants(s: "MobiusSurface") -> dict:
        """Словарь всех инвариантов для данной поверхности."""
        return {
            "class":       s.surface_class(),
            "twists":      s.twists,
            "chi":         s.euler_characteristic(),
            "orientable":  s.is_orientable(),
            "boundaries":  s.num_boundary_components(),
            "writhe":      s.writhing_number(),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Аналитическая теория поверхностей Мёбиуса (PDF 65b)"
    )
    p.add_argument("--twists", type=int, default=1,
                   help="Скручиваний N (0=цилиндр, 1=Мёбиус, 2=Клейн)")
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--topology", action="store_true",
                   help="Показать топологические инварианты")
    p.add_argument("--compare", type=int, nargs="+", metavar="N",
                   help="Сравнить поверхности с N1 N2 ... скручиваниями")
    p.add_argument("--table", action="store_true",
                   help="Стандартная таблица N=0..3")
    p.add_argument("--export", choices=["obj", "stl"],
                   help="Экспорт в формат")
    p.add_argument("--output", type=str, default=None,
                   help="Имя выходного файла")
    p.add_argument("--ascii", action="store_true",
                   help="ASCII-проекция поверхности")
    p.add_argument("--view", choices=["xy", "xz", "yz"], default="xz",
                   help="Вид проекции")
    args = p.parse_args(argv)

    if args.table:
        print(SurfaceTopology.standard_classification())
        return

    if args.compare:
        surfaces = [MobiusSurface(R=args.radius, width=args.width, twists=N)
                    for N in args.compare]
        print(SurfaceTopology().compare(surfaces))
        return

    mb = MobiusSurface(R=args.radius, width=args.width, twists=args.twists)

    if args.topology:
        print(mb.summary())
    elif args.ascii:
        print(mb.ascii_project(width=60, view=args.view))
    elif args.export == "obj":
        out = args.output or "mobius.obj"
        mb.to_obj(out)
        print(f"Экспортировано в OBJ: {out}")
    elif args.export == "stl":
        out = args.output or "mobius.stl"
        mb.to_stl(out)
        print(f"Экспортировано в STL: {out}")
    else:
        print(mb.summary())


if __name__ == "__main__":
    main()
