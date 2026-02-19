"""hexellipse.py — Скрытые параметры эллипса и линия катастроф.

Источник: PDF «Линия катастроф. Скрытые параметры эллипса» Германа.

Параметры эллипса x²/a² + y²/b² = 1 (a > b > 0):
  c = √(a²-b²)                фокусное расстояние
  e = c/a                     эксцентриситет
  p₁ = b²/a                   фокальный параметр (радиус кривизны в ±a)
  p₂ = a²/b                   радиальный параметр (= большой радиус описан. окружн.)
  p  = ab/c                   осевой параметр (полупараметр фокусной хорды)

Фундаментальное тождество: a·b = c·p  (где p = ab/c)

Линия катастроф — особая параллельная кривая при q = p₁:
  происходит схлопывание (особые точки у кривой)

Радиус описанной окружности задачи вписанных окружностей = p₂ = a²/b.
"""
import math
import sys
import argparse


class EllipseAnalysis:
    """Аналитическое исследование эллипса x²/a² + y²/b² = 1."""

    def __init__(self, a: float, b: float):
        if a <= 0 or b <= 0:
            raise ValueError("a и b должны быть > 0")
        if a < b:
            a, b = b, a   # a — большая полуось
        self.a = a
        self.b = b
        self._c = math.sqrt(a ** 2 - b ** 2) if a > b else 0.0

    # ── основные параметры ─────────────────────────────────────────────────────

    def c(self) -> float:
        """Фокусное расстояние c = √(a²-b²)."""
        return self._c

    def eccentricity(self) -> float:
        """Эксцентриситет e = c/a."""
        if self.a == 0:
            return 0.0
        return self._c / self.a

    def focal_parameter(self) -> float:
        """Фокальный параметр p₁ = b²/a (радиус кривизны в вершинах ±a)."""
        return self.b ** 2 / self.a

    def radial_parameter(self) -> float:
        """Радиальный параметр p₂ = a²/b."""
        return self.a ** 2 / self.b

    def axial_parameter(self) -> float:
        """Осевой параметр p = ab/c (полупараметр)."""
        if self._c == 0:
            return float("inf")
        return self.a * self.b / self._c

    def verify_identity(self, tol: float = 1e-9) -> dict:
        """Проверить тождество a·b = c·p."""
        p = self.axial_parameter()
        lhs = self.a * self.b
        rhs = self._c * p if self._c > 0 else float("inf")
        return {
            "a*b": lhs,
            "c*p": rhs,
            "ok": abs(lhs - rhs) < tol if self._c > 0 else (self.a == self.b),
        }

    # ── эквидистанта (параллельная кривая) ───────────────────────────────────

    def equidistant(self, q: float, n_points: int = 200) -> list[tuple[float, float]]:
        """Эквидистанта эллипса на расстоянии q.

        Параметрически: (x₀ + q·nₓ, y₀ + q·nᵧ), где n — внешняя нормаль.
        """
        points = []
        for i in range(n_points):
            t = 2 * math.pi * i / n_points
            x0 = self.a * math.cos(t)
            y0 = self.b * math.sin(t)
            # Нормаль к эллипсу: направление (x/a², y/b²), нормализованная
            nx = x0 / self.a ** 2
            ny = y0 / self.b ** 2
            norm = math.sqrt(nx ** 2 + ny ** 2)
            if norm < 1e-14:
                continue
            nx /= norm
            ny /= norm
            points.append((x0 + q * nx, y0 + q * ny))
        return points

    # ── линия катастроф ───────────────────────────────────────────────────────

    def catastrophe_curve(self, n_points: int = 200) -> list[tuple[float, float]]:
        """Линия катастроф: эквидистанта при q = p₁ = b²/a."""
        return self.equidistant(self.focal_parameter(), n_points)

    def at_catastrophe(self) -> dict:
        """Параметры в точке катастрофы (q = p₁)."""
        p1 = self.focal_parameter()
        # Особые точки при cos(t) = ±a·p₁/c² = ±b²/c²·1
        # При a == b (окружность): нет особых точек
        if self._c < 1e-14:
            return {"q": p1, "type": "circle_no_catastrophe"}
        cos_t = -(self.b ** 2 / (self.a * self._c ** 2)) * self._c ** 2 / self.a
        # Упрощённо: особые точки при cos(t) = -(b/a)²  (вогнутые точки)
        cos_t_val = -(self.b / self.a) ** 2
        cos_t_val = max(-1.0, min(1.0, cos_t_val))
        t_special = [math.acos(cos_t_val), -math.acos(cos_t_val)]
        inflection_pts = [(self.a * math.cos(t), self.b * math.sin(t))
                          for t in t_special]
        return {
            "q": p1,
            "type": "cusp",
            "t_special": t_special,
            "inflection_points": inflection_pts,
        }

    # ── вписанные окружности ──────────────────────────────────────────────────

    def inscribed_circle_radius(self) -> float:
        """Радиус описанной окружности в задаче вписанных окружностей = p₂ = a²/b."""
        return self.radial_parameter()

    def inscribed_system_solution(self) -> dict:
        """Решение системы уравнений для вписанных окружностей."""
        p2 = self.radial_parameter()
        p1 = self.focal_parameter()
        return {
            "R_outer": p2,   # внешний радиус = a²/b
            "r_inner": p1,   # внутренний радиус = b²/a
            "R/r": p2 / p1,  # отношение = (a/b)²
            "product": p1 * p2,   # p₁·p₂ = ab
        }

    # ── конструкции ───────────────────────────────────────────────────────────

    def focal_parameter_construction(self) -> str:
        """Инструкция геометрического построения p₁."""
        return (f"Построение p₁ = b²/a = {self.focal_parameter():.4f}:\n"
                f"  1. Построить полуось b = {self.b:.4f}\n"
                f"  2. Построить b² = {self.b**2:.4f} (средняя пропорциональная)\n"
                f"  3. Разделить на a = {self.a:.4f} → p₁ = {self.focal_parameter():.4f}")

    def radial_parameter_construction(self) -> str:
        """Инструкция геометрического построения p₂."""
        return (f"Построение p₂ = a²/b = {self.radial_parameter():.4f}:\n"
                f"  1. Построить полуось a = {self.a:.4f}\n"
                f"  2. Построить a² = {self.a**2:.4f}\n"
                f"  3. Разделить на b = {self.b:.4f} → p₂ = {self.radial_parameter():.4f}")

    # ── визуализация ─────────────────────────────────────────────────────────

    def plot_ascii(self, width: int = 60) -> str:
        """ASCII-рисунок эллипса с эквидистантами."""
        height = max(12, width // 3)
        rows = [[" "] * width for _ in range(height)]

        scale_x = (width - 4) / 2 / (self.a + self.focal_parameter() + 0.5)
        scale_y = (height - 2) / 2 / (self.b + self.focal_parameter() + 0.5)
        cx = width // 2
        cy = height // 2

        def plot_curve(pts, char):
            for x, y in pts:
                col = int(cx + x * scale_x)
                row = int(cy - y * scale_y)
                if 0 <= row < height and 0 <= col < width:
                    rows[row][col] = char

        # Эллипс
        plot_curve([(self.a * math.cos(t), self.b * math.sin(t))
                    for t in [2 * math.pi * i / 80 for i in range(81)]], "*")
        # Линия катастроф
        plot_curve(self.catastrophe_curve(100), ".")
        # Фокусы
        for sign in [1, -1]:
            c = int(cx + sign * self._c * scale_x)
            if 0 <= c < width:
                rows[cy][c] = "F"

        lines = [f"Эллипс a={self.a}, b={self.b}  (* = эллипс, . = линия катастроф, F = фокус)"]
        for r in rows:
            lines.append("│" + "".join(r))
        lines.append("└" + "─" * width)
        return "\n".join(lines)

    def summary(self) -> str:
        """Полная сводка по параметрам эллипса."""
        ident = self.verify_identity()
        lines = [
            f"Эллипс: a={self.a}, b={self.b}",
            f"  c  = {self.c():.6f}  (фокусное расстояние)",
            f"  e  = {self.eccentricity():.6f}  (эксцентриситет)",
            f"  p₁ = {self.focal_parameter():.6f}  = b²/a",
            f"  p₂ = {self.radial_parameter():.6f}  = a²/b",
            f"  p  = {self.axial_parameter():.6f}  = ab/c",
            f"  a·b = c·p: {ident['a*b']:.6f} = {ident['c*p']:.6f}  ✓={ident['ok']}",
            f"  p₁·p₂ = {self.focal_parameter() * self.radial_parameter():.6f}  (≈ a·b = {self.a*self.b:.6f})",
        ]
        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexellipse — скрытые параметры эллипса (теория Германа)")
    parser.add_argument("--ellipse", nargs=2, type=float, metavar=("A", "B"),
                        help="Задать эллипс (большая ось, малая ось)")
    parser.add_argument("--parameters", action="store_true",
                        help="Показать все параметры")
    parser.add_argument("--equidistant", action="store_true",
                        help="Показать параметры эквидистанты")
    parser.add_argument("--q", type=float, default=1.0,
                        help="Расстояние для эквидистанты")
    parser.add_argument("--catastrophe", action="store_true",
                        help="Линия катастроф (при q=p₁)")
    parser.add_argument("--inscribed-circle", action="store_true",
                        help="Задача вписанных окружностей")
    parser.add_argument("--plot", action="store_true",
                        help="ASCII-рисунок")
    parser.add_argument("--width", type=int, default=60)
    args = parser.parse_args()

    a, b = (args.ellipse if args.ellipse else (5.0, 3.0))
    ea = EllipseAnalysis(a, b)

    if args.parameters:
        print(ea.summary())

    if args.equidistant:
        pts = ea.equidistant(args.q, n_points=20)
        print(f"Эквидистанта при q={args.q} (первые 5 точек):")
        for x, y in pts[:5]:
            print(f"  ({x:.4f}, {y:.4f})")

    if args.catastrophe:
        info = ea.at_catastrophe()
        print(f"Линия катастроф при q = p₁ = {info['q']:.4f}:")
        print(f"  Тип: {info.get('type', '?')}")
        if "inflection_points" in info:
            for pt in info["inflection_points"]:
                print(f"  Особая точка: ({pt[0]:.4f}, {pt[1]:.4f})")

    if args.inscribed_circle:
        sol = ea.inscribed_system_solution()
        print(f"Вписанные окружности:")
        print(f"  R = p₂ = {sol['R_outer']:.6f}")
        print(f"  r = p₁ = {sol['r_inner']:.6f}")
        print(f"  R/r = {sol['R/r']:.6f} = (a/b)²")

    if args.plot:
        print(ea.plot_ascii(args.width))

    if len(sys.argv) == 1 or (not any([args.parameters, args.equidistant,
                                        args.catastrophe, args.inscribed_circle,
                                        args.plot])):
        print(ea.summary())


if __name__ == "__main__":
    _main()
