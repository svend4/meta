"""hexbuffon.py — Обобщённая формула Бюффона для произвольного паркета.

Источник: PDF «Формула паркета» Франца Германа.

Классическая задача Бюффона: бросить иглу длиной L на паркет.
W = ожидаемое число пересечений с линиями паркета.

Обобщённая формула (U = периметр плитки, F = площадь):
    W = L * U / (π * F)

Частные случаи:
    прямоугольник a×b:   W = (L/π) * (1/a + 1/b)
    квадрат a×a:         W = 4L / (πa)
    правильный шестиугольник (сторона r): W = 2L√3 / (πr)

Золотая формула (прямоугольник a×aφ, игла L = ae/2):
    W = φ·e/π ≈ 1.4008...
"""
import math
import random
import sys
import argparse

# Константы
_PHI = (1 + math.sqrt(5)) / 2    # золотое сечение φ
_E   = math.e
_PI  = math.pi


# ── общая формула ─────────────────────────────────────────────────────────────

def buffon_general(needle: float, perimeter: float, area: float) -> float:
    """Обобщённая формула Бюффона: W = L·U / (π·F)."""
    if area <= 0:
        raise ValueError("area должна быть > 0")
    return needle * perimeter / (_PI * area)


# ── основной класс ────────────────────────────────────────────────────────────

class BuffonParquet:
    """Обобщённая задача Бюффона для различных плиток."""

    # ── конкретные плитки ─────────────────────────────────────────────────────

    def general_formula(self, L: float, perimeter: float, area: float) -> float:
        """W = L·U / (π·F)."""
        return buffon_general(L, perimeter, area)

    def rectangular(self, a: float, b: float, needle: float) -> float:
        """Прямоугольник a×b: W = (L/π)·(1/a + 1/b)."""
        perimeter = 2 * (a + b)
        area = a * b
        return buffon_general(needle, perimeter, area)

    def square(self, a: float, needle: float) -> float:
        """Квадрат a×a: W = 4L / (πa)."""
        return self.rectangular(a, a, needle)

    def hexagonal(self, r: float, needle: float) -> float:
        """Правильный шестиугольник со стороной r.

        U = 6r,  F = 3√3/2 · r²
        W = 2L√3 / (πr)
        """
        perimeter = 6 * r
        area = 3 * math.sqrt(3) / 2 * r ** 2
        return buffon_general(needle, perimeter, area)

    def triangular(self, side: float, needle: float) -> float:
        """Правильный треугольник со стороной side.

        U = 3·side,  F = √3/4 · side²
        """
        perimeter = 3 * side
        area = math.sqrt(3) / 4 * side ** 2
        return buffon_general(needle, perimeter, area)

    # ── золотой прямоугольник ─────────────────────────────────────────────────

    def golden_rectangle(self, a: float) -> float:
        """Прямоугольник a × a·φ с иглой L = a·e/2.

        W = φ·e/π ≈ 1.4008...  (не зависит от a!)
        """
        b = a * _PHI
        needle = a * _E / 2
        return self.rectangular(a, b, needle)

    def golden_rectangle_verify(self) -> dict:
        """Аналитически показать: W = φ·e/π независимо от a."""
        exact = _PHI * _E / _PI
        computed = self.golden_rectangle(1.0)
        return {
            "exact":    exact,
            "computed": computed,
            "formula":  "φ·e/π",
            "phi":      _PHI,
            "e":        _E,
            "pi":       _PI,
            "ok":       abs(exact - computed) < 1e-12
        }

    # ── симуляция Монте-Карло ─────────────────────────────────────────────────

    def simulate(self, tile: str, needle: float, n: int = 100_000,
                 seed: int | None = None, **tile_params) -> dict:
        """Монте-Карло симуляция.

        tile: 'square', 'rectangle', 'hexagonal', 'golden'
        tile_params: a=, b= (для rectangle), r= (для hexagonal)
        """
        rng = random.Random(seed)
        hits = 0

        if tile == "square":
            a = tile_params.get("a", 1.0)
            exact_W = self.square(a, needle)
            for _ in range(n):
                # Угол иглы и позиция центра
                angle = rng.uniform(0, _PI)
                # Кол-во пересечений с горизонтальными линиями
                projection_y = needle * abs(math.sin(angle))
                projection_x = needle * abs(math.cos(angle))
                cy = rng.uniform(0, a)
                cx = rng.uniform(0, a)
                cnt = 0
                # Горизонтальные линии
                if cy < projection_y / 2 or cy > a - projection_y / 2:
                    cnt += 1
                # Вертикальные линии
                if cx < projection_x / 2 or cx > a - projection_x / 2:
                    cnt += 1
                hits += cnt

        elif tile == "rectangle":
            a = tile_params.get("a", 1.0)
            b = tile_params.get("b", 2.0)
            exact_W = self.rectangular(a, b, needle)
            for _ in range(n):
                angle = rng.uniform(0, _PI)
                proj_y = needle * abs(math.sin(angle))
                proj_x = needle * abs(math.cos(angle))
                cy = rng.uniform(0, a)
                cx = rng.uniform(0, b)
                cnt = 0
                if cy < proj_y / 2 or cy > a - proj_y / 2:
                    cnt += 1
                if cx < proj_x / 2 or cx > b - proj_x / 2:
                    cnt += 1
                hits += cnt

        elif tile == "golden":
            a = tile_params.get("a", 1.0)
            exact_W = self.golden_rectangle(a)
            b = a * _PHI
            needle_len = a * _E / 2
            for _ in range(n):
                angle = rng.uniform(0, _PI)
                proj_y = needle_len * abs(math.sin(angle))
                proj_x = needle_len * abs(math.cos(angle))
                cy = rng.uniform(0, a)
                cx = rng.uniform(0, b)
                cnt = 0
                if cy < proj_y / 2 or cy > a - proj_y / 2:
                    cnt += 1
                if cx < proj_x / 2 or cx > b - proj_x / 2:
                    cnt += 1
                hits += cnt

        else:
            raise ValueError(f"Неизвестный тип плитки: {tile}")

        estimated = hits / n
        error_pct = abs(estimated - exact_W) / max(abs(exact_W), 1e-12) * 100
        return {
            "tile": tile,
            "n": n,
            "estimated_W": estimated,
            "exact_W": exact_W,
            "error_pct": error_pct
        }

    # ── обратная задача ───────────────────────────────────────────────────────

    def find_needle_length(self, target_W: float, tile: str = "square",
                           **tile_params) -> float:
        """Найти длину иглы L для заданного W.

        Из W = L·U/(π·F) → L = W·π·F/U
        """
        a = tile_params.get("a", 1.0)
        b = tile_params.get("b", a)
        if tile == "square":
            perimeter = 4 * a
            area = a ** 2
        elif tile == "rectangle":
            perimeter = 2 * (a + b)
            area = a * b
        elif tile == "hexagonal":
            r = tile_params.get("r", 1.0)
            perimeter = 6 * r
            area = 3 * math.sqrt(3) / 2 * r ** 2
        else:
            raise ValueError(f"Неизвестный тип плитки: {tile}")
        return target_W * _PI * area / perimeter

    # ── связь φ, e, π в одной формуле ────────────────────────────────────────

    def plot_formula_relation(self) -> str:
        """ASCII-схема: φ·e/π = W (золотая формула Бюффона)."""
        val = _PHI * _E / _PI
        lines = [
            "Золотая формула Бюффона: W = φ · e / π",
            "",
            f"  φ = {_PHI:.10f}  (золотое сечение)",
            f"  e = {_E:.10f}  (число Эйлера)",
            f"  π = {_PI:.10f}  (число Пи)",
            f"  ─────────────────────────────",
            f"  W = {val:.10f}",
            "",
            "Интерпретация: бросить иглу длиной L = a·e/2 на паркет",
            "из золотых прямоугольников a × a·φ → ожидаемое число",
            "пересечений с линиями паркета = φ·e/π ≈ 1.4008.",
        ]
        return "\n".join(lines)

    # ── для Q6 ────────────────────────────────────────────────────────────────

    def hexagonal_q6(self, r: float = 1.0, needle: float = 1.0) -> float:
        """W для шестиугольной решётки (как у hexbio/Q6 геометрии)."""
        return self.hexagonal(r, needle)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexbuffon — обобщённая формула Бюффона")
    parser.add_argument("--rectangular", nargs=2, type=float, metavar=("A", "B"),
                        help="Прямоугольник A×B")
    parser.add_argument("--square", type=float, metavar="A",
                        help="Квадрат A×A")
    parser.add_argument("--hexagonal", type=float, metavar="R",
                        help="Шестиугольник со стороной R")
    parser.add_argument("--needle", type=float, default=1.0,
                        help="Длина иглы (по умолчанию 1.0)")
    parser.add_argument("--golden", type=float, metavar="A",
                        help="Золотой прямоугольник со стороной A")
    parser.add_argument("--golden-verify", action="store_true",
                        help="Проверить формулу φe/π")
    parser.add_argument("--simulate", type=str, metavar="TILE",
                        help="Симуляция для TILE (square/rectangle/golden)")
    parser.add_argument("--n", type=int, default=100_000,
                        help="Число бросков для симуляции")
    parser.add_argument("--find-needle", action="store_true",
                        help="Найти иглу для заданного W")
    parser.add_argument("--target-w", type=float, default=1.0,
                        help="Целевое значение W")
    parser.add_argument("--formula", action="store_true",
                        help="Показать золотую формулу φe/π")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=2.0)
    args = parser.parse_args()

    bp = BuffonParquet()

    if args.rectangular is not None:
        a, b = args.rectangular
        W = bp.rectangular(a, b, args.needle)
        print(f"Прямоугольник {a}×{b}, игла={args.needle}: W = {W:.6f}")

    if args.square is not None:
        W = bp.square(args.square, args.needle)
        print(f"Квадрат {args.square}×{args.square}, игла={args.needle}: W = {W:.6f}")

    if args.hexagonal is not None:
        W = bp.hexagonal(args.hexagonal, args.needle)
        print(f"Шестиугольник r={args.hexagonal}, игла={args.needle}: W = {W:.6f}")

    if args.golden is not None:
        W = bp.golden_rectangle(args.golden)
        print(f"Золотой прямоугольник a={args.golden}: W = {W:.10f}")
        print(f"  (φ·e/π = {_PHI * _E / _PI:.10f})")

    if args.golden_verify:
        res = bp.golden_rectangle_verify()
        print(f"Формула {res['formula']}:")
        print(f"  φ = {res['phi']:.10f}")
        print(f"  e = {res['e']:.10f}")
        print(f"  π = {res['pi']:.10f}")
        print(f"  Точно:    {res['exact']:.10f}")
        print(f"  Вычисл.: {res['computed']:.10f}")
        print(f"  Совпадает: {res['ok']}")

    if args.simulate:
        result = bp.simulate(args.simulate, needle=args.needle, n=args.n,
                             a=args.a, b=args.b)
        print(f"Симуляция '{args.simulate}', n={args.n}:")
        print(f"  Оценка W:  {result['estimated_W']:.5f}")
        print(f"  Точное W:  {result['exact_W']:.5f}")
        print(f"  Ошибка:    {result['error_pct']:.2f}%")

    if args.find_needle:
        L = bp.find_needle_length(args.target_w, tile="square", a=args.a)
        print(f"Игла для W={args.target_w}, квадрат a={args.a}: L = {L:.6f}")

    if args.formula:
        print(bp.plot_formula_relation())

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
