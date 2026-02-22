"""
hexkub — Арифметическая геометрия: система КУБ Касаткина В. В.

Источник: «Арифметическая геометрия Касаткина» (журнальная статья ~1988 г.)
Автор системы: Касаткин Вячеслав Владимирович
Прибор ПШР, патент №805737

Центральная идея:
  Каждому натуральному числу N ставится в соответствие геометрический
  объект — «КУБ заданного числа» с тремя координатами (a, b, c).
  Арифметические операции = геометрические преобразования куба.

Ключевое тождество с Q6-монорепо:
  64 = 2^6  →  Q6: 6-мерный гиперкуб (вся экосистема)
  64 = 4^3  →  КУБ(4): куб числа 4 в системе Касаткина
  Одно число — два геометрических воплощения.
"""

import math
import sys
import argparse
from itertools import product as iproduct


# ---------------------------------------------------------------------------
# KubNumber — основное число в системе КУБ
# ---------------------------------------------------------------------------

class KubNumber:
    """Натуральное число в геометрической системе КУБ Касаткина."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("КУБ определён для натуральных чисел (n >= 1)")
        self.n = n

    # --- базовые свойства ---------------------------------------------------

    def is_perfect_cube(self) -> bool:
        """Является ли n точным кубом (n = k^3 для целого k)?"""
        k = round(self.n ** (1/3))
        return k ** 3 == self.n

    def cube_root(self) -> float:
        """Кубический корень: n^(1/3)."""
        return self.n ** (1/3)

    def integer_cube_root(self) -> int:
        """Целая часть кубического корня."""
        return int(self.n ** (1/3) + 1e-9)

    def coordinates(self) -> tuple:
        """
        Координаты КУБа числа n: (a, b, c) — «правильный» куб.
        Для совершенного куба k³: возвращает (k, k, k).
        Для остальных: (k, k, k) где k = ближайший целый ∛n,
        плюс остаток.
        """
        k = self.integer_cube_root()
        if k ** 3 == self.n:
            return (k, k, k)
        # Ближайший куб снизу
        remainder = self.n - k**3
        return (k, k, k, remainder)  # (a,b,c, остаток)

    def nearest_cubes(self) -> tuple:
        """Два ближайших точных куба: (снизу, сверху)."""
        k = self.integer_cube_root()
        below = k ** 3
        above = (k + 1) ** 3
        return (below, above)

    def three_factor_decompositions(self) -> list:
        """
        Все разложения n = a * b * c (a <= b <= c, a,b,c >= 1).
        Это все возможные «прямоугольные параллелепипеды» объёма n.
        """
        result = []
        n = self.n
        for a in range(1, int(n**(1/3)) + 1):
            if n % a != 0:
                continue
            rest = n // a
            for b in range(a, int(rest**0.5) + 1):
                if rest % b != 0:
                    continue
                c = rest // b
                result.append((a, b, c))
        return result

    def most_cubic_decomposition(self) -> tuple:
        """
        Наиболее «кубическое» разложение n=a*b*c:
        минимизирует отклонение от куба (max/min → 1).
        """
        decomps = self.three_factor_decompositions()
        if not decomps:
            return (1, 1, self.n)
        best = min(decomps, key=lambda t: t[2] / t[0])  # c/a → min
        return best

    def q6_relation(self) -> str:
        """Описание связи n с Q6-монорепо."""
        lines = []
        lines.append(f"n = {self.n}")
        # Степень двойки?
        if self.n > 0 and (self.n & (self.n - 1)) == 0:
            k = int(math.log2(self.n))
            lines.append(f"  = 2^{k}  (гиперкуб Q{k})")
        # Точный куб?
        if self.is_perfect_cube():
            k = self.integer_cube_root()
            lines.append(f"  = {k}^3  (КУБ числа {k} по Касаткину)")
        # Точный квадрат?
        sq = int(math.isqrt(self.n))
        if sq * sq == self.n:
            lines.append(f"  = {sq}^2  (совершенный квадрат)")
        # Специально для 64
        if self.n == 64:
            lines.append("  ТОЖДЕСТВО: Q6 = КУБ(4) = 64 глифа = 64 кодона ДНК")
        return "\n".join(lines)

    def __repr__(self):
        return f"KubNumber({self.n})"


# ---------------------------------------------------------------------------
# KubGeometry — геометрия куба
# ---------------------------------------------------------------------------

class KubGeometry:
    """Геометрические свойства куба и его связь с Q6."""

    def __init__(self, a: float = 1.0):
        self.a = a  # длина ребра

    def edge(self) -> float:
        return self.a

    def face_diagonal(self) -> float:
        """Диагональ грани: a√2."""
        return self.a * math.sqrt(2)

    def space_diagonal(self) -> float:
        """Пространственная диагональ: a√3."""
        return self.a * math.sqrt(3)

    def inscribed_sphere_r(self) -> float:
        """Радиус вписанной сферы: a/2."""
        return self.a / 2

    def circumscribed_sphere_R(self) -> float:
        """Радиус описанной сферы: a√3/2."""
        return self.a * math.sqrt(3) / 2

    def sphere_cube_volume_ratio(self) -> float:
        """Отношение объёма куба к объёму описанной сферы."""
        V_cube = self.a ** 3
        R = self.circumscribed_sphere_R()
        V_sphere = (4/3) * math.pi * R**3
        return V_cube / V_sphere

    def diagonal_projection_hexagon(self) -> list:
        """
        Проекция куба вдоль главной диагонали (1,1,1)/√3.
        Возвращает 6 вершин правильного шестиугольника в 2D.
        Это 6 «промежуточных» вершин куба (yang=1 и yang=2 в Q3).
        """
        # Вершины куба, видимые с направления (1,1,1)
        # Это рёбра: (0,0,1)-(0,1,1), (0,0,1)-(1,0,1), (0,1,0)-(0,1,1),
        #            (1,0,0)-(1,0,1), (1,0,0)-(1,1,0), (0,1,0)-(1,1,0)
        # Проецируем 6 промежуточных вершин на плоскость, перп. (1,1,1)
        a = self.a
        intermediate = [
            (a, 0, 0), (0, a, 0), (0, 0, a),
            (a, a, 0), (a, 0, a), (0, a, a),
        ]
        # Ортогональный базис плоскости, перпендикулярной (1,1,1)
        # e1 = (1,-1,0)/√2,  e2 = (1,1,-2)/√6
        e1 = (1/math.sqrt(2), -1/math.sqrt(2), 0)
        e2 = (1/math.sqrt(6),  1/math.sqrt(6), -2/math.sqrt(6))

        pts2d = []
        for v in intermediate:
            x2 = sum(v[i]*e1[i] for i in range(3))
            y2 = sum(v[i]*e2[i] for i in range(3))
            pts2d.append((round(x2, 4), round(y2, 4)))
        return pts2d

    def golden_section_in_cube(self) -> dict:
        """
        Золотое сечение в геометрии куба.
        Икосаэдр, вписанный в куб, использует φ-пропорции.
        """
        phi = (1 + math.sqrt(5)) / 2
        a = self.a
        return {
            "phi": phi,
            "edge_icosahedron": a / phi,
            "diagonal_face": a * math.sqrt(2),
            "diagonal_space": a * math.sqrt(3),
            "ratio_R_r": math.sqrt(3),         # R/r описанной к вписанной
            "ratio_face_diag_edge": math.sqrt(2),
            "comment": (
                f"Икосаэдр вписан в куб ребром {a:.4f}/φ = {a/phi:.4f}. "
                f"12 вершин икосаэдра лежат на рёбрах куба в пропорции φ."
            ),
        }

    def ascii_cube(self, label: str = "") -> str:
        """ASCII-рисунок куба с подписями диагоналей."""
        lines = [
            f"    КУБ (ребро a={self.a})",
            "",
            "     *-------*",
            "    /|      /|",
            "   / |     / |",
            "  *-------*  |",
            "  |  *----|--*",
            "  | /     | /",
            "  |/      |/",
            "  *-------*",
            "",
            f"  Ребро:                a      = {self.a:.4f}",
            f"  Диагональ грани:      a√2    = {self.face_diagonal():.4f}",
            f"  Пространств. диагон.: a√3    = {self.space_diagonal():.4f}",
            f"  Вписанная сфера r:    a/2    = {self.inscribed_sphere_r():.4f}",
            f"  Описанная сфера R:    a√3/2  = {self.circumscribed_sphere_R():.4f}",
            f"  V_куб / V_сфера:             = {self.sphere_cube_volume_ratio():.4f}",
        ]
        if label:
            lines.insert(0, label)
        return "\n".join(lines)

    def ascii_diagonal_projection(self) -> str:
        """ASCII визуализация проекции куба вдоль диагонали → шестиугольник."""
        pts = self.diagonal_projection_hexagon()
        # Сортируем по углу
        import cmath
        pts_sorted = sorted(pts, key=lambda p: cmath.phase(complex(p[0], p[1])))

        # Нормализуем для рисунка
        max_r = max(abs(complex(p[0], p[1])) for p in pts_sorted) or 1
        W, H = 41, 21
        lines = [[" "] * W for _ in range(H)]
        cx, cy = W // 2, H // 2

        labels = ["◆", "◆", "◆", "◆", "◆", "◆"]
        for i, (px, py) in enumerate(pts_sorted):
            ix = int(cx + px / max_r * (W//2 - 2))
            iy = int(cy - py / max_r * (H//2 - 1))
            ix = max(0, min(W-1, ix))
            iy = max(0, min(H-1, iy))
            lines[iy][ix] = labels[i % len(labels)]

        # Рисуем стрелку в центре (диагональ = точка)
        lines[cy][cx] = "●"

        result = [
            "Проекция куба вдоль главной диагонали (1,1,1)/√3:",
            "Видны 6 промежуточных вершин → правильный шестиугольник",
            "Центр ● = диагональ куба = ось вращения",
            "",
        ]
        result.extend("".join(row) for row in lines)
        result.extend([
            "",
            "Связь с Q6: yang-слои куба 4×4×4 = слои Q6",
            "  yang=0: 1 вершина (0,0,0)",
            "  yang=1: 6 вершин  → 6 точек шестиугольника",
            "  yang=2: 15 вершин → внутренний слой",
            "  yang=3: 20 вершин → экватор (максимум)",
        ])
        return "\n".join(result)


# ---------------------------------------------------------------------------
# KubArithmetic — арифметика через КУБ
# ---------------------------------------------------------------------------

class KubArithmetic:
    """Арифметические операции в геометрической интерпретации Касаткина."""

    def cube_sum(self, a: int, b: int) -> dict:
        """
        Сумма двух кубических чисел: a³ + b³.
        Проверяет теорему Ферма: a³ + b³ ≠ c³ для натуральных c.
        """
        s = a**3 + b**3
        c_approx = s ** (1/3)
        c_int = round(c_approx)
        is_cube = (c_int ** 3 == s)
        return {
            "a": a, "b": b,
            "a_cube": a**3, "b_cube": b**3,
            "sum": s,
            "cube_root_approx": c_approx,
            "is_perfect_cube": is_cube,
            "fermat_satisfied": not is_cube,
            "comment": (
                f"{a}³ + {b}³ = {a**3} + {b**3} = {s}. "
                + ("✓ Не является кубом — теорема Ферма подтверждена."
                   if not is_cube else
                   f"⚠ Является кубом {c_int}³! (вырожд. случай?)")
            ),
        }

    def fermat_check(self, a: int, b: int, c: int, n: int = 3) -> dict:
        """Проверка: a^n + b^n == c^n?"""
        lhs = a**n + b**n
        rhs = c**n
        return {
            "equation": f"{a}^{n} + {b}^{n} = {c}^{n}?",
            "lhs": lhs, "rhs": rhs,
            "holds": lhs == rhs,
            "difference": lhs - rhs,
            "fermat_theorem": (
                "Теорема Ферма: при n>2 решений нет (доказано Уайлсом, 1995)"
                if n > 2 else
                "При n=2: пифагоровы тройки существуют"
            ),
        }

    def taxicab_number(self, n: int) -> list:
        """
        Числа такси (Рамануджан): представимые как сумма двух кубов
        двумя разными способами.
        Первое: 1729 = 1³+12³ = 9³+10³
        """
        results = []
        for a in range(1, int(n**(1/3)) + 1):
            for b in range(a, int(n**(1/3)) + 1):
                if a**3 + b**3 == n:
                    results.append((a, b))
        return results

    def cube_root_geometric(self, n: int) -> dict:
        """Кубический корень через геометрическое построение."""
        cr = n ** (1/3)
        k = KubNumber(n)
        return {
            "n": n,
            "cube_root": cr,
            "is_integer": k.is_perfect_cube(),
            "integer_root": k.integer_cube_root() if k.is_perfect_cube() else None,
            "nearest_cubes": k.nearest_cubes(),
            "method": "Геометрически: ребро куба объёма n = ∛n",
        }

    def golden_section_construction(self) -> dict:
        """Золотое сечение: геометрическое построение через диагональ."""
        phi = (1 + math.sqrt(5)) / 2
        return {
            "phi": phi,
            "1/phi": 1/phi,
            "phi^2": phi**2,
            "construction": (
                "Золотое сечение через куб:\n"
                "1. Возьмём куб с ребром 1\n"
                "2. Диагональ грани = √2\n"
                "3. Диагональ куба  = √3\n"
                "4. Икосаэдр в кубе: рёбра = 1/φ ≈ 0.618\n"
                "5. φ = (1+√5)/2 ≈ 1.618"
            ),
        }

    def sum_of_three_cubes(self, target: int, max_k: int = 50) -> list:
        """
        Задача Варинга для кубов: n = a³ + b³ + c³.
        Возвращает найденные разложения (a,b,c могут быть отрицательными).
        """
        solutions = []
        for a in range(-max_k, max_k + 1):
            for b in range(a, max_k + 1):
                rem = target - a**3 - b**3
                c_approx = abs(rem) ** (1/3)
                c = round(c_approx)
                if c > max_k:
                    continue
                for c_try in [c, -c]:
                    if a**3 + b**3 + c_try**3 == target:
                        triple = tuple(sorted([a, b, c_try]))
                        if triple not in solutions:
                            solutions.append(triple)
        return solutions


# ---------------------------------------------------------------------------
# Q6-интеграция
# ---------------------------------------------------------------------------

def q6_cube_identity() -> str:
    """Полная демонстрация тождества Q6 = КУБ(4)."""
    lines = [
        "=" * 60,
        "ТОЖДЕСТВО: Q6 = КУБ числа 4",
        "=" * 60,
        "",
        "  64 = 2^6   →  Q6: 6-мерный булев гиперкуб",
        "               64 вершины, 192 ребра, диаметр 6",
        "               Граф Кэли группы (Z₂)^6",
        "",
        "  64 = 4^3   →  КУБ(4) по Касаткину",
        "               Куб со стороной 4",
        "               4×4×4 = 64 клетки",
        "",
        "  64 = 8^2   →  Квадрат числа 8",
        "               8×8 = шахматная доска",
        "               8×8 сетка глифов Q6",
        "",
        "Единое пространство 64 элементов:",
        "",
        "  yang=0:  C(6,0)=1   вершин  ←→  (0,0,0) в кубе 4×4×4",
        "  yang=1:  C(6,1)=6   вершин  ←→  6 соседей центра",
        "  yang=2:  C(6,2)=15  вершин  ←→  второй слой",
        "  yang=3:  C(6,3)=20  вершин  ←→  экватор (максимум)",
        "  yang=4:  C(6,4)=15  вершин  ←→  симметрично yang=2",
        "  yang=5:  C(6,5)=6   вершин  ←→  симметрично yang=1",
        "  yang=6:  C(6,6)=1   вершин  ←→  вершина (3,3,3) куба",
        "",
        "  Сумма: 1+6+15+20+15+6+1 = 64 = 4^3 ✓",
        "",
        "Проекция Q6 вдоль диагонали 0→63:",
        "  Даёт 7 слоёв (yang=0..6)",
        "  Аналог: вид на куб вдоль пространственной диагонали",
        "  Промежуточные 6 вершин образуют шестиугольник",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def yang_layers_as_3d_cube() -> str:
    """Отображение 64 глифов Q6 в 4×4×4 куб Касаткина."""
    lines = [
        "64 глифа Q6 как КУБ числа 4 (4×4×4 = 64):",
        "",
        "Кодировка: h = x + 4*y + 16*z,  x,y,z ∈ {0,1,2,3}",
        "",
        "Слой z=0 (нижний):",
    ]
    # Рисуем 4 слоя куба
    for z in range(4):
        lines.append(f"\n  Слой z={z}:")
        lines.append("   x: 0    1    2    3")
        for y in range(4):
            row = []
            for x in range(4):
                h = x + 4*y + 16*z
                yang = bin(h).count('1')
                row.append(f" {h:2d}({'█'*yang+'·'*(6-yang)})")
            lines.append(f"  y={y}:{''.join(row)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexkub — Арифметическая геометрия: система КУБ Касаткина",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexkub.py --kub 64         КУБ числа 64 и связь с Q6
  python hexkub.py --kub 27         КУБ числа 27 (= 3^3)
  python hexkub.py --geometry       Геометрия единичного куба
  python hexkub.py --geometry 2.5   Геометрия куба со стороной 2.5
  python hexkub.py --projection     Проекция куба вдоль диагонали
  python hexkub.py --fermat 3 4 5   Теорема Ферма: 3^3+4^3=5^3?
  python hexkub.py --taxicab 1729   Число такси Рамануджана
  python hexkub.py --q6             Тождество Q6 = КУБ(4)
  python hexkub.py --3d             64 глифа как куб 4×4×4
  python hexkub.py --golden         Золотое сечение в кубе
  python hexkub.py --cubesum 3 5    Сумма кубов 3^3+5^3 и теорема Ферма
        """,
    )
    parser.add_argument("--kub", type=int, metavar="N",
                        help="Показать КУБ числа N")
    parser.add_argument("--geometry", type=float, nargs="?", const=1.0, metavar="A",
                        help="Геометрия куба с ребром A (по умолчанию 1.0)")
    parser.add_argument("--projection", action="store_true",
                        help="ASCII-проекция куба вдоль диагонали → шестиугольник")
    parser.add_argument("--fermat", type=int, nargs=3, metavar=("A","B","C"),
                        help="Проверить a^3+b^3=c^3? (теорема Ферма)")
    parser.add_argument("--taxicab", type=int, metavar="N",
                        help="Число такси: N = a^3+b^3 двумя способами?")
    parser.add_argument("--q6", action="store_true",
                        help="Тождество Q6 = КУБ(4) = 64")
    parser.add_argument("--3d", action="store_true", dest="threed",
                        help="64 глифа Q6 как куб 4×4×4")
    parser.add_argument("--golden", action="store_true",
                        help="Золотое сечение в геометрии куба")
    parser.add_argument("--cubesum", type=int, nargs=2, metavar=("A","B"),
                        help="Сумма кубов a^3+b^3 и проверка теоремы Ферма")
    parser.add_argument("--waring", type=int, metavar="N",
                        help="Разложение N = a^3+b^3+c^3 (задача Варинга)")

    args = parser.parse_args()

    ka = KubArithmetic()

    if args.kub is not None:
        k = KubNumber(args.kub)
        print(f"\n{'='*50}")
        print(f"КУБ числа {args.kub}")
        print(f"{'='*50}")
        print(f"\nСовершенный куб:  {k.is_perfect_cube()}")
        print(f"Кубический корень: {k.cube_root():.6f}")
        print(f"Координаты КУБа:  {k.coordinates()}")
        print(f"Ближайшие кубы:   {k.nearest_cubes()}")
        print(f"\n3-факторные разложения ({args.kub} = a×b×c):")
        for d in k.three_factor_decompositions():
            marker = " ← наиболее кубическое" if d == k.most_cubic_decomposition() else ""
            print(f"  {d[0]} × {d[1]} × {d[2]} = {d[0]*d[1]*d[2]}{marker}")
        print(f"\nСвязь с Q6:")
        print(k.q6_relation())

    elif args.geometry is not None:
        kg = KubGeometry(args.geometry)
        print("\n" + kg.ascii_cube())
        g = kg.golden_section_in_cube()
        print(f"\nЗолотое сечение в кубе:")
        print(f"  φ = {g['phi']:.6f}")
        print(f"  Икосаэдр в кубе, ребро = ребро_куба/φ = {g['edge_icosahedron']:.4f}")
        print(f"  {g['comment']}")

    elif args.projection:
        kg = KubGeometry(1.0)
        print("\n" + kg.ascii_diagonal_projection())

    elif args.fermat:
        a, b, c = args.fermat
        result = ka.fermat_check(a, b, c, n=3)
        print(f"\n{'='*50}")
        print("Теорема Ферма: a³ + b³ = c³?")
        print(f"{'='*50}")
        print(f"\nУравнение: {result['equation']}")
        print(f"Левая часть:  {a}³ + {b}³ = {a**3} + {b**3} = {result['lhs']}")
        print(f"Правая часть: {c}³ = {result['rhs']}")
        print(f"Выполняется:  {result['holds']}")
        print(f"Разница:      {result['difference']}")
        print(f"\n{result['fermat_theorem']}")

    elif args.taxicab is not None:
        n = args.taxicab
        pairs = ka.taxicab_number(n)
        print(f"\n{'='*50}")
        print(f"Число такси Рамануджана: {n}")
        print(f"{'='*50}")
        if len(pairs) >= 2:
            print(f"\n{n} — число такси! {len(pairs)} разложения:")
        elif len(pairs) == 1:
            print(f"\n{n} — одно разложение (не число такси):")
        else:
            print(f"\n{n} — не представимо как сумма двух кубов в диапазоне.")
        for a, b in pairs:
            print(f"  {a}³ + {b}³ = {a**3} + {b**3} = {a**3+b**3}")
        if n == 1729:
            print("\nЭто ПЕРВОЕ число такси (Hardy-Ramanujan number).")
            print("«Каждое такси-число — замечательный повод для разговора.»")

    elif args.q6:
        print(q6_cube_identity())

    elif args.threed:
        print(yang_layers_as_3d_cube())

    elif args.golden:
        kg = KubGeometry(1.0)
        g = kg.golden_section_in_cube()
        res = ka.golden_section_construction()
        print(f"\n{'='*50}")
        print("Золотое сечение в геометрии куба")
        print(f"{'='*50}")
        print(f"\nφ = {res['phi']:.10f}")
        print(f"1/φ = {res['1/phi']:.10f}")
        print(f"φ² = {res['phi^2']:.10f}")
        print(f"\n{res['construction']}")
        print(f"\nИкосаэдр, вписанный в куб (ребро куба = 1):")
        print(f"  Ребро икосаэдра  = 1/φ = {g['edge_icosahedron']:.6f}")
        print(f"  {g['comment']}")

    elif args.cubesum:
        a, b = args.cubesum
        result = ka.cube_sum(a, b)
        print(f"\n{'='*50}")
        print(f"Сумма кубов: {a}³ + {b}³")
        print(f"{'='*50}")
        print(f"\n{result['comment']}")
        print(f"\nКубический корень суммы: ∛{result['sum']} ≈ {result['cube_root_approx']:.6f}")

    elif args.waring:
        n = args.waring
        sols = ka.sum_of_three_cubes(n)
        print(f"\n{'='*50}")
        print(f"Задача Варинга: {n} = a³ + b³ + c³")
        print(f"{'='*50}")
        if sols:
            print(f"\nНайдено {len(sols)} разложений:")
            for s in sols:
                print(f"  {s[0]}³ + {s[1]}³ + {s[2]}³ = "
                      f"{s[0]**3} + {s[1]**3} + {s[2]**3} = {sum(x**3 for x in s)}")
        else:
            print(f"\nРазложений в диапазоне ±50 не найдено.")

    else:
        # По умолчанию — демонстрация ключевого тождества
        print(q6_cube_identity())
        print()
        kg = KubGeometry(1.0)
        print(kg.ascii_cube())
        print()
        k = KubNumber(64)
        print("КУБ числа 64:")
        print(k.q6_relation())


if __name__ == "__main__":
    main()
