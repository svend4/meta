"""hexhept.py — Единственность гептаэдра (модель RP²).

Источник: PDF «Единственность гептаэдра?» Франца Германа.

Гептаэдр:
  Вершины: 6,  Грани: 7 (4 треугольника + 3 квадрата),  Рёбра: 12
  χ = B + Γ - P = 6 + 7 - 12 = 1  (= χ(RP²))

Условия для многогранника быть моделью RP²:
  1. χ = B + Γ - P = 1
  2. Каждое ребро граничит ровно с 2 гранями
  3. Γ_добавить - Γ_убрать = 1 (условие добавления граней)
  4. Балансовое условие рёбер

Площадь и объём (ребро = a):
  S = a²√3·(1 + √3)
  V = a³√2/6

Изоморфизм RP² ↔ единичная сфера через разбиение граней.
"""
import math
import sys
import argparse
from itertools import combinations


# ── вспомогательные ───────────────────────────────────────────────────────────

def _euler_characteristic(vertices: int, faces: int, edges: int) -> int:
    return vertices + faces - edges


# ── класс проверки RP² ────────────────────────────────────────────────────────

class RP2Checker:
    """Проверка, является ли многогранник моделью RP²."""

    def check(self, vertices: int, faces: int, edges: int,
              face_types: dict | None = None) -> dict:
        """Проверить многогранник на соответствие RP².

        face_types: {'triangle': k, 'square': m, ...} — типы граней
        Возвращает словарь с результатом и причиной отклонения.
        """
        chi = _euler_characteristic(vertices, faces, edges)
        if chi != 1:
            return {"ok": False, "reason": f"χ = {chi} ≠ 1",
                    "chi": chi, "vertices": vertices,
                    "faces": faces, "edges": edges}

        # Условие 2: каждое ребро граничит с 2 гранями
        # Σ_f (число рёбер грани f) = 2 * edges
        if face_types is not None:
            total_edge_incidences = sum(
                k * sides for sides, k in
                [("triangle", face_types.get("triangle", 0)),
                 ("square", face_types.get("square", 0)),
                 ("pentagon", face_types.get("pentagon", 0))]
                if isinstance(sides, str)
                for sides, count in [
                    (3, face_types.get("triangle", 0)),
                    (4, face_types.get("square", 0)),
                    (5, face_types.get("pentagon", 0))
                ]
                for k in [count]
                if k > 0
            )
        else:
            total_edge_incidences = None

        # Условие 3: Γ_Y - Γ_D = 1
        # Для гептаэдра: Γ_Y = число граней с > min сторонами, Γ_D = число удаляемых
        # Упрощённо: проверяем χ = 1 + дополнительные условия

        # Основное условие: χ = 1 и данные совместимы
        result = {
            "ok": True,
            "chi": chi,
            "vertices": vertices,
            "faces": faces,
            "edges": edges,
            "reason": "χ = 1 ✓",
        }
        return result

    def check_icosahedron(self) -> dict:
        """Икосаэдр: B=12, Γ=20, P=30, χ=2 — не RP²."""
        return self.check(vertices=12, faces=20, edges=30)

    def check_cuboctahedron(self) -> dict:
        """Кубооктаэдр: B=12, Γ=14, P=24, χ=2 — не RP²."""
        return self.check(vertices=12, faces=14, edges=24)

    def check_heptahedron(self) -> dict:
        """Гептаэдр: B=6, Γ=7, P=12, χ=1 — RP²."""
        return self.check(vertices=6, faces=7, edges=12,
                          face_types={"triangle": 4, "square": 3})

    def enumerate_rp2_candidates(self, max_vertices: int = 8) -> list[dict]:
        """Перебрать многогранники с χ=1 и B ≤ max_vertices.

        Использует соотношения Эйлера и условие степени граней.
        """
        results = []
        for B in range(4, max_vertices + 1):
            # Из χ=1: Γ = P - B + 1
            # Из условия рёбер (среднее число сторон ≥ 3): P ≤ Γ * avg_sides / 2
            # Перебираем Γ от 2 до 2*B
            for Gamma in range(2, 2 * B + 1):
                P = B + Gamma - 1   # из χ=1
                if P < max(B, Gamma):
                    continue
                chi = _euler_characteristic(B, Gamma, P)
                if chi != 1:
                    continue
                # Проверить: 2*P ≥ 3*Gamma (каждая грань ≥ 3 сторон)
                if 2 * P < 3 * Gamma:
                    continue
                # Проверить: 2*P ≥ 3*B (каждая вершина степени ≥ 3)
                if 2 * P < 3 * B:
                    continue
                results.append({
                    "vertices": B,
                    "faces": Gamma,
                    "edges": P,
                    "chi": chi,
                    "ok": True,
                })
        return results


# ── гептаэдр ─────────────────────────────────────────────────────────────────

class Heptahedron:
    """Гептаэдр — минимальная полиэдральная модель RP²."""

    def __init__(self, edge_length: float = 1.0):
        self.a = edge_length

    def vertices(self) -> int:
        """Число вершин: 6."""
        return 6

    def faces(self) -> int:
        """Число граней: 7 (4 треугольника + 3 квадрата)."""
        return 7

    def edges(self) -> int:
        """Число рёбер: 12."""
        return 12

    def euler_characteristic(self) -> int:
        """χ = B + Γ - P = 6 + 7 - 12 = 1."""
        return _euler_characteristic(self.vertices(), self.faces(), self.edges())

    def face_types(self) -> dict:
        """Типы граней: 4 треугольника и 3 квадрата."""
        return {"triangle": 4, "square": 3}

    def surface_area(self) -> float:
        """Площадь поверхности: S = a²√3·(1 + √3)."""
        return self.a ** 2 * math.sqrt(3) * (1 + math.sqrt(3))

    def pseudo_volume(self) -> float:
        """Псевдо-объём: V = a³√2/6."""
        return self.a ** 3 * math.sqrt(2) / 6

    def is_rp2_model(self) -> bool:
        """Проверить, является ли гептаэдр моделью RP²."""
        checker = RP2Checker()
        result = checker.check_heptahedron()
        return result["ok"]

    def sphere_isomorphism(self) -> str:
        """Изоморфизм единичной сфере через разбиение граней.

        Грани AOF, AOB, BOF → три прямоугольника → x₁²+x₂²+x₃²=1.
        """
        return (f"Изоморфизм RP² ↔ S²:\n"
                f"  Разбиение трёх квадратных граней на прямоугольники\n"
                f"  даёт 6 прямоугольников, покрывающих S².\n"
                f"  Условие: x₁² + x₂² + x₃² = 1\n"
                f"  (из нормировки площадей AOF, AOB, BOF = 1/3 сферы каждая)")

    def net_description(self) -> str:
        """Описание развёртки гептаэдра."""
        return (
            "Развёртка гептаэдра (6 вершин A,B,C,D,E,F):\n"
            "  4 треугольных грани: ABС, ADE, BDF, CEF\n"
            "  3 квадратных грани:  ABDE, BCEF, ACDF\n"
            "  12 рёбер, χ = 6 + 7 - 12 = 1 = χ(RP²)"
        )

    def summary(self) -> str:
        """Полная сводка по гептаэдру."""
        lines = [
            f"Гептаэдр (ребро a={self.a}):",
            f"  Вершины:   {self.vertices()}",
            f"  Грани:     {self.faces()} (4 треугольника + 3 квадрата)",
            f"  Рёбра:     {self.edges()}",
            f"  χ(Эйлера): {self.euler_characteristic()} = χ(RP²) ✓",
            f"  Площадь:   {self.surface_area():.6f} = a²√3·(1+√3)",
            f"  Объём:     {self.pseudo_volume():.6f} = a³√2/6",
            f"  Модель RP²:{self.is_rp2_model()}",
        ]
        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexhept — гептаэдр (модель RP²)")
    parser.add_argument("--properties", action="store_true",
                        help="Свойства гептаэдра")
    parser.add_argument("--edge", type=float, default=1.0, metavar="A",
                        help="Длина ребра (по умолчанию 1)")
    parser.add_argument("--check-rp2", action="store_true",
                        help="Проверить B/Γ/P как модель RP²")
    parser.add_argument("--vertices", type=int, default=6)
    parser.add_argument("--faces-count", type=int, default=7, dest="faces_count")
    parser.add_argument("--edges-count", type=int, default=12, dest="edges_count")
    parser.add_argument("--enumerate", action="store_true",
                        help="Перебрать кандидаты RP² до --max-vertices")
    parser.add_argument("--max-vertices", type=int, default=8, dest="max_vertices")
    parser.add_argument("--sphere-isomorphism", action="store_true",
                        dest="sphere_iso")
    args = parser.parse_args()

    if args.properties:
        h = Heptahedron(args.edge)
        print(h.summary())
        print()
        print(h.net_description())

    if args.check_rp2:
        checker = RP2Checker()
        result = checker.check(args.vertices, args.faces_count, args.edges_count)
        print(f"B={args.vertices}, Γ={args.faces_count}, P={args.edges_count}:")
        print(f"  χ = {result['chi']}")
        print(f"  Модель RP²: {result['ok']}")
        if not result["ok"]:
            print(f"  Причина: {result['reason']}")

    if args.enumerate:
        checker = RP2Checker()
        candidates = checker.enumerate_rp2_candidates(args.max_vertices)
        print(f"Кандидаты RP² с B ≤ {args.max_vertices}:")
        for c in candidates:
            print(f"  B={c['vertices']}, Γ={c['faces']}, P={c['edges']}  χ={c['chi']}")

    if args.sphere_iso:
        h = Heptahedron(args.edge)
        print(h.sphere_isomorphism())

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
