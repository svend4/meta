"""hexpolyenum.py — Перебор многогранников и экс-додекаэдр.

Источники:
  PDF `677` — «Формула Эйлера и топология многогранников»
  PDF `87c` — «Экс-додекаэдр» (Герман)

Правильные сферические многогранники (χ=2, C_B — степень вершины, C_Γ — степень грани):
  B = 4·C_Γ / (2(C_B+C_Γ) - C_B·C_Γ)
  Γ = 4·C_B / (2(C_B+C_Γ) - C_B·C_Γ)
  P = 2·C_B·C_Γ / (2(C_B+C_Γ) - C_B·C_Γ)

Тороидальные (χ=0): 2(C_B+C_Γ) = C_B·C_Γ
  Решения: (3,6), (4,4), (6,3)

Экс-додекаэдр:
  B=32, Γ=24, P=54, χ=2
  Объём: V = (a³/2)·(4 + 3φ),  φ = золотое сечение
  Число диагоналей: D = B(B-1)/2 - P
"""
import math
import sys
import argparse

_PHI = (1 + math.sqrt(5)) / 2


# ── перебор правильных многогранников ────────────────────────────────────────

class PolyhedronRecord:
    """Описание правильного или квазиправильного многогранника."""

    def __init__(self, name: str, cb: int, cg: int,
                 vertices: int, faces: int, edges: int):
        self.name = name
        self.cb = cb       # степень вершины (число граней вокруг вершины)
        self.cg = cg       # степень грани (число сторон грани)
        self.vertices = vertices
        self.faces = faces
        self.edges = edges

    def euler(self) -> int:
        return self.vertices + self.faces - self.edges

    def diagonal_count(self) -> int:
        """Число диагоналей: D = B(B-1)/2 - P."""
        return self.vertices * (self.vertices - 1) // 2 - self.edges

    def __repr__(self):
        return (f"{self.name}: B={self.vertices}, Γ={self.faces}, "
                f"P={self.edges}, χ={self.euler()}, "
                f"(C_B={self.cb}, C_Γ={self.cg})")


class PolyhedronEnumerator:
    """Перебор правильных многогранников по формуле Эйлера."""

    def enumerate_spherical(self, max_degree: int = 10) -> list[PolyhedronRecord]:
        """Перебрать правильные сферические многогранники (χ=2, C_B,C_Γ ≥ 3).

        Формулы: B = 4·C_Γ/D, Γ = 4·C_B/D, P = 2·C_B·C_Γ/D,
        где D = 2(C_B+C_Γ) - C_B·C_Γ
        """
        results = []
        for cb in range(3, max_degree + 1):
            for cg in range(3, max_degree + 1):
                D = 2 * (cb + cg) - cb * cg
                if D <= 0:
                    continue
                if (4 * cg) % D != 0 or (4 * cb) % D != 0:
                    continue
                B = 4 * cg // D
                Gamma = 4 * cb // D
                P = 2 * cb * cg // D
                if B < 4 or Gamma < 4 or P < 6:
                    continue
                chi = B + Gamma - P
                if chi != 2:
                    continue
                name = _platonic_name(cb, cg)
                rec = PolyhedronRecord(name, cb, cg, B, Gamma, P)
                if not any(r.vertices == B and r.faces == Gamma for r in results):
                    results.append(rec)
        return sorted(results, key=lambda r: r.edges)

    def enumerate_toroidal(self) -> list[dict]:
        """Тороидальные многогранники (χ=0): 2(C_B+C_Γ) = C_B·C_Γ."""
        results = []
        for cb in range(3, 10):
            for cg in range(3, 10):
                if 2 * (cb + cg) == cb * cg:
                    results.append({"cb": cb, "cg": cg,
                                    "type": f"({cb},{cg})-tiling"})
        return results

    def from_degrees(self, face_degree: int, vertex_degree: int) -> PolyhedronRecord | None:
        """Получить многогранник по (C_Γ, C_B)."""
        recs = self.enumerate_spherical()
        for r in recs:
            if r.cg == face_degree and r.cb == vertex_degree:
                return r
        return None

    def check_euler(self, B: int, Gamma: int, P: int) -> dict:
        """Проверить формулу Эйлера."""
        chi = B + Gamma - P
        return {"B": B, "Gamma": Gamma, "P": P, "chi": chi,
                "spherical": chi == 2, "toroidal": chi == 0, "rp2": chi == 1}

    def diagonal_count(self, B: int, Gamma: int, P: int) -> int:
        """Число диагоналей: D = B(B-1)/2 - P."""
        return B * (B - 1) // 2 - P

    def compare_table(self) -> str:
        """ASCII-таблица правильных многогранников."""
        recs = self.enumerate_spherical()
        lines = [
            f"{'Имя':<20} {'C_B':>4} {'C_Γ':>4} {'B':>5} {'Γ':>5} "
            f"{'P':>5} {'χ':>3} {'Диаг':>6}",
            "─" * 60,
        ]
        for r in recs:
            lines.append(
                f"{r.name:<20} {r.cb:>4} {r.cg:>4} {r.vertices:>5} "
                f"{r.faces:>5} {r.edges:>5} {r.euler():>3} {r.diagonal_count():>6}"
            )
        return "\n".join(lines)


def _platonic_name(cb: int, cg: int) -> str:
    names = {
        (3, 3): "Тетраэдр",
        (3, 4): "Куб",
        (4, 3): "Октаэдр",
        (3, 5): "Додекаэдр",
        (5, 3): "Икосаэдр",
    }
    return names.get((cb, cg), f"({cb},{cg})-многогранник")


# ── экс-додекаэдр ─────────────────────────────────────────────────────────────

class ExDodecahedron:
    """Экс-додекаэдр Германа: B=32, Γ=24, P=54.

    Конструкция: додекаэдр → вырезать 4 ромба → вогнуть вершины.
    """

    def __init__(self, a: float = 1.0):
        self.a = a
        self._B = 32
        self._Gamma = 24
        self._P = 54

    def vertices(self) -> int:
        return self._B

    def faces(self) -> int:
        return self._Gamma

    def edges(self) -> int:
        return self._P

    def euler(self) -> int:
        return self._B + self._Gamma - self._P

    def volume(self) -> float:
        """V = (a³/2)·(4 + 3φ)."""
        return (self.a ** 3 / 2) * (4 + 3 * _PHI)

    def diagonal_count(self) -> int:
        """D = B(B-1)/2 - P."""
        return self._B * (self._B - 1) // 2 - self._P

    def construction_steps(self) -> list[str]:
        """Пошаговый алгоритм построения из додекаэдра."""
        return [
            "1. Взять правильный додекаэдр (12 пятиугольных граней).",
            "2. Каждую пятиугольную грань разбить: вставить диагональ.",
            "3. В 4 симметричных гранях вырезать ромб (параллелограмм).",
            "4. Сдвинуть вогнутые вершины внутрь (вогнуть).",
            "5. Получить 32 вершины, 24 грани, 54 ребра, χ = 2.",
        ]

    def face_types(self) -> dict:
        """Типы граней экс-додекаэдра."""
        return {"triangle": 8, "rhombus": 12, "pentagon": 4}

    def summary(self) -> str:
        lines = [
            f"Экс-додекаэдр (ребро a={self.a}):",
            f"  Вершины:    {self.vertices()}",
            f"  Грани:      {self.faces()} ({self.face_types()})",
            f"  Рёбра:      {self.edges()}",
            f"  χ(Эйлера):  {self.euler()} = 2 ✓",
            f"  Объём:      {self.volume():.6f} = (a³/2)(4+3φ)",
            f"  Диагоналей: {self.diagonal_count()}",
        ]
        return "\n".join(lines)

    def to_obj(self, filename: str) -> None:
        """Заглушка: экспорт в OBJ (требует реальной 3D-параметризации)."""
        # В полной реализации здесь были бы точные 3D-координаты
        with open(filename, "w") as f:
            f.write("# ExDodecahedron OBJ placeholder\n")
            f.write(f"# a={self.a}, B={self._B}, Gamma={self._Gamma}, P={self._P}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexpolyenum — перебор многогранников + экс-додекаэдр")
    parser.add_argument("--spherical", action="store_true",
                        help="Перечислить правильные сферические многогранники")
    parser.add_argument("--toroidal", action="store_true",
                        help="Тороидальные многогранники (χ=0)")
    parser.add_argument("--from-degrees", nargs=2, type=int, metavar=("CG", "CB"),
                        help="Многогранник по (C_Γ, C_B)")
    parser.add_argument("--diagonals", nargs=3, type=int, metavar=("B", "G", "P"),
                        help="Число диагоналей многогранника B Γ P")
    parser.add_argument("--ex-dodecahedron", action="store_true",
                        help="Свойства экс-додекаэдра")
    parser.add_argument("--edge", type=float, default=1.0,
                        help="Длина ребра")
    parser.add_argument("--compare-table", action="store_true",
                        help="Сравнительная таблица многогранников")
    args = parser.parse_args()

    pe = PolyhedronEnumerator()

    if args.spherical:
        recs = pe.enumerate_spherical()
        print("Правильные сферические многогранники:")
        for r in recs:
            print(f"  {r}")

    if args.toroidal:
        toroidal = pe.enumerate_toroidal()
        print("Тороидальные плитки (χ=0):")
        for t in toroidal:
            print(f"  C_B={t['cb']}, C_Γ={t['cg']} → {t['type']}")

    if args.from_degrees is not None:
        cg, cb = args.from_degrees
        rec = pe.from_degrees(cg, cb)
        if rec:
            print(rec)
        else:
            print(f"Правильного многогранника ({cb},{cg}) не найдено")

    if args.diagonals is not None:
        B, G, P = args.diagonals
        d = pe.diagonal_count(B, G, P)
        e = pe.check_euler(B, G, P)
        print(f"B={B}, Γ={G}, P={P}: χ={e['chi']}, диагоналей={d}")

    if args.ex_dodecahedron:
        ed = ExDodecahedron(args.edge)
        print(ed.summary())
        print("\nШаги конструкции:")
        for s in ed.construction_steps():
            print(f"  {s}")

    if args.compare_table:
        print(pe.compare_table())

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
