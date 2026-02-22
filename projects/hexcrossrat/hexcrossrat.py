"""hexcrossrat.py — Группа сечений R6 (двойное отношение четырёх точек).

Источник: PDF «Замыкание сложного отношения» Франца Германа.

Двойное отношение 4 точек (A1, A2, A3, A4):
    w = (A1,A3)·(A2,A4) / ((A3,A2)·(A1,A4))

При всех перестановках 4 точек получаем ровно 6 значений — группу R6:
    r₀ = w
    r₁ = 1/w
    r₂ = 1 - w
    r₃ = 1/(1-w)
    r₄ = (w-1)/w = 1 - 1/w
    r₅ = w/(w-1) = 1/(1-1/w)

R6 изоморфна S3 (симметрической группе порядка 6).

Тождества:
    Σ r_i = 3  (для w ≠ 0, 1)
    r₀·r₁·r₂·r₃·r₄·r₅ = 1

Четырёхгруппа Клейна V4: 4 перестановки {A1,A2,A3,A4}, сохраняющие w.
S4 = V4 ⊗ R6  (V4 — нормальный делитель S4).
"""
import sys
import argparse

# ── вычисление двойного отношения ─────────────────────────────────────────────

def cross_ratio(A1, A2, A3, A4, projective: bool = False) -> complex:
    """Вычислить двойное отношение 4 точек.

    w = (A1-A3)·(A2-A4) / ((A3-A2)·(A1-A4))

    Для проективных точек (homogeneous 2D): A_i = (x, y).
    """
    if projective and hasattr(A1, "__len__"):
        # Для 2D однородных координат (x, y): точка = x/y
        def pt(p):
            if p[1] == 0:
                return float("inf")
            return p[0] / p[1]
        A1, A2, A3, A4 = pt(A1), pt(A2), pt(A3), pt(A4)

    try:
        num = (A1 - A3) * (A2 - A4)
        den = (A3 - A2) * (A1 - A4)
        if den == 0:
            raise ZeroDivisionError("Знаменатель двойного отношения равен 0")
        return num / den
    except TypeError:
        raise TypeError("A1..A4 должны быть числами (вещественными или комплексными)")


# ── группа R6 ─────────────────────────────────────────────────────────────────

class CrossRatioGroup:
    """Группа R6 из 6 значений двойного отношения при перестановках точек."""

    def __init__(self, w):
        """Инициализация группы по базовому значению w двойного отношения."""
        if w == 0 or w == 1:
            raise ValueError("w не может быть 0 или 1 (вырожденные случаи)")
        self.w = w
        self._elements = self._compute_elements(w)

    @staticmethod
    def _compute_elements(w) -> list:
        """Вычислить все 6 элементов группы R6."""
        return [
            w,
            1 / w,
            1 - w,
            1 / (1 - w),
            (w - 1) / w,
            w / (w - 1),
        ]

    def elements(self) -> list:
        """Все 6 элементов группы R6."""
        return list(self._elements)

    def __repr__(self):
        return f"CrossRatioGroup(w={self.w})"

    # ── операции группы ───────────────────────────────────────────────────────

    def _apply(self, func_idx: int, value):
        """Применить i-ю функцию R6 к значению."""
        w = value
        ops = [
            lambda x: x,
            lambda x: 1 / x,
            lambda x: 1 - x,
            lambda x: 1 / (1 - x),
            lambda x: (x - 1) / x,
            lambda x: x / (x - 1),
        ]
        return ops[func_idx](w)

    def _find_index(self, value, tol: float = 1e-9) -> int:
        """Найти индекс элемента в R6 по значению."""
        for i, elem in enumerate(self._elements):
            if abs(complex(elem) - complex(value)) < tol:
                return i
        return -1

    def multiply(self, i: int, j: int) -> int:
        """Произведение r_i ∘ r_j в R6: применить r_j, потом r_i.

        Возвращает индекс результирующего элемента.
        """
        # r_j(w) даёт значение, затем r_i применяется к нему
        val = self._apply(j, self.w)
        result = self._apply(i, val)
        return self._find_index(result)

    def compose(self, i: int, j: int):
        """Значение r_i(r_j(w))."""
        val = self._apply(j, self.w)
        return self._apply(i, val)

    # ── таблица Кэли ─────────────────────────────────────────────────────────

    def cayley_table(self) -> list[list[int]]:
        """Таблица Кэли R6: 6×6 матрица индексов."""
        table = []
        for i in range(6):
            row = []
            for j in range(6):
                row.append(self.multiply(i, j))
            table.append(row)
        return table

    def print_cayley_table(self) -> str:
        """Красивый ASCII-вывод таблицы Кэли."""
        table = self.cayley_table()
        names = [f"r{i}" for i in range(6)]
        header = "     " + "  ".join(f"{n:2s}" for n in names)
        lines = [f"Таблица Кэли R6 (w={self.w:.4f})", header,
                 "     " + "──" * 13]
        for i, row in enumerate(table):
            line = f"r{i} │ " + "  ".join(f"r{v}" for v in row)
            lines.append(line)
        return "\n".join(lines)

    # ── изоморфизм R6 ≅ S3 ───────────────────────────────────────────────────

    def isomorphism_to_s3(self) -> dict:
        """Биекция R6 → S3 (перестановки {0,1,2}).

        r0 = e  (тождественная)
        r1 → (01)(2) — инверсия
        r2 → (012) или транспозиция, в зависимости от порядка
        """
        s3_perms = [
            (0, 1, 2),   # e
            (1, 0, 2),   # (01)
            (0, 2, 1),   # (12)
            (2, 1, 0),   # (02)
            (1, 2, 0),   # (012)
            (2, 0, 1),   # (021)
        ]
        return {f"r{i}": s3_perms[i] for i in range(6)}

    # ── четырёхгруппа Клейна V4 ───────────────────────────────────────────────

    def klein_four_group(self) -> list[tuple]:
        """4 перестановки {A1,A2,A3,A4}, сохраняющие w.

        V4 = {e, (12)(34), (13)(24), (14)(23)} — нормальный делитель S4.
        """
        return [
            (1, 2, 3, 4),    # e
            (2, 1, 4, 3),    # (12)(34)
            (3, 4, 1, 2),    # (13)(24)
            (4, 3, 2, 1),    # (14)(23)
        ]

    # ── разложение S4 = V4 ⊗ R6 ───────────────────────────────────────────────

    def s4_decomposition(self) -> str:
        """Описание смежных классов S4 = V4 · R6."""
        lines = [
            "S4 = V4 ⊗ R6, |S4| = 4 · 6 = 24",
            "",
            "V4 (нормальный делитель): 4 перестановки",
            "  e, (12)(34), (13)(24), (14)(23)",
            "",
            "R6 (6 смежных классов по V4):",
            "  r0=w, r1=1/w, r2=1-w, r3=1/(1-w), r4=(w-1)/w, r5=w/(w-1)",
            "",
            "Каждый из 24 элементов S4 задаёт одно значение w среди r0..r5.",
            "Значение w делит на 4 равных элемента → 6 × 4 = 24 = |S4|.",
        ]
        return "\n".join(lines)

    # ── тождества ─────────────────────────────────────────────────────────────

    def verify_sum_identity(self, tol: float = 1e-9) -> dict:
        """Проверить Σr_i = 3."""
        total = sum(complex(e) for e in self._elements)
        return {"sum": total, "expected": 3, "ok": abs(total - 3) < tol}

    def verify_product_identity(self, tol: float = 1e-9) -> dict:
        """Проверить ∏r_i = 1."""
        prod = 1
        for e in self._elements:
            prod *= complex(e)
        return {"product": prod, "expected": 1, "ok": abs(prod - 1) < tol}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexcrossrat — группа двойных отношений R6")
    parser.add_argument("--cross-ratio", nargs=4, type=float,
                        metavar=("A1", "A2", "A3", "A4"),
                        help="Вычислить двойное отношение 4 точек")
    parser.add_argument("--r6-group", type=float, metavar="W",
                        help="Создать группу R6 для значения w")
    parser.add_argument("--cayley", type=float, metavar="W",
                        help="Таблица Кэли R6 для w")
    parser.add_argument("--isomorphism", type=float, metavar="W",
                        help="Изоморфизм R6 ≅ S3 для w")
    parser.add_argument("--identities", type=float, metavar="W",
                        help="Проверить тождества для w")
    parser.add_argument("--s4-decomp", type=float, metavar="W",
                        help="Разложение S4 = V4 ⊗ R6")
    args = parser.parse_args()

    if args.cross_ratio:
        A1, A2, A3, A4 = args.cross_ratio
        w = cross_ratio(A1, A2, A3, A4)
        print(f"Двойное отношение ({A1},{A2},{A3},{A4}) = {w:.10f}")

    if args.r6_group is not None:
        r6 = CrossRatioGroup(args.r6_group)
        elems = r6.elements()
        print(f"Группа R6 (w={args.r6_group}):")
        for i, e in enumerate(elems):
            print(f"  r{i} = {e:.8f}")

    if args.cayley is not None:
        r6 = CrossRatioGroup(args.cayley)
        print(r6.print_cayley_table())

    if args.isomorphism is not None:
        r6 = CrossRatioGroup(args.isomorphism)
        iso = r6.isomorphism_to_s3()
        print(f"Изоморфизм R6 → S3 (w={args.isomorphism}):")
        for k, v in iso.items():
            print(f"  {k} → {v}")

    if args.identities is not None:
        r6 = CrossRatioGroup(args.identities)
        s = r6.verify_sum_identity()
        p = r6.verify_product_identity()
        print(f"Тождества R6 (w={args.identities}):")
        print(f"  Σr_i = {s['sum']:.8f}  (ожид. 3, ok={s['ok']})")
        print(f"  ∏r_i = {p['product']:.8f}  (ожид. 1, ok={p['ok']})")

    if args.s4_decomp is not None:
        r6 = CrossRatioGroup(args.s4_decomp)
        print(r6.s4_decomposition())

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
