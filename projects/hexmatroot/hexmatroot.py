"""hexmatroot.py — Нелинейная теория матриц (аналитические корни 2×2 матриц).

Источник: PDF «Нелинейная теория матриц» Франца Германа.

Формулы для матрицы A = [[a, b], [c, d]], det(A) = D = ad - bc:

1. Оператор инверсии Aₒ: A · Aₒ = A⁻¹
   Aₒ = (A⁻¹)² = A⁻²

2. Квадратный корень (два сопряжённых корня):
   √A₁ = +(A + E·√D) / √(tr(A) + 2√D)
   √A₂ = -(A - E·√D) / √(tr(A) - 2√D)

3. Идемпотентность: A² = A ⟺ det(A) = 0 И tr(A) = 1

4. Матричное уравнение X² + bX + C = 0 (b — скаляр):
   X = -(b/2)·E ± √((b/2)²·E - C)
"""
import math
import sys
import argparse


# ── 2×2 матрица как список списков ────────────────────────────────────────────

def _mat(rows) -> list[list[float]]:
    """Создать 2×2 матрицу из rows = [[a,b],[c,d]]."""
    return [[float(rows[0][0]), float(rows[0][1])],
            [float(rows[1][0]), float(rows[1][1])]]


def _det(A) -> float:
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def _tr(A) -> float:
    return A[0][0] + A[1][1]


def _add(A, B):
    return [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]


def _sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(2)] for i in range(2)]


def _mul_mat(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)]


def _scale(s, A):
    return [[s * A[i][j] for j in range(2)] for i in range(2)]


def _eye() -> list[list[float]]:
    return [[1.0, 0.0], [0.0, 1.0]]


def _inv(A) -> list[list[float]]:
    D = _det(A)
    if abs(D) < 1e-14:
        raise ValueError("Матрица вырождена (det = 0)")
    return [[A[1][1] / D, -A[0][1] / D],
            [-A[1][0] / D, A[0][0] / D]]


def _mat_eq(A, B, tol: float = 1e-9) -> bool:
    return all(abs(A[i][j] - B[i][j]) < tol for i in range(2) for j in range(2))


def _mat_str(A, name: str = "A") -> str:
    return (f"{name} = [[{A[0][0]:.6f}, {A[0][1]:.6f}],\n"
            f"     [{A[1][0]:.6f}, {A[1][1]:.6f}]]")


# ── основной класс ────────────────────────────────────────────────────────────

class MatrixAlgebra:
    """Нелинейная алгебра 2×2 матриц по Францу Герману."""

    # ── оператор инверсии ─────────────────────────────────────────────────────

    def inversion_operator(self, A) -> list[list[float]]:
        """Вычислить оператор инверсии Aₒ = (A⁻¹)².

        Свойства: A · Aₒ = A⁻¹,  Aₒ⁻¹ = A².
        """
        inv = _inv(A)
        return _mul_mat(inv, inv)

    def verify_inversion_operator(self, A, tol: float = 1e-9) -> dict:
        """Проверить: A · Aₒ == A⁻¹."""
        Ao = self.inversion_operator(A)
        inv = _inv(A)
        product = _mul_mat(A, Ao)
        return {
            "A": A,
            "Ao": Ao,
            "A_inv": inv,
            "A_times_Ao": product,
            "ok": _mat_eq(product, inv, tol),
        }

    # ── квадратный корень матрицы ─────────────────────────────────────────────

    def sqrt_matrix(self, A, tol: float = 1e-12) -> list[list[list[float]]]:
        """Вычислить два аналитических квадратных корня 2×2 матрицы A.

        Формула: √Aₖ = ±(A ± E·√D) / √(tr(A) ± 2√D)
        Возвращает список [√A₁, √A₂] (те, что существуют).
        """
        D = _det(A)
        tr = _tr(A)
        E = _eye()

        if D < 0:
            # Комплексный det: вернём приближённое вещественное (None)
            raise ValueError(f"det(A) = {D:.6f} < 0, вещественный √A не существует")

        sqD = math.sqrt(D)
        roots = []

        for sign in [1.0, -1.0]:
            denom_sq = tr + sign * 2 * sqD
            if abs(denom_sq) < tol:
                continue
            if denom_sq < 0:
                continue
            denom = math.sqrt(denom_sq)
            core = _add(A, _scale(sign * sqD, E))
            root = _scale(1.0 / denom, core)
            # Перед добавлением проверим: root^2 ≈ A
            if _mat_eq(_mul_mat(root, root), A, tol=1e-6):
                roots.append(root)

        return roots

    def has_sqrt(self, A) -> bool:
        """Проверить, существует ли вещественный квадратный корень A."""
        try:
            return len(self.sqrt_matrix(A)) > 0
        except ValueError:
            return False

    # ── идемпотентность ───────────────────────────────────────────────────────

    def is_idempotent(self, A, tol: float = 1e-9) -> bool:
        """Проверить A² = A (идемпотентность).

        Условие: det(A) ≈ 0  И  tr(A) ≈ 1.
        """
        return abs(_det(A)) < tol and abs(_tr(A) - 1) < tol

    def is_idempotent_verify(self, A, tol: float = 1e-9) -> dict:
        """Подробная проверка A² = A."""
        A2 = _mul_mat(A, A)
        return {
            "A2": A2,
            "A": A,
            "A2_eq_A": _mat_eq(A2, A, tol),
            "det": _det(A),
            "tr": _tr(A),
            "condition_det_zero": abs(_det(A)) < tol,
            "condition_tr_one": abs(_tr(A) - 1) < tol,
        }

    # ── матрицы Паули ─────────────────────────────────────────────────────────

    @staticmethod
    def pauli(k: int) -> list[list[float]]:
        """k-я матрица Паули (k = 1, 2, 3).

        S¹ = [[0,1],[1,0]],  S² = [[0,-1],[1,0]],  S³ = [[1,0],[0,-1]]
        """
        matrices = {
            1: [[0.0, 1.0], [1.0, 0.0]],
            2: [[0.0, -1.0], [1.0, 0.0]],
            3: [[1.0, 0.0], [0.0, -1.0]],
        }
        if k not in matrices:
            raise ValueError(f"k должен быть 1, 2 или 3")
        return matrices[k]

    def cyclic_group_of_4(self, S: list[list[float]]) -> dict:
        """Построить циклическую группу {E, √S, S, √S^3} для матрицы S.

        Возвращает словарь: {'E': ..., 'sqrt_S': ..., 'S': ..., 'S3': ...}
        """
        E = _eye()
        roots = self.sqrt_matrix(S)
        if not roots:
            raise ValueError("√S не существует в вещественных числах")
        sqrt_S = roots[0]
        S2 = _mul_mat(S, S)
        S3 = _mul_mat(S2, S)
        return {"E": E, "sqrt_S": sqrt_S, "S": S, "S3": S3}

    # ── решение X² + bX + C = 0 ───────────────────────────────────────────────

    def solve_quadratic(self, b: float, C: list[list[float]]) -> list[list[list[float]]]:
        """Решить X² + bX + C = 0 (b — скаляр, C — 2×2 матрица).

        X = -(b/2)E ± √((b/2)²E - C)
        """
        E = _eye()
        half_b = b / 2.0
        b2_E = _scale(half_b ** 2, E)
        discriminant = _sub(b2_E, C)
        roots_of_disc = self.sqrt_matrix(discriminant)
        shift = _scale(-half_b, E)
        solutions = []
        for r in roots_of_disc:
            solutions.append(_add(shift, r))
            solutions.append(_sub(shift, r))
        return solutions

    # ── семейство корней единицы ───────────────────────────────────────────────

    def identity_sqrt_family(self, k_values: list[int]) -> list[list[list[float]]]:
        """Семейство корней E: матрицы X_k такие что X_k² = E.

        X_k = [[cos(kπ), sin(kπ)], [-sin(kπ), cos(kπ)]] (ротация на kπ)
        Для k=0: E; k=1: -E; k=1/2: ротация π/2
        """
        result = []
        for k in k_values:
            angle = k * math.pi
            c, s = math.cos(angle), math.sin(angle)
            Xk = [[c, s], [-s, c]]
            result.append(Xk)
        return result

    # ── декомпозиция матрицы по корням ────────────────────────────────────────

    def decompose_by_roots(self, A) -> tuple:
        """Теорема: A = k₁·√A₁ + k₂·√A₂ для подходящих k₁, k₂.

        Возвращает (k₁, k₂) такие что k₁·√A₁ + k₂·√A₂ = A.
        """
        roots = self.sqrt_matrix(A)
        if len(roots) < 2:
            raise ValueError("Нет двух корней для декомпозиции")
        r1, r2 = roots[0], roots[1]
        # Из системы: k1*r1 + k2*r2 = A
        # Для 2×2: решаем систему по компоненте [0,0] и [0,1]
        # r1[0][0]*k1 + r2[0][0]*k2 = A[0][0]
        # r1[0][1]*k1 + r2[0][1]*k2 = A[0][1]
        a11, a12 = r1[0][0], r2[0][0]
        b1 = A[0][0]
        a21, a22 = r1[0][1], r2[0][1]
        b2 = A[0][1]
        det_sys = a11 * a22 - a12 * a21
        if abs(det_sys) < 1e-12:
            return (1.0, 0.0)   # вырожденный случай
        k1 = (b1 * a22 - b2 * a12) / det_sys
        k2 = (a11 * b2 - a21 * b1) / det_sys
        return (k1, k2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_matrix(s: str) -> list[list[float]]:
    """Разобрать строку '[[a,b],[c,d]]' в 2×2 матрицу."""
    s = s.replace("[", "").replace("]", "")
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError("Матрица должна быть 2×2: '[[a,b],[c,d]]'")
    return [[vals[0], vals[1]], [vals[2], vals[3]]]


def _main():
    parser = argparse.ArgumentParser(
        description="hexmatroot — аналитические корни 2×2 матриц (теория Германа)")
    parser.add_argument("--sqrt", type=str, metavar="MATRIX",
                        help="Квадратные корни матрицы 2×2 (напр. '[[3,2],[4,3]]')")
    parser.add_argument("--inversion-op", type=str, metavar="MATRIX",
                        help="Оператор инверсии Aₒ")
    parser.add_argument("--idempotent-check", type=str, metavar="MATRIX",
                        help="Проверить идемпотентность (A²=A)")
    parser.add_argument("--solve-quadratic", action="store_true",
                        help="Решить X² + bX + C = 0")
    parser.add_argument("--b", type=float, default=2.0,
                        help="Коэффициент b в уравнении")
    parser.add_argument("--C", type=str, default="[[1,0],[0,1]]",
                        help="Матрица C в уравнении")
    parser.add_argument("--pauli-cyclic", type=int, metavar="K",
                        help="Циклическая группа порядка 4 из матрицы Паули K")
    args = parser.parse_args()

    ma = MatrixAlgebra()

    if args.sqrt is not None:
        A = _parse_matrix(args.sqrt)
        try:
            roots = ma.sqrt_matrix(A)
            print(f"Квадратные корни матрицы A:")
            print(_mat_str(A))
            for i, r in enumerate(roots):
                print(f"\n√A{i + 1}:")
                print(_mat_str(r, f"√A{i + 1}"))
                A2 = _mul_mat(r, r)
                print(f"  √A{i + 1}² ≈ A: {_mat_eq(A2, A, tol=1e-6)}")
        except ValueError as e:
            print(f"Нет вещественного корня: {e}")

    if args.inversion_op is not None:
        A = _parse_matrix(args.inversion_op)
        res = ma.verify_inversion_operator(A)
        print("Оператор инверсии:")
        print(_mat_str(A))
        print(_mat_str(res["Ao"], "Aₒ"))
        print(f"A · Aₒ == A⁻¹: {res['ok']}")

    if args.idempotent_check is not None:
        A = _parse_matrix(args.idempotent_check)
        res = ma.is_idempotent_verify(A)
        print(f"Идемпотентность A² = A: {res['A2_eq_A']}")
        print(f"  det(A) = {res['det']:.6f}  (≈0: {res['condition_det_zero']})")
        print(f"  tr(A)  = {res['tr']:.6f}  (≈1: {res['condition_tr_one']})")

    if args.solve_quadratic:
        C = _parse_matrix(args.C)
        solutions = ma.solve_quadratic(args.b, C)
        print(f"Решения X² + {args.b}X + C = 0:")
        for i, X in enumerate(solutions):
            print(_mat_str(X, f"X{i + 1}"))

    if args.pauli_cyclic is not None:
        S = MatrixAlgebra.pauli(args.pauli_cyclic)
        try:
            group = ma.cyclic_group_of_4(S)
            print(f"Циклическая группа {{{', '.join(group.keys())}}}:")
            for name, mat in group.items():
                print(f"\n{name}:")
                print(_mat_str(mat, name))
        except ValueError as e:
            print(f"Ошибка: {e}")

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
