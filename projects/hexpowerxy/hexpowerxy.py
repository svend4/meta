"""hexpowerxy.py — Уравнение X^Y = Y^X (Первая теорема Германа).

Источник: PDF «Уравнение X^Y = Y^X» Франца Германа.

Параметрическое решение (параметр t = Y/X > 1):
    X(t) = t^(1/(t-1))
    Y(t) = t^(t/(t-1))

Симметрия: при замене t → 1/t переменные X и Y меняются местами.

Исключение: X = e — нет другого действительного решения Y ≠ X.

Связь с золотым сечением φ = (1+√5)/2:
    φ^(1/(φ-1)) = ... и «золотые уравнения» имеют корень x = φ.
"""
import math
import sys
import argparse

# Константы
_PHI = (1 + math.sqrt(5)) / 2    # золотое сечение
_E   = math.e


# ── основной класс ────────────────────────────────────────────────────────────

class PowerXY:
    """Исследование уравнения X^Y = Y^X."""

    # ── параметрическая кривая ────────────────────────────────────────────────

    @staticmethod
    def xy_from_t(t: float) -> tuple[float, float]:
        """Получить пару (X, Y) по параметру t > 1.

        X(t) = t^(1/(t-1)),  Y(t) = t^(t/(t-1)).
        """
        if t <= 1:
            raise ValueError(f"t должно быть > 1, получено {t}")
        exp_x = 1.0 / (t - 1)
        X = t ** exp_x
        Y = t ** (t * exp_x)
        return X, Y

    def generate_curve(self, t_min: float = 1.01, t_max: float = 10.0,
                       steps: int = 100) -> list[tuple[float, float]]:
        """Сгенерировать список (X, Y) пар вдоль параметрической кривой."""
        import linspace_like
        result = []
        step = (t_max - t_min) / (steps - 1)
        for i in range(steps):
            t = t_min + i * step
            try:
                result.append(self.xy_from_t(t))
            except (ValueError, OverflowError):
                pass
        return result

    def generate_curve(self, t_min: float = 1.01, t_max: float = 10.0,
                       steps: int = 100) -> list[tuple[float, float]]:
        """Сгенерировать список (X, Y) пар вдоль параметрической кривой."""
        result = []
        step_size = (t_max - t_min) / max(steps - 1, 1)
        for i in range(steps):
            t = t_min + i * step_size
            try:
                result.append(self.xy_from_t(t))
            except (ValueError, OverflowError):
                pass
        return result

    # ── для конкретного X найти Y ─────────────────────────────────────────────

    def find_y(self, X: float, tol: float = 1e-10) -> float | None:
        """Найти Y ≠ X такое, что X^Y = Y^X (численно).

        Возвращает None, если X близко к e (исключение).
        """
        if abs(X - _E) < 1e-6:
            return None  # исключение: X = e
        if X <= 1:
            return None

        # Параметрически: t = Y/X, X = t^(1/(t-1)) → найти t
        # Решаем f(t) = t^(1/(t-1)) - X = 0 методом бисекции
        def f(t):
            try:
                return t ** (1.0 / (t - 1)) - X
            except (OverflowError, ZeroDivisionError):
                return float("inf")

        # Ищем t > 1 (X > e: t ∈ (1, e), X < e: t > e... вообще ищем)
        # Биссекция на (1.0001, 100)
        lo, hi = 1.0001, 1000.0
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return None

        for _ in range(200):
            mid = (lo + hi) / 2
            fmid = f(mid)
            if abs(fmid) < tol:
                break
            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid

        t = (lo + hi) / 2
        X_t, Y_t = self.xy_from_t(t)
        return Y_t

    # ── проверка ─────────────────────────────────────────────────────────────

    @staticmethod
    def verify(X: float, Y: float, tol: float = 1e-9) -> bool:
        """Проверить X^Y ≈ Y^X."""
        try:
            lhs = X ** Y
            rhs = Y ** X
            return abs(lhs - rhs) < tol * max(abs(lhs), 1)
        except (OverflowError, ZeroDivisionError, ValueError):
            return False

    # ── золотые уравнения ────────────────────────────────────────────────────

    @staticmethod
    def golden_eq1(x: float) -> float:
        """Вычислить |LHS - RHS| для уравнения x^(1/(x-1)) = (1 + 1/x)^x.

        Корень при x = φ.
        """
        try:
            lhs = x ** (1.0 / (x - 1))
            rhs = (1 + 1.0 / x) ** x
            return abs(lhs - rhs)
        except (OverflowError, ZeroDivisionError, ValueError):
            return float("inf")

    @staticmethod
    def golden_eq2(x: float) -> float:
        """Вычислить |LHS - RHS| для уравнения x^(x/(x-1)) = (1 + 1/x)^(x+1).

        Корень при x = φ.
        """
        try:
            lhs = x ** (x / (x - 1))
            rhs = (1 + 1.0 / x) ** (x + 1)
            return abs(lhs - rhs)
        except (OverflowError, ZeroDivisionError, ValueError):
            return float("inf")

    def find_golden_root(self, tol: float = 1e-12) -> float:
        """Численно найти корень «золотого уравнения» (= φ)."""
        # Бисекция для golden_eq1 (≈ 0 в точке φ)
        # φ ≈ 1.618, ищем в (1.1, 3.0)
        lo, hi = 1.1, 3.0
        for _ in range(100):
            mid = (lo + hi) / 2
            # Хотим найти, где LHS - RHS меняет знак
            def signed(x):
                try:
                    return x ** (1.0 / (x - 1)) - (1 + 1.0 / x) ** x
                except Exception:
                    return float("nan")
            if signed(mid) * signed(lo) < 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    # ── анализ исключения ────────────────────────────────────────────────────

    @staticmethod
    def is_exception(X: float, tol: float = 1e-6) -> bool:
        """Проверить, является ли X = e исключением (нет другого Y ≠ X)."""
        return abs(X - _E) < tol

    @staticmethod
    def exception_constant() -> float:
        """Вернуть e как константу исключения."""
        return _E

    # ── таблица замечательных пар ─────────────────────────────────────────────

    def notable_pairs(self) -> list[tuple[float, float]]:
        """Список замечательных пар (X, Y) с X^Y = Y^X (только проверенные)."""
        pairs = [(2.0, 4.0)]   # 2^4 = 4^2 = 16
        # Параметрические пары (все гарантированно удовлетворяют уравнению)
        for t in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            try:
                x, y = self.xy_from_t(t)
                if self.verify(x, y, tol=1e-8):
                    pairs.append((round(x, 8), round(y, 8)))
            except Exception:
                pass
        return pairs

    # ── визуализация ─────────────────────────────────────────────────────────

    def plot_curve(self, t_min: float = 1.01, t_max: float = 6.0,
                   steps: int = 80, width: int = 60, ascii: bool = True) -> str:
        """ASCII-график кривой решений X^Y = Y^X."""
        curve = self.generate_curve(t_min, t_max, steps)
        if not curve:
            return "(нет точек)"

        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        height = 20

        rows = [[" "] * width for _ in range(height)]

        def col(x):
            return int((x - x_min) / (x_max - x_min + 1e-12) * (width - 1))

        def row(y):
            return height - 1 - int((y - y_min) / (y_max - y_min + 1e-12) * (height - 1))

        for x, y in curve:
            c, r = col(x), row(y)
            if 0 <= r < height and 0 <= c < width:
                rows[r][c] = "*"

        # Диагональ X = Y
        for x in [x_min + (x_max - x_min) * i / (width - 1) for i in range(width)]:
            if y_min <= x <= y_max:
                c = col(x)
                r = row(x)
                if 0 <= r < height and 0 <= c < width:
                    if rows[r][c] == " ":
                        rows[r][c] = "-"

        lines = [f"Y  (кривая X^Y = Y^X, t=[{t_min:.2f},{t_max:.2f}])"]
        for r in rows:
            lines.append("│" + "".join(r))
        lines.append("└" + "─" * width + "→ X")
        lines.append(f"  X=[{x_min:.2f},{x_max:.2f}]  Y=[{y_min:.2f},{y_max:.2f}]  * = кривая  - = X=Y")
        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexpowerxy — уравнение X^Y = Y^X (теория Германа)")
    parser.add_argument("--find-y", type=float, metavar="X",
                        help="Найти Y ≠ X, такое что X^Y = Y^X")
    parser.add_argument("--curve", action="store_true",
                        help="Показать параметрическую кривую (--steps)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--golden-root", action="store_true",
                        help="Найти корень золотого уравнения (= φ)")
    parser.add_argument("--plot", action="store_true",
                        help="ASCII-график кривой решений")
    parser.add_argument("--verify", nargs=2, type=float, metavar=("X", "Y"),
                        help="Проверить X^Y == Y^X")
    parser.add_argument("--notable", action="store_true",
                        help="Таблица замечательных пар")
    args = parser.parse_args()

    pxy = PowerXY()

    if args.find_y is not None:
        Y = pxy.find_y(args.find_y)
        if Y is None:
            print(f"X = {args.find_y} — исключение (X ≈ e), другого Y нет")
        else:
            ok = pxy.verify(args.find_y, Y)
            print(f"X = {args.find_y},  Y = {Y:.10f}  [X^Y=Y^X: {ok}]")

    if args.curve:
        curve = pxy.generate_curve(steps=args.steps)
        print(f"Параметрическая кривая ({len(curve)} точек):")
        for x, y in curve:
            ok = pxy.verify(x, y)
            print(f"  X={x:.6f}  Y={y:.6f}  ok={ok}")

    if args.golden_root:
        phi = pxy.find_golden_root()
        print(f"Корень золотого уравнения: x = {phi:.15f}")
        print(f"Золотое сечение φ:         φ = {_PHI:.15f}")
        print(f"Разница: {abs(phi - _PHI):.2e}")

    if args.plot:
        print(pxy.plot_curve())

    if args.verify:
        X, Y = args.verify
        ok = pxy.verify(X, Y)
        print(f"{X}^{Y} == {Y}^{X}: {ok}")

    if args.notable:
        print("Замечательные пары (X, Y) с X^Y = Y^X:")
        for x, y in pxy.notable_pairs():
            ok = pxy.verify(x, y)
            print(f"  ({x:.6f}, {y:.6f})  ok={ok}")

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
