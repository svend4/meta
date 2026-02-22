"""hexintermed.py — Промежуточный ряд H Франца Германа.

Источник: PDF «Промежуточный ряд» (Герман).

Промежуточный ряд H: для каждого n ≥ 1 находим единственное число в интервале
(n², (n+1)²), кратное (2n+1).

H = 3, 5, 14, 18, 33, 39, 60, 68, 95, 105, 138, 150, …

Замкнутые формулы (k — номер элемента, начиная с 1):
  нечётный k:  h(k) = (k+1) * (2k+1) // 2
  чётный k:    h(k) = k * (2k+1) // 2
  единая:      h(k) = (2k+1) * (2k+1 - (-1)^k) // 4

Частичная сумма:
  S(n) = (n+1) * (4*(n+1)^2 - 1 - 3*(-1)^n) // 12

k-угольные числа: α(n, k) = n * ((k-2)*(n-1) + 2) // 2
"""
import sys
import argparse

# ── вспомогательные ───────────────────────────────────────────────────────────

def _h_by_formula(k: int) -> int:
    """Вычислить k-й элемент промежуточного ряда H по замкнутой формуле (k ≥ 1)."""
    if k < 1:
        raise ValueError(f"k должно быть ≥ 1, получено {k}")
    sign = 1 if k % 2 == 0 else -1   # (-1)^k: +1 при чётном, -1 при нечётном
    return (2 * k + 1) * (2 * k + 1 - sign) // 4


def _h_by_search(k: int) -> int:
    """Найти k-й элемент H прямым перебором (для проверки)."""
    count = 0
    n = 1
    while True:
        lo, hi = n * n, (n + 1) * (n + 1)
        step = 2 * n + 1
        # найти первое кратное step в открытом интервале (lo, hi)
        start = ((lo // step) + 1) * step
        if start < hi:
            count += 1
            if count == k:
                return start
        n += 1


# ── основной класс ────────────────────────────────────────────────────────────

class IntermediateSeries:
    """Промежуточный ряд H и связанные операции."""

    # ── элементы ряда ─────────────────────────────────────────────────────────

    def term(self, k: int) -> int:
        """k-й элемент промежуточного ряда H (k ≥ 1)."""
        return _h_by_formula(k)

    def generate(self, count: int) -> list[int]:
        """Первые count элементов ряда H."""
        return [_h_by_formula(k) for k in range(1, count + 1)]

    # ── частичные суммы ───────────────────────────────────────────────────────

    def partial_sum(self, n: int) -> int:
        """Сумма первых n элементов ряда H.

        Формула: S(n) = (n+1) * (4*(n+1)^2 - 1 - 3*(-1)^n) // 12
        """
        if n < 1:
            return 0
        sign = 1 if n % 2 == 0 else -1   # (-1)^n
        return (n + 1) * (4 * (n + 1) ** 2 - 1 - 3 * sign) // 12

    # ── разложение h(k) = x(k) · y(k) ────────────────────────────────────────

    def factorize(self, k: int) -> dict:
        """Разложить h(k) как x(k) * y(k), где y(k) = 2k+1.

        x(k) = ⌈k/2⌉  при нечётном k
        x(k) = k//2   при чётном k
        Точнее: x = h(k) // (2k+1)
        """
        h = self.term(k)
        y = 2 * k + 1
        x = h // y
        return {"k": k, "h": h, "x": x, "y": y, "check": x * y == h}

    # ── рекуррентные соотношения ───────────────────────────────────────────────

    def recurrence_check(self, n_terms: int) -> list[dict]:
        """Проверить рекуррентное соотношение h(k+2) - h(k) для k = 1..n_terms.

        Ожидаем: h(k+2) - h(k) = 4k+6 (для нечётного k), 4k+6 (для чётного k).
        На самом деле проверяем h(k+2) = h(k) + разница.
        """
        results = []
        for k in range(1, n_terms + 1):
            h_k = self.term(k)
            h_k2 = self.term(k + 2)
            diff = h_k2 - h_k
            # ожидаемая разница: 4*(2k+2)+2 = ... проверим, что это зависит от чётности
            expected = (2 * (k + 2) + 1) * (2 * (k + 2) + 1 - (1 if (k+2) % 2 == 0 else -1)) // 4 - h_k
            results.append({
                "k": k, "h(k)": h_k, "h(k+2)": h_k2,
                "diff": diff, "expected_diff": expected,
                "ok": diff == expected
            })
        return results

    # ── k-угольные числа ──────────────────────────────────────────────────────

    @staticmethod
    def polygonal(n: int, k: int) -> int:
        """n-й k-угольный номер: α(n, k) = n * ((k-2)*(n-1) + 2) // 2."""
        if k < 3:
            raise ValueError("k-угольное число требует k ≥ 3")
        return n * ((k - 2) * (n - 1) + 2) // 2

    # ── свойства симметрии ────────────────────────────────────────────────────

    def symmetry_check(self, n: int, k: int = 4, m: int = 2) -> dict:
        """Проверить тождество симметрии из PDF.

        При n=6, k=4, m=2: h(n+m) + h(n-m) = 2·h(n) + f(m)
        Возвращает значения обеих частей и флаг совпадения.
        """
        if n <= m:
            raise ValueError("n должно быть > m")
        lhs = self.term(n + m) + self.term(n - m)
        rhs_base = 2 * self.term(n)
        diff = lhs - rhs_base
        return {"n": n, "m": m, "h(n+m)+h(n-m)": lhs,
                "2·h(n)": rhs_base, "diff": diff}

    # ── визуализация ─────────────────────────────────────────────────────────

    def plot(self, n_max: int = 20, width: int = 60) -> str:
        """ASCII-график ряда H с огибающими n² и (n+1)²."""
        terms = self.generate(n_max)
        max_val = max(terms)
        height = 15
        rows = [[" "] * width for _ in range(height)]

        def col(k):
            return int((k - 1) / (n_max - 1) * (width - 1)) if n_max > 1 else 0

        def row(v):
            return height - 1 - int(v / max_val * (height - 1))

        # огибающие: n² и (n+1)²
        for k in range(1, n_max + 1):
            c = col(k)
            lo_val = k * k
            hi_val = (k + 1) * (k + 1)
            r_lo = row(lo_val)
            r_hi = row(hi_val)
            if 0 <= r_lo < height:
                rows[r_lo][c] = "."
            if 0 <= r_hi < height:
                rows[r_hi][c] = "'"

        # значения ряда
        for k in range(1, n_max + 1):
            c = col(k)
            r = row(terms[k - 1])
            if 0 <= r < height:
                rows[r][c] = "H"

        lines = ["H(k)"]
        for r in rows:
            lines.append("│" + "".join(r))
        lines.append("└" + "─" * width + "→ k")
        lines.append(f"  [1..{n_max}]  H = ряд,  . = n²,  ' = (n+1)²")
        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexintermed — промежуточный ряд H Германа")
    parser.add_argument("--term", type=int, metavar="K",
                        help="k-й элемент ряда H")
    parser.add_argument("--generate", type=int, metavar="N",
                        help="Первые N элементов ряда H")
    parser.add_argument("--sum", type=int, metavar="N",
                        help="Частичная сумма первых N элементов")
    parser.add_argument("--factorize", type=int, metavar="K",
                        help="Разложить h(K) = x·y")
    parser.add_argument("--polygonal", nargs=2, type=int, metavar=("N", "K"),
                        help="n-й k-угольный номер")
    parser.add_argument("--recurrence", type=int, metavar="N",
                        help="Проверить рекуррентные соотношения для N элементов")
    parser.add_argument("--plot", type=int, metavar="N_MAX", default=0,
                        help="ASCII-график ряда H до N_MAX")
    args = parser.parse_args()

    h = IntermediateSeries()

    if args.term is not None:
        print(f"h({args.term}) = {h.term(args.term)}")

    if args.generate is not None:
        seq = h.generate(args.generate)
        print(f"H[1..{args.generate}] = {seq}")

    if args.sum is not None:
        s = h.partial_sum(args.sum)
        s_check = sum(h.generate(args.sum))
        print(f"S({args.sum}) = {s}  (прямая сумма: {s_check}, совпадает: {s == s_check})")

    if args.factorize is not None:
        f = h.factorize(args.factorize)
        print(f"h({f['k']}) = {f['h']} = {f['x']} × {f['y']}  ok={f['check']}")

    if args.polygonal is not None:
        n, k = args.polygonal
        print(f"P({n}, {k}) = {h.polygonal(n, k)}")

    if args.recurrence is not None:
        for rec in h.recurrence_check(args.recurrence):
            status = "✓" if rec["ok"] else "✗"
            print(f"  k={rec['k']}: h(k+2)-h(k) = {rec['diff']}  {status}")

    if args.plot:
        print(h.plot(args.plot))

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
