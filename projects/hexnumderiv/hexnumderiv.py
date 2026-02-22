"""hexnumderiv.py — Производная числа (дифференциальная/генетическая теория чисел).

Источник: PDF «Дифференциальная (генетическая) теория чисел» Франца Германа.

Оператор дифференцирования числа n:
    ∂n = 1 + сумма всех собственных делителей n  (делители строго меньше n)
    ∂1 = 1
    ∂P = 2  (P — простое)

Классы чисел:
  - совершенные (perfect): ∂n = n
  - супер-числа  (super):  цепочка ∂-производных растёт
  - обычные     (ordinary): цепочка убывает до 1

Огибающие кривые в координатах (N, ∂N):
    верхняя:  Y = 2√X + 1
    нижняя:   Y =  √X + 1  (достигается на точных квадратах)
"""
import math
import sys
import argparse

# ── вспомогательные функции ───────────────────────────────────────────────────

def _proper_divisors(n: int) -> list[int]:
    """Возвращает список всех собственных делителей n (делители < n, включая 1)."""
    if n <= 1:
        return []
    divs = [1]
    i = 2
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)


# ── основной класс ────────────────────────────────────────────────────────────

class NumberDerivative:
    """Оператор производной числа ∂ по Францу Герману."""

    # ── вычисление ∂n ─────────────────────────────────────────────────────────

    def derivative(self, n: int) -> int:
        """Вычислить производную числа n: ∂n = 1 + Σ{собственных делителей n}.

        Особые случаи: ∂1 = 1, ∂p = 2 (простое).
        """
        if n < 1:
            raise ValueError(f"n должно быть ≥ 1, получено {n}")
        if n == 1:
            return 1
        divs = _proper_divisors(n)
        return 1 + sum(divs)

    # ── цепочка производных ───────────────────────────────────────────────────

    def chain(self, n: int, max_steps: int = 200) -> list[int]:
        """Построить цепочку n → ∂n → ∂(∂n) → … до 1 или цикла.

        Возвращает список включая начальный элемент n.
        Для супер-чисел обрывается через max_steps шагов.
        """
        if n < 1:
            raise ValueError(f"n должно быть ≥ 1")
        result = [n]
        seen = {n}
        current = n
        for _ in range(max_steps):
            nxt = self.derivative(current)
            result.append(nxt)
            if nxt == 1 or nxt in seen:
                break
            seen.add(nxt)
            current = nxt
        return result

    # ── классификация ─────────────────────────────────────────────────────────

    def classify(self, n: int, max_steps: int = 200) -> str:
        """Классифицировать число n: "perfect", "super" или "ordinary".

        perfect:  ∂n = n (неподвижная точка)
        super:    цепочка содержит значение строго больше n (растёт)
        ordinary: цепочка убывает (все значения ≤ n, не растёт)
        """
        if n < 1:
            raise ValueError(f"n должно быть ≥ 1")
        dn = self.derivative(n)
        if dn == n:
            return "perfect"
        ch = self.chain(n, max_steps)
        # Супер-число: цепочка выходит за пределы n
        if any(v > n for v in ch[1:]):
            return "super"
        return "ordinary"

    # ── специальные списки ────────────────────────────────────────────────────

    def perfect_numbers(self, limit: int) -> list[int]:
        """Найти все совершенные числа n ≤ limit, т.е. ∂n = n."""
        return [n for n in range(1, limit + 1) if self.derivative(n) == n]

    def super_numbers(self, limit: int, max_steps: int = 200) -> list[int]:
        """Найти все супер-числа n ≤ limit."""
        return [n for n in range(2, limit + 1)
                if self.classify(n, max_steps) == "super"]

    # ── правило Лейбница ──────────────────────────────────────────────────────

    def leibniz_rule(self, k: int, m: int) -> dict:
        """Проверить аналог правила Лейбница для взаимно простых k и m.

        ∂(k·m) = k·∂m + m·∂k + ∂k·∂m
        Возвращает словарь с ∂(km), правой частью и флагом совпадения.
        """
        if math.gcd(k, m) != 1:
            raise ValueError(f"k={k} и m={m} не взаимно просты (gcd={math.gcd(k, m)})")
        dk = self.derivative(k)
        dm = self.derivative(m)
        lhs = self.derivative(k * m)
        rhs = k * dm + m * dk + dk * dm
        return {"km": k * m, "dk": dk, "dm": dm,
                "lhs": lhs, "rhs": rhs, "holds": lhs == rhs}

    # ── огибающие кривые ──────────────────────────────────────────────────────

    @staticmethod
    def upper_envelope(n: int) -> float:
        """Верхняя огибающая: Y = 2√n + 1."""
        return 2 * math.sqrt(n) + 1

    @staticmethod
    def lower_envelope(n: int) -> float:
        """Нижняя огибающая (совершенные квадраты): Y = √n + 1."""
        return math.sqrt(n) + 1

    # ── визуализация ─────────────────────────────────────────────────────────

    def plot_universe(self, n_max: int = 100, width: int = 60) -> str:
        """ASCII-график «числовой вселенной» (N, ∂N) с огибающими.

        Возвращает строку для вывода.
        """
        if n_max < 2:
            return ""

        max_dn = max(self.derivative(n) for n in range(2, n_max + 1))
        max_dn = max(max_dn, int(self.upper_envelope(n_max)) + 1)

        height = 20
        rows = [[" "] * width for _ in range(height)]

        def col(n):
            return int((n - 2) / (n_max - 2) * (width - 1))

        def row(dn):
            return height - 1 - int(dn / max_dn * (height - 1))

        # огибающие
        for n in range(2, n_max + 1):
            c = col(n)
            u = row(self.upper_envelope(n))
            l = row(self.lower_envelope(n))
            if 0 <= u < height:
                rows[u][c] = "^"
            if 0 <= l < height:
                rows[l][c] = "_"

        # точки (∂N, N)
        for n in range(2, n_max + 1):
            dn = self.derivative(n)
            c = col(n)
            r = row(dn)
            if 0 <= r < height and 0 <= c < width:
                cls = self.classify(n, max_steps=50)
                rows[r][c] = {"perfect": "P", "super": "S"}.get(cls, ".")

        lines = ["∂N"]
        for r in rows:
            lines.append("│" + "".join(r))
        lines.append("└" + "─" * width + "→ N")
        lines.append(f"  [2..{n_max}]  ^ = верхняя огибающая  _ = нижняя  P=совершенное  S=супер")
        return "\n".join(lines)

    def spectrum(self, n_max: int = 50) -> str:
        """Цветной ASCII-спектр цепочек: длина цепочки каждого числа."""
        lines = [f"Спектр цепочек ∂ для n = 2..{n_max}:"]
        for n in range(2, n_max + 1):
            ch = self.chain(n, max_steps=50)
            length = len(ch)
            cls = self.classify(n, max_steps=50)
            bar = "#" * min(length, 40)
            mark = {"perfect": "[P]", "super": "[S]"}.get(cls, "   ")
            lines.append(f"  {n:4d} {mark} len={length:3d} {bar}")
        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="hexnumderiv — производная числа (теория Германа)")
    parser.add_argument("--derivative", type=int, metavar="N",
                        help="Вычислить ∂N")
    parser.add_argument("--chain", type=int, metavar="N",
                        help="Построить цепочку производных N → ∂N → …")
    parser.add_argument("--classify", type=int, metavar="N",
                        help="Классифицировать N (perfect/super/ordinary)")
    parser.add_argument("--perfect", type=int, metavar="LIMIT",
                        help="Найти совершенные числа ≤ LIMIT")
    parser.add_argument("--super", type=int, metavar="LIMIT",
                        help="Найти супер-числа ≤ LIMIT")
    parser.add_argument("--leibniz", nargs=2, type=int, metavar=("K", "M"),
                        help="Проверить правило Лейбница для K и M (взаимно простых)")
    parser.add_argument("--spectrum", type=int, metavar="N_MAX", default=0,
                        help="Спектр цепочек для n=2..N_MAX")
    parser.add_argument("--plot-universe", type=int, metavar="N_MAX", default=0,
                        dest="plot_universe",
                        help="ASCII-график числовой вселенной до N_MAX")
    args = parser.parse_args()

    nd = NumberDerivative()

    if args.derivative is not None:
        print(f"∂{args.derivative} = {nd.derivative(args.derivative)}")

    if args.chain is not None:
        ch = nd.chain(args.chain)
        print(f"Цепочка {args.chain}: {' → '.join(map(str, ch))}")

    if args.classify is not None:
        print(f"Класс {args.classify}: {nd.classify(args.classify)}")

    if args.perfect is not None:
        perfects = nd.perfect_numbers(args.perfect)
        print(f"Совершенные числа ≤ {args.perfect}: {perfects}")

    if args.super is not None:
        supers = nd.super_numbers(args.super)
        print(f"Супер-числа ≤ {args.super}: {supers}")

    if args.leibniz:
        k, m = args.leibniz
        res = nd.leibniz_rule(k, m)
        print(f"Правило Лейбница для {k} и {m}:")
        print(f"  ∂{k} = {res['dk']},  ∂{m} = {res['dm']}")
        print(f"  ∂({k}·{m}) = ∂{res['km']} = {res['lhs']}")
        print(f"  {k}·∂{m} + {m}·∂{k} + ∂{k}·∂{m} = {res['rhs']}")
        print(f"  Выполняется: {res['holds']}")

    if args.spectrum:
        print(nd.spectrum(args.spectrum))

    if args.plot_universe:
        print(nd.plot_universe(args.plot_universe))

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    _main()
