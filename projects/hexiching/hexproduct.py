"""
hexproduct.py — Q6 как произведение Q3 × Q3

Каждая гексаграмма = (верхняя триграмма) × (нижняя триграмма).
Q6 = Q3 × Q3 как граф — произведение двух 3-мерных гиперкубов.

h = lower + 8 * upper  (lower=бит0-2, upper=бит3-5)

Триграммы Q3 (3-битные числа):
  0 ☷ Кунь  (000) Земля
  1 ☳ Чжэнь (001) Гром
  2 ☵ Кань  (010) Вода
  3 ☱ Дуй   (011) Озеро
  4 ☶ Гэнь  (100) Гора
  5 ☴ Сюнь  (101) Ветер
  6 ☲ Ли    (110) Огонь
  7 ☰ Цянь  (111) Небо

Таблица 8×8: строка = верхняя триграмма, столбец = нижняя.
Диагональ (верх=низ) = 8 «чистых» гексаграмм.

Связь с Касаткиным:
  lower (биты 0-2) = Крюков-зоны ВЛ,ВП,СЛ + нижние оси x,y_low
  upper (биты 3-5) = Крюков-зоны СП,НЛ,НП + верхние оси y_hi,z

Связь с Лю-Синь:
  lower = R(бит0)+Y(бит1)+G(бит2)  «тёплые» цвета
  upper = C(бит3)+B(бит4)+M(бит5)  «холодные» цвета
  Диагональ = «тёплые=холодные» (сбалансированные гексаграммы)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, upper_trigram, lower_trigram

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
from hexiching import Hexagram, KW_FROM_H

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}

# Символы триграмм
TRIGRAM_SYM  = ["☷","☳","☵","☱","☶","☲","☴","☰"]
TRIGRAM_PIN  = ["Кунь","Чжэнь","Кань","Дуй","Гэнь","Ли","Сюнь","Цянь"]
TRIGRAM_CHIN = ["坤","震","坎","兌","艮","離","巽","乾"]
TRIGRAM_ENG  = ["Earth","Thunder","Water","Lake","Mountain","Fire","Wind","Heaven"]
TRIGRAM_YANG = [yang_count(t) for t in range(8)]  # ян в триграмме (0..3)


# ---------------------------------------------------------------------------
# 8×8 таблица
# ---------------------------------------------------------------------------

def show_table(use_color: bool = True) -> str:
    """Таблица 8×8 всех гексаграмм по (верх × низ) триграммам."""
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    dim = DIM if use_color else ""

    # Заголовок: нижние триграммы (столбцы)
    header = f"  {' ' * 12}│"
    for low in range(8):
        tc = YANG_ANSI[TRIGRAM_YANG[low]] if use_color else ""
        header += f" {tc}{TRIGRAM_SYM[low]}{TRIGRAM_PIN[low][:4]:<4}{r}│"

    sep = "  " + "─" * 12 + "┼" + ("───────┼" * 8)

    lines = [
        "",
        f"{'═'*78}",
        f"  {bold}Q6 = Q3 × Q3 — Таблица произведений триграмм{r}",
        f"  Строка = верхняя триграмма (天/небо), столбец = нижняя (地/земля)",
        f"{'═'*78}",
        "",
        header,
        sep,
    ]

    # Суммы ян по строкам
    row_yang_sums = []

    for upper in range(8):
        yc_row = YANG_ANSI[TRIGRAM_YANG[upper]] if use_color else ""
        row_str = (
            f"  {yc_row}{TRIGRAM_SYM[upper]}{TRIGRAM_PIN[upper]:<8}{r}"
            f"  {dim}{TRIGRAM_YANG[upper]}ян{r} │"
        )
        yang_sum = 0
        for low in range(8):
            h = low + 8 * upper
            hx = Hexagram(h)
            yc = YANG_ANSI[hx.yang] if use_color else ""
            is_diag = (upper == low)
            is_antipode_diag = (upper + low == 7)
            prefix = bold if is_diag else ""
            suffix = r if (is_diag and use_color) else ""
            row_str += f" {prefix}{yc}{hx.sym}{r}{'*' if is_diag else ' '}{hx.yang}{suffix} │"
            yang_sum += hx.yang
        row_yang_sums.append(yang_sum)
        row_str += f" Σян={yang_sum}"
        lines.append(row_str)
        lines.append(sep)

    # Суммы ян по столбцам
    col_yang_sums = []
    for low in range(8):
        s = sum(yang_count(low + 8 * upper) for upper in range(8))
        col_yang_sums.append(s)
    footer = f"  {'Σян':>12}│"
    for s in col_yang_sums:
        footer += f"  {s:>3}   │"
    footer += f" Σ={sum(col_yang_sums)}"
    lines += [footer, ""]

    # Статистика
    total_yang = sum(yang_count(h) for h in range(64))
    lines += [
        f"  {bold}Свойства таблицы:{r}",
        f"  * Главная диагональ (верх=низ): 8 «чистых» гексаграмм ({bold}☷²☳²☵²☱²☶²☴²☲²☰²{r})",
        f"    Их ян: " + ", ".join(
            f"{TRIGRAM_SYM[k]}={yang_count(k + 8*k)}"
            for k in range(8)
        ),
        f"  * Антидиагональ (верх+низ=7): пары-антиподы верх/низ",
        f"  * Каждая строка и столбец — полный путь Q3 (8 вершин)",
        f"  * Сумма ян по строке: {sorted(set(row_yang_sums))}",
        f"  * Сумма ян по стлбцу: {sorted(set(col_yang_sums))}",
        f"  * Общая сумма ян = {total_yang} = 64 × 3 (среднее = 3.0)",
        "",
        f"  {bold}Лю-Синь разбиение:{r}",
        f"  lower trigram = биты 0-2 = R+Y+G  (тёплые: Огонь, Земля, Дерево)",
        f"  upper trigram = биты 3-5 = C+B+M  (холодные: Вода, Металл, Эфир)",
        f"  Диагональ: тёплый ян = холодный ян → {bold}«сбалансированные»{r}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Чистые гексаграммы (диагональ)
# ---------------------------------------------------------------------------

def show_diagonal(use_color: bool = True) -> str:
    """8 «чистых» гексаграмм с двойными триграммами."""
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    lines = [
        f"{bold}8 «чистых» гексаграмм (верхняя = нижняя триграмма){r}",
        "(Единственные гексаграммы с upper=lower — «самоподобные»)",
        "",
        f"  {'k':>2}  {'h':>3}  {'триграмма':>10}  {'ян':>3}  {'символ':>6}  {'KW#':>4}  Имя",
        "  " + "─" * 58,
    ]
    for k in range(8):
        h = k + 8 * k
        hx = Hexagram(h)
        yc = YANG_ANSI[hx.yang] if use_color else ""
        tc = YANG_ANSI[TRIGRAM_YANG[k]] if use_color else ""
        warm_yang = yang_count(k)         # ян в нижней (тёплой)
        cold_yang = yang_count(k)         # = ян в верхней (холодной) — они равны!
        lines.append(
            f"  {k:>2}  {h:>3}  "
            f"{tc}{TRIGRAM_SYM[k]}{TRIGRAM_SYM[k]}{r} {TRIGRAM_PIN[k][:6]:<6}"
            f"  {yc}{hx.yang:>3}{r}  "
            f"{hx.sym} {h:06b}  {hx.kw:>4}  {hx.name_pin[:20]}"
        )
    lines += [
        "",
        "  Заметим: ян каждой чистой гексаграммы = 2 × (ян триграммы).",
        "  Путь 0→9→18→27→36→45→54→63 — диагональ таблицы,",
        "  это и есть главная диагональ куба Касаткина: (0,0,0)→(3,3,3)!",
        "  V(k,k,k) = (k+1)³: 1, 8, 27, 64 — совершенные кубы.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Таблица KW-номеров
# ---------------------------------------------------------------------------

def show_kw_table(use_color: bool = True) -> str:
    """Те же 64 гексаграммы, но показывает КВ-номер."""
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    dim = DIM if use_color else ""

    header = f"  {'':12}│"
    for low in range(8):
        tc = YANG_ANSI[TRIGRAM_YANG[low]] if use_color else ""
        header += f" {tc}{TRIGRAM_SYM[low]:<6}{r}│"
    sep = "  " + "─"*12 + "┼" + ("────────┼"*8)

    lines = [
        f"{bold}Таблица КВ-номеров (Вэнь-ван){r}",
        "",
        header, sep,
    ]
    for upper in range(8):
        yc_row = YANG_ANSI[TRIGRAM_YANG[upper]] if use_color else ""
        row = f"  {yc_row}{TRIGRAM_SYM[upper]}{TRIGRAM_PIN[upper]:<10}{r}│"
        for low in range(8):
            h = low + 8 * upper
            hx = Hexagram(h)
            yc = YANG_ANSI[hx.yang] if use_color else ""
            is_diag = (upper == low)
            prefix = bold if is_diag else ""
            row += f" {prefix}{yc}{hx.sym}#{hx.kw:>2}{r}  │"
        lines += [row, sep]

    lines += [
        "",
        "  Заметим: никакой монотонной «симметрии» KW-номеров нет —",
        "  порядок Вэнь-вана не совпадает с двоичным порядком триграмм.",
        "  Но пары-антиподы (КВ1↔КВ2, КВ3↔КВ4...) — всегда antipode(h).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexproduct — Q6=Q3×Q3, таблица произведений триграмм",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexproduct.py           8×8 таблица (символы + ян)
  python hexproduct.py --kw      8×8 таблица КВ-номеров
  python hexproduct.py --diag    8 «чистых» гексаграмм (диагональ)
        """,
    )
    parser.add_argument("--kw",       action="store_true")
    parser.add_argument("--diag",     action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.kw:
        print()
        print(show_kw_table(use_color))
    elif args.diag:
        print()
        print(show_diagonal(use_color))
    else:
        print(show_table(use_color))


if __name__ == "__main__":
    main()
