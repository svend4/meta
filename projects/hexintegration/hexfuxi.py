"""
hexfuxi.py — 先天八卦: Ранненебесное расположение (Фу Си) и связь с двоичным счётом

═══════════════════════════════════════════════════════════════════════════
ДВА РАСПОЛОЖЕНИЯ ТРИГРАММ
═══════════════════════════════════════════════════════════════════════════

1. 先天八卦 Ранненебесное (Fú Xī / Leibniz):
   Восемь триграмм в октагоне; противоположные — дополнения (XOR = 7).
   Порядок по двоичному убыванию (Цянь=7 → Кунь=0).
   Открыто Лейбницем в 1701 г. как двоичная арифметика.

2. 後天八卦 Позднненебесное (Вэнь-ван / Ло-Шу):
   Восемь триграмм привязаны к направлениям через квадрат Ло-Шу.
   Основа практики Фэн-шуй и системы Куа (hexkua.py).

═══════════════════════════════════════════════════════════════════════════
РАННЕНЕБЕСНЫЙ ОКТАГОН (по часовой стрелке от Юга)
═══════════════════════════════════════════════════════════════════════════

         Ю: ☰ 7 (Цянь, 111)
    ЮЗ: ☴ 6       ЮВ: ☱ 3
  З: ☵ 2    ◈    В: ☲ 5
    СЗ: ☶ 4       СВ: ☳ 1
         С: ☷ 0 (Кунь, 000)

Двоичная последовательность (по часовой): 7, 3, 5, 1, 0, 4, 2, 6
= bit-реверс последовательности: 7, 6, 5, 4, 3, 2, 1, 0

Ключевое свойство: противоположные позиции дополняют друг друга:
  Цянь(7) ↔ Кунь(0)    Дуй(3) ↔ Гэнь(4)
  Ли(5)   ↔ Кань(2)    Чжэнь(1) ↔ Сюнь(6)

═══════════════════════════════════════════════════════════════════════════
ЛЕЙБНИЦ И ДВОИЧНАЯ АРИФМЕТИКА (1701)
═══════════════════════════════════════════════════════════════════════════

В 1701 г. Готфрид Вильгельм Лейбниц получил от иезуитского миссионера
Буве рисунок 64-гексаграммного квадрата Фу Си. Лейбниц немедленно
опознал в нём бинарную систему счисления:

  Ян(━) = 1,  Инь(╌) = 0;  гексаграмма = 6-разрядное двоичное число.

  Кунь (000000) = 0  =  Нуль
  Фу   (000001) = 1
  Ши   (000010) = 2
  ...
  Цянь (111111) = 63  =  Максимум

Квадрат Фу Си ДО Лейбница воплощал двоичную систему.
Лейбниц написал статью «Explication de l'Arithmétique Binaire» (1703),
в которой отметил совпадение с И-Цзин.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../libs/hexcore"))
from hexcore import yang_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
from hexiching import Hexagram

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}
GRN  = "\033[92m"
YLW  = "\033[93m"
CYN  = "\033[96m"

HX   = {h: Hexagram(h) for h in range(64)}

# ---------------------------------------------------------------------------
# Данные: два расположения триграмм
# ---------------------------------------------------------------------------

TRIG_SYMS  = {0:"☷",1:"☳",2:"☵",3:"☱",4:"☶",5:"☲",6:"☴",7:"☰"}
TRIG_NAMES = {0:"Кунь",1:"Чжэнь",2:"Кань",3:"Дуй",
              4:"Гэнь", 5:"Ли",   6:"Сюнь", 7:"Цянь"}

# Ранненебесный порядок (по часовой от Юга): 7,3,5,1,0,4,2,6
FUXI_ORDER = [7, 3, 5, 1, 0, 4, 2, 6]
FUXI_DIRS  = ["Ю", "ЮВ", "В", "СВ", "С", "СЗ", "З", "ЮЗ"]

# Позднненебесный порядок (Ло-Шу / Куа):
LATER_ORDER = [1, 8, 3, 4, 9, 2, 7, 6]  # Куа-номера (1-9 без 5)
LATER_DIRS  = ["С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ"]
LATER_TRIG  = {1:2, 2:0, 3:1, 4:6, 6:7, 7:3, 8:4, 9:5}  # куа → trig_h


def rev3(t: int) -> int:
    """Bit-реверс 3-битного числа."""
    return ((t & 1) << 2) | (t & 2) | ((t >> 2) & 1)


# ---------------------------------------------------------------------------
# ASCII-октагон
# ---------------------------------------------------------------------------

def show_octagon(arrangement: str = "fuxi", use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    is_fuxi = arrangement == "fuxi"
    title   = "先天八卦 Ранненебесное (Фу Си)" if is_fuxi else "後天八卦 Позднненебесное (Вэнь-ван)"

    if is_fuxi:
        order = FUXI_ORDER
        dirs  = FUXI_DIRS
    else:
        order = [LATER_TRIG[k] for k in LATER_ORDER]
        dirs  = LATER_DIRS

    def cell(pos: int) -> str:
        t  = order[pos]
        yc = YANG_ANSI[yang_count(t)] if use_color else ""
        return f"{yc}{TRIG_SYMS[t]}{t}{r}"

    # 8 позиций: 0=Ю, 1=ЮВ, 2=В, 3=СВ, 4=С, 5=СЗ, 6=З, 7=ЮЗ
    # ASCII layout:
    #        (0) Ю
    #   (7)ЮЗ      (1)ЮВ
    # (6)З   ◈   (2)В
    #   (5)СЗ      (3)СВ
    #        (4) С

    lines = [
        f"  {bold}{title}{r}",
        "",
        f"         {dirs[0]}: {cell(0)} ({TRIG_NAMES[order[0]]})",
        f"   {dirs[7]}:{cell(7)}        {dirs[1]}:{cell(1)}",
        f" {dirs[6]}:{cell(6)}    ◈    {dirs[2]}:{cell(2)}",
        f"   {dirs[5]}:{cell(5)}        {dirs[3]}:{cell(3)}",
        f"         {dirs[4]}: {cell(4)} ({TRIG_NAMES[order[4]]})",
        "",
    ]

    if is_fuxi:
        lines += [
            f"  {dim}Порядок по часовой: "
            + " → ".join(f"{TRIG_SYMS[order[i]]}{order[i]}" for i in range(8))
            + f" → ...{r}",
            f"  {dim}Двоичная последовательность: "
            + ", ".join(str(order[i]) for i in range(8))
            + f"  (bit-реверс: 7,6,5,4,0,1,2,3){r}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Квадрат Фу Си 8×8 (64 гексаграммы)
# ---------------------------------------------------------------------------

def show_fuxi_square(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    # Строки = верхняя триграмма в ранненебесном порядке (сверху = Цянь=7)
    # Столбцы = нижняя триграмма в ранненебесном порядке (слева = Цянь=7)
    row_trigs = FUXI_ORDER        # верхние
    col_trigs = FUXI_ORDER        # нижние

    lines = [
        f"  {bold}Квадрат Фу Си: 8×8 гексаграмм (64 = 2⁶){r}",
        f"  {dim}Строки: верхняя триграмма; столбцы: нижняя триграмма (оба в ранненебесном порядке){r}",
        "",
    ]

    # Заголовок столбцов
    head = "      " + "  ".join(
        f"{(YANG_ANSI[yang_count(t)] if use_color else '')}{TRIG_SYMS[t]}{RESET if use_color else ''}"
        for t in col_trigs
    )
    lines.append(head)
    lines.append("  " + "─"*42)

    for up in row_trigs:
        ycu = YANG_ANSI[yang_count(up)] if use_color else ""
        row = f"  {ycu}{TRIG_SYMS[up]}{r}  "
        for lo in col_trigs:
            h    = lo | (up << 3)
            hx   = HX[h]
            yc   = YANG_ANSI[hx.yang] if use_color else ""
            row += f"{yc}{hx.sym}{r} "
        row += f"  {ycu}{TRIG_NAMES[up]}{r}"
        lines.append(row)

    lines += [
        "",
        f"  {dim}Главная диагональ: удвоенные триграммы (同 тóng):",
        "  " + "  ".join(
            f"{(YANG_ANSI[yang_count(t)] if use_color else '')}{TRIG_SYMS[t]}"
            f"{HX[t|(t<<3)].sym}{RESET if use_color else ''}"
            for t in FUXI_ORDER
        ),
        "",
        f"  Центральная симметрия: h и comp(h) расположены симметрично",
        f"  относительно центра квадрата (h XOR 63 = зеркальная позиция).{r}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Двоичное сравнение: Фу Си vs натуральный порядок
# ---------------------------------------------------------------------------

def show_binary_table(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    lines = [
        f"  {bold}Двоичный счёт и И-Цзин (Лейбниц, 1701){r}",
        "",
        f"  {'h':>4}  {'двоич.':>6}  {'ян':>3}  {'символ':>4}  {'КВ#':>4}  Имя  "
        f"{'  Ранненеб.'}",
        "  " + "─"*54,
    ]

    fuxi_seq = []
    for i in range(8):
        for j in range(8):
            up = FUXI_ORDER[i]
            lo = FUXI_ORDER[j]
            fuxi_seq.append(lo | (up << 3))

    # Первые 16 и последние 4 в натуральном порядке
    show_h = list(range(10)) + [21, 42, 62, 63]
    for h in show_h:
        hx = HX[h]
        yc = YANG_ANSI[hx.yang] if use_color else ""
        fuxi_pos = fuxi_seq.index(h) if h in fuxi_seq else -1
        row_f, col_f = divmod(fuxi_pos, 8)
        fp = f"({TRIG_SYMS[FUXI_ORDER[row_f]]},{TRIG_SYMS[FUXI_ORDER[col_f]]})" if fuxi_pos >= 0 else ""
        lines.append(
            f"  {yc}{h:>4}  {h:06b}  {hx.yang}   {hx.sym}  КВ#{hx.kw:>2}  "
            f"{hx.name_pin[:14]:<14}{r}  {dim}{fp}{r}"
        )
    lines += [
        "  ...",
        "",
        f"  {bold}Наблюдение Лейбница:{r}",
        "  Если ян(━)=1, инь(╌)=0, а линии нумеровать снизу вверх,",
        "  то каждая гексаграмма — шестибитное двоичное число.",
        "  Кунь(000000)=0, Цянь(111111)=63.",
        "  64 гексаграммы = первые 64 двоичных числа.",
        "",
        f"  {bold}Единство двух открытий:{r}",
        "  Фу Си создал систему тысячи лет назад.",
        "  Лейбниц изобрёл двоичную арифметику в 1679 г.",
        "  В 1701 г. он увидел: они идентичны.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Сравнение двух расположений
# ---------------------------------------------------------------------------

def show_comparison(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    lines = [
        f"  {bold}Сравнение двух расположений восьми триграмм{r}",
        "",
        f"  {'Направл.':>8}  {'Ранненеб.':^16}  {'Позднненеб.':^16}",
        "  " + "─"*52,
    ]

    later_by_dir = {}
    for i, kua in enumerate(LATER_ORDER):
        t = LATER_TRIG[kua]
        later_by_dir[LATER_DIRS[i]] = t

    for i in range(8):
        d   = FUXI_DIRS[i]
        tf  = FUXI_ORDER[i]
        tl  = later_by_dir.get(d, -1)
        yf  = YANG_ANSI[yang_count(tf)] if use_color else ""
        yl  = YANG_ANSI[yang_count(tl)] if use_color and tl >= 0 else ""
        same = " =" if tf == tl else ""
        f_str = f"{TRIG_SYMS[tf]}{tf} {TRIG_NAMES[tf]}"
        l_str = f"{TRIG_SYMS[tl]}{tl} {TRIG_NAMES[tl]}" if tl >= 0 else "—"
        lines.append(
            f"  {d:>8}   {yf}{f_str:<14}{r}  {yl}{l_str:<14}{r}{same}"
        )

    lines += [
        "",
        f"  {dim}Ранненебесное: структурная симметрия (математика){r}",
        f"  {dim}Позднненебесное: функциональная картина (практика){r}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexfuxi — 先天八卦: Фу Си, двоичный счёт, Лейбниц",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexfuxi.py              Ранненебесный октагон
  python hexfuxi.py --later      Позднненебесный октагон (Вэнь-ван)
  python hexfuxi.py --compare    Сравнение двух расположений
  python hexfuxi.py --square     Квадрат Фу Си 8×8 (64 гексаграммы)
  python hexfuxi.py --binary     Двоичный счёт и Лейбниц
        """,
    )
    parser.add_argument("--later",    action="store_true")
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--square",   action="store_true")
    parser.add_argument("--binary",   action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args   = parser.parse_args()
    use_color = not args.no_color

    if args.later:
        print()
        print(show_octagon("later", use_color))
    elif args.compare:
        print()
        print(show_comparison(use_color))
    elif args.square:
        print()
        print(show_fuxi_square(use_color))
    elif args.binary:
        print()
        print(show_binary_table(use_color))
    else:
        print()
        print(show_octagon("fuxi", use_color))
        print()
        print(show_binary_table(use_color))


if __name__ == "__main__":
    main()
