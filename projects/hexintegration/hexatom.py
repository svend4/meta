"""
hexatom.py — Q6 как векторное пространство GF(2)^6

Каждый из 64 h-кодов — это 6-битный вектор над полем {0,1}.
Операция: XOR (сложение в GF(2)^6).

АТОМЫ — 6 базисных векторов:
  a0 = 1  = 000001 = открыть ВЛ / включить R / шаг по оси X1
  a1 = 2  = 000010 = открыть ВП / включить Y / шаг по оси X2
  a2 = 4  = 000100 = открыть СЛ / включить G / шаг по оси Y1
  a3 = 8  = 001000 = открыть СП / включить C / шаг по оси Y2
  a4 = 16 = 010000 = открыть НЛ / включить B / шаг по оси Z1
  a5 = 32 = 100000 = открыть НП / включить M / шаг по оси Z2

Любое состояние h = XOR(ai | бит i установлен в h).
Расстояние Хэмминга d(h1,h2) = yang(h1 XOR h2).

Ключевые отношения:
  h XOR 63 = antipode(h)     (антипод = дополнение по всем битам)
  h XOR h  = 0               (нейтральный элемент = Кунь)
  h XOR 0  = h               (Кунь = единица группы)

Суверенный цикл = двойное применение атомов:
  0 -a0-> 1 -a1-> 3 -a2-> 7 -a3-> 15 -a4-> 31 -a5-> 63
  63 -a0-> 62 -a1-> 60 -a2-> 56 -a3-> 48 -a4-> 32 -a5-> 0

Подгруппы GF(2)^6:
  Размеры: 1, 2, 4, 8, 16, 32, 64 (все делители 64 = 2^6)
  Самая малая: {0} (тривиальная)
  Самая большая: все 64 (сама группа)
  Специальные: {0,63} (антиподная пара, размер 2)
               {0,9,18,27,36,45,54,63} (диагональ куба, размер 8)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, antipode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexliuxing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexboya"))

from hexiching import Hexagram
from hexliuxing import ELEMENTS
from hexboya import ZONE_NAMES

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}
ELEM_COLORS = ["\033[31m","\033[33m","\033[32m","\033[36m","\033[34m","\033[35m"]

ATOMS = [1 << i for i in range(6)]   # [1, 2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# Разложение на атомы
# ---------------------------------------------------------------------------

def decompose(h: int) -> list:
    """Возвращает список индексов установленных битов (атомов)."""
    return [i for i in range(6) if (h >> i) & 1]


def xor_sum(*args: int) -> int:
    result = 0
    for a in args:
        result ^= a
    return result


# ---------------------------------------------------------------------------
# 6 атомов
# ---------------------------------------------------------------------------

def show_atoms(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    dim = DIM if use_color else ""

    lines = [
        "",
        f"{'═'*68}",
        f"  {bold}6 АТОМОВ — БАЗИС GF(2)^6{r}",
        f"  Каждый атом = один бит = один «элементарный сдвиг» системы",
        f"{'═'*68}",
        "",
        f"  {dim}a   h   бинарный  ян  Крюков  Лю-Синь  Касаткин  Гексаграмма{r}",
        "  " + "─"*66,
    ]

    for i, atom in enumerate(ATOMS):
        hx = Hexagram(atom)
        ec = ELEM_COLORS[i] if use_color else ""
        zn = ZONE_NAMES[i]
        elem = ELEMENTS[i]
        yc = YANG_ANSI[1] if use_color else ""
        x = atom & 3; y = (atom >> 2) & 3; z = (atom >> 4) & 3
        vol = (x+1)*(y+1)*(z+1)

        lines.append(
            f"  {ec}a{i} = {atom:>2} = {atom:06b}{r}"
            f"  1  {ec}{zn}{r}    {ec}{elem['short']}{r}={elem['name'][:6]}"
            f"  ({x},{y},{z})V={vol}"
            f"  {hx.sym}#{hx.kw:>2} {hx.name_pin[:16]}"
        )

    lines += [
        "",
        f"  {bold}XOR-свойства атомов:{r}",
        f"  ai XOR ai = 0   (дважды применить = вернуться)",
        f"  ai XOR aj ≠ 0   (i≠j — разные атомы не компенсируют)",
        f"  ai XOR 63 = 63 XOR ai → снова атом (63 - ai)",
        f"  Сумма всех атомов: {xor_sum(*ATOMS)} = {xor_sum(*ATOMS):06b} = 63 = Цянь",
        "",
        f"  {bold}Суверенный цикл через атомы:{r}",
        f"  0 -a0→ 1 -a1→ 3 -a2→ 7 -a3→ 15 -a4→ 31 -a5→ 63 (открытие)",
        f"  63 -a0→ 62 -a1→ 60 -a2→ 56 -a3→ 48 -a4→ 32 -a5→ 0 (закрытие)",
        f"  Та же последовательность атомов a0,a1,a2,a3,a4,a5 — дважды!",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# XOR-разложение гексаграммы
# ---------------------------------------------------------------------------

def show_decomp(h: int, use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    hx = Hexagram(h)
    idx = decompose(h)
    yc = YANG_ANSI[hx.yang] if use_color else ""

    lines = [
        f"\n  {bold}XOR-разложение: h={h} ({h:06b}) = {hx.sym}#{hx.kw} «{hx.name_pin}»{r}",
        "",
    ]

    if not idx:
        lines.append(f"  h=0 = {bold}нейтральный элемент{r} (Кунь ䷁, все инь)")
    else:
        parts = []
        for i in idx:
            ec = ELEM_COLORS[i] if use_color else ""
            parts.append(f"{ec}a{i}(={ATOMS[i]}){r}")
        lines += [
            f"  {yc}h={h}{r} = " + " XOR ".join(parts),
            f"         = {' XOR '.join(str(ATOMS[i]) for i in idx)}",
            f"         = {' XOR '.join(f'{ATOMS[i]:06b}' for i in idx)}",
            "",
            f"  Задействованные атомы ({len(idx)} из 6):",
        ]
        for i in idx:
            ec = ELEM_COLORS[i] if use_color else ""
            zn = ZONE_NAMES[i]
            elem = ELEMENTS[i]
            lines.append(f"    {ec}a{i} = {ATOMS[i]:>2} = бит{i}  {zn}  {elem['short']} ({elem['name']}){r}")

    # Связанные состояния
    ap = antipode(h)
    ap_hx = Hexagram(ap)
    lines += [
        "",
        f"  Антипод: h XOR 63 = {h} XOR 63 = {yc}{ap}{r} = {ap_hx.sym}#{ap_hx.kw} «{ap_hx.name_pin}»",
        f"           Незадействованные атомы: " + " ".join(
            f"a{i}" for i in range(6) if i not in idx
        ),
    ]

    # Соседи (XOR с каждым атомом)
    lines += ["", f"  Соседи (XOR с атомом):"]
    for i in range(6):
        nb = h ^ ATOMS[i]
        nb_hx = Hexagram(nb)
        action = "закрыть" if (h >> i) & 1 else "открыть"
        ec = ELEM_COLORS[i] if use_color else ""
        yc2 = YANG_ANSI[nb_hx.yang] if use_color else ""
        lines.append(
            f"    h XOR a{i} = {yc2}{nb:>2}{r} = {nb_hx.sym}#{nb_hx.kw:>2}"
            f"  ({ec}{action} {ZONE_NAMES[i]}{r})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# XOR-таблица (малая: только атомы и 0)
# ---------------------------------------------------------------------------

def show_xor_table(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    selected = [0] + ATOMS + [63]

    header = f"  {'XOR':>4} │"
    for b in selected:
        yc = YANG_ANSI[yang_count(b)] if use_color else ""
        header += f" {yc}{b:>3}{r}│"

    sep = "  " + "─"*5 + "┼" + "─────┼"*len(selected)

    lines = [
        f"{bold}XOR-таблица: 0, атомы a0..a5, 63 (антипод){r}",
        "",
        header, sep,
    ]

    for a in selected:
        yca = YANG_ANSI[yang_count(a)] if use_color else ""
        row = f"  {yca}{a:>4}{r} │"
        for b in selected:
            c = a ^ b
            ycc = YANG_ANSI[yang_count(c)] if use_color else ""
            row += f" {ycc}{c:>3}{r}│"
        lines += [row, sep]

    lines += [
        "",
        f"  Структура: (GF(2)^6, XOR) — абелева группа порядка 64.",
        f"  Нейтральный элемент: 0 (Кунь ䷁).",
        f"  Обратный к h = сам h (h XOR h = 0).",
        f"  Порядок каждого элемента = 2 (кроме нейтрального).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Подгруппы
# ---------------------------------------------------------------------------

def show_subgroups(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    dim = DIM if use_color else ""

    # Специальные подгруппы
    subgroups = [
        ("Тривиальная",  [0],          "только Кунь"),
        ("{0,Цянь}",     [0, 63],      "антиподная пара"),
        ("Диагональ",    [0,9,18,27,36,45,54,63], "8 чистых гексаграмм"),
        ("Чётный ян",    [h for h in range(64) if yang_count(h) % 2 == 0], "32 гексаграммы"),
        ("Вся группа",   list(range(64)), "все 64"),
    ]

    lines = [
        f"{bold}Избранные подгруппы GF(2)^6{r}",
        "",
        f"  {dim}Имя               Размер  Описание{r}",
        "  " + "─"*55,
    ]

    for name, sg, desc in subgroups:
        # Проверка замкнутости (для малых)
        if len(sg) <= 8:
            closed = all((a ^ b) in sg for a in sg for b in sg)
            check = "✓" if closed else "✗"
        else:
            check = "?"
        lines.append(
            f"  {name:<18} {len(sg):>6}  {check} {desc}"
        )

    lines += [
        "",
        f"  {bold}Смежные классы (косеты) подгруппы {{0,63}}:{r}",
        "  (32 пары антиподов — разбивают все 64 на 32 косета размера 2)",
        "",
        f"  {'Косет':>4}  h1  h1_бин   h2  h2_бин   h1 XOR h2",
        "  " + "─"*50,
    ]

    seen = set()
    coset_n = 0
    for h in range(64):
        if h in seen:
            continue
        ap = antipode(h)
        seen |= {h, ap}
        coset_n += 1
        hx = Hexagram(h)
        ax = Hexagram(ap)
        yc = YANG_ANSI[yang_count(h)] if use_color else ""
        yca = YANG_ANSI[yang_count(ap)] if use_color else ""
        lines.append(
            f"  {coset_n:>4}  {yc}{h:>2} {h:06b}{r}  "
            f"{yca}{ap:>2} {ap:06b}{r}  "
            f"{h^ap:>3}  {hx.sym}↔{ax.sym}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexatom — GF(2)^6: атомы, XOR, подгруппы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexatom.py              Описание 6 атомов
  python hexatom.py --decomp 7   XOR-разложение h=7 (Тай)
  python hexatom.py --xor        XOR-таблица для атомов
  python hexatom.py --subgroups  Подгруппы GF(2)^6
        """,
    )
    parser.add_argument("--decomp",    type=int, metavar="H")
    parser.add_argument("--xor",       action="store_true")
    parser.add_argument("--subgroups", action="store_true")
    parser.add_argument("--no-color",  action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.decomp is not None:
        print(show_decomp(args.decomp, use_color))
    elif args.xor:
        print()
        print(show_xor_table(use_color))
    elif args.subgroups:
        print()
        print(show_subgroups(use_color))
    else:
        print(show_atoms(use_color))


if __name__ == "__main__":
    main()
