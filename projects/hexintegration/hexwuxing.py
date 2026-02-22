"""
hexwuxing.py — 五行 Wǔ Xíng: Пять Первоэлементов в гексаграммах Q6

Каждая из восьми триграмм традиционно связана с одним из пяти первоэлементов
по системе Поздненебесного расположения (後天 Hòu Tiān, квадрат Ло-Шу):

  ☳ Чжэнь (В)   → 木 Дерево
  ☴ Сюнь  (ЮВ)  → 木 Дерево
  ☲ Ли    (Ю)   → 火 Огонь
  ☷ Кунь  (ЮЗ)  → 土 Земля
  ☶ Гэнь  (СВ)  → 土 Земля
  ☰ Цянь  (СЗ)  → 金 Металл
  ☱ Дуй   (З)   → 金 Металл
  ☵ Кань  (С)   → 水 Вода

═══════════════════════════════════════════════════════════
ЦИКЛ ПОРОЖДЕНИЯ (相生 Xiāng Shēng):
  Дерево → Огонь → Земля → Металл → Вода → Дерево

ЦИКЛ ПОДАВЛЕНИЯ (相克 Xiāng Kè):
  Дерево → Земля → Вода → Огонь → Металл → Дерево
═══════════════════════════════════════════════════════════

Каждая из 64 гексаграмм = нижняя триграмма + верхняя триграмма.
Соотношение их элементов даёт ПЯТЬ ДИНАМИК:

  同氣 Тóng Qì   — Единство:    нижн. = верхн. (14 гекс.)
  生   Shēng     — Рождение:    нижн. порождает верхн. (12 гекс.)
  反生 Fǎn Shēng — Питание:     верхн. порождает нижн. (12 гекс.)
  克   Kè        — Управление:  нижн. подавляет верхн. (13 гекс.)
  反克 Fǎn Kè    — Противление: верхн. подавляет нижн. (13 гекс.)
"""

import sys
import os
import argparse
from collections import defaultdict

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
RED  = "\033[91m"
YLW  = "\033[93m"
BLU  = "\033[94m"
CYN  = "\033[96m"
MAG  = "\033[95m"
WHT  = "\033[97m"

# ---------------------------------------------------------------------------
# Данные: пять первоэлементов
# ---------------------------------------------------------------------------

ELEM_ID = {"wood": 0, "fire": 1, "earth": 2, "metal": 3, "water": 4}

ELEM_DATA = [
    {"name_cn": "木 Дерево", "name": "wood",  "sym": "☘",
     "ansi": "\033[92m",  "trig_h": [1, 6],   "dir": "В/ЮВ"},  # Zhen, Xun
    {"name_cn": "火 Огонь",  "name": "fire",  "sym": "🔥",
     "ansi": "\033[91m",  "trig_h": [5],       "dir": "Ю"},    # Li
    {"name_cn": "土 Земля",  "name": "earth", "sym": "⬛",
     "ansi": "\033[93m",  "trig_h": [0, 4],   "dir": "ЮЗ/СВ"}, # Kun, Gen
    {"name_cn": "金 Металл", "name": "metal", "sym": "◈",
     "ansi": "\033[97m",  "trig_h": [7, 3],   "dir": "СЗ/З"},  # Qian, Dui
    {"name_cn": "水 Вода",   "name": "water", "sym": "〰",
     "ansi": "\033[94m",  "trig_h": [2],       "dir": "С"},    # Kan
]

# Триграмма → элемент
TRIG_TO_ELEM: dict[int, int] = {}
for _eid, _ed in enumerate(ELEM_DATA):
    for _t in _ed["trig_h"]:
        TRIG_TO_ELEM[_t] = _eid

# Триграмма → символ
TRIG_SYMS = {0: "☷", 1: "☳", 2: "☵", 3: "☱", 4: "☶", 5: "☲", 6: "☴", 7: "☰"}
TRIG_NAMES = {0: "Кунь", 1: "Чжэнь", 2: "Кань", 3: "Дуй",
              4: "Гэнь",  5: "Ли",     6: "Сюнь",  7: "Цянь"}

# Динамика взаимодействия
DYN_SAME    = "same"    # нижн. = верхн.
DYN_GEN     = "gen"     # нижн. порождает верхн.
DYN_REVGEN  = "revgen"  # верхн. порождает нижн.
DYN_CTRL    = "ctrl"    # нижн. подавляет верхн.
DYN_REVCTRL = "revctrl" # верхн. подавляет нижн.

DYN_DATA = {
    DYN_SAME:    {"name": "同氣 Единство",   "short": "同", "ansi": "\033[92m"},
    DYN_GEN:     {"name": "生 Рождение",      "short": "生", "ansi": "\033[96m"},
    DYN_REVGEN:  {"name": "反生 Питание",     "short": "↑生","ansi": "\033[95m"},
    DYN_CTRL:    {"name": "克 Управление",    "short": "克", "ansi": "\033[93m"},
    DYN_REVCTRL: {"name": "反克 Противление", "short": "↑克","ansi": "\033[91m"},
}


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

def elem_of_trig(t: int) -> int:
    return TRIG_TO_ELEM[t]


def hex_dynamics(h: int) -> tuple[int, int, str]:
    """
    Возвращает (lower_elem, upper_elem, dynamics_key) для гексаграммы h.
    """
    lo = h & 7
    up = (h >> 3) & 7
    le = elem_of_trig(lo)
    ue = elem_of_trig(up)
    if le == ue:
        dyn = DYN_SAME
    elif (le + 1) % 5 == ue:
        dyn = DYN_GEN
    elif (ue + 1) % 5 == le:
        dyn = DYN_REVGEN
    elif (le + 2) % 5 == ue:
        dyn = DYN_CTRL
    elif (ue + 2) % 5 == le:
        dyn = DYN_REVCTRL
    else:
        dyn = DYN_SAME  # fallback (не должно случиться)
    return le, ue, dyn


def hex_elem_str(h: int, use_color: bool = True) -> str:
    le, ue, dyn = hex_dynamics(h)
    hx = Hexagram(h)
    d  = DYN_DATA[dyn]
    yc = YANG_ANSI[hx.yang] if use_color else ""
    ac = d["ansi"] if use_color else ""
    lc = ELEM_DATA[le]["ansi"] if use_color else ""
    uc = ELEM_DATA[ue]["ansi"] if use_color else ""
    r  = RESET if use_color else ""
    return (
        f"{yc}{hx.sym}{r} "
        f"{lc}{ELEM_DATA[le]['name_cn'][:1]}{r}"
        f"→"
        f"{uc}{ELEM_DATA[ue]['name_cn'][:1]}{r}"
        f" {ac}{d['short']}{r}"
    )


# ---------------------------------------------------------------------------
# Обзор системы
# ---------------------------------------------------------------------------

def show_overview(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    lines = [
        "",
        "═"*64,
        f"  {bold}五行 WǓ XÍNG — ПЯТЬ ПЕРВОЭЛЕМЕНТОВ В Q6{r}",
        "═"*64,
        "",
        f"  {bold}Триграммы и элементы:{r}",
        "",
    ]

    for eid, ed in enumerate(ELEM_DATA):
        ac  = ed["ansi"] if use_color else ""
        trigs = "  ".join(f"{TRIG_SYMS[t]} {TRIG_NAMES[t]}" for t in ed["trig_h"])
        lines.append(f"  {ac}{ed['name_cn']:<12}{r}  {trigs}")
    lines.append("")

    # Циклы
    gen_str  = " → ".join(ed["name_cn"][:1] for ed in ELEM_DATA) + " → ..."
    ctrl_str = " → ".join(ELEM_DATA[i]["name_cn"][:1]
                          for i in [0, 2, 4, 1, 3]) + " → ..."

    lines += [
        f"  {bold}Цикл порождения (相生):{r}",
        f"  {GRN if use_color else ''}{gen_str}{r}",
        "",
        f"  {bold}Цикл подавления (相克):{r}",
        f"  {RED if use_color else ''}{ctrl_str}{r}",
        "",
    ]

    # Статистика
    dyn_counts: dict[str, int] = defaultdict(int)
    for h in range(64):
        _, _, dyn = hex_dynamics(h)
        dyn_counts[dyn] += 1

    lines += [
        f"  {bold}Распределение 64 гексаграмм по динамикам:{r}",
        "",
    ]
    for key in [DYN_SAME, DYN_GEN, DYN_REVGEN, DYN_CTRL, DYN_REVCTRL]:
        d  = DYN_DATA[key]
        ac = d["ansi"] if use_color else ""
        bar = "█" * dyn_counts[key]
        lines.append(
            f"  {ac}{d['name']:<25}{r}  {dyn_counts[key]:>2}  {bar}"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Таблица 5×5 взаимодействий
# ---------------------------------------------------------------------------

def show_matrix(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    # Заполняем матрицу
    matrix = [[[] for _ in range(5)] for _ in range(5)]
    for h in range(64):
        le, ue, _ = hex_dynamics(h)
        matrix[le][ue].append(h)

    lines = [
        f"  {bold}Матрица 5×5: нижняя триграмма (строки) × верхняя (столбцы){r}",
        f"  {dim}Число гексаграмм в каждой ячейке (нижн.→ верхн. по циклам){r}",
        "",
    ]

    # Заголовок столбцов
    header = "         " + "  ".join(
        f"{(ELEM_DATA[e]['ansi'] if use_color else '')}{ELEM_DATA[e]['name_cn'][:1]}{r:>3}"
        for e in range(5)
    )
    lines.append(header)
    lines.append("  " + "─"*40)

    for lo_e in range(5):
        ed_lo = ELEM_DATA[lo_e]
        lc    = ed_lo["ansi"] if use_color else ""
        row   = f"  {lc}{ed_lo['name_cn'][:1]}{r}  "
        for hi_e in range(5):
            cell = matrix[lo_e][hi_e]
            dyn  = hex_dynamics(cell[0])[2] if cell else DYN_SAME
            ac   = DYN_DATA[dyn]["ansi"] if use_color else ""
            row += f"  {ac}{len(cell):>2}{r}"
        row += f"   {lc}{ed_lo['name_cn']}{r}"
        lines.append(row)

    lines += [
        "",
        f"  {dim}Диагональ (同 Единство): одинаковые элементы верх/низ{r}",
        f"  {DYN_DATA[DYN_GEN]['ansi'] if use_color else ''}"
        f"Надиагональ по циклу порождения = 生 Рождение{r}",
        f"  {DYN_DATA[DYN_CTRL]['ansi'] if use_color else ''}"
        f"Через одну позицию = 克 Управление{r}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Все гексаграммы по динамикам
# ---------------------------------------------------------------------------

def show_by_dynamics(dyn_key: str | None = None, use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    groups: dict[str, list[int]] = defaultdict(list)
    for h in range(64):
        _, _, dyn = hex_dynamics(h)
        groups[dyn].append(h)

    keys = [dyn_key] if dyn_key else [DYN_SAME, DYN_GEN, DYN_REVGEN, DYN_CTRL, DYN_REVCTRL]
    lines = []

    for key in keys:
        d    = DYN_DATA[key]
        ac   = d["ansi"] if use_color else ""
        hs   = groups[key]
        lines += [
            f"  {ac}{bold}{d['name']}  ({len(hs)} гексаграмм){r}",
            "",
        ]
        for h in hs:
            le, ue, _ = hex_dynamics(h)
            hx = Hexagram(h)
            yc = YANG_ANSI[hx.yang] if use_color else ""
            lc = ELEM_DATA[le]["ansi"] if use_color else ""
            uc = ELEM_DATA[ue]["ansi"] if use_color else ""
            lines.append(
                f"  {yc}{hx.sym} КВ#{hx.kw:>2} «{hx.name_pin}»{r}  "
                f"{lc}{ELEM_DATA[le]['name_cn']}{r} "
                f"{ac}→{r} "
                f"{uc}{ELEM_DATA[ue]['name_cn']}{r}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Анализ одной гексаграммы
# ---------------------------------------------------------------------------

def show_hexagram(h: int, use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    hx = Hexagram(h)
    lo = h & 7
    up = (h >> 3) & 7
    le, ue, dyn = hex_dynamics(h)
    ed_lo = ELEM_DATA[le]
    ed_up = ELEM_DATA[ue]
    d     = DYN_DATA[dyn]

    yc  = YANG_ANSI[hx.yang] if use_color else ""
    lc  = ed_lo["ansi"] if use_color else ""
    uc  = ed_up["ansi"] if use_color else ""
    ac  = d["ansi"] if use_color else ""

    DYN_DESC = {
        DYN_SAME:    "Нижняя и верхняя триграммы одного элемента — гармония и устойчивость.",
        DYN_GEN:     "Нижняя триграмма порождает верхнюю — рост, развитие, переход в следующую фазу.",
        DYN_REVGEN:  "Верхняя триграмма порождает нижнюю — питание, опора, возврат к истоку.",
        DYN_CTRL:    "Нижняя триграмма подавляет верхнюю — управление, дисциплина, ограничение.",
        DYN_REVCTRL: "Верхняя триграмма подавляет нижнюю — давление сверху, сопротивление, испытание.",
    }

    lines = [
        "",
        f"  {yc}{hx.sym} КВ#{hx.kw} «{hx.name_cn} {hx.name_pin}»  h={h}  ян={hx.yang}{r}",
        "",
        f"  Нижняя триграмма: {TRIG_SYMS[lo]} {TRIG_NAMES[lo]:>7}  →  "
        f"{lc}{ed_lo['name_cn']}{r}",
        f"  Верхняя триграмма: {TRIG_SYMS[up]} {TRIG_NAMES[up]:>7}  →  "
        f"{uc}{ed_up['name_cn']}{r}",
        "",
        f"  Динамика:  {ac}{d['name']}{r}",
        f"  {dim}{DYN_DESC[dyn]}{r}",
        "",
    ]

    # Показываем связанные элементы
    if dyn != DYN_SAME:
        gen_next_of_lo = ELEM_DATA[(le + 1) % 5]["name_cn"]
        ctrl_next_of_lo = ELEM_DATA[(le + 2) % 5]["name_cn"]
        lines += [
            f"  {dim}Из {ed_lo['name_cn']}: порождает {gen_next_of_lo}, "
            f"подавляет {ctrl_next_of_lo}{r}",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Путь по циклу порождения / подавления
# ---------------------------------------------------------------------------

def show_cycle_path(start_elem: int, n_steps: int = 5,
                     use_cycle: str = "gen", use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""

    step  = 1 if use_cycle == "gen" else 2
    cycle_name = "порождения (相生)" if use_cycle == "gen" else "подавления (相克)"
    lines = [
        f"  {bold}Цикл {cycle_name} от {ELEM_DATA[start_elem]['name_cn']}:{r}",
        "",
    ]
    elem = start_elem
    for i in range(n_steps + 1):
        ed = ELEM_DATA[elem]
        ac = ed["ansi"] if use_color else ""
        trigs_str = " ".join(f"{TRIG_SYMS[t]}{TRIG_NAMES[t]}" for t in ed["trig_h"])
        marker = " ← старт" if i == 0 else ""
        lines.append(f"  {ac}{ed['name_cn']:<14}{r}  {trigs_str}{marker}")
        if i < n_steps:
            arrow = " →" if use_color else " ->"
            lines.append(f"  {arrow}")
        elem = (elem + step) % 5

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexwuxing — 五行: пять первоэлементов в гексаграммах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexwuxing.py              Обзор: элементы, циклы, статистика
  python hexwuxing.py --matrix     Матрица 5×5 взаимодействий
  python hexwuxing.py --same       Все гексаграммы единства (同氣)
  python hexwuxing.py --gen        Все гексаграммы рождения (生)
  python hexwuxing.py --ctrl       Все гексаграммы управления (克)
  python hexwuxing.py --all        Все гексаграммы по динамикам
  python hexwuxing.py --h 7        Анализ гексаграммы h=7 (Тай)
  python hexwuxing.py --cycle gen  Цикл порождения от Дерева
  python hexwuxing.py --cycle ctrl Цикл подавления от Дерева
        """,
    )
    parser.add_argument("--matrix",   action="store_true")
    parser.add_argument("--same",     action="store_true")
    parser.add_argument("--gen",      action="store_true")
    parser.add_argument("--ctrl",     action="store_true")
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--h",        type=int, metavar="N")
    parser.add_argument("--cycle",    type=str, choices=["gen", "ctrl"])
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.matrix:
        print()
        print(show_matrix(use_color))
    elif args.same:
        print()
        print(show_by_dynamics(DYN_SAME, use_color))
    elif args.gen:
        print()
        print(show_by_dynamics(DYN_GEN, use_color))
    elif args.ctrl:
        print()
        print(show_by_dynamics(DYN_CTRL, use_color))
    elif args.all:
        print()
        print(show_by_dynamics(None, use_color))
    elif args.h is not None:
        if not 0 <= args.h <= 63:
            print("Ошибка: h должно быть 0..63")
            return
        print(show_hexagram(args.h, use_color))
    elif args.cycle:
        print()
        print(show_cycle_path(0, 5, args.cycle, use_color))
    else:
        print(show_overview(use_color))


if __name__ == "__main__":
    main()
