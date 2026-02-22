"""
hexkua.py — 九宮八卦: число Куа, Ло-Шу и система Восьми Дворцов (八宅 Bā Zhái)

Число Куа (卦數 guà shù) — личный нумерологический ключ в китайской метафизике.
Определяется по году рождения и полу. Связывает человека с одной из восьми
триграмм, восемью сторонами света и системой Ло-Шу (洛書 luò shū).

══════════════════════════════════════════════════════════════════════
МАГИЧЕСКИЙ КВАДРАТ ЛО-ШУ (洛書, основа Восьми Дворцов)
══════════════════════════════════════════════════════════════════════

      С(1·Кань)
 СЗ(6·Цянь) СВ(8·Гэнь)
З(7·Дуй)  ◈(5)  В(3·Чжэнь)
 ЮЗ(2·Кунь) ЮВ(4·Сюнь)
      Ю(9·Ли)

Сумма по любой линии = 15. Каждое число 1-9 встречается ровно один раз.
Числа 1-4,6-9 → 8 триграмм, число 5 → Земля-Центр.

══════════════════════════════════════════════════════════════════════
ВЫЧИСЛЕНИЕ ЧИСЛА КУА
══════════════════════════════════════════════════════════════════════

1. Суммируем цифры года рождения до однозначного (цифровой корень).
2. Для рождённых до 2000 г.:
   • Мужчины:   Куа = 10 − dr  (если 5 → 2)
   • Женщины:   Куа =  5 + dr  (если > 9 → −9; если 5 → 8)
3. Для рождённых с 2000 г.:
   • Мужчины:   Куа =  9 − dr  (если 0 → 9; если 5 → 2)
   • Женщины:   Куа =  6 + dr  (если > 9 → −9; если 5 → 8)

Число 5 не является стабильным Куа:
  Мужчины с Куа=5 → используют Куа=2 (Кунь)
  Женщины с Куа=5 → используют Куа=8 (Гэнь)

══════════════════════════════════════════════════════════════════════
ВОСТОЧНАЯ и ЗАПАДНАЯ ГРУППА
══════════════════════════════════════════════════════════════════════

Восточная группа (木/水/火 Дерево/Вода/Огонь):  Куа 1, 3, 4, 9
  Благоприятные направления: Север, Юг, Восток, Юго-Восток

Западная группа (土/金 Земля/Металл):           Куа 2, 6, 7, 8
  Благоприятные направления: Северо-Запад, Запад, Северо-Восток, Юго-Запад
"""

import sys
import os
import argparse
from datetime import datetime

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
GRN = "\033[92m"
YLW = "\033[93m"
CYN = "\033[96m"
RED = "\033[91m"
MAG = "\033[95m"

# ---------------------------------------------------------------------------
# Данные: Ло-Шу
# ---------------------------------------------------------------------------

# Направления компаса → номер Куа (и триграмма)
DIR_KUA = {
    "N":  1,   # Север     → Кань ☵
    "NE": 8,   # Северо-В. → Гэнь ☶
    "E":  3,   # Восток    → Чжэнь ☳
    "SE": 4,   # Юго-В.    → Сюнь ☴
    "S":  9,   # Юг        → Ли ☲
    "SW": 2,   # Юго-З.    → Кунь ☷
    "W":  7,   # Запад     → Дуй ☱
    "NW": 6,   # Северо-З. → Цянь ☰
    "C":  5,   # Центр     → Земля
}

# Ло-Шу как 3×3 сетка (строки: С → Ю, столбцы: З → В)
# Смотрим «по карте» с Севером вверху
LO_SHU = [
    [(6, "СЗ", "NW"), (1, "С",  "N"),  (8, "СВ", "NE")],
    [(7, "З",  "W"),  (5, "Ц",  "C"),  (3, "В",  "E")],
    [(2, "ЮЗ", "SW"), (9, "Ю",  "S"),  (4, "ЮВ", "SE")],
]

# Триграммы: значение Куа → (триграмма, h-lower, стихия, символ)
KUA_DATA = {
    # trig_h — 3-битный номер: 0=Кунь,1=Чжэнь,2=Кань,3=Дуй,4=Гэнь,5=Ли,6=Сюнь,7=Цянь
    1: {"name_cn": "坎 Кань",   "trig_h": 2,  "elem": "Вода 水",  "dir": "N",  "sym": "☵", "name": "Кань"},
    2: {"name_cn": "坤 Кунь",   "trig_h": 0,  "elem": "Земля 土", "dir": "SW", "sym": "☷", "name": "Кунь"},
    3: {"name_cn": "震 Чжэнь",  "trig_h": 1,  "elem": "Дерево 木","dir": "E",  "sym": "☳", "name": "Чжэнь"},
    4: {"name_cn": "巽 Сюнь",   "trig_h": 6,  "elem": "Дерево 木","dir": "SE", "sym": "☴", "name": "Сюнь"},
    5: {"name_cn": "中 Центр",  "trig_h": -1, "elem": "Земля 土", "dir": "C",  "sym": "☯", "name": "Центр"},
    6: {"name_cn": "乾 Цянь",   "trig_h": 7,  "elem": "Металл 金","dir": "NW", "sym": "☰", "name": "Цянь"},
    7: {"name_cn": "兌 Дуй",    "trig_h": 3,  "elem": "Металл 金","dir": "W",  "sym": "☱", "name": "Дуй"},
    8: {"name_cn": "艮 Гэнь",   "trig_h": 4,  "elem": "Земля 土", "dir": "NE", "sym": "☶", "name": "Гэнь"},
    9: {"name_cn": "離 Ли",     "trig_h": 5,  "elem": "Огонь 火", "dir": "S",  "sym": "☲", "name": "Ли"},
}

# Восточная/западная группа
EAST_GROUP = {1, 3, 4, 9}
WEST_GROUP = {2, 6, 7, 8}

EAST_DIRS = ["N", "S", "E", "SE"]
WEST_DIRS = ["NW", "W", "NE", "SW"]

# 八宅 Ba Zhai: [ШэнЦи, ТяньИ, ЯньНянь, ФуВэй, ХуоХай, ЛюШа, УГуй, ЦзюэМин]
BA_ZHAI = {
    1: ["SE", "E",  "S",  "N",  "SW", "NE", "W",  "NW"],
    2: ["NE", "W",  "NW", "SW", "E",  "SE", "N",  "S"],
    3: ["S",  "N",  "SE", "E",  "W",  "SW", "NE", "NW"],
    4: ["N",  "S",  "E",  "SE", "NW", "W",  "SW", "NE"],
    6: ["W",  "NE", "SW", "NW", "SE", "E",  "N",  "S"],
    7: ["NW", "SW", "NE", "W",  "E",  "S",  "SE", "N"],
    8: ["SW", "NW", "W",  "NE", "S",  "N",  "SE", "E"],
    9: ["E",  "SE", "N",  "S",  "NE", "NW", "SW", "W"],
}
BA_ZHAI_NAMES = ["生氣 Шэн Ци (витальность)", "天醫 Тянь И (здоровье)",
                 "延年 Янь Нянь (долголетие)", "伏位 Фу Вэй (стабильность)",
                 "禍害 Хуо Хай (неудачи)",    "六煞 Лю Ша (препятствия)",
                 "五鬼 У Гуй (хаос)",          "絕命 Цзюэ Мин (опасность)"]

DIR_RU = {"N": "С", "NE": "СВ", "E": "В", "SE": "ЮВ",
          "S": "Ю", "SW": "ЮЗ", "W": "З", "NW": "СЗ", "C": "Ц"}

# ---------------------------------------------------------------------------
# Вычисление числа Куа
# ---------------------------------------------------------------------------

def digital_root(year: int) -> int:
    s = sum(int(d) for d in str(abs(year)))
    while s > 9:
        s = sum(int(d) for d in str(s))
    return s


def kua_number(birth_year: int, gender: str) -> tuple[int, int]:
    """
    Возвращает (kua, raw) — число Куа (1-9, без 5) и сырое значение.
    gender: 'M' (мужской) или 'F' (женский).
    """
    dr = digital_root(birth_year)
    if birth_year >= 2000:
        k = (9 - dr) if gender.upper() == "M" else (6 + dr)
    else:
        k = (10 - dr) if gender.upper() == "M" else (5 + dr)

    if k > 9:
        k -= 9
    if k <= 0:
        k += 9
    raw = k
    if k == 5:
        k = 2 if gender.upper() == "M" else 8
    return k, raw


def personal_hex(kua: int) -> int | None:
    """Личная гексаграмма = удвоенная триграмма Куа."""
    t = KUA_DATA[kua]["trig_h"]
    if t < 0:
        return None
    return t | (t << 3)


# ---------------------------------------------------------------------------
# Отображение Ло-Шу
# ---------------------------------------------------------------------------

def show_loshu(highlight_kua: int | None = None, use_color: bool = True) -> str:
    """Отображает магический квадрат Ло-Шу с триграммами."""
    lines = []
    for row in LO_SHU:
        cells = []
        for kua_n, dir_ru, dir_en in row:
            data = KUA_DATA[kua_n]
            sym  = data["sym"]
            hl   = (kua_n == highlight_kua) and use_color
            if hl:
                cell = f"\033[7m{sym}{kua_n}{RESET}"
            else:
                yc = YANG_ANSI.get(yang_count(data["trig_h"]) if data["trig_h"] >= 0 else 3, "")
                rc = RESET if use_color else ""
                cell = f"{yc if use_color else ''}{sym}{kua_n}{rc}"
            cells.append(cell)
        lines.append("  " + "   ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Полное чтение
# ---------------------------------------------------------------------------

def full_reading(birth_year: int, gender: str, use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    kua, raw = kua_number(birth_year, gender)
    data     = KUA_DATA[kua]
    group    = "Восточная" if kua in EAST_GROUP else "Западная"
    fav_dirs = EAST_DIRS if kua in EAST_GROUP else WEST_DIRS

    trig_h   = data["trig_h"]
    hex_h    = personal_hex(kua)
    hx       = Hexagram(hex_h) if hex_h is not None else None

    dr = digital_root(birth_year)

    # цвет группы
    gc = GRN if use_color and kua in EAST_GROUP else (MAG if use_color else "")

    lines = [
        "",
        "═"*64,
        f"  {bold}九宮八卦 — ЛИЧНОЕ ЧИСЛО КУА{r}",
        f"  {dim}Год рождения: {birth_year}  |  Пол: {'Мужской' if gender.upper()=='M' else 'Женский'}{r}",
        "═"*64,
        "",
        f"  Цифровой корень года: {dr}",
        f"  {'10' if birth_year < 2000 else ' 9'} − {dr} = {raw}"
        if gender.upper() == "M"
        else f"  {'5' if birth_year < 2000 else '6'} + {dr} = {raw}",
        "",
        f"  {bold}Число Куа:{r}  {gc}{kua}  {data['name_cn']}  {data['sym']}{r}",
        f"  Стихия:         {data['elem']}",
        f"  Направление:    {DIR_RU[data['dir']]} ({data['dir']})",
        f"  Группа:         {gc}{group}{r}",
        "",
    ]

    # Личная гексаграмма
    if hx:
        yc = YANG_ANSI[hx.yang] if use_color else ""
        lines += [
            f"  {bold}Личная гексаграмма (удвоенная триграмма):{r}",
            f"  {yc}{hx.sym} КВ#{hx.kw:>2} «{hx.name_cn} {hx.name_pin}»  ян={hx.yang}  h={hex_h}{r}",
            "",
        ]

    # Ло-Шу
    lines += [
        f"  {bold}Магический квадрат Ло-Шу:{r}",
        f"  {dim}(ваше Куа выделено){r}",
        "",
        show_loshu(kua, use_color),
        "",
        f"  {'С' if use_color else ''}Сумма по любой линии = 15{r if use_color else ''}",
        "",
    ]

    # 八宅 Ba Zhai
    if kua in BA_ZHAI:
        bz = BA_ZHAI[kua]
        lines += [
            f"  {bold}八宅 Бā Жái — Восемь Дворцов:{r}",
            "",
        ]
        for i, (dir_en, bz_name) in enumerate(zip(bz, BA_ZHAI_NAMES)):
            d_ru   = DIR_RU.get(dir_en, dir_en)
            kua_d  = DIR_KUA.get(dir_en, 0)
            d_data = KUA_DATA.get(kua_d, {})
            good   = i < 4
            gc2    = (GRN if use_color else "") if good else (RED if use_color else "")
            sign   = "✦" if good else "✕"
            lines.append(
                f"  {gc2}{sign} {d_ru:>3} ({dir_en:>2})  "
                f"{d_data.get('sym','')}  {bz_name}{r}"
            )
        lines.append("")

    # Групповые направления (упрощённо)
    fav_str = " ".join(DIR_RU[d] for d in fav_dirs)
    unfav   = [d for d in ["N","NE","E","SE","S","SW","W","NW"] if d not in fav_dirs]
    unfav_str = " ".join(DIR_RU[d] for d in unfav)
    lines += [
        f"  {bold}Группа {group}:{r}",
        f"  {GRN if use_color else ''}Благоприятные:  {fav_str}{r}",
        f"  {RED if use_color else ''}Неблагоприятные:{unfav_str}{r}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Таблица всех Куа
# ---------------------------------------------------------------------------

def show_table(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    lines = [
        f"  {bold}Таблица чисел Куа (все 8 активных){r}",
        "",
        f"  {'Куа':>4}  {'Симв.':>5}  {'Триграмма':>12}  {'Стихия':>10}  "
        f"{'Напр.':>6}  {'Группа':>12}  Гексаграмма",
        "  " + "─"*72,
    ]

    for k in [1, 2, 3, 4, 6, 7, 8, 9]:
        d    = KUA_DATA[k]
        t    = d["trig_h"]
        h    = t | (t << 3) if t >= 0 else None
        hx   = Hexagram(h) if h is not None else None
        grp  = "Восточная" if k in EAST_GROUP else "Западная"
        yc   = YANG_ANSI[yang_count(t)] if use_color and t >= 0 else ""
        gc   = GRN if use_color and k in EAST_GROUP else (MAG if use_color else "")
        hstr = f"{hx.sym} КВ#{hx.kw:>2} «{hx.name_pin}»" if hx else "—"
        lines.append(
            f"  {yc}{k:>4}  {d['sym']:>5}  {d['name_cn']:>12}  "
            f"{d['elem']:>10}  {DIR_RU[d['dir']]:>4}({d['dir']:<2})  "
            f"{gc}{grp:<12}{r}  {yc}{hstr}{r}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexkua — 九宮八卦: число Куа и система Восьми Дворцов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexkua.py --born 1987 --gender M     Мужчина, 1987 г.р.
  python hexkua.py --born 1990 --gender F     Женщина, 1990 г.р.
  python hexkua.py --table                    Таблица всех 8 Куа
  python hexkua.py --loshu                    Квадрат Ло-Шу
  python hexkua.py --born 1956 --gender M --year 2026   + год календаря
        """,
    )
    parser.add_argument("--born",   type=int, metavar="YEAR")
    parser.add_argument("--gender", type=str, metavar="M|F", default="M")
    parser.add_argument("--table",  action="store_true")
    parser.add_argument("--loshu",  action="store_true")
    parser.add_argument("--year",   type=int, metavar="YEAR")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.table:
        print()
        print(show_table(use_color))
    elif args.loshu:
        print()
        print(f"  {'Магический квадрат Ло-Шу' if not use_color else BOLD+'Магический квадрат Ло-Шу'+RESET}")
        print()
        print(show_loshu(None, use_color))
        print()
    elif args.born:
        print(full_reading(args.born, args.gender, use_color))
        # Если указан год — добавить год по календарю
        if args.year:
            try:
                sys.path.insert(0, os.path.dirname(__file__))
                from hexcalendar import year_hex
                h_yr, kw_yr = year_hex(args.year)
                hx = Hexagram(h_yr)
                yc = YANG_ANSI[hx.yang] if use_color else ""
                print(
                    f"\n  {'Год '+str(args.year)+' (64-летний цикл)':}\n"
                    f"  {yc}{hx.sym} КВ#{kw_yr} «{hx.name_pin}»  h={h_yr}  ян={hx.yang}{RESET if use_color else ''}\n"
                )
            except ImportError:
                pass
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
