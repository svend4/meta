"""
hexcalendar.py — Практический гексаграммный календарь

Три временных уровня:

1. ЧАС  → один из 12 «двойных часов» (时辰 shí chén), каждый 2 ч.
   Соответствие: с полуночи (子 Zǐ 23:00) по порядку ян нарастает,
   затем убывает. Каждый двойной час = одна суверенная гексаграмма.

2. МЕСЯЦ → один из 12 лунных месяцев / 24 циклических узлов (节气)
   Суверенные гексаграммы расположены по ян-уровню: 0→1→...→6→5→...→1

3. ГОД  → 64-летний цикл (по 8×8 = последовательность Вэнь-вана)
   Эпоха: 2000 г. = КВ#1 (Цянь)

Двенадцать суверенных гексаграмм и их время:

  子 Zǐ    23–01  ䷗ Фу  (#24)  ян=1  (зимнее солнцестояние, полночь, новый импульс)
  丑 Chǒu  01–03  ䷒ Линь (#19)  ян=2
  寅 Yín   03–05  ䷊ Тай  (#11)  ян=3  (нижняя точка ночи, переход)
  卯 Mǎo   05–07  ䷡ Да Чжуан (#34) ян=4  (рассвет)
  辰 Chén  07–09  ䷪ Гуай (#43) ян=5
  巳 Sì    09–11  ䷀ Цянь (#1)  ян=6  (утренний максимум)
  午 Wǔ    11–13  ䷫ Гоу  (#44) ян=5  (полдень, первое убывание)
  未 Wèi   13–15  ䷠ Дунь (#33) ян=4
  申 Shēn  15–17  ䷋ Пи   (#12) ян=3
  酉 Yǒu   17–19  ䷓ Гуань (#20) ян=2  (закат)
  戌 Xū    19–21  ䷖ Бо   (#23) ян=1
  亥 Hài   21–23  ䷁ Кунь (#2)  ян=0  (полная ночь)
"""

import sys
import os
import argparse
from datetime import datetime, date, timezone

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

# 12 суверенных гексаграмм (в порядке суток / года)
# h-значения, ян-уровень 1→6→0
SOVEREIGN_H = [1, 3, 7, 15, 31, 63, 62, 60, 56, 48, 32, 0]

# Двойные часы (时辰): начало в 23:00 (子时 Zǐ)
DOUBLE_HOURS = [
    ("子 Zǐ",    23, 1,  "полночь, зарождение"),
    ("丑 Chǒu",   1, 3,  "глубокая ночь"),
    ("寅 Yín",    3, 7,  "до рассвета"),
    ("卯 Mǎo",    5, 15, "рассвет"),
    ("辰 Chén",   7, 31, "утро"),
    ("巳 Sì",     9, 63, "утренний максимум"),
    ("午 Wǔ",    11, 62, "полдень"),
    ("未 Wèi",   13, 60, "послеполдень"),
    ("申 Shēn",  15, 56, "день"),
    ("酉 Yǒu",   17, 48, "закат"),
    ("戌 Xū",    19, 32, "вечер"),
    ("亥 Hài",   21,  0, "ночь"),
]

# Месяцы (западный календарь → ближайшая суверенная гексаграмма)
MONTH_H = {
    1:  7,   # Январь  → Тай (#11)   (после зимнего солнцестояния)
    2:  15,  # Февраль → Да Чжуан (#34)
    3:  31,  # Март    → Гуай (#43)  (весеннее равноденствие)
    4:  63,  # Апрель  → Цянь (#1)
    5:  62,  # Май     → Гоу (#44)   (максимум ян, начало убывания)
    6:  60,  # Июнь    → Дунь (#33)  (летнее солнцестояние)
    7:  56,  # Июль    → Пи (#12)
    8:  48,  # Август  → Гуань (#20)
    9:  32,  # Сентябрь→ Бо (#23)    (осеннее равноденствие)
    10:  0,  # Октябрь → Кунь (#2)
    11:  1,  # Ноябрь  → Фу (#24)    (зимнее солнцестояние)
    12:  3,  # Декабрь → Линь (#19)
}

# Первый год 64-летнего цикла = 2000 г. → КВ#1 Цянь
EPOCH_YEAR = 2000
KW_TO_H = {Hexagram(h).kw: h for h in range(64)}


# ---------------------------------------------------------------------------
# Вычисление гексаграмм по времени
# ---------------------------------------------------------------------------

def hour_hex(hour: int) -> tuple[int, tuple]:
    """
    Возвращает (h, dh_entry) для данного часа (0-23).
    Зǐ-время начинается в 23:00.
    """
    # Нормализуем: 23:00 = индекс 0
    idx = ((hour - 23) % 24) // 2
    dh  = DOUBLE_HOURS[idx]
    return dh[2], dh


def month_hex(month: int) -> int:
    return MONTH_H.get(month, 63)


def year_hex(year: int) -> tuple[int, int]:
    """
    Возвращает (h, kw) для данного года в 64-летнем цикле.
    Эпоха: 2000 → KW#1 (Цянь).
    """
    idx = (year - EPOCH_YEAR) % 64
    kw  = idx + 1
    h   = KW_TO_H[kw]
    return h, kw


def age_hex(birth_year: int, current_year: int | None = None) -> tuple[int, int]:
    """
    Гексаграмма возраста: сколько лет прошло с рождения.
    Каждые 64 года = полный цикл.
    """
    if current_year is None:
        current_year = datetime.now().year
    age = current_year - birth_year
    kw  = (age % 64) + 1
    h   = KW_TO_H[kw]
    return h, age


# ---------------------------------------------------------------------------
# Текущий момент
# ---------------------------------------------------------------------------

def now_reading(use_color: bool = True, dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now()

    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    h_hr, dh_entry = hour_hex(dt.hour)
    h_mo           = month_hex(dt.month)
    h_yr, kw_yr    = year_hex(dt.year)

    def hx_line(h: int, extra: str = "") -> str:
        hx = Hexagram(h)
        yc = YANG_ANSI[hx.yang] if use_color else ""
        return (
            f"  {yc}{hx.sym}  КВ#{hx.kw:>2} «{hx.name_cn} {hx.name_pin}»"
            f"  ян={hx.yang}  h={h:>2} ({h:06b}){r}"
            + (f"  {dim}{extra}{r}" if extra else "")
        )

    lines = [
        "",
        "═"*64,
        f"  {bold}ГЕКСАГРАММНЫЙ МОМЕНТ{r}",
        f"  {dim}{dt.strftime('%Y-%m-%d  %H:%M')}{r}",
        "═"*64,
        "",
        f"  {bold}ЧАС   {dh_entry[0]} ({dh_entry[1]:02d}:00–{(dh_entry[1]+2)%24:02d}:00):{r}",
        hx_line(h_hr, dh_entry[3]),
        "",
        f"  {bold}МЕСЯЦ  {dt.strftime('%B (%m)')}:{r}",
        hx_line(h_mo),
        "",
        f"  {bold}ГОД    {dt.year}  (КВ#{kw_yr}, цикл {(dt.year-EPOCH_YEAR)//64+1}):{r}",
        hx_line(h_yr, f"2000+{(dt.year-EPOCH_YEAR)%64} из 64"),
        "",
    ]

    # Триада момента
    hx_h = Hexagram(h_hr)
    hx_m = Hexagram(h_mo)
    hx_y = Hexagram(h_yr)
    avg_yang = (hx_h.yang + hx_m.yang + hx_y.yang) / 3
    lines += [
        "─"*64,
        f"  {bold}ТРИАДА: час × месяц × год{r}",
        f"  {YANG_ANSI[hx_h.yang] if use_color else ''}{hx_h.sym}{r}"
        f" × {YANG_ANSI[hx_m.yang] if use_color else ''}{hx_m.sym}{r}"
        f" × {YANG_ANSI[hx_y.yang] if use_color else ''}{hx_y.sym}{r}",
        f"  Среднее ян: {avg_yang:.1f}  (из 6)",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Анализ дня рождения
# ---------------------------------------------------------------------------

def birthday_reading(birth_year: int, use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    current_year = datetime.now().year
    h_by, kw_by = year_hex(birth_year)
    h_ag, age   = age_hex(birth_year, current_year)

    hx_b = Hexagram(h_by)
    hx_a = Hexagram(h_ag)

    def hx_line(h, label=""):
        hx = Hexagram(h)
        yc = YANG_ANSI[hx.yang] if use_color else ""
        return (
            f"  {yc}{hx.sym}  КВ#{hx.kw:>2} «{hx.name_cn} {hx.name_pin}»"
            f"  ян={hx.yang}  h={h:>2}{r}"
            + (f"  {dim}{label}{r}" if label else "")
        )

    # Год рождения → КВ в 64-летнем цикле
    year_in_cycle = (birth_year - EPOCH_YEAR) % 64 + 1

    lines = [
        "",
        "═"*64,
        f"  {bold}ГЕКСАГРАММА РОЖДЕНИЯ И ВОЗРАСТА{r}",
        f"  {dim}Год рождения: {birth_year}  |  Возраст: {age} лет  |  Текущий: {current_year}{r}",
        "═"*64,
        "",
        f"  {bold}Гексаграмма года рождения (64-летний цикл):{r}",
        f"  {dim}({birth_year} − 2000) mod 64 + 1 = КВ#{kw_by}{r}",
        hx_line(h_by, f"год {birth_year}"),
        "",
        f"  {bold}Гексаграмма текущего возраста ({age} лет):{r}",
        f"  {dim}возраст mod 64 = {age%64} → КВ#{age%64+1}{r}",
        hx_line(h_ag, f"возраст {age}"),
        "",
    ]

    # Ядерная гексаграмма возраста
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from hexnuclear import nuclear
        h_nuc = nuclear(h_ag)
        hx_n  = Hexagram(h_nuc)
        yc = YANG_ANSI[hx_n.yang] if use_color else ""
        lines += [
            f"  {bold}Ядерная гексаграмма возраста (互卦):{r}",
            f"  {yc}{hx_n.sym}  КВ#{hx_n.kw:>2} «{hx_n.name_cn} {hx_n.name_pin}»"
            f"  ян={hx_n.yang}  h={h_nuc}{r}",
            "",
        ]
    except ImportError:
        pass

    # Суверенная гексаграмма месяца рождения (если задан)
    lines += [
        "─"*64,
        f"  {bold}64-летний цикл: взгляд на всю жизнь{r}",
        "",
    ]

    # Показываем 5-летние диапазоны вокруг текущего возраста
    cur_age_mod = age % 64
    for delta in range(-2, 8):
        a  = cur_age_mod + delta
        if not 0 <= a <= 63:
            continue
        kw = a + 1
        h  = KW_TO_H[kw]
        hx = Hexagram(h)
        yc = YANG_ANSI[hx.yang] if use_color else ""
        marker = " ◄ СЕЙЧАС" if delta == 0 else ""
        real_year = birth_year + age - cur_age_mod + a
        lines.append(
            f"  {yc}{hx.sym}{r} возраст {a:>2}  ({real_year:>4})  "
            f"{yc}КВ#{kw:>2} «{hx.name_pin}»{r}{marker}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Таблица двойных часов
# ---------------------------------------------------------------------------

def show_clock(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    lines = [
        f"  {bold}12 двойных часов (时辰) → суверенные гексаграммы{r}",
        "",
        f"  {'时辰':>10}  {'часы':>9}  {'ян':>3}  символ  КВ#  Имя             Описание",
        "  " + "─"*72,
    ]

    now_hour = datetime.now().hour
    for i, (zh, start, h, desc) in enumerate(DOUBLE_HOURS):
        end = (start + 2) % 24
        hx  = Hexagram(h)
        yc  = YANG_ANSI[hx.yang] if use_color else ""
        cur = now_hour >= start and now_hour < end
        cur = cur or (start == 23 and (now_hour >= 23 or now_hour < 1))
        marker = " ◄" if cur else ""
        lines.append(
            f"  {yc}{zh:>10}  {start:02d}:00–{end:02d}:00  {hx.yang}  "
            f"  {hx.sym}  #{hx.kw:>2}  {hx.name_pin[:16]:<16}  {dim}{desc}{r}"
            f"{yc}{marker}{r}"
        )

    return "\n".join(lines)


def show_year_cycle(start_year: int = 2000, use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    lines = [
        f"  {bold}64-летний цикл КВ (начало: {start_year}){r}",
        "",
        f"  {'Год':>6}  {'КВ':>4}  {'ян':>3}  Гексаграмма",
        "  " + "─"*50,
    ]

    for idx in range(64):
        year = start_year + idx
        kw   = idx + 1
        h    = KW_TO_H[kw]
        hx   = Hexagram(h)
        yc   = YANG_ANSI[hx.yang] if use_color else ""
        cur  = year == datetime.now().year
        marker = " ◄" if cur else ""
        lines.append(
            f"  {yc}{year:>6}  КВ#{kw:>2}  {hx.yang}  {hx.sym} «{hx.name_pin}»{r}{marker}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexcalendar — гексаграммный календарь",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexcalendar.py             Текущий момент (час + месяц + год)
  python hexcalendar.py --clock     Таблица 12 двойных часов
  python hexcalendar.py --cycle     64-летний цикл 2000-2063
  python hexcalendar.py --born 1987 Анализ для рождённого в 1987 г.
  python hexcalendar.py --year 2026 Гексаграмма конкретного года
  python hexcalendar.py --hour 15   Гексаграмма для 15:00
        """,
    )
    parser.add_argument("--clock",    action="store_true")
    parser.add_argument("--cycle",    action="store_true")
    parser.add_argument("--born",     type=int, metavar="YEAR")
    parser.add_argument("--year",     type=int, metavar="YEAR")
    parser.add_argument("--hour",     type=int, metavar="H")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.clock:
        print()
        print(show_clock(use_color))
    elif args.cycle:
        print()
        print(show_year_cycle(2000, use_color))
    elif args.born:
        print(birthday_reading(args.born, use_color))
    elif args.year:
        h, kw = year_hex(args.year)
        hx = Hexagram(h)
        yc = YANG_ANSI[hx.yang] if use_color else ""
        print(f"\n  {yc}{args.year} → КВ#{kw} {hx.sym} «{hx.name_cn} {hx.name_pin}»  h={h}  ян={hx.yang}{RESET if use_color else ''}\n")
    elif args.hour is not None:
        h, dh = hour_hex(args.hour)
        hx = Hexagram(h)
        yc = YANG_ANSI[hx.yang] if use_color else ""
        print(f"\n  {yc}{args.hour:02d}:xx → {dh[0]} {hx.sym} «{hx.name_pin}»  h={h}  ян={hx.yang}{RESET if use_color else ''}\n")
    else:
        print(now_reading(use_color))


if __name__ == "__main__":
    main()
