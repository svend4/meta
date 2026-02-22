"""
hexsovereign.py — 12 государевых гексаграмм (辟卦 pì guà)

В китайской традиции каждый лунный месяц «управляется» гексаграммой,
отражающей баланс ян/инь в природе. Эти 12 гексаграмм — единственный
монотонный путь Q6 от ян=0 (Кунь) до ян=6 (Цянь) и обратно.

h-путь: 0 → 1 → 3 → 7 → 15 → 31 → 63 → 62 → 60 → 56 → 48 → 32 → 0

Каждый шаг меняет РОВНО ОДИН бит — открывает или закрывает одну зону.
Порядок открытия зон Крюкова: ВЛ → ВП → СЛ → СП → НЛ → НП.
Порядок добавления цвета Лю-Синь:  R  →  Y  →  G  →  C  →  B  →  M.

Структура года:
  Зимнее солнцестояние (мес.11) → Кунь    ян=0  всё инь
  ...6 месяцев роста янской энергии...
  Летнее солнцестояние  (мес.5) → Цянь    ян=6  всё ян
  ...6 месяцев роста иньской энергии...
  → возврат к Куню

Сегодня: из даты выбирается управляющая государева гексаграмма.
"""

import sys
import os
import argparse
from datetime import date, timedelta
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, antipode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexliuxing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexboya"))

from hexiching import Hexagram, KW_FROM_H
from hexliuxing import ELEMENTS
from hexboya import ZONE_NAMES, BodyState

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}
ELEM_COLORS = ["\033[31m","\033[33m","\033[32m","\033[36m","\033[34m","\033[35m"]

# ---------------------------------------------------------------------------
# 12 государевых гексаграмм
# ---------------------------------------------------------------------------

# h-код, название, месяц (1=декабрь-январь, ...), ок. дата начала (Грег.)
SOVEREIGN = [
    # h     pin-yin     кит.   мес. рус.имя          approx_start (mm-dd)
    (0,  "Кунь",       "坤",   11,  "Земля",          "12-07"),
    (1,  "Фу",         "復",   12,  "Возврат",        "01-06"),
    (3,  "Линь",       "臨",    1,  "Приближение",    "02-04"),
    (7,  "Тай",        "泰",    2,  "Мир-Расцвет",    "03-06"),
    (15, "Да Чжуан",  "大壯",   3,  "Великая мощь",   "04-05"),
    (31, "Гуай",       "夬",    4,  "Решимость",      "05-06"),
    (63, "Цянь",       "乾",    5,  "Небо",           "06-06"),
    (62, "Гоу",        "姤",    6,  "Встреча",        "07-07"),
    (60, "Дунь",       "遯",    7,  "Отступление",    "08-07"),
    (56, "Пи",         "否",    8,  "Застой",         "09-08"),
    (48, "Гуань",      "觀",    9,  "Созерцание",     "10-08"),
    (32, "Бо",         "剝",   10,  "Распад",         "11-07"),
]

CHINESE_MONTHS = {
    11: "冬月 (зима)", 12: "腊月 (холод)",
     1: "正月 (весна)", 2: "二月",  3: "三月", 4: "四月",
     5: "午月 (лето)", 6: "未月",  7: "申月", 8: "酉月",
     9: "戌月 (осень)", 10: "亥月",
}

SEASON = {11: "зима", 12: "зима", 1: "весна", 2: "весна", 3: "весна",
          4: "весна", 5: "лето",  6: "лето",  7: "лето",  8: "осень",
          9: "осень", 10: "осень"}

PHASE = {  # ян или инь активна
    0: "дно инь", 1: "рост ян", 2: "рост ян", 3: "равновесие",
    4: "рост ян", 5: "рост ян", 6: "пик ян",
}


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _liu_str(h: int, use_color: bool = True) -> str:
    parts = []
    for i in range(6):
        if (h >> i) & 1:
            c = ELEM_COLORS[i] if use_color else ""
            parts.append(f"{c}{ELEMENTS[i]['short']}{RESET if use_color else ''}")
    return "+".join(parts) if parts else "∅"


def _zones_str(h: int, use_color: bool = True) -> str:
    parts = [ZONE_NAMES[i] for i in range(6) if (h >> i) & 1]
    return ",".join(parts) if parts else "∅"


def _date_to_sovereign(d: date) -> tuple:
    """Возвращает (индекс 0-11, %, h) государевой гексаграммы для даты d."""
    # Найдём ближайшее «начало» каждого месяца в текущем году
    def parse_mmdd(s, year):
        mm, dd = map(int, s.split("-"))
        return date(year, mm, dd)

    # Попробуем год d и год-1 для граничного случая
    best_idx = 0
    best_date = None
    for year in (d.year - 1, d.year):
        for idx, row in enumerate(SOVEREIGN):
            try:
                sd = parse_mmdd(row[5], year)
                if sd <= d:
                    if best_date is None or sd > best_date:
                        best_date = sd
                        best_idx = idx
            except ValueError:
                pass

    # Прогресс внутри месяца
    next_idx = (best_idx + 1) % 12
    try:
        next_date = parse_mmdd(SOVEREIGN[next_idx][5], d.year)
        if next_date < best_date:  # type: ignore
            next_date = parse_mmdd(SOVEREIGN[next_idx][5], d.year + 1)
        span = (next_date - best_date).days  # type: ignore
        elapsed = (d - best_date).days  # type: ignore
        pct = min(100, round(elapsed * 100 / span)) if span else 0
    except (ValueError, TypeError):
        pct = 0

    return best_idx, pct, SOVEREIGN[best_idx][0]


# ---------------------------------------------------------------------------
# Таблица 12 месяцев
# ---------------------------------------------------------------------------

def show_table(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    dim = DIM if use_color else ""

    lines = [
        "",
        f"{'═'*72}",
        f"  {bold}12 ГОСУДАРЕВЫХ ГЕКСАГРАММ — ГОДОВОЙ ЦИКЛ Q6{r}",
        f"  (辟卦 pì guà) · Путь h: 0→1→3→7→15→31→63→62→60→56→48→32→0",
        f"{'═'*72}",
        "",
        f"  {dim}Мес  Дата   h  ян  Гексаграмма        КВ#  Зоны Крюкова   Лю-Синь цвета{r}",
        "  " + "─" * 70,
    ]

    prev_h = 0
    for idx, (h, pin, chin, month, rname, start) in enumerate(SOVEREIGN):
        hx = Hexagram(h)
        yc = YANG_ANSI[hx.yang] if use_color else ""
        zones = _zones_str(h, use_color)
        liu = _liu_str(h, use_color)
        season = SEASON[month]
        chin_mo = CHINESE_MONTHS[month]

        # Изменённая зона в этом шаге
        if idx == 0:
            changed = "—"
        else:
            diff_bit = (prev_h ^ h).bit_length() - 1
            opened = (h >> diff_bit) & 1
            arrow = "▶" if opened else "◀"
            zn = ZONE_NAMES[diff_bit]
            ec = ELEM_COLORS[diff_bit] if use_color else ""
            changed = f"{ec}{arrow}{zn}{r}"
        prev_h = h

        lines.append(
            f"  {month:>2}  {start}  "
            f"{yc}{h:>2}({h:06b}){r}  "
            f"{yc}{hx.yang}{r}  "
            f"{hx.sym}{yc}{pin:<10}{r}  {chin}  "
            f"КВ#{hx.kw:>2}  "
            f"{changed:<12}  "
            f"[{liu}]"
        )

    # Замыкание: Bo→Kun
    diff_bit = (32 ^ 0).bit_length() - 1
    ec = ELEM_COLORS[diff_bit] if use_color else ""
    yc0 = YANG_ANSI[0] if use_color else ""
    lines += [
        "  " + "─" * 70,
        f"  11  12-07  {yc0} 0(000000){r}  "
        f"{yc0}0{r}  "
        f"{'䷁'}Кунь       坤  КВ#2  "
        f"{ec}◀{ZONE_NAMES[diff_bit]}{r}  → цикл замкнут",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Сегодня
# ---------------------------------------------------------------------------

def show_today(today: date = None, use_color: bool = True) -> str:
    if today is None:
        today = date.today()
    idx, pct, h = _date_to_sovereign(today)
    row = SOVEREIGN[idx]
    h, pin, chin, month, rname, start = row
    hx = Hexagram(h)

    yc = YANG_ANSI[hx.yang] if use_color else ""
    r = RESET if use_color else ""
    bold = BOLD if use_color else ""

    bar_len = 20
    filled = round(pct * bar_len / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    # Переход к следующей
    next_row = SOVEREIGN[(idx + 1) % 12]
    next_hx = Hexagram(next_row[0])
    diff_bit = (h ^ next_row[0]).bit_length() - 1
    ec = ELEM_COLORS[diff_bit] if use_color else ""
    opened = (next_row[0] >> diff_bit) & 1
    action = ("откроется" if opened else "закроется")
    elem_name = ELEMENTS[diff_bit]['name']
    zone_name = ZONE_NAMES[diff_bit]

    lines = [
        "",
        f"  {bold}СЕГОДНЯ: {today.strftime('%d.%m.%Y')}{r}",
        "",
        f"  Государева гексаграмма:  {yc}{hx.sym} {pin} ({chin}){r}",
        f"  КВ#{hx.kw:>2} «{rname}»  ян={yc}{hx.yang}{r}  h={h}({h:06b})",
        f"  Сезон: {SEASON[month]}  ({CHINESE_MONTHS[month]})",
        f"  Фаза:  {PHASE[hx.yang]}",
        "",
        f"  Прогресс месяца: [{yc}{bar}{r}] {pct}%",
        "",
        f"  Зоны Крюкова: {_zones_str(h, use_color) or '∅ (все закрыты)'}",
        f"  Цвета Лю-Синь: {_liu_str(h, use_color) or '∅'}",
        "",
        f"  Следующая: {next_hx.sym} {next_row[1]}  ({ec}{action} {zone_name} [{elem_name}]{r})",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Открытие зон — детальная разбивка
# ---------------------------------------------------------------------------

def show_zones_cycle(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    lines = [
        f"{bold}Цикл открытия зон Крюкова (янский полугод){r}",
        "",
        "  Зима → Лето: одна зона открывается каждый месяц.",
        "  Порядок: ВЛ→ВП→СЛ→СП→НЛ→НП  (сверху вниз, слева направо).",
        "  В теле это: верхний пояс → средний пояс → нижний пояс.",
        "",
        f"  {'Шаг':>4}  {'Зона':>4}  {'Цвет':>5}  {'Стихия':>12}  {'Месяц':>6}  {'h':>2} -> {'':<6}  {'Гексаграмма'}",
        "  " + "─" * 68,
    ]

    path = [row[0] for row in SOVEREIGN[:7]]
    for step in range(1, 7):
        h_from = path[step - 1]
        h_to   = path[step]
        bit    = (h_from ^ h_to).bit_length() - 1
        zn     = ZONE_NAMES[bit]
        elem   = ELEMENTS[bit]
        ec     = ELEM_COLORS[bit] if use_color else ""
        month  = SOVEREIGN[step][3]
        hx     = Hexagram(h_to)
        yc     = YANG_ANSI[hx.yang] if use_color else ""

        lines.append(
            f"  {step:>4}  {ec}{zn:>4}{r}  {ec}{elem['short']:>5}{r}"
            f"  {elem['name']:>12}"
            f"  мес.{month:>2}"
            f"  {h_from:>2}→{yc}{h_to:<6}{r}"
            f"  {hx.sym} {hx.kw:>3} {hx.name_pin[:16]}"
        )

    lines += [
        "",
        f"  {bold}Цикл закрытия зон (иньский полугод){r}",
        "  Лето → Зима: те же зоны закрываются в том же порядке.",
        "  (Одна и та же зона открывается и закрывается в своём месяце.)",
        "",
        f"  {'Шаг':>4}  {'Зона':>4}  {'Цвет':>5}  {'Стихия':>12}  {'Месяц':>6}  {'h':>2} -> {'':<6}  {'Гексаграмма'}",
        "  " + "─" * 68,
    ]

    path2 = [row[0] for row in SOVEREIGN[6:]]
    path2.append(0)
    for step in range(1, 7):
        h_from = path2[step - 1]
        h_to   = path2[step]
        bit    = (h_from ^ h_to).bit_length() - 1
        zn     = ZONE_NAMES[bit]
        elem   = ELEMENTS[bit]
        ec     = ELEM_COLORS[bit] if use_color else ""
        month  = SOVEREIGN[6 + step - 1][3] if 6 + step - 1 < 12 else 11
        hx     = Hexagram(h_to)
        yc     = YANG_ANSI[hx.yang] if use_color else ""

        lines.append(
            f"  {step:>4}  {ec}{zn:>4}{r}  {ec}{elem['short']:>5}{r}"
            f"  {elem['name']:>12}"
            f"  мес.{month:>2}"
            f"  {h_from:>2}→{yc}{h_to:<6}{r}"
            f"  {hx.sym} {hx.kw:>3} {hx.name_pin[:16]}"
        )

    lines += [
        "",
        "  ВЫВОД: каждая зона (= цвет = стихия) имеет свой",
        "  'янский месяц' (открытие) и 'иньский месяц' (закрытие).",
        "  Пара месяцев для зоны k = месяц (k+1) и месяц (12-k).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ян-синусоида: ASCII-диаграмма
# ---------------------------------------------------------------------------

def show_sinusoid(use_color: bool = True) -> str:
    hs   = [row[0] for row in SOVEREIGN] + [0]
    yangs = [yang_count(h) for h in hs]
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    lines = [
        f"{bold}Ян-синусоида государевых гексаграмм{r}",
        "",
    ]
    MAX_Y = 6
    for row_y in range(MAX_Y, -1, -1):
        row = f"  {row_y}│"
        for i, y in enumerate(yangs):
            yc = YANG_ANSI[row_y] if use_color else ""
            if y == row_y:
                row += f"{yc}◆{r}"
            elif y > row_y:
                if i == 0 or yangs[i-1] == y:
                    row += "│"
                else:
                    row += "│"
            else:
                row += " "
        lines.append(row)
    lines.append("   └" + "─" * 13 + "┬" + "─" * 13)
    # Подписи
    tick_row  = "    "
    label_row = "    "
    seasons = ["З","З","В","В","В","В","Л","Л","Л","О","О","О","З"]
    for i in range(13):
        tick_row  += f"{i+1:<1} "
        label_row += f"{seasons[i]:<1} "
    lines += [
        tick_row + "  (месяц)",
        label_row + "  З=зима В=весна Л=лето О=осень",
        "",
        "  ◆ = государева гексаграмма",
        "  Красивая: пик в мес.5 (Цянь ян=6), дно в мес.11 (Кунь ян=0).",
        "  Цикл строго симметричен: ян(месяц k) + ян(месяц k+6) = 6.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexsovereign — 12 государевых гексаграмм",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexsovereign.py               Таблица 12 месяцев
  python hexsovereign.py --today       Сегодняшняя гексаграмма
  python hexsovereign.py --date 2026-06-15  Гексаграмма на дату
  python hexsovereign.py --zones       Цикл открытия зон Крюкова
  python hexsovereign.py --wave        Ян-синусоида года
        """,
    )
    parser.add_argument("--today",    action="store_true")
    parser.add_argument("--date",     type=str, metavar="YYYY-MM-DD")
    parser.add_argument("--zones",    action="store_true")
    parser.add_argument("--wave",     action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.today:
        print(show_today(use_color=use_color))
    elif args.date:
        d = date.fromisoformat(args.date)
        print(show_today(today=d, use_color=use_color))
    elif args.zones:
        print()
        print(show_zones_cycle(use_color))
    elif args.wave:
        print()
        print(show_sinusoid(use_color))
    else:
        print(show_table(use_color))


if __name__ == "__main__":
    main()
