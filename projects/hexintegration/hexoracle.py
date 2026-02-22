"""
hexoracle.py — Аутентичная консультация И Цзин

Три метода бросания:

  «Монеты»  (3 монеты, современный):
    орёл=3, решка=2; сумма линии:
    6 = ррр (1/8)   — 老阴 lǎo yīn  (старый инь)  → меняется в ян
    7 = оор+оро+роо (3/8) — 少阳 shào yáng (молодой ян) → стабильная
    8 = ооо+оот+ото (3/8) — 少阴 shào yīn  (молодой инь) → стабильная
    9 = ооо (1/8)   — 老阳 lǎo yáng  (старый ян)  → меняется в инь

  «Тысячелистник» (yarrow stalks, традиционный):
    49 стеблей делятся трижды; распределение:
    6  = 1/16  (старый инь)
    7  = 5/16  (молодой ян)
    8  = 7/16  (молодой инь)
    9  = 3/16  (старый ян)

  «Равномерный» (для обучения / чистый Q6):
    6,7,8,9 с равной вероятностью 1/4

Линии строятся снизу вверх (позиция 0 = первая/нижняя линия).
Менее вероятные (старые) линии изменяются во вторичной гексаграмме:
  6 → 1 (был инь, станет ян)
  9 → 0 (был ян, станет инь)
  7,8 → остаются теми же
"""

import sys
import os
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../libs/hexcore"))
from hexcore import yang_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
from hexiching import Hexagram

# Попытка подключить модули интеграции
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from hexdna import h_to_codon, CODON_TABLE
    HAS_DNA = True
except ImportError:
    HAS_DNA = False

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
YANG_ANSI = {
    0: "\033[90m", 1: "\033[94m", 2: "\033[96m",
    3: "\033[92m", 4: "\033[93m", 5: "\033[95m", 6: "\033[97m",
}
RED  = "\033[91m"
GRN  = "\033[92m"
YLW  = "\033[93m"
MAG  = "\033[95m"
CYN  = "\033[96m"

# Описания шести позиций (снизу вверх)
LINE_POSITIONS = [
    "1-я (нижняя) — земля",
    "2-я",
    "3-я — переход",
    "4-я",
    "5-я — правитель",
    "6-я (верхняя) — небо",
]

LINE_NAMES = {
    6: ("老阴 lǎo yīn",  "старый инь",  "──── (○) ────", "меняется в ян"),
    7: ("少阳 shào yáng","молодой ян",  "─────────────", "стабильная"),
    8: ("少阴 shào yīn", "молодой инь", "────  ────", "стабильная"),
    9: ("老阳 lǎo yáng", "старый ян",   "──── (×) ────", "меняется в инь"),
}

LINE_YANG = {6: 0, 7: 1, 8: 0, 9: 1}   # является ли линия яном
LINE_MOVING = {6: True, 7: False, 8: False, 9: True}  # меняется ли


# ---------------------------------------------------------------------------
# Методы бросания
# ---------------------------------------------------------------------------

def throw_coins() -> int:
    """3 монеты: орёл=3, решка=2. Возвращает 6,7,8,9."""
    return sum(random.choice((2, 3)) for _ in range(3))


def throw_yarrow() -> int:
    """
    Имитация стеблей тысячелистника.
    Распределение: 6→1/16, 7→5/16, 8→7/16, 9→3/16.
    """
    p = random.randint(1, 16)
    if   p <= 1:  return 6
    elif p <= 6:  return 7
    elif p <= 13: return 8
    else:          return 9


def throw_uniform() -> int:
    """Равномерное: 6,7,8,9 с вероятностью 1/4."""
    return random.choice((6, 7, 8, 9))


METHODS = {
    "coins":   throw_coins,
    "yarrow":  throw_yarrow,
    "uniform": throw_uniform,
}

METHOD_PROBS = {
    "coins":   {6: "1/8",  7: "3/8", 8: "3/8", 9: "1/8"},
    "yarrow":  {6: "1/16", 7: "5/16", 8: "7/16", 9: "3/16"},
    "uniform": {6: "1/4",  7: "1/4",  8: "1/4",  9: "1/4"},
}


# ---------------------------------------------------------------------------
# Консультация
# ---------------------------------------------------------------------------

def consult(method: str = "coins", seed: int | None = None) -> dict:
    """
    Выполняет гадание: 6 бросков → 6 линий снизу вверх.
    Возвращает словарь с первичной и вторичной гексаграммами.
    """
    if seed is not None:
        random.seed(seed)

    throw_fn = METHODS.get(method, throw_coins)
    lines = [throw_fn() for _ in range(6)]   # lines[0] = нижняя

    # Первичная гексаграмма
    h_primary = sum(LINE_YANG[v] << i for i, v in enumerate(lines))

    # Вторичная гексаграмма (меняем подвижные линии)
    h_secondary = h_primary
    for i, v in enumerate(lines):
        if LINE_MOVING[v]:
            h_secondary ^= (1 << i)

    has_changes = any(LINE_MOVING[v] for v in lines)
    changing_positions = [i for i, v in enumerate(lines) if LINE_MOVING[v]]

    return {
        "method":    method,
        "lines":     lines,                    # [bottom..top]
        "h_primary": h_primary,
        "h_secondary": h_secondary if has_changes else None,
        "has_changes": has_changes,
        "changing":  changing_positions,
    }


# ---------------------------------------------------------------------------
# Чтение одной гексаграммы
# ---------------------------------------------------------------------------

def _hx_block(h: int, label: str, use_color: bool) -> list[str]:
    hx = Hexagram(h)
    yc = YANG_ANSI[hx.yang] if use_color else ""
    r  = RESET if use_color else ""
    bd = BOLD  if use_color else ""
    dm = DIM   if use_color else ""

    bits = f"{h:06b}"
    lines = [
        f"  {bd}{label}{r}",
        f"  {yc}{hx.sym}  КВ#{hx.kw:>2} «{hx.name_cn} {hx.name_pin}»{r}",
        f"  h={h}  ({bits})  ян={yc}{hx.yang}{r}",
        "",
    ]

    # Визуальный стек линий (сверху вниз для отображения)
    for i in range(5, -1, -1):
        bit = (h >> i) & 1
        line_str = "─────────────" if bit else "────  ────"
        pos_name = LINE_POSITIONS[i]
        lc = (YLW if bit else dm) if use_color else ""
        lines.append(f"  {lc}{line_str}  {pos_name}{r}")

    lines.append("")

    # Расширенные атрибуты
    lower_tg = h & 7
    upper_tg = (h >> 3) & 7
    tg_names = {0:"☷坤Кунь",1:"☳Чжэнь",2:"☵Кань",3:"☶Гэнь",
                4:"☴Сюнь",5:"☲Ли",6:"☱Дуй",7:"☰乾Цянь"}
    lines += [
        f"  {dm}Нижняя триграмма: {tg_names.get(lower_tg,'?')}",
        f"  Верхняя триграмма: {tg_names.get(upper_tg,'?')}{r}",
    ]

    # ДНК-кодон
    if HAS_DNA:
        codon = h_to_codon(h)
        aa_l, aa_en = CODON_TABLE.get(codon, ('?', '?'))
        lines.append(f"  {dm}РНК-кодон: {codon}  →  АК={aa_l} ({aa_en}){r}")

    return lines


# ---------------------------------------------------------------------------
# Полный вывод чтения
# ---------------------------------------------------------------------------

def render_reading(result: dict, use_color: bool = True) -> str:
    bd = BOLD  if use_color else ""
    dm = DIM   if use_color else ""
    r  = RESET if use_color else ""
    yc_mov = (RED if use_color else "")
    yc_stb = (GRN if use_color else "")

    method_ru = {"coins": "монеты", "yarrow": "тысячелистник", "uniform": "равномерный"}
    probs = METHOD_PROBS[result["method"]]

    out = [
        "",
        "═"*64,
        f"  {bd}КОНСУЛЬТАЦИЯ И ЦЗИН{r}  ({method_ru.get(result['method'],result['method'])})",
        "═"*64,
        "",
        f"  {bd}Метод{r}  P(6)={probs[6]}  P(7)={probs[7]}"
        f"  P(8)={probs[8]}  P(9)={probs[9]}",
        "",
        f"  {bd}Шесть линий (снизу вверх):{r}",
        "",
    ]

    for i, v in enumerate(result["lines"]):
        nm, desc, sym, change = LINE_NAMES[v]
        moving = LINE_MOVING[v]
        col = yc_mov if moving else yc_stb
        pos = LINE_POSITIONS[i]
        out.append(
            f"  {i+1}. {col}{sym:>16}{r}  {col}{nm:<20}{r}"
            f"  [{col}{change}{r}]  {dm}{pos}{r}"
        )

    out += ["", "─"*64, ""]
    out += _hx_block(result["h_primary"], "ПЕРВИЧНАЯ ГЕКСАГРАММА", use_color)

    if result["has_changes"]:
        pos_strs = [LINE_POSITIONS[i] for i in result["changing"]]
        out += [
            f"  {bd}Подвижные линии:{r} {', '.join(str(i+1) for i in result['changing'])}",
            f"  {dm}({', '.join(pos_strs)}){r}",
            "",
            "─"*64,
            "",
        ]
        out += _hx_block(result["h_secondary"], "ВТОРИЧНАЯ ГЕКСАГРАММА (после изменений)", use_color)
    else:
        out += [
            f"  {dm}Подвижных линий нет.{r}",
            f"  {dm}Ситуация стабильна; вторичная гексаграмма отсутствует.{r}",
        ]

    out += [
        "",
        "═"*64,
        f"  {bd}КЛЮЧЕВЫЕ СИМВОЛЫ:{r}",
    ]

    h1 = result["h_primary"]
    hx1 = Hexagram(h1)
    out.append(
        f"  {YANG_ANSI[hx1.yang] if use_color else ''}Первичная: {hx1.sym} {hx1.name_pin}"
        f"  (КВ#{hx1.kw}, ян={hx1.yang}){r}"
    )

    if result["has_changes"]:
        h2  = result["h_secondary"]
        hx2 = Hexagram(h2)
        # Соотношение между ними
        from hexsymm import comp, rev6
        rel = []
        if h2 == comp(h1):  rel.append("дополнение (错)")
        if h2 == rev6(h1):  rel.append("обращение (综)")
        if h2 == (h1 ^ 63): rel.append("антипод")
        if not rel:         rel.append("трансформация")
        out.append(
            f"  {YANG_ANSI[hx2.yang] if use_color else ''}Вторичная: {hx2.sym} {hx2.name_pin}"
            f"  (КВ#{hx2.kw}, ян={hx2.yang})  [{', '.join(rel)}]{r}"
        )

    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexoracle — консультация И Цзин",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexoracle.py                 Бросок монетами (по умолчанию)
  python hexoracle.py --method yarrow Тысячелистник (традиционный)
  python hexoracle.py --method uniform Равномерный
  python hexoracle.py --seed 42        Воспроизводимый результат
  python hexoracle.py --h 6            Прямая гексаграмма h=6 (без гадания)
  python hexoracle.py --n 3            Три консультации подряд
        """,
    )
    parser.add_argument("--method",   choices=["coins","yarrow","uniform"],
                        default="coins")
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--h",        type=int, default=None,
                        help="Показать гексаграмму без гадания")
    parser.add_argument("--n",        type=int, default=1,
                        help="Число консультаций")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.h is not None:
        if not 0 <= args.h <= 63:
            print("Ошибка: h должно быть 0..63")
            return
        print()
        print("\n".join(_hx_block(args.h, f"h={args.h}", use_color)))
        return

    for i in range(args.n):
        seed = args.seed if args.n == 1 else (None if args.seed is None else args.seed + i)
        result = consult(method=args.method, seed=seed)
        print(render_reading(result, use_color))


if __name__ == "__main__":
    main()
