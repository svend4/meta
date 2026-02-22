"""
hexgray.py — Код Грея как 64-шаговое путешествие по И-цзин

Стандартный бинарный отражённый код Грея для 6 бит:
  обходит все 64 вершины Q6, меняя ровно 1 бит за шаг.
  Это Гамильтонов ЦИКЛ на Q6 (замыкается обратно в h=0).

Смысл: каждый шаг = одна «меняющаяся черта».
Весь путь = полное путешествие через все 64 состояния системы.

Структура путешествия:
  Шаги 1-8:   первая «октава» (биты 0,1,2)  ян 0→1→2→1→2→3→2→1
  Шаги 9-16:  вторая октава               ян 1→2→3→2→3→4→3→2
  ...
  Шаги 41-48: пик — здесь достигается ян=6 (Цянь ䷀)
  Шаги 57-64: финальная октава → возврат к ян=1

В И-цзин: каждый шаг = переход по ребру Q6 = одна меняющаяся черта.
Весь цикл = «великое путешествие» через все 64 гексаграммы.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, gray_code, antipode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexliuxing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexboya"))

from hexiching import Hexagram, KW_FROM_H
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


# ---------------------------------------------------------------------------
# Ступени путешествия
# ---------------------------------------------------------------------------

def journey_step(gc: list, step_i: int, use_color: bool = True) -> str:
    """Описание одного шага кода Грея."""
    h_prev = gc[(step_i - 1) % 64]
    h_curr = gc[step_i]
    diff_bit = (h_prev ^ h_curr).bit_length() - 1

    hx = Hexagram(h_curr)
    elem = ELEMENTS[diff_bit]["short"]
    zone = ZONE_NAMES[diff_bit]
    opened = (h_curr >> diff_bit) & 1

    yc = YANG_ANSI[hx.yang] if use_color else ""
    ec = ELEM_COLORS[diff_bit] if use_color else ""
    r = RESET if use_color else ""

    liu_active = [
        (ELEM_COLORS[i] if use_color else "") + ELEMENTS[i]["short"] + r
        for i in range(6) if (h_curr >> i) & 1
    ]
    liu_str = "+".join(liu_active) if liu_active else "∅"

    x = h_curr & 3
    y2 = (h_curr >> 2) & 3
    z = (h_curr >> 4) & 3
    vol = (x+1)*(y2+1)*(z+1)

    return (
        f"  {step_i+1:>3}. {yc}h={h_curr:>2} {h_curr:06b}{r}"
        f"  ян={yc}{hx.yang}{r}"
        f"  {ec}{'▶' if opened else '◀'}{zone}{r}"
        f"  {hx.sym}#{hx.kw:>2} {hx.name_pin[:12]:<12}"
        f"  [{liu_str}{r}]"
        f"  V={vol}"
    )


# ---------------------------------------------------------------------------
# Полное путешествие
# ---------------------------------------------------------------------------

def full_journey(use_color: bool = True, chapter_size: int = 8) -> str:
    gc = gray_code()
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    dim = DIM if use_color else ""

    lines = [
        "",
        f"{'═'*70}",
        f"  {bold}ВЕЛИКОЕ ПУТЕШЕСТВИЕ: Код Грея как 64 гексаграммы{r}",
        f"  Каждый шаг = одна меняющаяся черта (ребро Q6)",
        f"{'═'*70}",
        "",
        f"  {dim}Шаг  h    бинарный  ян  действие  гексаграмма         [цвета]  V{r}",
    ]

    for chapter in range(64 // chapter_size):
        start = chapter * chapter_size
        yang_range = [yang_count(gc[i]) for i in range(start, start + chapter_size)]
        ch_min, ch_max = min(yang_range), max(yang_range)
        yc_ch = YANG_ANSI[round((ch_min+ch_max)/2)] if use_color else ""
        lines.append(
            f"\n  {yc_ch}── Глава {chapter+1}  "
            f"(шаги {start+1}–{start+chapter_size}, ян {ch_min}–{ch_max}){r}"
        )
        for i in range(start, start + chapter_size):
            lines.append(journey_step(gc, i, use_color))

    # Замыкание
    d_close = hamming(gc[-1], gc[0])
    close_bit = (gc[-1] ^ gc[0]).bit_length() - 1
    lines += [
        "",
        f"  {'─'*60}",
        f"  {bold}Замыкание:{r} h={gc[-1]}→h={gc[0]}"
        f"  бит{close_bit}={ZONE_NAMES[close_bit]}"
        f"  d={d_close}  {'(цикл!)' if d_close==1 else '(НЕ цикл)'}",
        f"  Возврат: {Hexagram(gc[-1]).sym}→{Hexagram(gc[0]).sym}  "
        f"({Hexagram(gc[-1]).name_pin}→{Hexagram(gc[0]).name_pin})",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Только ключевые события
# ---------------------------------------------------------------------------

def journey_milestones(use_color: bool = True) -> str:
    """Ключевые события путешествия."""
    gc = gray_code()
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    events = []
    yang_seq = [yang_count(h) for h in gc]

    # Первое достижение каждого ян-уровня
    first_reached = {}
    for i, h in enumerate(gc):
        y = yang_seq[i]
        if y not in first_reached:
            first_reached[y] = i

    # Максимумы и минимумы
    for i in range(64):
        y = yang_seq[i]
        y_prev = yang_seq[(i-1) % 64]
        y_next = yang_seq[(i+1) % 64]
        if y > y_prev and y > y_next:   # локальный максимум
            events.append(("пик", i, y))
        elif y < y_prev and y < y_next:  # локальный минимум
            events.append(("дно", i, y))

    lines = [
        f"{bold}Ключевые события путешествия Грея{r}",
        "",
        f"  {bold}Первое достижение каждого ян-уровня:{r}",
    ]
    for y in range(7):
        i = first_reached[y]
        h = gc[i]
        hx = Hexagram(h)
        yc = YANG_ANSI[y] if use_color else ""
        lines.append(
            f"    ян={yc}{y}{r}: шаг {i+1:>3}  {hx.sym}#{hx.kw:>2} «{hx.name_ru[:18]}»"
            f"  h={h}({h:06b})"
        )

    lines += ["", f"  {bold}Пики и долины (локальные экстремумы):{r}"]
    for kind, i, y in events:
        h = gc[i]
        hx = Hexagram(h)
        yc = YANG_ANSI[y] if use_color else ""
        diff_bit = (gc[(i-1)%64] ^ h).bit_length() - 1
        lines.append(
            f"    {'▲' if kind=='пик' else '▼'} {yc}ян={y}{r}  шаг {i+1:>3}"
            f"  {hx.sym}#{hx.kw:>2} «{hx.name_ru[:16]}»"
            f"  {ZONE_NAMES[diff_bit]}"
        )

    lines += [
        "",
        f"  {bold}Симметрия:{r}",
        f"    Половина 1 (шаги  1-32): ян-сумма={sum(yang_seq[:32])}",
        f"    Половина 2 (шаги 33-64): ян-сумма={sum(yang_seq[32:])}",
        f"    Всего: {sum(yang_seq)}  среднее: {sum(yang_seq)/64:.2f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Сравнение: Gray vs KW
# ---------------------------------------------------------------------------

def gray_vs_kw_yang(use_color: bool = True) -> str:
    """Ян-профили: код Грея vs порядок Вэнь-вана."""
    from hexiching import _KW_TO_H
    gc = gray_code()
    kw = _KW_TO_H

    yang_gray = [yang_count(h) for h in gc]
    yang_kw   = [yang_count(h) for h in kw]

    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    lines = [
        f"{bold}Ян-профиль: Код Грея vs Вэнь-ван{r}",
        "",
        f"  {'#':>3}  {'Gray-ян':>7}  {'KW-ян':>6}  {'разн.':>6}",
        "  " + "─" * 30,
    ]
    diffs = []
    for i in range(64):
        d = yang_gray[i] - yang_kw[i]
        diffs.append(d)
        yg = YANG_ANSI[yang_gray[i]] if use_color else ""
        yk = YANG_ANSI[yang_kw[i]] if use_color else ""
        dc = "\033[91m" if d != 0 else "\033[90m"
        dc = dc if use_color else ""
        lines.append(
            f"  {i+1:>3}  {yg}{yang_gray[i]:>7}{r}  {yk}{yang_kw[i]:>6}{r}"
            f"  {dc}{d:>+6}{r}"
        )
    lines += [
        "  " + "─" * 30,
        f"  Gray avg={sum(yang_gray)/64:.2f}  KW avg={sum(yang_kw)/64:.2f}",
        f"  |diff| avg={sum(abs(d) for d in diffs)/64:.2f}",
        f"  Совпадений (d=0): {sum(1 for d in diffs if d==0)}/64",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexgray — Код Грея как И-цзин путешествие",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexgray.py                  Полное путешествие (64 шага)
  python hexgray.py --milestones     Ключевые события (пики, долины)
  python hexgray.py --compare        Ян-профиль Gray vs Вэнь-ван
  python hexgray.py --chapter 5      Только глава 5 (шаги 33-40)
        """,
    )
    parser.add_argument("--milestones", action="store_true")
    parser.add_argument("--compare",    action="store_true")
    parser.add_argument("--chapter",    type=int, metavar="N")
    parser.add_argument("--no-color",   action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.milestones:
        print()
        print(journey_milestones(use_color))
    elif args.compare:
        print()
        print(gray_vs_kw_yang(use_color))
    elif args.chapter:
        gc = gray_code()
        n = args.chapter - 1
        bold = BOLD if use_color else ""
        r = RESET if use_color else ""
        print(f"\n  {bold}Глава {args.chapter} (шаги {n*8+1}–{n*8+8}){r}")
        for i in range(n*8, n*8+8):
            print(journey_step(gc, i, use_color))
    else:
        print(full_journey(use_color))


if __name__ == "__main__":
    main()
