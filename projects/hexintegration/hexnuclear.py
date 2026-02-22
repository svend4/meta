"""
hexnuclear.py — 互卦 Hù guà: ядерная гексаграмма и её динамика

互卦 (hù guà) = «взаимная / ядерная гексаграмма» — классическая операция И Цзин.
Из гексаграммы с линиями b0..b5 (снизу вверх) берём:
  - Нижняя ядерная триграмма = линии 2,3,4 = биты 1,2,3
  - Верхняя ядерная триграмма = линии 3,4,5 = биты 2,3,4

N(h) = ((h>>1) & 7) | (((h>>2) & 7) << 3)

=============================================================
ГЛАВНОЕ ОТКРЫТИЕ: Динамика N на Q6
=============================================================

Три аттрактора N:
  {0}      — Кунь  (ян=0): устойчивая неподвижная точка
  {63}     — Цянь  (ян=6): устойчивая неподвижная точка
  {21, 42} — Цзи Цзи ↔ Вэй Цзи (ян=3): устойчивый 2-цикл!

Каждая из 64 гексаграмм попадает в один из трёх бассейнов
за не более чем 3 шага.

Критерий бассейна: определяется ОДНИМ условием — равенством
3-й и 4-й линий (бит2 и бит3, граница нижней/верхней триграммы):

  bit2 = bit3 = 0  →  Кунь  (16 гексаграмм)
  bit2 = bit3 = 1  →  Цянь  (16 гексаграмм)
  bit2 ≠ bit3      →  Цзи Цзи ↔ Вэй Цзи  (32 гексаграммы)

Граница между нижней и верхней триграммами определяет судьбу!
  • Согласованная граница (линии 3=4) → система очищается до чистоты
  • Несогласованная граница (линии 3≠4) → вечная пульсация между
    «Уже завершено» (Цзи Цзи, #63) и «Ещё не завершено» (Вэй Цзи, #64)

Заметное совпадение: {21, 42} — также 2-цикл при rev6:
  rev6(21) = rev6(010101) = 101010 = 42
  rev6(42) = rev6(101010) = 010101 = 21
  comp(21) = 42
И они же — последняя пара в последовательности Вэнь-вана (#63-64)!
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
RED = "\033[91m"
GRN = "\033[92m"
YLW = "\033[93m"
MAG = "\033[95m"
CYN = "\033[96m"

HX = {h: Hexagram(h) for h in range(64)}


# ---------------------------------------------------------------------------
# Ядерная операция
# ---------------------------------------------------------------------------

def nuclear(h: int) -> int:
    """
    N(h): ядерная гексаграмма.
    Нижняя ядерная триграмма = биты 1,2,3 = (h>>1)&7
    Верхняя ядерная триграмма = биты 2,3,4 = (h>>2)&7
    """
    return ((h >> 1) & 7) | (((h >> 2) & 7) << 3)


def nuclear_chain(h: int, max_iter: int = 20) -> list[int]:
    """Цепочка итераций N(h) до первого повтора."""
    seen = set()
    seq  = []
    cur  = h
    for _ in range(max_iter):
        if cur in seen:
            break
        seen.add(cur)
        seq.append(cur)
        cur = nuclear(cur)
    seq.append(cur)   # финальный (повторный) элемент
    return seq


def attractor(h: int) -> str:
    """
    Определяет бассейн притяжения h:
    'kun', 'qian', или 'cycle' ({21,42}).
    """
    b2 = (h >> 2) & 1
    b3 = (h >> 3) & 1
    if b2 == b3 == 0:
        return "kun"
    if b2 == b3 == 1:
        return "qian"
    return "cycle"


def depth(h: int) -> int:
    """Число шагов до аттрактора."""
    return len(nuclear_chain(h)) - 2


# ---------------------------------------------------------------------------
# Основной показ
# ---------------------------------------------------------------------------

def show_overview(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    lines = [
        "",
        "═"*68,
        f"  {bold}互卦 HÙ GUÀ — ЯДЕРНАЯ ГЕКСАГРАММА{r}",
        "═"*68,
        "",
        "  N(h) = нижняя (биты 1,2,3) + верхняя (биты 2,3,4) триграммы",
        "",
        f"  {bold}Динамика N на Q6: три аттрактора{r}",
        "",
    ]

    # Визуализация трёх аттракторов
    a0  = HX[0]
    a63 = HX[63]
    a21 = HX[21]
    a42 = HX[42]

    lines += [
        f"  {YANG_ANSI[0] if use_color else ''}  {a0.sym} Кунь (h=0)   ←─ N ─→   N(h=0) = 0{r}",
        f"  {YANG_ANSI[6] if use_color else ''}  {a63.sym} Цянь (h=63) ←─ N ─→   N(h=63) = 63{r}",
        f"  {YANG_ANSI[3] if use_color else ''}  {a21.sym} Цзи Цзи (h=21) ←─ N ─→   {a42.sym} Вэй Цзи (h=42){r}",
        f"  {YANG_ANSI[3] if use_color else ''}                             N ↑       ↓ N{r}",
        f"                                  2-цикл  ∞",
        "",
        f"  {bold}Бассейны притяжения (критерий: биты 2 и 3):{r}",
        "",
    ]

    basins = {"kun": [], "qian": [], "cycle": []}
    for h in range(64):
        basins[attractor(h)].append(h)

    for name, col, idx in [("kun", "\033[90m", 0), ("qian", "\033[97m", 63), ("cycle", "\033[92m", 21)]:
        hs    = basins[name]
        syms  = "".join(HX[h].sym for h in hs)
        label = {"kun": f"Кунь (h=0):  bit2=bit3=0",
                 "qian": f"Цянь (h=63): bit2=bit3=1",
                 "cycle": f"Цзи↔Вэй:  bit2≠bit3"}[name]
        lc = col if use_color else ""
        lines.append(f"  {lc}{len(hs):>2} гекс.  {label}{r}")
        lines.append(f"  {lc}{syms}{r}")
        lines.append("")

    lines += [
        f"  {bold}Ключевое наблюдение:{r}",
        f"  Граница нижней и верхней триграмм (линии 3 и 4, биты 2 и 3)",
        f"  определяет долгосрочную судьбу ЛЮБОЙ гексаграммы:",
        f"  • Согласована (==0 или ==1) → система очищается до крайности",
        f"  • Рассогласована (бит2≠бит3) → вечное колебание Цзи Цзи ↔ Вэй Цзи",
    ]
    return "\n".join(lines)


def show_chains(use_color: bool = True, only_attr: str | None = None) -> str:
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    ATTR_COL = {"kun": "\033[90m", "qian": "\033[97m", "cycle": "\033[92m"}
    ATTR_LAB = {"kun": "→ Кунь",   "qian": "→ Цянь",  "cycle": "↺ {Цзи↔Вэй}"}

    lines = [
        f"{bold}Ядерные цепочки всех 64 гексаграмм{r}",
        f"{dim}  h  бит.  ян  гексаграмма          цепочка N(h)...{r}",
        "  " + "─"*62,
    ]

    depths_count = defaultdict(int)

    for h in range(64):
        at = attractor(h)
        if only_attr and at != only_attr:
            continue

        chain = nuclear_chain(h)
        d     = depth(h)
        depths_count[d] += 1
        hx    = HX[h]

        ac  = ATTR_COL[at] if use_color else ""
        yc  = YANG_ANSI[hx.yang] if use_color else ""

        # Визуализация цепочки
        chain_str = " → ".join(
            f"{HX[v].sym}{'*' if v in (21,42) else ''}"
            for v in chain
        )

        lines.append(
            f"  {yc}{h:>2} ({h:06b}) {hx.yang}{r}"
            f"  {hx.sym}#{hx.kw:>2} {hx.name_pin[:16]:<16}"
            f"  {dim}d={d}{r}  {ac}{chain_str}  {ATTR_LAB[at]}{r}"
        )

    if not only_attr:
        lines += [
            "",
            f"  {bold}Глубины:{r}",
            "  " + "  ".join(f"d={k}: {depths_count[k]} гекс." for k in sorted(depths_count)),
        ]
    return "\n".join(lines)


def show_image(use_color: bool = True) -> str:
    """Показывает 16-элементный образ N."""
    bold = BOLD if use_color else ""
    r    = RESET if use_color else ""
    dim  = DIM   if use_color else ""

    image = sorted(set(nuclear(h) for h in range(64)))
    lines = [
        f"  {bold}Образ N (image): {len(image)} гексаграмм{r}",
        f"  {dim}Условие: bit1(h) = bit3(h) И bit2(h) = bit4(h){r}",
        f"  {dim}(средний бит нижней триграммы = верхний бит нижней половины){r}",
        "",
        f"  {'h':>4}  {'биты':>6}  {'ян':>3}  {'символ':>4}  {'КВ':>4}  Имя",
        "  " + "─"*50,
    ]

    for h in image:
        hx = HX[h]
        b1 = (h >> 1) & 1
        b3 = (h >> 3) & 1
        b2 = (h >> 2) & 1
        b4 = (h >> 4) & 1
        yc = YANG_ANSI[hx.yang] if use_color else ""
        lines.append(
            f"  {yc}{h:>4}  ({h:06b})  {hx.yang}  {hx.sym}"
            f"  КВ#{hx.kw:>2}  {hx.name_pin}{r}"
        )

    # N(image) → проверим, что N(image) ⊂ {0,21,42,63}
    n_image = sorted(set(nuclear(h) for h in image))
    lines += [
        "",
        f"  N(image) = {n_image}  → {{0, 21, 42, 63}} (все аттракторы) ✓",
        "",
        f"  Так: N³(h) ∈ {{0, 21, 42, 63}} для ЛЮБЫХ h.",
        f"  После 3 итераций все 64 гексаграммы достигают аттрактора.",
    ]
    return "\n".join(lines)


def show_hexagram(h: int, use_color: bool = True) -> str:
    """Полный анализ ядерных итераций гексаграммы h."""
    hx   = HX[h]
    at   = attractor(h)
    yc   = YANG_ANSI[hx.yang] if use_color else ""
    r    = RESET if use_color else ""
    bold = BOLD if use_color else ""

    chain  = nuclear_chain(h)
    d      = depth(h)

    ATTR_COL = {"kun": "\033[90m", "qian": "\033[97m", "cycle": "\033[92m"}
    ATTR_LAB = {
        "kun":   "Кунь — чистый инь",
        "qian":  "Цянь — чистый ян",
        "cycle": "Цзи Цзи ↔ Вэй Цзи — вечное колебание",
    }
    ac = ATTR_COL[at] if use_color else ""

    lines = [
        "",
        f"  {yc}h={h} ({h:06b})  {hx.sym}#{hx.kw} «{hx.name_cn} {hx.name_pin}»  ян={hx.yang}{r}",
        f"  Бассейн притяжения: {ac}{ATTR_LAB[at]}{r}",
        f"  Глубина: {d} шаг{'а' if 2<=d<=4 else '' if d==1 else 'ов'}",
        "",
        f"  {bold}Ядерная цепочка:{r}",
    ]

    for i, v in enumerate(chain):
        hxv = HX[v]
        ycv = YANG_ANSI[hxv.yang] if use_color else ""
        marker = "★" if v in (0, 63) else ("↺" if v in (21, 42) else "→")
        label  = "" if i == 0 else f" ← N({'∞' if i==len(chain)-1 else str(i)})"
        lines.append(
            f"  {' ' if i == 0 else f'  N({i}) ='}"
            f"  {ycv}{v:>2} ({v:06b}) {hxv.sym}#{hxv.kw:>2} «{hxv.name_pin}» {marker}{r}"
        )

    # Показываем биты 2 и 3
    b2 = (h >> 2) & 1
    b3 = (h >> 3) & 1
    lines += [
        "",
        f"  bit2(h) = {b2}  bit3(h) = {b3}  "
        f"{'→ равны ({bval}), система очищается'.format(bval=b2) if b2==b3 else '→ разные, система осциллирует'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexnuclear — 互卦 ядерная гексаграмма",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexnuclear.py              Обзор трёх бассейнов притяжения
  python hexnuclear.py --chains     Все 64 ядерных цепочки
  python hexnuclear.py --kun        Только цепочки бассейна Кунь
  python hexnuclear.py --qian       Только цепочки бассейна Цянь
  python hexnuclear.py --cycle      Только цепочки бассейна Цзи Цзи ↔ Вэй Цзи
  python hexnuclear.py --image      16-элементный образ N
  python hexnuclear.py --h 7        Анализ гексаграммы h=7 (Тай)
  python hexnuclear.py --h 21       Анализ Цзи Цзи (в цикле)
        """,
    )
    parser.add_argument("--chains",   action="store_true")
    parser.add_argument("--kun",      action="store_true")
    parser.add_argument("--qian",     action="store_true")
    parser.add_argument("--cycle",    action="store_true")
    parser.add_argument("--image",    action="store_true")
    parser.add_argument("--h",        type=int, metavar="N")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.chains:
        print()
        print(show_chains(use_color))
    elif args.kun:
        print()
        print(show_chains(use_color, only_attr="kun"))
    elif args.qian:
        print()
        print(show_chains(use_color, only_attr="qian"))
    elif args.cycle:
        print()
        print(show_chains(use_color, only_attr="cycle"))
    elif args.image:
        print()
        print(show_image(use_color))
    elif args.h is not None:
        if not 0 <= args.h <= 63:
            print("Ошибка: h должно быть 0..63")
            return
        print(show_hexagram(args.h, use_color))
    else:
        print(show_overview(use_color))


if __name__ == "__main__":
    main()
