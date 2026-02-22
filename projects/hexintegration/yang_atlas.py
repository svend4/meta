"""
yang_atlas.py — Атлас Q6 по ян-уровням (сферические слои)

Q6 = 64 вершины, разбитые на 7 слоёв по числу единичных битов:

  Слой 0:  C(6,0)= 1 вершина  — ян=0  (Кунь, всё инь)
  Слой 1:  C(6,1)= 6 вершин   — ян=1  (шесть базовых единиц)
  Слой 2:  C(6,2)=15 вершин   — ян=2
  Слой 3:  C(6,3)=20 вершин   — ян=3  «экватор», максимум
  Слой 4:  C(6,4)=15 вершин   — ян=4
  Слой 5:  C(6,5)= 6 вершин   — ян=5
  Слой 6:  C(6,6)= 1 вершина  — ян=6  (Цянь, всё ян)

Слои — это сферические оболочки Q6:
  Крюков:  Слой 1 = МВС (вписанная сфера, контакт)
           Слой 2 = СВС (средняя сфера, ближний)
           Слой 3+ = БВС (описанная сфера, дальний)

Слои-антиподы:
  Слой 0 ↔ Слой 6  (h=0 ↔ h=63)
  Слой 1 ↔ Слой 5  (ян=1 ↔ ян=5)
  Слой 2 ↔ Слой 4  (ян=2 ↔ ян=4)
  Слой 3 ↔ Слой 3  (ян=3 самодополняющий)
"""

import sys
import os
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, antipode, gray_code

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexliuxing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexboya"))

from hexiching import Hexagram, KW_FROM_H, TRIGRAMS
from hexliuxing import LiuElement, ELEMENTS
from hexboya import BodyState, ZONE_NAMES

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

YANG_ANSI = {
    0: "\033[90m",      # тёмно-серый
    1: "\033[94m",      # синий
    2: "\033[96m",      # голубой
    3: "\033[92m",      # зелёный
    4: "\033[93m",      # жёлтый
    5: "\033[95m",      # пурпурный
    6: "\033[97m",      # белый
}

ELEM_COLORS = ["\033[31m","\033[33m","\033[32m","\033[36m","\033[34m","\033[35m"]


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _vertices_by_yang(y: int) -> list:
    return sorted(h for h in range(64) if yang_count(h) == y)


def _liu_str(h: int, use_color: bool = True) -> str:
    elems = [LiuElement(i) for i in range(6) if (h >> i) & 1]
    if not elems:
        return "∅"
    parts = []
    for e in elems:
        c = ELEM_COLORS[e.idx] if use_color else ""
        parts.append(f"{c}{e.short}{RESET if use_color else ''}")
    return "+".join(parts)


def _xyz(h: int) -> tuple:
    return (h & 3, (h >> 2) & 3, (h >> 4) & 3)


def _vol(h: int) -> int:
    x, y, z = _xyz(h)
    return (x+1)*(y+1)*(z+1)


# ---------------------------------------------------------------------------
# Один слой
# ---------------------------------------------------------------------------

def show_layer(y: int, use_color: bool = True) -> str:
    verts = _vertices_by_yang(y)
    yc = YANG_ANSI[y] if use_color else ""
    r = RESET if use_color else ""
    bold = BOLD if use_color else ""
    dim = DIM if use_color else ""

    # Имена слоя
    sphere = {0: "центр", 1: "МВС", 2: "СВС", 3: "БВС", 4: "БВС", 5: "БВС", 6: "вне БВС"}[y]
    elem_count = math.comb(6, y)

    lines = [
        f"{yc}{bold}━━━ ЯН = {y}  ({elem_count} вершин)  [{sphere}]{r}",
        f"{dim}  {'h':>3}  {'бин':>8}  {'xyz'}  {'V':>3}  "
        f"{'гекс':>4}  {'KW':>3}  {'зоны':>14}  цвета{r}",
    ]

    for h in verts:
        hx = Hexagram(h)
        xyz = _xyz(h)
        vol = _vol(h)
        bs = BodyState(h)
        zones = ",".join(bs.open_zones()) or "∅"
        liu = _liu_str(h, use_color)
        ap = antipode(h)
        ap_hx = Hexagram(ap)

        lines.append(
            f"  {yc}{h:>3}{r}  {h:08b}  "
            f"({xyz[0]},{xyz[1]},{xyz[2]})  {vol:>3}  "
            f"{hx.sym}{hx.kw:>3}  {zones:<14}  {liu}"
            + (f"  ↔{ap_hx.sym}" if y <= 2 else "")
        )

    # Сводка слоя
    vols = [_vol(h) for h in verts]
    lines += [
        f"",
        f"  {dim}Объёмы (Касаткин): min={min(vols)} max={max(vols)} "
        f"avg={sum(vols)/len(vols):.1f}  "
        f"сумма={sum(vols)}{r}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Полный атлас
# ---------------------------------------------------------------------------

def full_atlas(use_color: bool = True) -> str:
    lines = [
        "",
        f"{'═'*65}",
        "  АТЛАС Q6 — 64 вершины по ян-уровням",
        f"  Касаткин × Крюков × Лю-Синь × И-цзин",
        f"{'═'*65}",
        "",
    ]
    for y in range(7):
        lines.append(show_layer(y, use_color))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Симметрия: антиподные пары по слоям
# ---------------------------------------------------------------------------

def antipode_shells(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    lines = [
        f"{bold}Антиподные пары слоёв Q6{r}",
        "",
        "  Слой k ↔ Слой (6-k): каждая вершина h в слое k",
        "  имеет антипод (63-h) в слое (6-k).",
        "",
    ]
    for y in range(4):
        y2 = 6 - y
        verts_y = _vertices_by_yang(y)
        lines.append(
            f"  Слой {y} ↔ Слой {y2}:  "
            f"{math.comb(6,y)} пар"
            + ("  (самодополняющий)" if y == 3 else "")
        )
        for h in verts_y[:3]:  # первые 3 примера
            ap = antipode(h)
            hx = Hexagram(h)
            ax = Hexagram(ap)
            yc_h = YANG_ANSI[y] if use_color else ""
            yc_a = YANG_ANSI[y2] if use_color else ""
            lines.append(
                f"    {yc_h}h={h:>2}({h:06b}) {hx.sym}#{hx.kw:>2} «{hx.name_pin[:10]}»{r}"
                f"  ↔  "
                f"{yc_a}h={ap:>2}({ap:06b}) {ax.sym}#{ax.kw:>2} «{ax.name_pin[:10]}»{r}"
            )
        if len(verts_y) > 3:
            lines.append(f"    ... (ещё {len(verts_y)-3})")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Профиль ян по коду Грея
# ---------------------------------------------------------------------------

def gray_yang_profile(use_color: bool = True) -> str:
    """Ян-уровень в каждом шаге кода Грея — ASCII-гора."""
    gc = gray_code()
    yang_seq = [yang_count(h) for h in gc]

    bold = BOLD if use_color else ""
    r = RESET if use_color else ""

    # ASCII-гора (высота = yang_count)
    MAX_Y = 6
    lines = [
        f"{bold}Профиль ян-уровня по коду Грея (64 шага){r}",
        "(каждый столбец = один шаг, высота = yang_count)",
        "",
    ]
    for row_y in range(MAX_Y, -1, -1):
        row = f"  {row_y}│"
        for y in yang_seq:
            yc = YANG_ANSI[row_y] if use_color else ""
            if y == row_y:
                row += f"{yc}█{r}"
            elif y > row_y:
                row += "│"
            else:
                row += " "
        lines.append(row)

    # Подпись: каждые 8 шагов
    lines.append("   └" + "─"*64)
    tick_row = "    "
    for i in range(0, 64, 8):
        tick_row += f"{i+1:<8}"
    lines.append(tick_row)

    # Статистика
    peaks = [(i, yang_seq[i]) for i in range(len(yang_seq))
             if (i == 0 or yang_seq[i] > yang_seq[i-1])
             and (i == len(yang_seq)-1 or yang_seq[i] >= yang_seq[i+1])
             and yang_seq[i] >= 5]
    valleys = [(i, yang_seq[i]) for i in range(len(yang_seq))
               if (i == 0 or yang_seq[i] < yang_seq[i-1])
               and (i == len(yang_seq)-1 or yang_seq[i] <= yang_seq[i+1])
               and yang_seq[i] <= 1]
    lines += [
        "",
        f"  Пики (ян≥5):  " +
        ", ".join(
            f"шаг{i+1}:{Hexagram(gc[i]).sym}ян={y}"
            for i, y in peaks
        ),
        f"  Долины (ян≤1):" +
        ", ".join(
            f"шаг{i+1}:{Hexagram(gc[i]).sym}ян={y}"
            for i, y in valleys
        ),
        "",
        f"  Среднее: {sum(yang_seq)/len(yang_seq):.2f}",
        f"  Максимальное: {max(yang_seq)} (Цянь ䷀, все ян)",
        f"  Минимальное:  {min(yang_seq)} (Кунь ䷁, все инь)",
        f"  Замыкание: шаг64→шаг1  d={hamming(gc[-1],gc[0])} "
        f"({'цикл!' if hamming(gc[-1],gc[0])==1 else 'НЕ цикл'})",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Таблица объёмов (Касаткин) по слоям
# ---------------------------------------------------------------------------

def volume_by_layer(use_color: bool = True) -> str:
    bold = BOLD if use_color else ""
    r = RESET if use_color else ""
    lines = [
        f"{bold}Объёмы Касаткина по ян-слоям{r}",
        "",
        f"  {'ян':>3}  {'N':>3}  {'min V':>6}  {'max V':>6}  {'avg V':>7}  "
        f"{'sum V':>7}  Совершенные кубы",
        "  " + "─"*65,
    ]
    for y in range(7):
        verts = _vertices_by_yang(y)
        vols = [_vol(h) for h in verts]
        perfect = [v for v in vols if round(v**(1/3))**3 == v]
        yc = YANG_ANSI[y] if use_color else ""
        perf_str = str(sorted(set(perfect))) if perfect else "—"
        lines.append(
            f"  {yc}{y:>3}  {len(verts):>3}  {min(vols):>6}  {max(vols):>6}  "
            f"{sum(vols)/len(vols):>7.1f}  {sum(vols):>7}  {perf_str}{r}"
        )
    total_sum = sum(_vol(h) for h in range(64))
    lines += [
        "  " + "─"*65,
        f"  Сумма всех объёмов: {total_sum}",
        f"  = (1+2+3+4)³ = 10³  (каждая ось даёт Σ(x+1)=1+2+3+4=10)",
        "",
        "  ЗАМЕЧАНИЕ: V(h=21)=8=2³, V(h=42)=27=3³, V(h=63)=64=4³",
        "  Это кубы числа куба Касаткина при координатах (k,k,k)!",
        "  V(k,k,k) = (k+1)³  →  V(0,0,0)=1=1³  V(1,1,1)=8=2³",
        "             V(2,2,2)=27=3³  V(3,3,3)=64=4³",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="yang_atlas — Атлас Q6 по ян-уровням",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python yang_atlas.py               Полный атлас (все 7 слоёв)
  python yang_atlas.py --layer 3     Только слой ян=3 (20 вершин)
  python yang_atlas.py --antipodes   Антиподные пары слоёв
  python yang_atlas.py --gray        Профиль ян по коду Грея
  python yang_atlas.py --volumes     Таблица объёмов Касаткина
        """,
    )
    parser.add_argument("--layer",    type=int, metavar="Y")
    parser.add_argument("--antipodes",action="store_true")
    parser.add_argument("--gray",     action="store_true")
    parser.add_argument("--volumes",  action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    use_color = not args.no_color

    if args.layer is not None:
        print()
        print(show_layer(args.layer, use_color))
    elif args.antipodes:
        print()
        print(antipode_shells(use_color))
    elif args.gray:
        print()
        print(gray_yang_profile(use_color))
    elif args.volumes:
        print()
        print(volume_by_layer(use_color))
    else:
        print(full_atlas(use_color))


if __name__ == "__main__":
    main()
