"""
star_walk.py — Навигация по Звезде Давида (Лю-Синь)

Три цикла движения на шестиугольнике:

  ТВОРЕНИЕ   (шаг 1): R→Y→G→C→B→M→R  (обходим по кругу)
  СОХРАНЕНИЕ (шаг 2): два треугольника
                        △₁ R→G→B→R  (аддитивные)
                        △₂ Y→C→M→Y  (субтрактивные)
  РАЗРУШЕНИЕ (шаг 3): три диаметра
                        R↔C, Y↔B, G↔M  (дополнительные пары)

Практика (как в У-Синь, но 6 элементов):
  «Я нахожусь в элементе X.
   Один шаг создаёт Y.
   Два шага поддерживают Z.
   Три шага разрушают W (W = дополнение X).»

Связь с Q6:
  Шаг по шестиугольнику = флип одного бита в Q6
  Путь длины k = маршрут в Q6 через k рёбер
  Все пути творения образуют 6-цикл Грея в подграфе Q6
"""

import sys
import os
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, flip

from hexliuxing import LiuElement, LiuSystem, ELEMENTS, RELATIONS, RESET


# ---------------------------------------------------------------------------
# Карта шестиугольника: какой бит Q6 соответствует каждому шагу
# ---------------------------------------------------------------------------

# Элемент i ↔ бит i в Q6
# Шаг с позиции i на позицию j = флип битов i и j одновременно
# (но мы идём по рёбрам шестиугольника, где рёбра = соседи)

# Рёбра шестиугольника (1 шаг = творение):
HEXAGON_EDGES = [(i, (i+1)%6) for i in range(6)]

# Рёбра двух треугольников (2 шага = сохранение):
TRIANGLE1_EDGES = [(0,2), (2,4), (4,0)]   # R-G-B
TRIANGLE2_EDGES = [(1,3), (3,5), (5,1)]   # Y-C-M

# Диаметры (3 шага = разрушение):
DIAMETER_EDGES = [(0,3), (1,4), (2,5)]    # R↔C, Y↔B, G↔M

# Цвета ANSI для шагов
STEP_COLORS = {
    1: "\033[92m",   # зелёный — творение
    2: "\033[93m",   # жёлтый  — сохранение
    3: "\033[91m",   # красный — разрушение
    4: "\033[90m",   # серый   — угасание
    5: "\033[35m",   # фиолет  — деградация
}


# ---------------------------------------------------------------------------
# StarWalk — навигатор по звезде Давида
# ---------------------------------------------------------------------------

class StarWalk:
    """Движение по шестиугольнику Лю-Синь."""

    def __init__(self, start: int = 0):
        self.pos = start % 6
        self.history = [start % 6]
        self.sys6 = LiuSystem()

    @property
    def current(self) -> LiuElement:
        return LiuElement(self.pos)

    def step(self, n: int = 1) -> "StarWalk":
        """Сделать n шагов вперёд по циклу творения."""
        self.pos = (self.pos + n) % 6
        self.history.append(self.pos)
        return self

    def step_back(self, n: int = 1) -> "StarWalk":
        """Сделать n шагов назад."""
        self.pos = (self.pos - n) % 6
        self.history.append(self.pos)
        return self

    def jump_to(self, target: int) -> "StarWalk":
        """Перейти к элементу target (кратчайшим путём)."""
        d_fwd = (target - self.pos) % 6
        d_bwd = (self.pos - target) % 6
        if d_fwd <= d_bwd:
            for _ in range(d_fwd):
                self.step(1)
        else:
            for _ in range(d_bwd):
                self.step_back(1)
        return self

    def state(self) -> dict:
        """Текущее состояние: элемент + все отношения."""
        cur = self.current
        return {
            "pos": self.pos,
            "element": cur,
            "creates":    LiuElement((self.pos + 1) % 6),
            "preserves":  LiuElement((self.pos + 2) % 6),
            "destroys":   LiuElement((self.pos + 3) % 6),
            "decays":     LiuElement((self.pos + 4) % 6),
            "degrades":   LiuElement((self.pos + 5) % 6),
            "created_by": LiuElement((self.pos - 1) % 6),
            "q6_bit":     self.pos,
            "q6_h_single": 1 << self.pos,
        }

    def describe_state(self, use_color: bool = True) -> str:
        """Текстовое описание текущего состояния."""
        s = self.state()
        cur = s["element"]
        c = cur.color if use_color else ""
        r = RESET if use_color else ""

        def fmt(e, dist):
            ec = e.color if use_color else ""
            rel = RELATIONS[dist]
            step_c = STEP_COLORS.get(dist, "") if use_color else ""
            return f"{step_c}{rel[2]}{r} {ec}{e.name}{r} ({rel[0]})"

        lines = [
            f"Текущий элемент: {c}● {cur.name} ({cur.short}){r}",
            f"  Q6-бит: {s['q6_bit']},  h_чистый = {s['q6_h_single']:06b} ({s['q6_h_single']})",
            f"  {cur.desc}",
            "",
            f"  1 шаг вперёд  → {fmt(s['creates'],   1)}",
            f"  2 шага вперёд → {fmt(s['preserves'],  2)}",
            f"  3 шага        → {fmt(s['destroys'],   3)}  (дополнение)",
            f"  4 шага        → {fmt(s['decays'],     4)}",
            f"  5 шагов       → {fmt(s['degrades'],   5)}",
            f"  ← создан из   ← {fmt(s['created_by'], 5)}",
        ]
        return "\n".join(lines)

    def trace(self, use_color: bool = True) -> str:
        """История пройденных шагов."""
        if len(self.history) < 2:
            return "Нет истории движений."
        lines = ["Маршрут:"]
        for i in range(len(self.history) - 1):
            a = LiuElement(self.history[i])
            b = LiuElement(self.history[i+1])
            rel = a.relation_to(b)
            ca = a.color if use_color else ""
            cb = b.color if use_color else ""
            d = rel["steps_forward"]
            step_c = STEP_COLORS.get(d, "") if use_color else ""
            lines.append(
                f"  {ca}{a.short}{RESET} {step_c}{rel['symbol']}{RESET} {cb}{b.short}{RESET}"
                f"  ({rel['relation']}, {d} шаг{'а' if 2<=d<=4 else 'ов' if d>=5 else ''})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Все циклы и пути
# ---------------------------------------------------------------------------

def show_all_cycles(use_color: bool = True) -> str:
    """Три типа циклов на шестиугольнике."""
    sys6 = LiuSystem()
    e = sys6.elements

    def ec(elem):
        return (elem.color if use_color else "") + elem.short + (RESET if use_color else "")

    lines = [
        "=" * 58,
        "Три цикла Лю-Синь на шестиугольнике:",
        "=" * 58,
        "",
        "1. ТВОРЕНИЕ (шаг 1) — 6-цикл, обход по кругу:",
        "   " + " → ".join(ec(e[i]) for i in range(6)) + f" → {ec(e[0])}",
        "",
        "   Каждый элемент порождает следующего:",
    ]
    for i in range(6):
        a, b = e[i], e[(i+1)%6]
        mix = a.mix_with(b)
        lines.append(
            f"   {ec(a)} → {ec(b)}"
            f"  (смесь: {mix['result']})"
        )

    lines += [
        "",
        "2. СОХРАНЕНИЕ (шаг 2) — два треугольника △:",
        "",
        f"   △₁ аддитивные:    {ec(e[0])} → {ec(e[2])} → {ec(e[4])} → {ec(e[0])}",
        f"   △₂ субтрактивные: {ec(e[1])} → {ec(e[3])} → {ec(e[5])} → {ec(e[1])}",
        "",
        "   Смешение вершин △₁ (три смешанных цвета):",
    ]
    for a_i, b_i in [(0,2), (2,4), (4,0)]:
        a, b = e[a_i], e[b_i]
        mix = a.mix_with(b)
        lines.append(f"   {ec(a)} + {ec(b)} = {mix['result']}")
    lines.append("   Смешение вершин △₂:")
    for a_i, b_i in [(1,3), (3,5), (5,1)]:
        a, b = e[a_i], e[b_i]
        mix = a.mix_with(b)
        lines.append(f"   {ec(a)} + {ec(b)} = {mix['result']}")

    lines += [
        "",
        "3. РАЗРУШЕНИЕ (шаг 3) — три диаметра (дополнительные пары):",
        "",
    ]
    for a_i, b_i in [(0,3),(1,4),(2,5)]:
        a, b = e[a_i], e[b_i]
        lines.append(
            f"   {ec(a)} ↔ {ec(b)}"
            f"  (смесь: Белый аддит. / Серый субтракт.)"
        )
    lines += [
        "",
        "Итог: Шестиугольник = Звезда Давида = Колесо цветов:",
        "  Внешний контур: цикл творения (6-цикл)",
        "  Два треугольника: сохранение (два 3-цикла)",
        "  Диаметры: разрушение (три пары)",
    ]
    return "\n".join(lines)


def paths_from(start_idx: int, max_steps: int = 6, use_color: bool = True) -> str:
    """Все пути из заданного элемента (до max_steps шагов)."""
    sys6 = LiuSystem()
    e = sys6.elements
    start = e[start_idx]
    c0 = start.color if use_color else ""

    lines = [
        f"Все пути из {c0}{start.name}{RESET} (шаги 1..{max_steps}):",
        "",
    ]
    for k in range(1, max_steps + 1):
        target = LiuElement((start_idx + k) % 6)
        rel = start.relation_to(target)
        step_c = STEP_COLORS.get(k, "") if use_color else ""
        ct = target.color if use_color else ""
        lines.append(
            f"  {k} {'шаг ' if k==1 else 'шага' if 2<=k<=4 else 'шагов'}:"
            f" {step_c}{rel['symbol']}{RESET}"
            f" {ct}{target.name}{RESET}"
            f"  — {rel['relation']}"
            + (" (дополнение!)" if k == 3 else "")
            + (" (возврат)" if k == 6 else "")
        )
    return "\n".join(lines)


def hexagon_as_graph(use_color: bool = True) -> str:
    """Матрица отношений 6×6."""
    sys6 = LiuSystem()
    e = sys6.elements
    lines = ["Матрица отношений Лю-Синь (строка FROM, столбец TO):", ""]

    # Заголовок
    header = "       "
    for el in e:
        c = el.color if use_color else ""
        header += f" {c}{el.short}{RESET}  "
    lines.append(header)

    for i, a in enumerate(e):
        ca = a.color if use_color else ""
        row = f" {ca}{a.short}={a.name[:5]:<5}{RESET} "
        for j, b in enumerate(e):
            if i == j:
                row += " ∅   "
            else:
                d = (j - i) % 6
                rel = RELATIONS[d]
                step_c = STEP_COLORS.get(d, "") if use_color else ""
                row += f" {step_c}{rel[2]}{d}{RESET}  "
        lines.append(row)

    lines += [
        "",
        "Обозначения:",
    ]
    for d, (name, desc, sym) in RELATIONS.items():
        if d > 0:
            step_c = STEP_COLORS.get(d, "") if use_color else ""
            lines.append(f"  {step_c}{sym}{d}{RESET}  {name}: {desc}")
    return "\n".join(lines)


def wuxing_vs_liuxing(use_color: bool = True) -> str:
    """Детальное сравнение У-Синь и Лю-Синь."""
    lines = [
        "=" * 58,
        "У-Синь (5 элементов) vs Лю-Синь (6 элементов)",
        "=" * 58,
        "",
        "У-СИНЬ (пятиугольник):",
        "  Элементы: Дерево(木) Огонь(火) Земля(土) Металл(金) Вода(水)",
        "  Творение (相生, шэн): 1 шаг по часовой",
        "    Дерево→Огонь→Земля→Металл→Вода→Дерево",
        "  Разрушение (相克, кэ): 2 шага (через один)",
        "    Дерево→Земля→Вода→Огонь→Металл→Дерево",
        "  Нет третьего цикла (5 нечётное — нет диаметра).",
        "",
        "ЛЮ-СИНЬ (шестиугольник):",
        "  Элементы: R  Y  G  C  B  M  (6 цветов радуги)",
        "  Творение   (1 шаг): R→Y→G→C→B→M→R",
        "  Сохранение (2 шага): △₁(R→G→B) + △₂(Y→C→M)   ← НОВЫЙ ЦИКЛ",
        "  Разрушение (3 шага): R↔C  Y↔B  G↔M             (дополнительные)",
        "",
        "КЛЮЧЕВОЕ ОТЛИЧИЕ:",
        "  В У-Синь нет «среднего» цикла — нет сохранения.",
        "  В Лю-Синь цикл сохранения = Звезда Давида:",
        "    Два треугольника (аддитивные + субтрактивные цвета)",
        "    Вместе дают полный шестиугольник.",
        "",
        "ФИЗИЧЕСКИЙ СМЫСЛ (цвета художника):",
        "  Творение: смешиваем соседей → промежуточный цвет",
        "    R+Y=оранжевый, Y+G=жёлто-зелёный, G+C=бирюзовый...",
        "  Сохранение: смешиваем через одного → вторичный цвет",
        "    R+G=жёлтый, G+B=голубой, B+R=пурпурный (RGB модель!)",
        "  Разрушение: смешиваем дополнительные → белый/серый",
        "    R+C=белый (аддитивно), нейтрализация",
        "",
        "СИМВОЛИЧЕСКАЯ КАРТА (Крюков + Лю-Синь):",
        "  6 элементов = 6 окон тела (боевая система)",
        "  Творение = открытие зоны (1 движение)",
        "  Разрушение = атака в противоположную зону (3 движения)",
        "  Сохранение = поддержание треугольной защиты (2 движения)",
    ]
    return "\n".join(lines)


def q6_hexagon_path() -> str:
    """6-цикл творения как маршрут в Q6."""
    # yang=1 вершины: h = 1,2,4,8,16,32 (по одному биту)
    yang1 = [h for h in range(64) if yang_count(h) == 1]
    # Упорядочиваем по позиции бита
    yang1_sorted = sorted(yang1, key=lambda h: h.bit_length() - 1)

    lines = [
        "Цикл творения Лю-Синь как 6-цикл в Q6:",
        "",
        "Yang=1 вершины Q6 (по одному биту):",
    ]
    elem_names = ["R", "Y", "G", "C", "B", "M"]
    for i, h in enumerate(yang1_sorted):
        bit = h.bit_length() - 1
        lines.append(
            f"  бит {bit}: h={h:2d} ({h:06b}) = элемент {elem_names[i]} ({ELEMENTS[i]['name']})"
        )
    lines += [
        "",
        "6-цикл творения в Q6:",
        "  h=1(R) → h=2(Y) → h=4(G) → h=8(C) → h=16(B) → h=32(M) → h=1(R)",
        "",
        "  Это подграф Q6: 6-цикл на yang=1 вершинах.",
        "  Каждый шаг = выключить один бит, включить соседний.",
        "",
        "Два треугольника сохранения в Q6:",
        "  △₁ (RGB): h=1 → h=4 → h=16 → h=1",
        "  △₂ (YCM): h=2 → h=8 → h=32 → h=2",
        "",
        "Три диаметра разрушения в Q6:",
        "  R↔C: h=1 (000001) ↔ h=8  (001000)  бит0 ↔ бит3",
        "  Y↔B: h=2 (000010) ↔ h=16 (010000)  бит1 ↔ бит4",
        "  G↔M: h=4 (000100) ↔ h=32 (100000)  бит2 ↔ бит5",
        "",
        "  Дополнительность в Q6: h XOR 0b111000 = антипод в подкубе?",
        "  Нет, это не антипод Q6 (антипод = 63-h),",
        "  но это дополнение в группе битов {0,3}, {1,4}, {2,5}.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="star_walk — Движение по Звезде Давида (Лю-Синь)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python star_walk.py                  Все три цикла шестиугольника
  python star_walk.py --from R         Все пути из Красного
  python star_walk.py --from G --to B  Путь от Зелёного к Синему
  python star_walk.py --walk R 1 2 3   Шаги: старт R, шаги 1,2,3
  python star_walk.py --matrix         Матрица отношений 6×6
  python star_walk.py --compare        Сравнение У-Синь и Лю-Синь
  python star_walk.py --q6             Циклы как маршруты в Q6
        """,
    )
    parser.add_argument("--from", dest="from_elem", type=str, metavar="ELEM",
                        help="Показать все пути из элемента (R/Y/G/C/B/M или имя)")
    parser.add_argument("--to", dest="to_elem", type=str, metavar="ELEM",
                        help="Целевой элемент (вместе с --from)")
    parser.add_argument("--walk", type=str, nargs="+", metavar="...",
                        help="Маршрут: СТАРТ ШАГ1 ШАГ2 ... (шаги в числах)")
    parser.add_argument("--matrix", action="store_true",
                        help="Матрица отношений 6×6")
    parser.add_argument("--compare", action="store_true",
                        help="У-Синь (5) vs Лю-Синь (6)")
    parser.add_argument("--q6", action="store_true",
                        help="Циклы как маршруты в Q6")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    sys6 = LiuSystem()
    use_color = not args.no_color

    if args.from_elem:
        elem = sys6.get(args.from_elem)
        if args.to_elem:
            target = sys6.get(args.to_elem)
            walker = StarWalk(elem.idx)
            walker.jump_to(target.idx)
            print(f"\n{walker.describe_state(use_color)}")
            print()
            print(walker.trace(use_color))
        else:
            print("\n" + paths_from(elem.idx, use_color=use_color))
            print()
            w = StarWalk(elem.idx)
            print(w.describe_state(use_color))

    elif args.walk:
        if not args.walk:
            print("Укажите старт и шаги: --walk R 1 2 -1")
            return
        start_name = args.walk[0]
        steps = [int(x) for x in args.walk[1:]] if len(args.walk) > 1 else []
        start_elem = sys6.get(start_name)
        walker = StarWalk(start_elem.idx)
        for step in steps:
            if step >= 0:
                walker.step(step)
            else:
                walker.step_back(-step)
        print(f"\n{walker.describe_state(use_color)}")
        print()
        print(walker.trace(use_color))

    elif args.matrix:
        print("\n" + hexagon_as_graph(use_color))

    elif args.compare:
        print("\n" + wuxing_vs_liuxing(use_color))

    elif args.q6:
        print("\n" + q6_hexagon_path())

    else:
        print("\n" + show_all_cycles(use_color))


if __name__ == "__main__":
    main()
