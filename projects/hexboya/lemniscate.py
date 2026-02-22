"""
lemniscate.py — Лемниската (восьмёрка) в Q6

Базовое движение системы Крюкова:
  Из нейтральной позиции (h=центр) боец проходит
  два круговых маршрута (4-цикла) с общей вершиной-центром,
  образуя в пространстве Q6 символ «∞» (лемниската Бернулли).

СТРУКТУРА ВОСЬМЁРКИ в Q6:
  4-цикл = квадрат в двух измерениях:
    h → h⊕a → h⊕a⊕b → h⊕b → h
    (открыть зону A, открыть зону B, закрыть A, закрыть B)

  Восьмёрка = два 4-цикла с общей вершиной (центром):
    Петля 1: центр → центр⊕a → центр⊕a⊕b → центр⊕b → центр
    Петля 2: центр → центр⊕c → центр⊕c⊕d → центр⊕d → центр

ТЕЛЕСНАЯ ИНТЕРПРЕТАЦИЯ (Крюков, 6 окон):
  Лемниската — движение через два парных окна тела:
  Верхняя петля: ВЛ↔ВП (правая и левая рука верх)
  Нижняя петля:  НЛ↔НП (правая и левая рука низ)
  Или по вертикали:
    Левая петля:  ВЛ↔НЛ (вся левая сторона)
    Правая петля: ВП↔НП (вся правая сторона)

ЛЮ-СИНЬ ИНТЕРПРЕТАЦИЯ (6 цветов):
  Петля 1 через R,Y:  ∅ → R → R+Y → Y → ∅
  Петля 2 через B,M:  ∅ → B → B+M → M → ∅

СВЯЗЬ С ЛЕМНИСКАТОЙ БЕРНУЛЛИ:
  Уравнение: (x²+y²)² = 2a²(x²-y²)
  В Q6: два 4-цикла, «пережатых» в центральной вершине.
  Число вершин: 4+4-1 = 7 (центр + 6 периферийных).
  Число рёбер: 4+4 = 8 (по 4 в каждом квадрате).
"""

import sys
import os
import argparse
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, neighbors, shortest_path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexboya"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexliuxing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexiching"))

from hexboya import BodyState, ZONE_NAMES
from hexliuxing import LiuElement, ELEMENTS
from hexiching import Hexagram

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

# Цвета для каждой из шести зон
ZONE_COLORS = {
    "ВЛ": "\033[34m",   # синий
    "ВП": "\033[32m",   # зелёный
    "СЛ": "\033[36m",   # голубой
    "СП": "\033[33m",   # жёлтый
    "НЛ": "\033[35m",   # пурпурный
    "НП": "\033[31m",   # красный
}
ELEM_COLORS = [
    "\033[31m",  # R
    "\033[33m",  # Y
    "\033[32m",  # G
    "\033[36m",  # C
    "\033[34m",  # B
    "\033[35m",  # M
]


# ---------------------------------------------------------------------------
# Lemniscate — пара 4-циклов с общей вершиной
# ---------------------------------------------------------------------------

class Lemniscate:
    """
    Лемниската в Q6: два 4-цикла с общей центральной вершиной.

    Параметры:
      center: int — центральная вершина (общая для обоих петель)
      bits1: (a, b) — биты первой петли
      bits2: (c, d) — биты второй петли
    """

    def __init__(self, center: int, bits1: tuple, bits2: tuple):
        self.center = center
        self.a, self.b = bits1
        self.c, self.d = bits2

        # Проверяем, что биты различны
        all_bits = [self.a, self.b, self.c, self.d]
        if len(set(all_bits)) != 4:
            raise ValueError("Четыре бита лемнискаты должны быть различны.")

    def loop1(self) -> list:
        """Вершины первой петли (включая центр дважды)."""
        c = self.center
        a, b = 1 << self.a, 1 << self.b
        return [c, c ^ a, c ^ a ^ b, c ^ b, c]

    def loop2(self) -> list:
        """Вершины второй петли."""
        c = self.center
        x, y = 1 << self.c, 1 << self.d
        return [c, c ^ x, c ^ x ^ y, c ^ y, c]

    def full_path(self) -> list:
        """Полный путь: петля1 + петля2 (без дублирования центра)."""
        return self.loop1() + self.loop2()[1:]

    def all_vertices(self) -> set:
        """Все уникальные вершины лемнискаты."""
        return set(self.loop1() + self.loop2())

    def area(self) -> int:
        """«Площадь» петли в Q6: число уникальных вершин за вычетом центра."""
        return len(self.all_vertices()) - 1

    def describes_as_body(self) -> list:
        """Описание полного пути в терминах Крюкова."""
        zone_names = ZONE_NAMES
        path = self.full_path()
        result = []
        for i in range(len(path) - 1):
            a_h, b_h = path[i], path[i + 1]
            diff_bit = (a_h ^ b_h).bit_length() - 1
            action = "открыть" if (b_h >> diff_bit) & 1 else "закрыть"
            zone = zone_names[diff_bit]
            result.append({
                "from": a_h,
                "to": b_h,
                "bit": diff_bit,
                "zone": zone,
                "action": action,
            })
        return result

    def describes_as_liu(self) -> list:
        """Описание полного пути в терминах Лю-Синь."""
        elem_shorts = [e["short"] for e in ELEMENTS]
        path = self.full_path()
        result = []
        for i in range(len(path) - 1):
            a_h, b_h = path[i], path[i + 1]
            diff_bit = (a_h ^ b_h).bit_length() - 1
            action = "активировать" if (b_h >> diff_bit) & 1 else "деактивировать"
            elem = elem_shorts[diff_bit]
            result.append({
                "from": a_h,
                "to": b_h,
                "bit": diff_bit,
                "elem": elem,
                "action": action,
            })
        return result

    def render(self, use_color: bool = True) -> str:
        """ASCII-диаграмма лемнискаты."""
        zone_names = ZONE_NAMES
        elem_shorts = [e["short"] for e in ELEMENTS]

        c = self.center
        a_mask, b_mask = 1 << self.a, 1 << self.b
        x_mask, y_mask = 1 << self.c, 1 << self.d

        # Вершины
        v = {
            "c":  c,
            "a":  c ^ a_mask,
            "ab": c ^ a_mask ^ b_mask,
            "b":  c ^ b_mask,
            "x":  c ^ x_mask,
            "xy": c ^ x_mask ^ y_mask,
            "y":  c ^ y_mask,
        }

        def label(h):
            bs = BodyState(h)
            zones = ",".join(bs.open_zones()) or "∅"
            colors = ",".join(
                (ELEM_COLORS[i] if use_color else "") + elem_shorts[i] + (RESET if use_color else "")
                for i in range(6) if (h >> i) & 1
            ) or "∅"
            return f"h={h:2d}[{zones}]"

        bold = BOLD if use_color else ""
        r    = RESET if use_color else ""

        loop1_color = "\033[34m" if use_color else ""  # синий
        loop2_color = "\033[31m" if use_color else ""  # красный
        cen_color   = "\033[33m" if use_color else ""  # жёлтый

        lines = [
            f"{bold}Лемниската: центр={c}, петля1=биты({self.a},{self.b}), петля2=биты({self.c},{self.d}){r}",
            "",
            f"  Петля 1 ({loop1_color}──{r}) через зоны [{zone_names[self.a]}, {zone_names[self.b]}]:",
            f"    {loop1_color}{label(v['a'])}{r}",
            f"        ↗         ↘",
            f"  {cen_color}{label(v['c'])}{r}           {loop1_color}{label(v['ab'])}{r}",
            f"        ↖         ↙",
            f"    {loop1_color}{label(v['b'])}{r}",
            "",
            f"  Петля 2 ({loop2_color}──{r}) через зоны [{zone_names[self.c]}, {zone_names[self.d]}]:",
            f"    {loop2_color}{label(v['x'])}{r}",
            f"        ↗         ↘",
            f"  {cen_color}{label(v['c'])}{r}           {loop2_color}{label(v['xy'])}{r}",
            f"        ↖         ↙",
            f"    {loop2_color}{label(v['y'])}{r}",
        ]
        return "\n".join(lines)

    def render_path(self, use_color: bool = True) -> str:
        """Шаги полного пути с описанием в трёх системах."""
        zone_names = ZONE_NAMES
        elem_shorts = [e["short"] for e in ELEMENTS]
        path = self.full_path()

        loop1_color = "\033[34m" if use_color else ""
        loop2_color = "\033[31m" if use_color else ""
        center_color = "\033[33m" if use_color else ""
        r = RESET if use_color else ""
        bold = BOLD if use_color else ""

        lines = [
            f"{bold}Полный путь лемнискаты ({len(path)-1} шагов):{r}",
            "",
        ]
        for step, (h1, h2) in enumerate(zip(path[:-1], path[1:])):
            diff_bit = (h1 ^ h2).bit_length() - 1
            zone = zone_names[diff_bit]
            elem = elem_shorts[diff_bit]
            opened = (h2 >> diff_bit) & 1

            is_loop2 = step >= 4
            sc = loop2_color if is_loop2 else loop1_color
            zc = ZONE_COLORS.get(zone, "") if use_color else ""

            liu_active = [
                (ELEM_COLORS[i] if use_color else "") + elem_shorts[i] + r
                for i in range(6) if (h2 >> i) & 1
            ]
            liu_str = "+".join(liu_active) if liu_active else "∅"

            hex_sym = Hexagram(h2).sym
            kw_num = Hexagram(h2).kw

            lines.append(
                f"  {sc}шаг {step+1}{r}: h={h1:2d}→{h2:2d}"
                f"  {zc}{'▶' if opened else '◀'} {zone}{r}"
                f"  {'активир.' if opened else 'деактив.'} {elem}"
                f"  [{liu_str}{r}]"
                f"  {hex_sym}#{kw_num}"
            )

        lines += [
            "",
            f"  Петля 1: шаги 1-4  (биты {self.a},{self.b})",
            f"  Петля 2: шаги 5-8  (биты {self.c},{self.d})",
            f"  Возврат в центр h={self.center} после 8 шагов.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Предустановленные лемнискаты
# ---------------------------------------------------------------------------

def kryukov_standard(center: int = 0) -> Lemniscate:
    """
    Стандартная лемниската Крюкова:
      Петля 1: верхние зоны (ВЛ=бит0, ВП=бит1)
      Петля 2: нижние зоны  (НЛ=бит4, НП=бит5)
    """
    return Lemniscate(center, (0, 1), (4, 5))


def kryukov_vertical(center: int = 0) -> Lemniscate:
    """
    Вертикальная лемниската Крюкова:
      Петля 1: левая сторона (ВЛ=бит0, НЛ=бит4)
      Петля 2: правая сторона (ВП=бит1, НП=бит5)
    """
    return Lemniscate(center, (0, 4), (1, 5))


def kryukov_cross(center: int = 0) -> Lemniscate:
    """
    Крестовая лемниската:
      Петля 1: диагональ ВЛ-НП (бит0, бит5)
      Петля 2: диагональ ВП-НЛ (бит1, бит4)
    """
    return Lemniscate(center, (0, 5), (1, 4))


def liuxing_lemniscate(center: int = 0) -> Lemniscate:
    """
    Лемниската Лю-Синь:
      Петля 1: R-Y цикл (биты 0,1)
      Петля 2: B-M цикл (биты 4,5)
      Соответствует: △₁ RGB + △₂ YCM через центр.
    """
    return Lemniscate(center, (0, 1), (4, 5))


# ---------------------------------------------------------------------------
# Перебор всех лемнискат от центра
# ---------------------------------------------------------------------------

def all_lemniscates_from(center: int) -> list:
    """Все возможные лемнискаты из данного центра (C(6,2) * C(4,2) / 2 = 45)."""
    result = []
    bits = list(range(6))
    for ab in itertools.combinations(bits, 2):
        remaining = [b for b in bits if b not in ab]
        for cd in itertools.combinations(remaining, 2):
            if ab < cd:  # убираем дубли
                result.append(Lemniscate(center, ab, cd))
    return result


def find_lemniscates_by_yang(center: int, yang_range: tuple) -> list:
    """
    Найти лемнискаты, все вершины которых имеют yang в заданном диапазоне.
    """
    lo, hi = yang_range
    result = []
    for lem in all_lemniscates_from(center):
        verts = lem.all_vertices()
        if all(lo <= yang_count(v) <= hi for v in verts):
            result.append(lem)
    return result


# ---------------------------------------------------------------------------
# Двойная восьмёрка (символ «∞∞» — два узла)
# ---------------------------------------------------------------------------

class DoubleLemniscate:
    """
    Двойная восьмёрка: четыре 4-цикла с общим центром.
    Проходит через все 6 бит (по 2 в каждой петле).
    """

    def __init__(self, center: int):
        self.center = center
        # Три пары для трёх «колец» (bipartite edges)
        self.lemn1 = Lemniscate(center, (0, 1), (2, 3))
        self.lemn2 = Lemniscate(center, (4, 5), (0, 2))
        # Стандартный вариант с тремя парами:
        self._bits_pairs = [(0,1), (2,3), (4,5)]

    def full_path(self) -> list:
        """Путь через все три пары петель."""
        c = self.center
        path = [c]
        for a, b in self._bits_pairs:
            a_mask = 1 << a
            b_mask = 1 << b
            path += [c ^ a_mask, c ^ a_mask ^ b_mask, c ^ b_mask, c]
        return path

    def render_summary(self, use_color: bool = True) -> str:
        """Сводка двойной восьмёрки."""
        zone_names = ZONE_NAMES
        path = self.full_path()
        bold = BOLD if use_color else ""
        r = RESET if use_color else ""
        lines = [
            f"{bold}Двойная лемниската (три петли, 12 шагов): центр=h={self.center}{r}",
            "",
        ]
        colors = ["\033[34m", "\033[32m", "\033[35m"] if use_color else ["","",""]
        for loop_i, (a, b) in enumerate(self._bits_pairs):
            sc = colors[loop_i]
            z_a, z_b = zone_names[a], zone_names[b]
            c = self.center
            v1 = c ^ (1 << a)
            v2 = c ^ (1 << a) ^ (1 << b)
            v3 = c ^ (1 << b)
            lines.append(
                f"  {sc}Петля {loop_i+1}: [{z_a},{z_b}]  "
                f"{c}→{v1}→{v2}→{v3}→{c}{r}"
            )
        lines += [
            "",
            f"  Всего: 12 шагов, возврат в центр h={self.center}.",
            f"  Охват: все 6 битов активированы хотя бы раз.",
            f"  Вершины: {len(set(path))} уникальных (центр + 6 периферийных).",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="lemniscate — Лемниската (восьмёрка) в Q6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python lemniscate.py                  Стандартная лемниската Крюкова
  python lemniscate.py --vertical       Вертикальная лемниската
  python lemniscate.py --cross          Крестовая лемниската
  python lemniscate.py --double         Двойная восьмёрка (три петли)
  python lemniscate.py --center 21      Лемниската с центром h=21
  python lemniscate.py --all --center 0 Все лемнискаты из h=0 (45 штук)
  python lemniscate.py --bits 0 1 4 5   Лемниската с заданными парами битов
        """,
    )
    parser.add_argument("--center",   type=int, default=0,   metavar="H")
    parser.add_argument("--vertical", action="store_true")
    parser.add_argument("--cross",    action="store_true")
    parser.add_argument("--double",   action="store_true")
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--bits",     type=int, nargs=4, metavar=("A","B","C","D"))
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    use_color = not args.no_color
    c = args.center

    if args.bits:
        a, b, cc, d = args.bits
        lem = Lemniscate(c, (a, b), (cc, d))
        print()
        print(lem.render(use_color))
        print()
        print(lem.render_path(use_color))

    elif args.vertical:
        lem = kryukov_vertical(c)
        print()
        print(lem.render(use_color))
        print()
        print(lem.render_path(use_color))

    elif args.cross:
        lem = kryukov_cross(c)
        print()
        print(lem.render(use_color))
        print()
        print(lem.render_path(use_color))

    elif args.double:
        dl = DoubleLemniscate(c)
        print()
        print(dl.render_summary(use_color))
        print()
        # Стандартная лемниската для детального просмотра
        lem = kryukov_standard(c)
        print(lem.render_path(use_color))

    elif args.all:
        lems = all_lemniscates_from(c)
        print(f"\nВсе лемнискаты из h={c}: {len(lems)} штук\n")
        for i, lem in enumerate(lems[:10]):
            zone_names = ZONE_NAMES
            za, zb = zone_names[lem.a], zone_names[lem.b]
            zc, zd = zone_names[lem.c], zone_names[lem.d]
            verts = sorted(lem.all_vertices())
            print(
                f"  {i+1:>3}. биты({lem.a},{lem.b})/({lem.c},{lem.d})"
                f"  [{za},{zb}]/[{zc},{zd}]"
                f"  вершины={verts}"
            )
        if len(lems) > 10:
            print(f"  ... (ещё {len(lems)-10})")

    else:
        lem = kryukov_standard(c)
        print()
        print(lem.render(use_color))
        print()
        print(lem.render_path(use_color))


if __name__ == "__main__":
    main()
