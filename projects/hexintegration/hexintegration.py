"""
hexintegration — Синтез Касаткин × Крюков × Лю-Синь × Q6

Три источника описывают одно и то же пространство 64 элементов,
каждый со своей стороны:

  КАСАТКИН  — КУБ числа 4 (4³ = 64):  Числа в пространстве куба
  КРЮКОВ    — 6 окон тела (2⁶ = 64):  Состояния бойца в пространстве боя
  ЛЮ-СИНЬ   — 6 цветов/элементов:     Качества в пространстве гексагона

Все три = вершины Q6 = 6-мерного булева гиперкуба.

КЛЮЧЕВЫЕ ТОЖДЕСТВА:
  64 = 4³ = 2⁶ = 8²
  КУБ(4) = Q6 = 64 глифа И-цзин = 6 окон × ОТКР/ЗАКР = 6 цветов × вкл/выкл

ЕДИНАЯ ИНТЕРПРЕТАЦИЯ ВЕРШИНЫ h ∈ {0..63}:
  Касаткин: h = позиция в кубе 4×4×4,  x=h%4, y=(h//4)%4, z=h//16
  Крюков:   h = боевое состояние тела,  бит i = окно i открыто
  Лю-Синь:  h = подмножество активных цветов, бит i = цвет i активен
  И-цзин:   h = гексаграмма (6 черт инь/ян)

ДВИЖЕНИЯ (ребро Q6 = смена 1 бита):
  Касаткин: 1 шаг в 3D кубе (ΔV = ±1 по одной оси)
  Крюков:   1 движение (открыть/закрыть одно окно)
  Лю-Синь:  1 шаг по шестиугольнику (смена одного элемента)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, antipode, neighbors, shortest_path

# Импортируем компоненты из трёх проектов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexkub"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexboya"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../hexliuxing"))

from hexkub import KubNumber
from hexboya import BodyState, SphereSystem
from hexliuxing import LiuElement, LiuSystem, ELEMENTS

RESET = "\033[0m"


# ---------------------------------------------------------------------------
# UnifiedVertex — единая точка Q6 в трёх интерпретациях
# ---------------------------------------------------------------------------

class UnifiedVertex:
    """
    Вершина h ∈ {0..63} Q6 в трёх интерпретациях одновременно.
    """

    def __init__(self, h: int):
        if not 0 <= h <= 63:
            raise ValueError(f"h ∈ {{0..63}}, получено {h}")
        self.h = h

    # --- Касаткин: КУБ числа 4 -------------------------------------------

    def kasatkin_xyz(self) -> tuple:
        """Координаты в кубе 4×4×4 (x,y,z ∈ {0,1,2,3})."""
        x = self.h & 3
        y = (self.h >> 2) & 3
        z = (self.h >> 4) & 3
        return (x, y, z)

    def kasatkin_volume(self) -> int:
        """«Объём» в системе Касаткина: произведение координат (x+1)(y+1)(z+1)."""
        x, y, z = self.kasatkin_xyz()
        return (x + 1) * (y + 1) * (z + 1)

    def kasatkin_cube_root(self) -> float:
        """Кубический корень объёма по Касаткину."""
        return self.kasatkin_volume() ** (1/3)

    def kasatkin_is_perfect_cube(self) -> bool:
        """Является ли объём точным кубом?"""
        return KubNumber(self.kasatkin_volume()).is_perfect_cube()

    # --- Крюков: система окон -------------------------------------------

    def kryukov_state(self) -> BodyState:
        """Боевое состояние тела по Крюкову."""
        return BodyState(self.h)

    def kryukov_open_zones(self) -> list:
        """Открытые зоны тела."""
        return self.kryukov_state().open_zones()

    def kryukov_sphere(self, center: int = 0) -> str:
        """К какой сфере относится от центра h=0."""
        return SphereSystem(center).classify(self.h)

    # --- Лю-Синь: активные цвета/элементы --------------------------------

    def liuxing_active(self) -> list:
        """Список активных элементов Лю-Синь."""
        return [LiuElement(i) for i in range(6) if (self.h >> i) & 1]

    def liuxing_color_mix(self) -> str:
        """Получившийся цвет при смешении активных элементов."""
        active = self.liuxing_active()
        if not active:
            return "Чёрный (∅)"
        if len(active) == 6:
            return "Белый (все)"
        rgb = [sum((e._data["rgb"][c] for e in active), 0) for c in range(3)]
        result = tuple(1 if v > 0 else 0 for v in rgb)
        from hexliuxing import _rgb_to_name
        return _rgb_to_name(result)

    def liuxing_dominant(self) -> str:
        """Доминирующий элемент (с наибольшим весом)."""
        active = self.liuxing_active()
        if not active:
            return "нет"
        if len(active) == 1:
            return active[0].name
        # Наименьший индекс по кругу — ведущий элемент
        return active[yang_count(self.h) // 2].name

    # --- Q6 свойства -------------------------------------------------------

    def yang(self) -> int:
        return yang_count(self.h)

    def antipode(self) -> "UnifiedVertex":
        return UnifiedVertex(antipode(self.h))

    def neighbors(self) -> list:
        return [UnifiedVertex(n) for n in neighbors(self.h)]

    # --- Единое описание ---------------------------------------------------

    def describe(self, use_color: bool = True) -> str:
        x, y, z = self.kasatkin_xyz()
        state = self.kryukov_state()
        active_liu = self.liuxing_active()
        color_mix = self.liuxing_color_mix()

        # Цвет ANSI по yang-уровню
        yang_colors = {
            0: "\033[90m", 1: "\033[34m", 2: "\033[36m",
            3: "\033[32m", 4: "\033[33m", 5: "\033[35m", 6: "\033[37m"
        }
        yc = yang_colors[self.yang()] if use_color else ""
        r = RESET if use_color else ""

        lines = [
            f"{yc}h = {self.h:2d}  ({self.h:06b})  yang={self.yang()}{r}",
            "",
            f"  Касаткин (КУБ 4×4×4):  ({x},{y},{z})  "
            f"объём={self.kasatkin_volume()}  "
            f"{'(совершенный куб!)' if self.kasatkin_is_perfect_cube() else ''}",
            f"  Крюков (6 окон):        [{', '.join(state.open_zones()) or 'все закрыты'}]  "
            f"сфера={self.kryukov_sphere()}",
            f"  Лю-Синь (6 цветов):     [{', '.join(e.short for e in active_liu) or '∅'}]  "
            f"смесь={color_mix}",
            f"  Q6 (И-цзин):            гексаграмма #{self.h}  "
            f"антипод={antipode(self.h)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Движение по Q6 в трёх интерпретациях
# ---------------------------------------------------------------------------

def describe_move(h1: int, h2: int, use_color: bool = True) -> str:
    """
    Описать переход h1→h2 в Q6 в трёх интерпретациях.
    """
    path = shortest_path(h1, h2)
    d = len(path) - 1

    lines = [
        f"Переход {h1} → {h2}  (расстояние Хэмминга = {d})",
        f"Маршрут: {' → '.join(str(h) for h in path)}",
        "",
    ]

    for step_i in range(len(path) - 1):
        a, b = path[step_i], path[step_i+1]
        # Какой бит изменился
        diff = a ^ b
        bit = diff.bit_length() - 1

        # Три интерпретации шага
        ax, ay, az = UnifiedVertex(a).kasatkin_xyz()
        bx, by, bz = UnifiedVertex(b).kasatkin_xyz()
        zone_names = ["ВЛ", "ВП", "СЛ", "СП", "НЛ", "НП"]
        elem_names = [e["short"] for e in ELEMENTS]

        opened = (b >> bit) & 1
        liu_action = f"{elem_names[bit]} {'активирован' if opened else 'деактивирован'}"
        kryukov_action = (
            f"{'открыть' if opened else 'закрыть'} окно {zone_names[bit]}"
        )

        lines.append(
            f"  Шаг {step_i+1}: {a:06b}→{b:06b}  бит={bit}"
        )
        lines.append(f"    Касаткин: ({ax},{ay},{az})→({bx},{by},{bz})")
        lines.append(f"    Крюков:   {kryukov_action}")
        lines.append(f"    Лю-Синь:  {liu_action}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Сводная таблица
# ---------------------------------------------------------------------------

def unified_table(h_list: list = None, use_color: bool = True) -> str:
    """Сводная таблица по списку h (по умолчанию ключевые значения)."""
    if h_list is None:
        h_list = [0, 1, 2, 4, 7, 21, 42, 63, 8, 16, 32]

    yang_colors = {
        0: "\033[90m", 1: "\033[34m", 2: "\033[36m",
        3: "\033[32m", 4: "\033[33m", 5: "\033[35m", 6: "\033[37m"
    }

    lines = [
        f"{'h':>4}  {'bin':>8}  {'y':>2}  {'xyz':>10}  "
        f"{'V':>4}  {'зоны':>14}  {'цвета':>10}  {'сфера':>6}",
        "─" * 75,
    ]
    for h in h_list:
        v = UnifiedVertex(h)
        yc = yang_colors[v.yang()] if use_color else ""
        r = RESET if use_color else ""
        xyz = v.kasatkin_xyz()
        zones = ",".join(v.kryukov_open_zones()) or "—"
        colors = ",".join(e.short for e in v.liuxing_active()) or "∅"
        lines.append(
            f"{yc}{h:>4}  {h:08b}  {v.yang():>2}  "
            f"({xyz[0]},{xyz[1]},{xyz[2]})::{v.kasatkin_volume():>3}  "
            f"{zones:<14}  {colors:<10}  {v.kryukov_sphere()}{r}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ASCII-синтез
# ---------------------------------------------------------------------------

def synthesis_diagram() -> str:
    """Диаграмма синтеза всех трёх источников."""
    lines = [
        "=" * 62,
        "СИНТЕЗ: Касаткин × Крюков × Лю-Синь → Q6",
        "=" * 62,
        "",
        "           64 = 4³ = 2⁶ = 8²",
        "",
        "  ┌─────────────────────────────────────────────────┐",
        "  │                     Q6                          │",
        "  │         6-мерный булев гиперкуб                 │",
        "  │              64 вершины                         │",
        "  │                                                  │",
        "  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │",
        "  │  │КАСАТКИН  │  │КРЮКОВ    │  │ЛЮ-СИНЬ       │  │",
        "  │  │          │  │          │  │              │  │",
        "  │  │КУБ 4×4×4 │  │6 окон    │  │6 цветов      │  │",
        "  │  │h=x+4y+16z│  │тела бойца│  │радуги        │  │",
        "  │  │          │  │          │  │              │  │",
        "  │  │ 64 клетки│  │64 состоян│  │64 подмножест.│  │",
        "  │  └──────────┘  └──────────┘  └──────────────┘  │",
        "  │                                                  │",
        "  │  ДВИЖЕНИЕ = смена 1 бита = ребро Q6             │",
        "  │    Касаткин: шаг по кубу (±1 по оси)            │",
        "  │    Крюков:   открыть/закрыть одно окно           │",
        "  │    Лю-Синь:  активировать/деактивировать цвет    │",
        "  └─────────────────────────────────────────────────┘",
        "",
        "YANG = число активных «единиц» в h:",
        "  Касаткин: yang = число ненулевых бит (активных измерений)",
        "  Крюков:   yang = число открытых окон (0=защита, 6=атака)",
        "  Лю-Синь:  yang = число активных цветов (0=чёрный, 6=белый)",
        "",
        "АНТИПОД = 63-h = инверсия всех 6 бит:",
        "  Касаткин: зеркало куба относительно центра (1.5,1.5,1.5)",
        "  Крюков:   полная инверсия: открытое↔закрытое",
        "  Лю-Синь:  дополнительный цвет всей палитры",
        "",
        "ВОСЬМЁРКА = два 4-цикла с общей вершиной:",
        "  Касаткин: обход двух граней куба (две петли)",
        "  Крюков:   базовое движение (лемниската Крюкова)",
        "  Лю-Синь:  цикл творения △₁ + цикл творения △₂",
        "",
        "ТРИ СФЕРЫ КУБА ↔ МВС/СВС/БВС ↔ радиусы 1:√2:√3:",
        "  ball(h,1)=7   = вписанная сфера  = МВС = контакт",
        "  ball(h,2)=22  = средняя сфера    = СВС = ближний",
        "  ball(h,3)=42  = описанная сфера  = БВС = дальний",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexintegration — Синтез Касаткин × Крюков × Лю-Синь × Q6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexintegration.py              Диаграмма синтеза
  python hexintegration.py --vertex 21  Вершина h=21 в трёх интерпретациях
  python hexintegration.py --move 0 21  Переход 0→21 в трёх интерпретациях
  python hexintegration.py --table      Сводная таблица ключевых вершин
  python hexintegration.py --table 0 7 21 42 63
        """,
    )
    parser.add_argument("--vertex", type=int, metavar="H",
                        help="Описать вершину h в трёх системах")
    parser.add_argument("--move", type=int, nargs=2, metavar=("H1","H2"),
                        help="Описать переход h1→h2 в трёх системах")
    parser.add_argument("--table", type=int, nargs="*", metavar="H",
                        help="Сводная таблица (без аргументов = ключевые вершины)")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    use_color = not args.no_color

    if args.vertex is not None:
        v = UnifiedVertex(args.vertex)
        print()
        print(v.describe(use_color))

    elif args.move is not None:
        h1, h2 = args.move
        print()
        print(describe_move(h1, h2, use_color))

    elif args.table is not None:
        h_list = args.table if args.table else None
        print()
        print(unified_table(h_list, use_color))

    else:
        print()
        print(synthesis_diagram())


if __name__ == "__main__":
    main()
