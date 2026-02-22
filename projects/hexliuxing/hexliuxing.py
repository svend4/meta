"""
hexliuxing — Шесть элементов (六行, Лю-Синь)

Расширение У-Синь (五行, 5 элементов, пятиугольник)
до 6 элементов на правильном ШЕСТИУГОЛЬНИКЕ.

МАТЕМАТИЧЕСКАЯ ОСНОВА:

1. У-Синь (5 элементов):
     Цикл творения:    1 шаг по кругу (каждый порождает следующего)
     Цикл разрушения:  2 шага (через один = каждый гасит через одного)

2. Лю-Синь (6 элементов):
     1 шаг  = ТВОРЕНИЕ   (сосед: порождает)
     2 шага = СОХРАНЕНИЕ (через один: поддерживает)  ← новый цикл
     3 шага = РАЗРУШЕНИЕ (противоположный: дополнительный цвет)

   6 шагов = возврат к себе (обновление / рождение заново)

ЦВЕТА РАДУГИ = 6 ЭЛЕМЕНТОВ:
   Позиция 0: Красный   (R)  ↔  дополн. Голубой   (C) поз.3
   Позиция 1: Жёлтый    (Y)  ↔  дополн. Синий     (B) поз.4
   Позиция 2: Зелёный   (G)  ↔  дополн. Пурпурный (M) поз.5
   Позиция 3: Голубой   (C)  ↔  дополн. Красный   (R) поз.0
   Позиция 4: Синий     (B)  ↔  дополн. Жёлтый    (Y) поз.1
   Позиция 5: Пурпурный (M)  ↔  дополн. Зелёный   (G) поз.2

ЗВЕЗДА ДАВИДА = ДВА ТРЕУГОЛЬНИКА:
   △₁ (чётные 0,2,4): R, G, B  — аддитивные первичные
   △₂ (нечётные 1,3,5): Y, C, M — субтрактивные первичные

ДИАГОНАЛЬ КУБА → ШЕСТИУГОЛЬНИК:
   Куб (0,0,0)→(1,1,1), проекция вдоль (1,1,1)/√3:
   6 промежуточных вершин = 6 точек правильного шестиугольника
   Центральная точка = диагональ куба = ось симметрии

СВЯЗЬ С Q6:
   6 элементов = 6 бит = 6 измерений Q6
   Бит i открыт → элемент i «активен»
   h (0..63) = подмножество активных элементов = вершина Q6
"""

import math
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from libs.hexcore.hexcore import yang_count, hamming, flip, antipode


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# 6 элементов = 6 цветов колеса (аддитивный + субтрактивный)
ELEMENTS = [
    {"name": "Красный",   "short": "R", "ansi": "\033[91m", "hex": "#FF0000",
     "rgb": (1, 0, 0), "desc": "Огонь, активность, начало"},
    {"name": "Жёлтый",   "short": "Y", "ansi": "\033[93m", "hex": "#FFFF00",
     "rgb": (1, 1, 0), "desc": "Земля, устойчивость, центр"},
    {"name": "Зелёный",  "short": "G", "ansi": "\033[92m", "hex": "#00FF00",
     "rgb": (0, 1, 0), "desc": "Дерево, рост, жизнь"},
    {"name": "Голубой",  "short": "C", "ansi": "\033[96m", "hex": "#00FFFF",
     "rgb": (0, 1, 1), "desc": "Вода, течение, глубина"},
    {"name": "Синий",    "short": "B", "ansi": "\033[94m", "hex": "#0000FF",
     "rgb": (0, 0, 1), "desc": "Металл, ясность, холод"},
    {"name": "Пурпурный","short": "M", "ansi": "\033[95m", "hex": "#FF00FF",
     "rgb": (1, 0, 1), "desc": "Эфир, трансформация, мост"},
]

RELATIONS = {
    0: ("тождество",   "возврат к себе", "♾"),
    1: ("творение",    "порождает следующий", "→"),
    2: ("сохранение",  "поддерживает через одного", "⟹"),
    3: ("разрушение",  "гасит противоположный", "✕"),
    4: ("угасание",    "обратное сохранение", "⟸"),
    5: ("деградация",  "обратное творение", "←"),
}

RESET = "\033[0m"


# ---------------------------------------------------------------------------
# LiuElement — один из шести элементов
# ---------------------------------------------------------------------------

class LiuElement:
    """Один элемент на шестиугольнике Лю-Синь (0..5)."""

    def __init__(self, idx: int):
        self.idx = idx % 6
        self._data = ELEMENTS[self.idx]

    @property
    def name(self) -> str:
        return self._data["name"]

    @property
    def short(self) -> str:
        return self._data["short"]

    @property
    def color(self) -> str:
        return self._data["ansi"]

    @property
    def desc(self) -> str:
        return self._data["desc"]

    def complement(self) -> "LiuElement":
        """Дополнительный элемент (противоположный, 3 шага)."""
        return LiuElement((self.idx + 3) % 6)

    def creates(self) -> "LiuElement":
        """Следующий элемент (творение, 1 шаг по часовой)."""
        return LiuElement((self.idx + 1) % 6)

    def created_by(self) -> "LiuElement":
        """Предыдущий элемент (обратное творение)."""
        return LiuElement((self.idx - 1) % 6)

    def preserves(self) -> "LiuElement":
        """Элемент через одного (сохранение, 2 шага)."""
        return LiuElement((self.idx + 2) % 6)

    def distance_to(self, other: "LiuElement") -> int:
        """Расстояние по шестиугольнику (0..3, симметричное)."""
        d = abs(self.idx - other.idx)
        return min(d, 6 - d)

    def relation_to(self, other: "LiuElement") -> dict:
        """Отношение к другому элементу (по направлению и расстоянию)."""
        dist_fwd = (other.idx - self.idx) % 6
        dist_bwd = (self.idx - other.idx) % 6
        rel = RELATIONS[dist_fwd]
        return {
            "from": self.name,
            "to": other.name,
            "steps_forward": dist_fwd,
            "steps_backward": dist_bwd,
            "distance": self.distance_to(other),
            "relation": rel[0],
            "description": rel[1],
            "symbol": rel[2],
        }

    def mix_with(self, other: "LiuElement") -> dict:
        """Смешение двух элементов (цвета художника)."""
        r = (self._data["rgb"][0] + other._data["rgb"][0]) / 2
        g_v = (self._data["rgb"][1] + other._data["rgb"][1]) / 2
        b = (self._data["rgb"][2] + other._data["rgb"][2]) / 2
        # Определяем получившийся цвет
        result_rgb = (1 if r > 0 else 0, 1 if g_v > 0 else 0, 1 if b > 0 else 0)
        mix_name = _rgb_to_name(result_rgb)
        return {
            "a": self.name,
            "b": other.name,
            "mixed_rgb": (r, g_v, b),
            "result": mix_name,
            "distance": self.distance_to(other),
        }

    def q6_bit(self) -> int:
        """Номер бита Q6, соответствующего этому элементу."""
        return self.idx

    def __repr__(self):
        return f"LiuElement({self.idx}, {self.name})"

    def __eq__(self, other):
        return isinstance(other, LiuElement) and self.idx == other.idx


def _rgb_to_name(rgb: tuple) -> str:
    """(1,0,0)→'Красный' и т.д."""
    table = {
        (1, 0, 0): "Красный", (1, 1, 0): "Жёлтый",
        (0, 1, 0): "Зелёный", (0, 1, 1): "Голубой",
        (0, 0, 1): "Синий",   (1, 0, 1): "Пурпурный",
        (1, 1, 1): "Белый",   (0, 0, 0): "Чёрный",
    }
    return table.get(rgb, f"RGB{rgb}")


# ---------------------------------------------------------------------------
# LiuSystem — вся система шести элементов
# ---------------------------------------------------------------------------

class LiuSystem:
    """
    Система шести элементов на правильном шестиугольнике.
    Расширение У-Синь (5 элементов) до 6.
    """

    def __init__(self):
        self.elements = [LiuElement(i) for i in range(6)]

    def get(self, idx_or_name) -> LiuElement:
        """Получить элемент по индексу или имени."""
        if isinstance(idx_or_name, int):
            return LiuElement(idx_or_name % 6)
        name = str(idx_or_name).lower()
        for e in self.elements:
            if e.name.lower().startswith(name) or e.short.lower() == name:
                return e
        raise KeyError(f"Элемент не найден: {idx_or_name}")

    def creation_cycle(self) -> list:
        """Цикл творения: каждый порождает следующего."""
        return [e.name for e in self.elements] + [self.elements[0].name]

    def preservation_cycle(self) -> list:
        """Цикл сохранения: через одного (2 шага)."""
        # Два треугольника
        tri1 = [self.elements[i].name for i in [0, 2, 4, 0]]
        tri2 = [self.elements[i].name for i in [1, 3, 5, 1]]
        return {"triangle_1": tri1, "triangle_2": tri2}

    def destruction_pairs(self) -> list:
        """Пары разрушения (дополнительные, 3 шага)."""
        return [(self.elements[i].name, self.elements[(i+3)%6].name)
                for i in range(3)]

    def all_relations(self) -> list:
        """Все 30 отношений между элементами (направленные)."""
        result = []
        for i in range(6):
            for j in range(6):
                if i != j:
                    result.append(self.elements[i].relation_to(self.elements[j]))
        return result

    def compare_wuxing(self) -> str:
        """Сравнение с классической У-Синь (5 элементов)."""
        lines = [
            "Сравнение У-Синь (5) и Лю-Синь (6):",
            "",
            "У-Синь (пятиугольник, 5 элементов):",
            "  Дерево → Огонь → Земля → Металл → Вода → (Дерево)",
            "  Творение: 1 шаг  (Дерево→Огонь→Земля→...)",
            "  Разрушение: 2 шага (Дерево→Земля→Вода→...)",
            "",
            "Лю-Синь (шестиугольник, 6 элементов):",
            "  " + " → ".join(e.name for e in self.elements) + " → " + self.elements[0].name,
            "  Творение:    1 шаг  " + " → ".join(e.short for e in self.elements),
            "  Сохранение:  2 шага " + " → ".join(self.elements[i].short for i in [0,2,4,0,2]),
            "  Разрушение:  3 шага " + " — ".join(f"{self.elements[i].short}↔{self.elements[(i+3)%6].short}" for i in range(3)),
            "",
            "НОВЫЙ ЦИКЛ в Лю-Синь:",
            "  Сохранение (2 шага): два треугольника звезды Давида",
            "  △₁ (RGB): " + " → ".join(self.elements[i].short for i in [0,2,4,0]),
            "  △₂ (YCM): " + " → ".join(self.elements[i].short for i in [1,3,5,1]),
        ]
        return "\n".join(lines)

    def ascii_hexagon(self, use_color: bool = True) -> str:
        """ASCII правильный шестиугольник с 6 элементами."""
        e = self.elements
        c = [el.color if use_color else "" for el in e]
        r = RESET if use_color else ""

        lines = [
            "        Шестиугольник Лю-Синь:",
            "",
            f"              {c[0]}{e[0].short}={e[0].name}{r}",
            f"             /         \\",
            f"    {c[5]}{e[5].short}={e[5].name:<8}{r}   {c[1]}{e[1].short}={e[1].name}{r}",
            f"            |     ☯     |",
            f"    {c[4]}{e[4].short}={e[4].name:<8}{r}   {c[2]}{e[2].short}={e[2].name}{r}",
            f"             \\         /",
            f"              {c[3]}{e[3].short}={e[3].name}{r}",
            "",
            "  Расстояния:",
            f"  1 шаг  = ТВОРЕНИЕ   ({e[0].short}→{e[1].short}→{e[2].short}→{e[3].short}→{e[4].short}→{e[5].short}→{e[0].short})",
            f"  2 шага = СОХРАНЕНИЕ  △₁({e[0].short}→{e[2].short}→{e[4].short}) △₂({e[1].short}→{e[3].short}→{e[5].short})",
            f"  3 шага = РАЗРУШЕНИЕ  {e[0].short}↔{e[3].short}  {e[1].short}↔{e[4].short}  {e[2].short}↔{e[5].short}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StarOfDavid — Звезда Давида = шестиугольник + диагонали
# ---------------------------------------------------------------------------

class StarOfDavid:
    """
    Звезда Давида = правильный шестиугольник + оба диагональных треугольника.

    Два треугольника:
      △₁ (аддитивные первичные): R(0), G(2), B(4)
      △₂ (субтрактивные первичные): Y(1), C(3), M(5)

    Центральная точка = диагональ куба.
    """

    def __init__(self):
        self.sys = LiuSystem()
        self.triangle_rgb = [LiuElement(i) for i in [0, 2, 4]]
        self.triangle_ycm = [LiuElement(i) for i in [1, 3, 5]]

    def is_rgb(self, e: LiuElement) -> bool:
        return e.idx % 2 == 0

    def is_ycm(self, e: LiuElement) -> bool:
        return e.idx % 2 == 1

    def inner_hexagon(self) -> list:
        """6 вершин внутреннего шестиугольника (пересечения треугольников)."""
        # В звезде Давида два треугольника пересекаются в 6 точках,
        # образующих малый шестиугольник
        # Символически: смеси соседних элементов
        result = []
        for i in range(6):
            a = LiuElement(i)
            b = LiuElement((i + 1) % 6)
            mix = a.mix_with(b)
            result.append({
                "between": (a.name, b.name),
                "mixed": mix["result"],
                "position": i,
            })
        return result

    def cube_diagonal_connection(self) -> str:
        """Связь звезды Давида с проекцией куба вдоль диагонали."""
        lines = [
            "Звезда Давида ↔ Диагональная проекция куба:",
            "",
            "Куб (0,0,0)→(1,1,1), взгляд вдоль (1,1,1)/√3:",
            "",
            "  Вершина (0,0,0): центральная точка (дальняя)  = yang=0",
            "  yang=1 вершины: (1,0,0),(0,1,0),(0,0,1) → △₁ (RGB)",
            "  yang=2 вершины: (1,1,0),(1,0,1),(0,1,1) → △₂ (YCM)",
            "  Вершина (1,1,1): центральная точка (ближняя) = yang=3",
            "",
            "  △₁ R,G,B  = yang=1 вершины куба (1 бит = 1)",
            "  △₂ Y,C,M  = yang=2 вершины куба (2 бита = 1)",
            "  Центр ●   = главная диагональ = ось вращения шестиугольника",
            "",
            "  Звезда Давида = проекция куба вдоль диагонали",
            "               = шестиугольник из 6 вершин куба",
            "               = два треугольника аддитивных и субтрактивных цветов",
        ]
        return "\n".join(lines)

    def q6_interpretation(self) -> str:
        """Интерпретация звезды Давида в терминах Q6."""
        lines = [
            "Звезда Давида в Q6:",
            "",
            "6 вершин шестиугольника = 6 «единичных» вершин Q6:",
            f"  △₁ (RGB): h=1 (000001), h=4 (000100), h=16 (010000)",
            f"  △₂ (YCM): h=2 (000010), h=8 (001000), h=32 (100000)",
            "",
            "yang=1 слой Q6: " + str([h for h in range(64) if yang_count(h)==1]),
            "  = 6 вершин = 6 «пёрышек» вокруг центра (h=0)",
            "",
            "yang=2 слой Q6: 15 вершин (все пары бит)",
            "  Пары смежных бит (шестиугольник):",
            "  h=3(00..11), h=6(00.110), h=12(01100), h=24(11000), h=33, h=48",
            "",
            "Маршрут творения (1 шаг) = ребро Q6 между yang=1 вершинами:",
            "  h=1 → h=2 → h=4 → h=8 → h=16 → h=32 → h=1",
            "  Это 6-цикл в Q6 (подграф шестиугольника)",
        ]
        return "\n".join(lines)

    def ascii_star(self, use_color: bool = True) -> str:
        """ASCII звезда Давида с элементами."""
        e = self.sys.elements
        c = [el.color if use_color else "" for el in e]
        r = RESET if use_color else ""
        lines = [
            "      Звезда Давида = Лю-Синь:",
            "",
            f"            {c[0]} R {r}",
            f"           / \\",
            f"    {c[5]} M {r} --- {c[1]} Y {r}",
            f"     \\ / \\ /",
            f"      X  ☯  X",
            f"     / \\ / \\",
            f"    {c[4]} B {r} --- {c[2]} G {r}",
            f"           \\ /",
            f"            {c[3]} C {r}",
            "",
            f"  △₁ ({c[0]}R{r},{c[2]}G{r},{c[4]}B{r}): аддитивные первичные цвета",
            f"  △₂ ({c[1]}Y{r},{c[3]}C{r},{c[5]}M{r}): субтрактивные первичные цвета",
            f"  ☯  центр: главная диагональ куба",
            "",
            "  Дополнительные пары (3 шага, разрушение):",
        ]
        for i in range(3):
            a = e[i]
            b = e[i+3]
            lines.append(f"    {c[i]}{a.short}={a.name}{r} ↔ {c[i+3]}{b.short}={b.name}{r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RainbowWheel — Колесо цветов = шестиугольник Q6
# ---------------------------------------------------------------------------

class RainbowWheel:
    """
    Колесо 6 цветов радуги = Лю-Синь = Q6.

    Правила смешения:
      Соседние цвета (1 шаг): дают промежуточный оттенок
      Дополнительные (3 шага): дают белый (аддитивно) или грязь (субтрактивно)
    """

    def __init__(self):
        self.sys = LiuSystem()

    def mix(self, name_a: str, name_b: str) -> dict:
        """Смешать два цвета."""
        a = self.sys.get(name_a)
        b = self.sys.get(name_b)
        return a.mix_with(b)

    def color_table(self) -> str:
        """Таблица смешений: что получается при соединении соседних цветов."""
        e = self.sys.elements
        lines = ["Таблица смешения цветов (1 шаг = соседи):"]
        for i in range(6):
            a = e[i]
            b = e[(i+1) % 6]
            mix = a.mix_with(b)
            lines.append(
                f"  {a.color}{a.short}{RESET} + {b.color}{b.short}{RESET}"
                f" = {mix['result']}"
            )
        lines.append("\nДополнительные пары (3 шага = разрушение/нейтрализация):")
        for i in range(3):
            a = e[i]
            b = e[i+3]
            lines.append(
                f"  {a.color}{a.name}{RESET} + {b.color}{b.name}{RESET}"
                f" = Белый (аддит.) / Серый (субтракт.)"
            )
        return "\n".join(lines)

    def q6_color_encoding(self) -> str:
        """Кодирование 64 состояний Q6 через 6 цветов."""
        lines = [
            "Q6 = 64 состояния через 6 цветов:",
            "",
            "h (0..63) = подмножество активных цветов",
            "",
        ]
        # Показываем примеры
        examples = [0, 1, 2, 4, 8, 16, 32, 63, 21, 42]
        for h in examples:
            active = [ELEMENTS[i]["short"] for i in range(6) if (h >> i) & 1]
            colors_str = "".join(
                ELEMENTS[i]["ansi"] + ELEMENTS[i]["short"] + RESET
                for i in range(6) if (h >> i) & 1
            )
            lines.append(
                f"  h={h:2d} ({h:06b}): [{colors_str or '∅':^20}]"
                f"  yang={yang_count(h)}"
            )
        lines.extend([
            "",
            "Интерпретация yang-уровней:",
            "  yang=0: ни один цвет не активен (чёрный)",
            "  yang=1: один чистый цвет (6 состояний = 6 вершин шестиугольника)",
            "  yang=2: два цвета смешаны (15 состояний)",
            "  yang=3: три цвета (20 состояний = экватор Q6)",
            "  yang=4: четыре цвета (15 состояний)",
            "  yang=5: все кроме одного (6 состояний)",
            "  yang=6: все шесть активны (белый свет, 1 состояние)",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Связь с кубом и системой сфер
# ---------------------------------------------------------------------------

def cube_diagonal_hexagon() -> str:
    """
    Проекция куба вдоль главной диагонали → шестиугольник.
    Связь с Лю-Синь.
    """
    lines = [
        "=" * 58,
        "Диагональ куба → Шестиугольник → Звезда Давида",
        "=" * 58,
        "",
        "Куб [0,1]³, взгляд вдоль вектора (1,1,1)/√3:",
        "",
        "  Вершина (0,0,0) ──── yang=0 ──── центр 'ближний'",
        "  Вершины yang=1:  (100),(010),(001) ──── △₁ (R,G,B)",
        "  Вершины yang=2:  (110),(101),(011) ──── △₂ (Y,C,M)",
        "  Вершина (1,1,1) ──── yang=3 ──── центр 'дальний'",
        "",
        "  Проекция на плоскость ⊥ (1,1,1):",
        "  e₁ = (1,−1,0)/√2,  e₂ = (1,1,−2)/√6",
        "",
    ]
    # Вычисляем проекции
    import math
    e1 = (1/math.sqrt(2), -1/math.sqrt(2), 0)
    e2 = (1/math.sqrt(6),  1/math.sqrt(6), -2/math.sqrt(6))
    vertices_yang1 = [(1,0,0), (0,1,0), (0,0,1)]
    vertices_yang2 = [(1,1,0), (1,0,1), (0,1,1)]
    elem_names = ["R", "G", "B", "Y", "C", "M"]
    lines.append("  Координаты 6 вершин шестиугольника:")
    for i, v in enumerate(vertices_yang1 + vertices_yang2):
        x2 = sum(v[k]*e1[k] for k in range(3))
        y2 = sum(v[k]*e2[k] for k in range(3))
        tri = "△₁" if i < 3 else "△₂"
        lines.append(f"    {elem_names[i]} {tri} {v} → ({x2:.4f}, {y2:.4f})")
    lines.extend([
        "",
        "  Шесть вершин образуют ПРАВИЛЬНЫЙ ШЕСТИУГОЛЬНИК.",
        "  Три вершины yang=1 и три вершины yang=2 чередуются.",
        "  Это ЗВЕЗДА ДАВИДА (два треугольника).",
        "",
        "Связь с Q6:",
        "  Q6 → проекция вдоль диагонали 0→63:",
        "  yang=1 слой: 6 вершин = 6 точек шестиугольника",
        "  yang=2 слой: 15 вершин (включает стороны звезды)",
        "  yang=3 слой: 20 вершин (экватор Q6)",
    ])
    return "\n".join(lines)


def two_cubes_figure8() -> str:
    """
    Два куба / два шара = восьмёрка = Q6 × Q6.
    """
    lines = [
        "=" * 58,
        "Два куба / два шара = Восьмёрка = Q6 × Q6",
        "=" * 58,
        "",
        "Восьмёрка (лемниската ∞) = ДВА шара/куба, соединённых вместе:",
        "",
        "  Физические аналогии:",
        "    — Гантели (две гири на штанге)",
        "    — Песочные часы",
        "    — Две руки человека",
        "    — Два захваченных объекта манипулятора",
        "",
        "  Математически: Q6 × Q6 = Q12",
        "    128 вершин = 2 × 64 состояния",
        "    Состояние пары: (h₁, h₂), h₁,h₂ ∈ {0..63}",
        "",
        "  В системе Крюкова (hexboya):",
        "    h₁ = состояние правой руки (6 окон)",
        "    h₂ = состояние левой руки  (6 окон)",
        "    Переход восьмёрки: (h₁,h₂) → (h₂,h₁) через Q12",
        "",
        "  Пропорция 3:1 Крюкова: Q6 = Q3(руки) × Q3(ноги)",
        "    Q3 = 8 вершин (3 бита)",
        "    Q6 = Q3 × Q3 = 64 вершины",
        "    Q12 = Q6 × Q6 = 4096 состояний двух тел",
        "",
        "  Восьмёрка как маршрут в Q12:",
        "    Петля 1: (h₁,h₂) → флип бит i в h₁ → флип бит j в h₁ → (h₁,h₂)",
        "    Петля 2: (h₁,h₂) → флип бит i в h₂ → флип бит j в h₂ → (h₁,h₂)",
        "    Перекрест ✕ = возврат к исходному (h₁,h₂)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexliuxing — 6 элементов (Лю-Синь): расширение У-Синь",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexliuxing.py --hexagon       Шестиугольник 6 элементов
  python hexliuxing.py --star          Звезда Давида
  python hexliuxing.py --compare       Сравнение с У-Синь (5 элементов)
  python hexliuxing.py --colors        Колесо цветов + таблица смешений
  python hexliuxing.py --q6            Q6 = 64 состояния через 6 цветов
  python hexliuxing.py --diagonal      Диагональ куба → шестиугольник
  python hexliuxing.py --two-cubes     Два куба = восьмёрка = Q6×Q6
  python hexliuxing.py --relation R C  Отношение Красный → Голубой
        """,
    )
    parser.add_argument("--hexagon", action="store_true",
                        help="Шестиугольник 6 элементов")
    parser.add_argument("--star", action="store_true",
                        help="Звезда Давида = два треугольника")
    parser.add_argument("--compare", action="store_true",
                        help="Сравнение с У-Синь (5 элементов)")
    parser.add_argument("--colors", action="store_true",
                        help="Колесо цветов и таблица смешений")
    parser.add_argument("--q6", action="store_true",
                        help="Q6 через 6 цветов")
    parser.add_argument("--diagonal", action="store_true",
                        help="Диагональ куба → шестиугольник")
    parser.add_argument("--two-cubes", action="store_true", dest="twocubes",
                        help="Два куба = восьмёрка = Q6×Q6")
    parser.add_argument("--relation", type=str, nargs=2, metavar=("FROM", "TO"),
                        help="Отношение между двумя элементами")
    parser.add_argument("--no-color", action="store_true",
                        help="Без цвета")
    args = parser.parse_args()

    sys6 = LiuSystem()
    star = StarOfDavid()
    wheel = RainbowWheel()
    use_color = not args.no_color

    if args.hexagon:
        print("\n" + sys6.ascii_hexagon(use_color))

    elif args.star:
        print("\n" + star.ascii_star(use_color))
        print()
        print(star.cube_diagonal_connection())
        print()
        print(star.q6_interpretation())

    elif args.compare:
        print("\n" + sys6.compare_wuxing())

    elif args.colors:
        print("\n" + wheel.color_table())

    elif args.q6:
        print("\n" + wheel.q6_color_encoding())

    elif args.diagonal:
        print("\n" + cube_diagonal_hexagon())

    elif args.twocubes:
        print("\n" + two_cubes_figure8())

    elif args.relation:
        a_name, b_name = args.relation
        a = sys6.get(a_name)
        b = sys6.get(b_name)
        rel = a.relation_to(b)
        print(f"\nОтношение: {a.name} → {b.name}")
        print(f"  Шагов вперёд: {rel['steps_forward']}")
        print(f"  Расстояние:   {rel['distance']}")
        print(f"  Отношение:    {rel['symbol']} {rel['relation']}")
        print(f"  Описание:     {rel['description']}")
        print(f"\nОбратно: {b.name} → {a.name}")
        rel2 = b.relation_to(a)
        print(f"  Отношение: {rel2['symbol']} {rel2['relation']} ({rel2['description']})")

    else:
        # По умолчанию — полная демонстрация
        print()
        print(sys6.ascii_hexagon(use_color))
        print()
        print(star.ascii_star(use_color))
        print()
        print(sys6.compare_wuxing())
        print()
        print("Запуск: python hexliuxing.py --help")


if __name__ == "__main__":
    main()
