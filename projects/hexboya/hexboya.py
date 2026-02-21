"""
hexboya — Тотальная система боя (Крюков В.В.): Q6-модель

Источник: «Тотальная система боя», ООО «ТОТАЛ», В.В. Крюков.

ЦЕНТРАЛЬНАЯ ИДЕЯ:
  Тело бойца разбито на 6 зон-«окон» (по Крюкову):
    бит 0: верхнее левое  (ВЛ)
    бит 1: верхнее правое (ВП)
    бит 2: среднее левое  (СЛ)
    бит 3: среднее правое (СП)
    бит 4: нижнее левое   (НЛ)
    бит 5: нижнее правое  (НП)

  Каждое окно: ОТКРЫТО=1 (атака/уязвимость) или ЗАКРЫТО=0 (защита).
  h ∈ {0..63}  =  6-битовый номер  =  вершина Q6  =  боевое состояние.

  64 состояния тела = 64 вершины Q6 = 64 глифа И-цзин.

ЗАКОНЫ:
  Закон чёрного ящика:    ≤3 движений в одной атаке (путь в Q6 длины ≤3)
  Закон двух движений:    ≤2 движений для поражения цели (hamming ≤2)
  Закон памяти:           суммарный путь техники ≤9
  Закон нечётных:         нечётный путь меняет паритет ян
  Закон сильного справа:  круговая доминанта стилей = Грей-код Q6

ВОСЬМЁРКА (лемниската):
  Базовое движение системы. В Q6 = два 4-цикла с общей вершиной.
  15 возможных восьмёрок в Q6 = C(6,2) = число 2D-граней гиперкуба.

СИСТЕМА СФЕР:
  МВС: ball(h, 1) — 7 вершин  (контакт)
  СВС: ball(h, 2) — 22 вершины (ближний)
  БВС: ball(h, 3) — 42 вершины (дальний)
"""

import math
import sys
import argparse
from itertools import combinations

# ---------------------------------------------------------------------------
# Импорт Q6-ядра
# ---------------------------------------------------------------------------

sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), "../../"))

from libs.hexcore.hexcore import (
    yang_count, hamming, neighbors, antipode,
    ball, sphere, to_bits, from_bits, flip,
    shortest_path, gray_code,
)


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

ZONE_NAMES = ["ВЛ", "ВП", "СЛ", "СП", "НЛ", "НП"]
ZONE_NAMES_EN = ["upper_left", "upper_right", "mid_left", "mid_right", "low_left", "low_right"]
ZONE_LABELS = {
    "ВЛ": "верхнее левое",
    "ВП": "верхнее правое",
    "СЛ": "среднее левое",
    "СП": "среднее правое",
    "НЛ": "нижнее левое",
    "НП": "нижнее правое",
}

ANIMALS = [
    ("тигр",    "жёсткий, прямая сила"),
    ("журавль", "мягкий, баланс и дистанция"),
    ("змея",    "текучий, захваты"),
    ("богомол", "резкий, серийные удары"),
    ("обезьяна","хаотичный, уклоны"),
    ("леопард", "взрывной, скорость"),
    ("медведь", "силовой, пресс"),
    ("утка",    "низкий, подсечки"),
]

COMBAT_MODES = {
    "фехтование":         {"sphere": "БВС", "yang": (1, 2), "desc": "Дистанционные укалывания"},
    "строительство":      {"sphere": "СВС", "yang": (2, 4), "desc": "Послойная атака зон"},
    "пугливая обезьяна":  {"sphere": "МВС", "yang": (1, 3), "desc": "Уклоны + контратака"},
    "удары богомола":     {"sphere": "СВС", "yang": (3, 5), "desc": "Серийные удары руками"},
    "вьетнамский ключ":   {"sphere": "МВС", "yang": (1, 2), "desc": "Захваты + рычаги"},
}


# ---------------------------------------------------------------------------
# BodyState — состояние тела бойца (система окон)
# ---------------------------------------------------------------------------

class BodyState:
    """
    Боевое состояние тела бойца по системе Крюкова.

    h (0..63) — 6-битовый номер = вершина Q6.
    Каждый бит: 0=закрыто, 1=открыто.
    """

    def __init__(self, h: int):
        if not 0 <= h <= 63:
            raise ValueError(f"h должно быть в диапазоне 0..63, получено {h}")
        self.h = h

    # --- состояние зон ------------------------------------------------------

    def zones(self) -> dict:
        """Словарь: зона → состояние (True=открыто)."""
        return {ZONE_NAMES[i]: bool(self.h >> i & 1) for i in range(6)}

    def open_zones(self) -> list:
        """Список открытых зон."""
        return [ZONE_NAMES[i] for i in range(6) if self.h >> i & 1]

    def closed_zones(self) -> list:
        """Список закрытых зон."""
        return [ZONE_NAMES[i] for i in range(6) if not (self.h >> i & 1)]

    def yang_level(self) -> int:
        """Число открытых окон (0=полная защита, 6=полная атака)."""
        return yang_count(self.h)

    def is_open(self, zone: str) -> bool:
        """Открыто ли окно zone (по коду 'ВЛ', 'ВП', ...)."""
        idx = ZONE_NAMES.index(zone)
        return bool(self.h >> idx & 1)

    # --- переходы -----------------------------------------------------------

    def open_zone(self, zone: str) -> "BodyState":
        """Открыть окно zone → новое состояние."""
        idx = ZONE_NAMES.index(zone)
        return BodyState(self.h | (1 << idx))

    def close_zone(self, zone: str) -> "BodyState":
        """Закрыть окно zone → новое состояние."""
        idx = ZONE_NAMES.index(zone)
        return BodyState(self.h & ~(1 << idx))

    def flip_zone(self, zone: str) -> "BodyState":
        """Переключить окно zone (одно движение)."""
        idx = ZONE_NAMES.index(zone)
        return BodyState(flip(self.h, idx))

    # --- Q6 операции --------------------------------------------------------

    def antipode(self) -> "BodyState":
        """Антипод: полная инверсия всех окон."""
        return BodyState(antipode(self.h))

    def neighbors(self) -> list:
        """6 соседних состояний (одно движение)."""
        return [BodyState(n) for n in neighbors(self.h)]

    def attack_path(self, target: "BodyState") -> list:
        """Кратчайший путь атаки от текущего к целевому состоянию."""
        return shortest_path(self.h, target.h)

    def distance_to(self, other: "BodyState") -> int:
        """Гаммингово расстояние = минимальное число движений."""
        return hamming(self.h, other.h)

    def can_attack_in_two(self, target: "BodyState") -> bool:
        """Закон двух движений: цель достижима за ≤2 шага?"""
        return self.distance_to(target) <= 2

    # --- описание -----------------------------------------------------------

    def ascii_body(self) -> str:
        """ASCII-схема тела с открытыми/закрытыми зонами."""
        z = self.zones()
        def cell(name):
            return "■" if z[name] else "□"
        lines = [
            f"       ╔═══════╦═══════╗",
            f"       ║  ВЛ   ║  ВП   ║   ■=открыто □=закрыто",
            f"       ║  {cell('ВЛ')}    ║  {cell('ВП')}    ║   yang={self.yang_level()}",
            f"       ╠═══════╬═══════╣",
            f"       ║  СЛ   ║  СП   ║",
            f"       ║  {cell('СЛ')}    ║  {cell('СП')}    ║",
            f"       ╠═══════╬═══════╣",
            f"       ║  НЛ   ║  НП   ║",
            f"       ║  {cell('НЛ')}    ║  {cell('НП')}    ║",
            f"       ╚═══════╩═══════╝",
            f"       h={self.h:06b}({self.h})  [{', '.join(self.open_zones()) or 'нет открытых'}]",
        ]
        return "\n".join(lines)

    def describe(self) -> str:
        """Текстовое описание состояния."""
        y = self.yang_level()
        chars = {
            0: "Полная защита — все окна закрыты",
            1: "Минимальная атака — одно открытое окно",
            2: "Лёгкая атака — два открытых окна",
            3: "Баланс атаки и защиты",
            4: "Активная атака — четыре открытых окна",
            5: "Максимальная атака — одно закрытое",
            6: "Полная атака — все окна открыты (максимальная уязвимость)",
        }
        return f"h={self.h} (yang={y}): {chars[y]}\nОткрыто: {self.open_zones() or ['—']}"

    def __repr__(self):
        return f"BodyState({self.h}, yang={self.yang_level()}, open={self.open_zones()})"

    def __eq__(self, other):
        return isinstance(other, BodyState) and self.h == other.h

    def __hash__(self):
        return hash(self.h)


# ---------------------------------------------------------------------------
# FigureEight — восьмёрка (лемниската) как маршрут в Q6
# ---------------------------------------------------------------------------

class FigureEight:
    """
    Восьмёрка (лемниската, ∞) — базовая траектория движения по Крюкову.
    В Q6: два 4-цикла (квадрата) с общей вершиной.

    Параметрические уравнения лемнискаты:
      x(t) = r · sin(t)
      y(t) = r · sin(t) · cos(t)

    В Q6: восьмёрка по паре битов (i, j):
      start → flip_i → flip_i+flip_j → flip_j → start (петля 1)
      start → flip_j → flip_j+flip_i → flip_i → start (петля 2)
    """

    def __init__(self, start: int = 0, axis_pair: tuple = (0, 1)):
        self.start = start
        self.i, self.j = axis_pair
        if not (0 <= self.i < 6 and 0 <= self.j < 6 and self.i != self.j):
            raise ValueError("axis_pair: два различных индекса 0..5")

    def loop1(self) -> list:
        """Первая петля восьмёрки: 4-цикл по биту i."""
        h = self.start
        a = flip(h, self.i)
        b = flip(a, self.j)
        c = flip(h, self.j)
        return [h, a, b, c, h]

    def loop2(self) -> list:
        """Вторая петля восьмёрки: 4-цикл по биту j."""
        h = self.start
        a = flip(h, self.j)
        b = flip(a, self.i)
        c = flip(h, self.i)
        return [h, a, b, c, h]

    def full_path(self) -> list:
        """Полный маршрут восьмёрки: петля1 + петля2 (без повтора центра)."""
        return self.loop1()[:-1] + self.loop2()

    def trajectory_xy(self, r: float = 1.0, n: int = 200) -> list:
        """
        Точки лемнискаты в 2D: x(t)=r·sin(t), y(t)=r·sin(t)·cos(t).
        """
        pts = []
        for k in range(n):
            t = 2 * math.pi * k / n
            x = r * math.sin(t)
            y = r * math.sin(t) * math.cos(t)
            pts.append((x, y))
        return pts

    @staticmethod
    def all_figure8s() -> list:
        """
        Все 15 восьмёрок в Q6 (C(6,2) = 15 пар осей).
        Каждая восьмёрка — 2D-грань гиперкуба.
        """
        return list(combinations(range(6), 2))

    def ascii_trajectory(self, r: float = 1.0) -> str:
        """ASCII-рисунок лемнискаты (∞)."""
        W, H = 51, 21
        canvas = [[" "] * W for _ in range(H)]
        cx, cy = W // 2, H // 2
        pts = self.trajectory_xy(r, 800)
        max_x = max(abs(p[0]) for p in pts) or 1
        max_y = max(abs(p[1]) for p in pts) or 1
        for x, y in pts:
            ix = int(cx + x / max_x * (W // 2 - 2))
            iy = int(cy - y / max_y * (H // 2 - 2))
            ix = max(0, min(W-1, ix))
            iy = max(0, min(H-1, iy))
            canvas[iy][ix] = "·"
        canvas[cy][cx] = "✕"
        lines = [
            f"Восьмёрка (лемниската ∞) по осям {ZONE_NAMES[self.i]}/{ZONE_NAMES[self.j]}:",
            f"Q6-маршрут: {' → '.join(str(h) for h in self.full_path())}",
            "",
        ]
        lines.extend("".join(row) for row in canvas)
        lines.extend([
            "",
            f"✕ = перекрест восьмёрки (h={self.start})",
            f"Петля 1: {self.loop1()}",
            f"Петля 2: {self.loop2()}",
            f"Всего восьмёрок в Q6: C(6,2) = 15",
        ])
        return "\n".join(lines)

    def zone_names(self) -> tuple:
        return (ZONE_NAMES[self.i], ZONE_NAMES[self.j])


# ---------------------------------------------------------------------------
# SphereSystem — система сфер МВС/СВС/БВС
# ---------------------------------------------------------------------------

class SphereSystem:
    """
    Три концентрические боевые сферы вокруг бойца (по Крюкову):
      МВС (малая внутренняя сфера):  радиус 1 в Q6
      СВС (средняя внутренняя сфера): радиус 2 в Q6
      БВС (большая внешняя сфера):   радиус 3 в Q6
    """

    def __init__(self, center: int = 0):
        self.center = center

    def mvs(self) -> list:
        """МВС: шар радиуса 1 (7 вершин: центр + 6 соседей)."""
        return ball(self.center, 1)

    def svs(self) -> list:
        """СВС: шар радиуса 2 (22 вершины)."""
        return ball(self.center, 2)

    def bvs(self) -> list:
        """БВС: шар радиуса 3 (42 вершины)."""
        return ball(self.center, 3)

    def sphere_shell(self, radius: int) -> list:
        """Оболочка сферы (только вершины на расстоянии radius)."""
        return sphere(self.center, radius)

    def classify(self, h: int) -> str:
        """К какой сфере относится вершина h?"""
        d = hamming(self.center, h)
        if d == 0:
            return "центр"
        elif d == 1:
            return "МВС"
        elif d == 2:
            return "СВС"
        elif d == 3:
            return "БВС"
        else:
            return f"вне БВС (d={d})"

    def ascii_spheres(self) -> str:
        """ASCII-схема трёх сфер."""
        c = self.center
        mvs = self.mvs()
        svs_only = [h for h in self.svs() if h not in mvs]
        bvs_only = [h for h in self.bvs() if h not in self.svs()]
        outside = [h for h in range(64) if h not in self.bvs()]
        lines = [
            f"Система сфер вокруг h={c} (yang={yang_count(c)}):",
            "",
            "  ┌────────────────────────────────┐",
            "  │  БВС (r=3): 42 вершины         │",
            "  │  ┌──────────────────────────┐  │",
            "  │  │  СВС (r=2): 22 вершины   │  │",
            "  │  │  ┌────────────────────┐  │  │",
            "  │  │  │  МВС (r=1): 7 в.   │  │  │",
            f"  │  │  │    ● h={c:2d}  yang={yang_count(c)}    │  │  │",
            "  │  │  │    + 6 соседей      │  │  │",
            "  │  │  └────────────────────┘  │  │",
            "  │  └──────────────────────────┘  │",
            "  └────────────────────────────────┘",
            "",
            f"  МВС ({len(mvs)} вершин, r≤1): {sorted(mvs)}",
            f"  СВС ({len(self.svs())} вершин, r≤2): "
            f"  +{len(svs_only)} новых на r=2",
            f"  БВС ({len(self.bvs())} вершин, r≤3): "
            f"  +{len(bvs_only)} новых на r=3",
            f"  Вне БВС: {len(outside)} вершин",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CombatLaws — законы системы боя
# ---------------------------------------------------------------------------

class CombatLaws:
    """
    Законы чёрного ящика и нечётных действий по Крюкову,
    переведённые в термины метрики Q6.
    """

    MAX_PER_ATTACK = 3   # ≤3 движений в одной атаке
    MAX_HAMMING = 2      # ≤2 движений для поражения одной цели
    MAX_TOTAL = 9        # суммарный путь техники ≤9

    def check_attack_path(self, path: list) -> dict:
        """Проверить маршрут атаки на соответствие законам."""
        length = len(path) - 1 if len(path) > 1 else 0
        h_start = path[0] if path else 0
        h_end = path[-1] if path else 0
        hd = hamming(h_start, h_end)
        return {
            "path": path,
            "length": length,
            "hamming": hd,
            "law1_ok": hd <= self.MAX_HAMMING,
            "law2_ok": length <= self.MAX_ATTACK,
            "is_odd": length % 2 == 1,
            "can_mirror": length % 2 == 1,
            "comment": self._comment(length, hd),
        }

    MAX_ATTACK = MAX_PER_ATTACK

    def _comment(self, length: int, hd: int) -> str:
        parts = []
        if length <= self.MAX_PER_ATTACK:
            parts.append("✓ закон 3 движений")
        else:
            parts.append(f"✗ превышение ({length}>{self.MAX_PER_ATTACK})")
        if hd <= self.MAX_HAMMING:
            parts.append("✓ закон 2 движений")
        else:
            parts.append(f"△ гамминг={hd} (допустимо для сложных техник)")
        if length % 2 == 1:
            parts.append("✓ нечётный (зеркалится)")
        else:
            parts.append("○ чётный (не зеркалится)")
        return "; ".join(parts)

    def is_valid_technique(self, paths: list) -> dict:
        """
        Проверить набор маршрутов (технику) на закон памяти.
        paths: список маршрутов (каждый = список h)
        """
        total = sum(len(p) - 1 for p in paths if len(p) > 1)
        return {
            "total_moves": total,
            "law_memory_ok": total <= self.MAX_TOTAL,
            "comment": (
                f"✓ закон памяти (≤{self.MAX_TOTAL})"
                if total <= self.MAX_TOTAL else
                f"✗ превышение памяти ({total}>{self.MAX_TOTAL})"
            ),
        }

    def optimal_arsenal(self, techniques: list) -> dict:
        """Проверить арсенал: оптимален ли (7-9 техник)?"""
        n = len(techniques)
        return {
            "count": n,
            "optimal": 7 <= n <= 9,
            "comment": (
                "✓ оптимальный арсенал (7-9 техник)" if 7 <= n <= 9 else
                f"{'мало' if n < 7 else 'много'}: {n} техник (норма: 7-9)"
            ),
        }

    def odd_action_analysis(self, path: list) -> str:
        """Анализ пути по закону нечётных действий."""
        length = len(path) - 1 if len(path) > 1 else 0
        y_start = yang_count(path[0]) if path else 0
        y_end = yang_count(path[-1]) if path else 0
        same_parity = y_start % 2 == y_end % 2
        lines = [
            f"Путь: {path}",
            f"Длина: {length} ({'нечётная' if length % 2 == 1 else 'чётная'})",
            f"yang({path[0]})={y_start}, yang({path[-1]})={y_end}",
            f"Паритет yang: {'одинаковый' if same_parity else 'разный'}",
            "",
        ]
        if length % 2 == 1:
            lines += [
                "✓ Нечётный путь:",
                "  — возможно зеркальное повторение без прерывания ритма",
                "  — возможна смена направления при вращении",
                "  — паритет yang изменился (инь↔ян)",
            ]
        else:
            lines += [
                "○ Чётный путь:",
                "  — нет возможности зеркального повторения",
                "  — паритет yang сохранился",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AnimalCycle — круговая доминанта стилей животных
# ---------------------------------------------------------------------------

class AnimalCycle:
    """
    Круговая доминанта боевых стилей (закон сильного справа).
    8 стилей в цикле: каждый сильнее предыдущего и слабее следующего.
    Нетранзитивное отношение (аналог камень-ножницы-бумага).
    """

    def __init__(self):
        self.styles = [a[0] for a in ANIMALS]
        self.n = len(self.styles)
        # Ассоциируем стиль i с Грей-кодом позиции i
        gc = gray_code()
        self.style_to_h = {self.styles[i]: gc[i] for i in range(self.n)}
        self.h_to_style = {v: k for k, v in self.style_to_h.items()}

    def beats(self, attacker: str, defender: str) -> bool:
        """Атакующий стиль побеждает защитника (закон сильного справа)?"""
        i = self.styles.index(attacker)
        j = self.styles.index(defender)
        return (i + 1) % self.n == j

    def dominance_chain(self) -> list:
        """Полная цепочка доминирования."""
        return [(self.styles[i], self.styles[(i+1) % self.n])
                for i in range(self.n)]

    def q6_glyph(self, style: str) -> int:
        """Q6-глиф (Грей-код), ассоциированный со стилем."""
        return self.style_to_h[style]

    def ascii_cycle(self) -> str:
        """ASCII-схема круговой доминанты."""
        n = self.n
        lines = [
            "Круговая доминанта стилей (закон сильного справа):",
            "",
        ]
        # Размещаем по кругу
        positions = []
        W, H = 51, 21
        cx, cy = W // 2, H // 2
        canvas = [[" "] * W for _ in range(H)]

        for i, (name, _) in enumerate(ANIMALS):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = int(cx + 18 * math.cos(angle))
            y = int(cy + 8 * math.sin(angle))
            positions.append((x, y))
            # Рисуем имя (первые 4 символа)
            label = name[:4]
            for k, ch in enumerate(label):
                lx = x - len(label)//2 + k
                if 0 <= lx < W and 0 <= y < H:
                    canvas[y][lx] = ch
            # Стрелка к следующему
            ni = (i + 1) % n
            angle_next = 2 * math.pi * ni / n - math.pi / 2
            nx = int(cx + 16 * math.cos((angle + 2*math.pi*ni/n - math.pi/2)/2))
            ny = int(cy + 7 * math.sin((angle + 2*math.pi*ni/n - math.pi/2)/2))

        # Центр
        canvas[cy][cx] = "☯"

        lines.extend("".join(row) for row in canvas)
        lines.append("")
        lines.append("Цепочка: " + " → ".join(self.styles) + f" → {self.styles[0]}")
        lines.append("")
        for i, (name, desc) in enumerate(ANIMALS):
            h = self.q6_glyph(name)
            lines.append(f"  {name:10s}: h={h:2d} (yang={yang_count(h)})  — {desc}")
        lines.extend([
            "",
            "Закон сильного справа:",
            "  A → B → C → ... → A (каждый побеждает предыдущего)",
            "  Нетранзитивность: A>B>C, но C>A",
            "  В Q6: стили = Грей-код = путь по рёбрам Q6",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def yin_yang_split() -> str:
    """Инь-ян разбиение Q6 по паритету yang."""
    yang_odd  = [h for h in range(64) if yang_count(h) % 2 == 1]
    yang_even = [h for h in range(64) if yang_count(h) % 2 == 0]
    lines = [
        "Инь-ян разбиение Q6 по паритету числа открытых окон:",
        "",
        f"ЯН (нечётный yang, {len(yang_odd)} состояний):",
        f"  yang=1: {[h for h in range(64) if yang_count(h)==1]}",
        f"  yang=3: {[h for h in range(64) if yang_count(h)==3]}",
        f"  yang=5: {[h for h in range(64) if yang_count(h)==5]}",
        "",
        f"ИНЬ (чётный yang, {len(yang_even)} состояний):",
        f"  yang=0: {[h for h in range(64) if yang_count(h)==0]}",
        f"  yang=2: {[h for h in range(64) if yang_count(h)==2]}",
        f"  yang=4: {[h for h in range(64) if yang_count(h)==4]}",
        f"  yang=6: {[h for h in range(64) if yang_count(h)==6]}",
        "",
        "Переход по ребру Q6 (одно движение) всегда переключает инь↔ян.",
        "→ Восьмёрка: 4 шага = чётный путь → возвращает к той же половине.",
    ]
    return "\n".join(lines)


def show_all_states_by_yang() -> str:
    """64 боевых состояния, сгруппированных по уровню ян."""
    lines = ["64 боевых состояния тела (система окон Крюкова):", ""]
    for y in range(7):
        states = [h for h in range(64) if yang_count(h) == y]
        desc = {
            0: "Полная защита",
            1: "Минимальная атака",
            2: "Лёгкая атака",
            3: "Баланс",
            4: "Активная атака",
            5: "Максимальная атака",
            6: "Полная атака",
        }[y]
        lines.append(f"yang={y} ({desc}): {len(states)} состояний")
        lines.append(f"  h = {states}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hexboya — Тотальная система боя Крюкова: Q6-модель",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python hexboya.py --state 21          Состояние h=21, открытые зоны
  python hexboya.py --state 0 63        Путь атаки от h=0 до h=63
  python hexboya.py --figure8 0 0 1     Восьмёрка от h=0 по осям ВЛ/ВП
  python hexboya.py --spheres 0         МВС/СВС/БВС вокруг h=0
  python hexboya.py --animals           Круговая доминанта стилей
  python hexboya.py --yin-yang          Инь-ян разбиение Q6
  python hexboya.py --all-states        Все 64 состояния по yang
  python hexboya.py --laws 0 1 3 2      Проверить путь 0→1→3→2
        """,
    )
    parser.add_argument("--state", type=int, nargs="+", metavar="H",
                        help="Показать состояние h (или путь от h1 до h2)")
    parser.add_argument("--figure8", type=int, nargs=3, metavar=("H","I","J"),
                        help="Восьмёрка: start=H, оси=I,J (0..5)")
    parser.add_argument("--spheres", type=int, metavar="H",
                        help="Система сфер МВС/СВС/БВС вокруг h")
    parser.add_argument("--animals", action="store_true",
                        help="Круговая доминанта стилей животных")
    parser.add_argument("--yin-yang", action="store_true", dest="yinyang",
                        help="Инь-ян разбиение Q6")
    parser.add_argument("--all-states", action="store_true", dest="allstates",
                        help="Все 64 состояния по уровню ян")
    parser.add_argument("--laws", type=int, nargs="+", metavar="H",
                        help="Проверить маршрут H1 H2 H3... на законы боя")

    args = parser.parse_args()

    if args.state:
        if len(args.state) == 1:
            s = BodyState(args.state[0])
            print()
            print(s.ascii_body())
            print()
            print(s.describe())
            print()
            anti = s.antipode()
            print(f"Антипод (полная инверсия): h={anti.h}")
            print(anti.ascii_body())
        else:
            h1, h2 = args.state[0], args.state[1]
            s1, s2 = BodyState(h1), BodyState(h2)
            path = s1.attack_path(s2)
            laws = CombatLaws()
            result = laws.check_attack_path(path)
            print(f"\nПуть атаки: {h1} → {h2}")
            print(f"Маршрут: {' → '.join(str(h) for h in path)}")
            print(f"Длина: {result['length']} движений")
            print(f"Гамминг: {result['hamming']}")
            print(f"Законы: {result['comment']}")
            print()
            print(s1.ascii_body())
            print("         ↓")
            print(s2.ascii_body())

    elif args.figure8 is not None:
        h, i, j = args.figure8
        f8 = FigureEight(h, (i, j))
        print()
        print(f8.ascii_trajectory())

    elif args.spheres is not None:
        sys_s = SphereSystem(args.spheres)
        print()
        print(sys_s.ascii_spheres())

    elif args.animals:
        ac = AnimalCycle()
        print()
        print(ac.ascii_cycle())

    elif args.yinyang:
        print()
        print(yin_yang_split())

    elif args.allstates:
        print()
        print(show_all_states_by_yang())

    elif args.laws:
        path = args.laws
        laws = CombatLaws()
        result = laws.check_attack_path(path)
        print(f"\nПроверка маршрута: {path}")
        print(f"Длина: {result['length']}")
        print(f"Гамминг (start→end): {result['hamming']}")
        print(f"Законы: {result['comment']}")
        print()
        print(laws.odd_action_analysis(path))

    else:
        # По умолчанию — демонстрация системы
        print("\n" + "=" * 60)
        print("hexboya — Тотальная система боя: Q6-модель")
        print("=" * 60)
        print()
        print("Система окон Крюкова → Q6:")
        print("  6 зон тела × ОТКРЫТО/ЗАКРЫТО = 2^6 = 64 состояния = Q6")
        print()
        print("Примеры состояний:")
        for h in [0, 21, 42, 63, 7, 56]:
            s = BodyState(h)
            print(f"  h={h:2d} ({h:06b}): yang={s.yang_level()}  {s.open_zones() or ['нет']}")
        print()
        print("Система сфер вокруг h=0:")
        sys_s = SphereSystem(0)
        print(f"  МВС (r=1): {len(sys_s.mvs())} вершин")
        print(f"  СВС (r=2): {len(sys_s.svs())} вершин")
        print(f"  БВС (r=3): {len(sys_s.bvs())} вершин")
        print()
        print("Восьмёрка по осям ВЛ/ВП (биты 0,1) от h=0:")
        f8 = FigureEight(0, (0, 1))
        print(f"  Петля 1: {f8.loop1()}")
        print(f"  Петля 2: {f8.loop2()}")
        print(f"  Всего восьмёрок в Q6: {len(FigureEight.all_figure8s())}")
        print()
        print("Запуск: python hexboya.py --help")


if __name__ == "__main__":
    main()
