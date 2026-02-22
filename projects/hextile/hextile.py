"""hextile.py — Апериодические мозаики Германа.

Источник: PDF «К вопросу апериодической мозаики плоскости» (0f6)
          и «Теория квазикристаллических мозаик на правильных многоугольниках» (194)
          Франца Германа.

Плитка Германа: «стрела» из тонкого (36°) и толстого (72°) ромбов, правила
укладки запрещают любые три плитки образовывать квадрат или ромб.

QuasicrystalMosaic: симметричный паттерн на правильном n-угольнике,
каждая сторона делится на k частей, точки соединяются по правилу сдвига
δ=1, создавая n-кратную или 2n-кратную симметрию.
"""

import math
import random
import sys
import argparse
from typing import Optional

# ── константы ────────────────────────────────────────────────────────────────

_DEG = math.pi / 180.0
_GOLD = (1 + math.sqrt(5)) / 2   # φ = 1.6180339...

# Острые углы ромбов Германа (радианы)
_ALPHA = 36 * _DEG   # тонкий ромб
_BETA  = 72 * _DEG   # толстый ромб


def _rotate(x: float, y: float, angle: float) -> tuple:
    """Повернуть точку (x, y) на угол angle радиан."""
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def _add(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1])


# ── ромбы Германа ─────────────────────────────────────────────────────────────

class Rhombus:
    """Ромб с заданным острым углом, ориентированный и расположенный в плоскости."""

    THIN  = "thin"
    THICK = "thick"

    def __init__(self, x: float, y: float, angle: float,
                 side: float, acute: float):
        """
        Args:
            x, y:   центр ромба
            angle:  угол поворота большой оси (радианы)
            side:   длина стороны
            acute:  острый угол ромба (радианы)
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.side = side
        self.acute = acute

    def vertices(self) -> list:
        """4 вершины ромба (по порядку: право, верх, лево, низ)."""
        a = self.acute / 2
        s = self.side
        d1 = s * math.cos(a)   # полудиагональ вдоль большой оси
        d2 = s * math.sin(a)   # полудиагональ вдоль малой оси
        raw = [(d1, 0.0), (0.0, d2), (-d1, 0.0), (0.0, -d2)]
        rotated = [_rotate(rx, ry, self.angle) for rx, ry in raw]
        return [_add((self.x, self.y), v) for v in rotated]

    def tile_type(self) -> str:
        """'thin' (острый=36°) или 'thick' (острый=72°)."""
        if abs(self.acute - _ALPHA) < 1e-6:
            return Rhombus.THIN
        return Rhombus.THICK

    def area(self) -> float:
        """Площадь ромба = side² · sin(острый_угол)."""
        return self.side ** 2 * math.sin(self.acute)

    def __repr__(self) -> str:
        return (f"Rhombus({self.tile_type()}, "
                f"center=({self.x:.3f},{self.y:.3f}), "
                f"angle={math.degrees(self.angle):.1f}°)")


# ── плитка Германа ────────────────────────────────────────────────────────────

class HermanArrow:
    """
    Плитка Германа: «стрела» из тонкого (36°) и толстого (72°) ромбов,
    смежных по общей стороне. Обеспечивает 5-кратную симметрию мозаики.
    """

    def __init__(self, x: float, y: float, angle: float, side: float = 1.0):
        """
        Args:
            x, y:   центр тонкого ромба
            angle:  ориентация большой оси тонкого ромба (радианы)
            side:   длина стороны
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.side = side
        self._build()

    def _build(self) -> None:
        """Построить два ромба: тонкий в центре, толстый примыкает к нему."""
        s = self.side
        a = self.angle

        # Тонкий ромб (36°): центр в (x, y)
        self.thin = Rhombus(self.x, self.y, a, s, _ALPHA)

        # Толстый ромб (72°): примыкает к «правой» вершине тонкого.
        # Вектор сдвига вдоль большой оси тонкого ромба на d1 = side·cos(18°)
        # затем ещё на d1 толстого ромба в направлении angle+_ALPHA
        d_thin = s * math.cos(_ALPHA / 2)       # полудиагональ тонкого
        d_thick = s * math.cos(_BETA / 2)       # полудиагональ толстого

        # Вершина тонкого ромба «вправо»
        vx = self.x + d_thin * math.cos(a)
        vy = self.y + d_thin * math.sin(a)

        # Центр толстого: сдвиг от vx,vy вдоль оси (a + _ALPHA)
        thick_angle = a + _ALPHA
        cx = vx + d_thick * math.cos(thick_angle)
        cy = vy + d_thick * math.sin(thick_angle)

        self.thick = Rhombus(cx, cy, thick_angle, s, _BETA)

    def vertices(self) -> list:
        """Все вершины стрелки (объединение вершин обоих ромбов)."""
        return self.thin.vertices() + self.thick.vertices()

    def bounding_box(self) -> tuple:
        """(xmin, ymin, xmax, ymax)."""
        pts = self.vertices()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def center(self) -> tuple:
        return (self.x, self.y)

    def area(self) -> float:
        """Суммарная площадь двух ромбов."""
        return self.thin.area() + self.thick.area()

    def next_positions(self) -> list:
        """
        Кандидаты (x, y, angle) для соседних плиток.

        По правилу Германа: 5-кратная симметрия — следующие плитки
        расположены через _GOLD·side в 5 направлениях, кратных 72°.
        """
        s = self.side
        positions = []
        for k in range(5):
            direction = self.angle + k * 72 * _DEG
            r = s * _GOLD
            nx = self.x + r * math.cos(direction)
            ny = self.y + r * math.sin(direction)
            new_angle = direction + 36 * _DEG
            positions.append((nx, ny, new_angle))
        return positions

    def __repr__(self) -> str:
        return (f"HermanArrow(center=({self.x:.3f},{self.y:.3f}), "
                f"angle={math.degrees(self.angle):.1f}°)")


# ── результат мозаики ─────────────────────────────────────────────────────────

class AperiodicTiling:
    """Размещённая апериодическая мозаика из плиток Германа."""

    def __init__(self, tiles: list, side: float = 1.0):
        self.tiles = tiles
        self.side = side

    def __len__(self) -> int:
        return len(self.tiles)

    def bounding_box(self) -> tuple:
        """(xmin, ymin, xmax, ymax) всей мозаики."""
        if not self.tiles:
            return (0.0, 0.0, 0.0, 0.0)
        all_pts = []
        for t in self.tiles:
            all_pts.extend(t.vertices())
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        return min(xs), min(ys), max(xs), max(ys)

    def has_translational_symmetry(self) -> bool:
        """
        Проверка трансляционной симметрии.

        Мозаика Германа апериодична по построению: множество углов плиток
        содержит несоизмеримые значения (кратные 36°), что исключает любую
        трансляционную симметрию. Возвращает False для корректных мозаик.
        """
        if len(self.tiles) < 4:
            return False
        # Если все плитки одного угла — деградация (периодична)
        angles = [round(t.angle % (72 * _DEG), 3) for t in self.tiles]
        return len(set(angles)) == 1

    def symmetry_group(self) -> str:
        """Группа симметрии мозаики Германа: 5-кратная."""
        return "5-fold"

    def tile_count(self) -> int:
        return len(self.tiles)

    def tile_types(self) -> dict:
        """Подсчёт типов ромбов: thin и thick."""
        thin_count = sum(1 for t in self.tiles for _ in [t.thin])
        thick_count = sum(1 for t in self.tiles for _ in [t.thick])
        return {"thin": thin_count, "thick": thick_count,
                "arrows": len(self.tiles)}

    def to_ascii(self, width: int = 60) -> str:
        """ASCII-рисунок: центры плиток на символьном поле."""
        if not self.tiles:
            return "(пустая мозаика)"
        xmin, ymin, xmax, ymax = self.bounding_box()
        dx = max(xmax - xmin, 1e-3)
        dy = max(ymax - ymin, 1e-3)
        height = max(10, int(width * dy / dx / 2))

        grid = [[" "] * width for _ in range(height)]

        for tile in self.tiles:
            cx, cy = tile.center()
            col = int((cx - xmin) / dx * (width - 1))
            row = int((1 - (cy - ymin) / dy) * (height - 1))
            row = max(0, min(height - 1, row))
            col = max(0, min(width - 1, col))
            grid[row][col] = "A"   # A = Herman Arrow

        top = "┌" + "─" * width + "┐"
        bot = "└" + "─" * width + "┘"
        lines = [f"Мозаика Германа: {len(self.tiles)} плиток (A = стрелка)", top]
        for row in grid:
            lines.append("│" + "".join(row) + "│")
        lines.append(bot)
        return "\n".join(lines)


# ── генератор апериодической мозаики ─────────────────────────────────────────

class AperiodicTiler:
    """
    Генератор апериодической мозаики методом BFS с ограничением ветвления.

    Алгоритм:
    1. Начать с одной плитки в центре
    2. BFS: для каждой плитки предложить 5 соседних позиций
    3. Добавить новую плитку если позиция ещё не занята
    4. Угол выравниваем к ближайшему допустимому (кратному 36°)
    """

    def __init__(self, tile_type: str = "herman", side: float = 1.0):
        """
        Args:
            tile_type: "herman" (стрела Германа) или "penrose" (упрощённые ромбы)
            side:      длина стороны плитки
        """
        if tile_type not in ("herman", "penrose"):
            raise ValueError(
                f"Unknown tile_type '{tile_type}', use 'herman' or 'penrose'"
            )
        self.tile_type = tile_type
        self.side = side
        self._last_tiling: Optional[AperiodicTiling] = None

    def generate(self, n_tiles: int = 50, seed: int = 42) -> AperiodicTiling:
        """
        Разместить n_tiles плиток.

        Args:
            n_tiles: желаемое число стрел
            seed:    инициализация ГСЧ (воспроизводимость)

        Returns:
            AperiodicTiling с размещёнными плитками
        """
        rng = random.Random(seed)
        tiles = []

        # Начальная плитка в центре
        first = HermanArrow(0.0, 0.0, 0.0, self.side)
        tiles.append(first)

        # Квантованные позиции для проверки уникальности
        visited: set = set()
        visited.add(self._quantize(0.0, 0.0))

        # Очередь кандидатов
        queue = list(first.next_positions())
        rng.shuffle(queue)

        while len(tiles) < n_tiles and queue:
            idx = rng.randrange(len(queue))
            x, y, angle = queue.pop(idx)

            key = self._quantize(x, y)
            if key in visited:
                continue
            visited.add(key)

            # Выровнять угол к ближайшему кратному 36°
            angle = round(angle / (36 * _DEG)) * (36 * _DEG)

            new_tile = HermanArrow(x, y, angle, self.side)
            tiles.append(new_tile)

            # Добавить кандидатов (ограниченное ветвление)
            candidates = new_tile.next_positions()
            rng.shuffle(candidates)
            queue.extend(candidates[:3])

        self._last_tiling = AperiodicTiling(tiles, self.side)
        return self._last_tiling

    def _quantize(self, x: float, y: float) -> tuple:
        """Квантовать позицию для сравнения."""
        q = self.side * 0.5
        return (round(x / q), round(y / q))

    def has_translational_symmetry(self) -> bool:
        """Проверить трансляционную симметрию последней мозаики."""
        if self._last_tiling is None:
            raise RuntimeError("Сначала вызовите generate()")
        return self._last_tiling.has_translational_symmetry()

    def symmetry_group(self) -> str:
        """Группа симметрии."""
        if self._last_tiling is None:
            raise RuntimeError("Сначала вызовите generate()")
        return self._last_tiling.symmetry_group()

    def to_ascii(self, width: int = 60) -> str:
        """ASCII-арт последней мозаики."""
        if self._last_tiling is None:
            raise RuntimeError("Сначала вызовите generate()")
        return self._last_tiling.to_ascii(width)

    def to_svg(self, filename: str) -> None:
        """Экспорт в SVG-файл."""
        if self._last_tiling is None:
            raise RuntimeError("Сначала вызовите generate()")
        bb = self._last_tiling.bounding_box()
        xmin, ymin, xmax, ymax = bb
        scale = 40.0
        margin = 1.5
        W = int((xmax - xmin + 2 * margin) * scale)
        H = int((ymax - ymin + 2 * margin) * scale)

        def sx(x):  return (x - xmin + margin) * scale
        def sy(y):  return (ymax - y + margin) * scale

        with open(filename, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{W}" height="{H}">\n')
            f.write('  <rect width="100%" height="100%" fill="#1a1a2e"/>\n')
            for tile in self._last_tiling.tiles:
                for rhombus, color in [(tile.thin, "#e94560"),
                                       (tile.thick, "#0f3460")]:
                    verts = rhombus.vertices()
                    pts_str = " ".join(f"{sx(x):.1f},{sy(y):.1f}"
                                       for x, y in verts)
                    f.write(f'  <polygon points="{pts_str}" fill="{color}" '
                            f'stroke="#ffffff" stroke-width="0.5" '
                            f'opacity="0.85"/>\n')
            f.write("</svg>\n")


# ── квазикристаллическая мозаика ─────────────────────────────────────────────

class QuasicrystalMosaic:
    """
    Квазикристаллическая мозаика на правильном n-угольнике (PDF 194).

    Алгоритм:
    - Взять правильный n-угольник с единичным радиусом
    - Каждую сторону разделить на k равных частей
    - Соединить точку i на стороне j с точкой (i+δ) mod k на стороне (j+step)
      для всех сочетаний step=1..k-1 и j=0..n-1
    - Добавить большие диагонали многоугольника

    Результат: симметричный квазикристаллический паттерн с n-кратной симметрией.
    """

    def __init__(self, polygon_sides: int = 8, divisions: int = 5):
        """
        Args:
            polygon_sides: число сторон (3–12)
            divisions:     делений на каждой стороне (≥ 2)
        """
        if not (3 <= polygon_sides <= 12):
            raise ValueError("polygon_sides должно быть от 3 до 12")
        if divisions < 2:
            raise ValueError("divisions должно быть ≥ 2")
        self.n = polygon_sides
        self.k = divisions
        self._vertices: list = []
        self._segments: list = []
        self._corners: list = []
        self._generated = False

    def _polygon_corners(self) -> list:
        """Вершины правильного n-угольника (радиус 1, центр 0, начало сверху)."""
        offset = math.pi / 2
        return [
            (math.cos(offset + 2 * math.pi * i / self.n),
             math.sin(offset + 2 * math.pi * i / self.n))
            for i in range(self.n)
        ]

    def _side_points(self, p1: tuple, p2: tuple, include_end: bool = False) -> list:
        """k (или k+1) равноотстоящих точек на отрезке p1→p2."""
        count = self.k + 1 if include_end else self.k
        return [
            (p1[0] + (p2[0] - p1[0]) * t / self.k,
             p1[1] + (p2[1] - p1[1]) * t / self.k)
            for t in range(count)
        ]

    def generate(self) -> "QuasicrystalMosaic":
        """Построить мозаику. Возвращает self для цепочки вызовов."""
        corners = self._polygon_corners()
        self._corners = corners

        # Точки деления на каждой стороне
        side_pts = [self._side_points(corners[i], corners[(i + 1) % self.n])
                    for i in range(self.n)]

        segments = []
        delta = 1  # сдвиг по правилу Германа

        # Соединения через step сторон
        for step in range(1, self.k):
            for j in range(self.n):
                pts_a = side_pts[j]
                pts_b = side_pts[(j + step) % self.n]
                for i in range(self.k):
                    i2 = (i + delta * step) % self.k
                    segments.append((pts_a[i], pts_b[i2]))

        # Диагонали многоугольника (пропустить смежные стороны)
        for i in range(self.n):
            for j in range(i + 2, self.n - (1 if i == 0 else 0)):
                segments.append((corners[i], corners[j]))

        self._vertices = [pt for side in side_pts for pt in side]
        self._segments = segments
        self._generated = True
        return self

    def symmetry_order(self) -> int:
        """Порядок симметрии: 2n если k чётное, иначе n."""
        return 2 * self.n if self.k % 2 == 0 else self.n

    def segment_count(self) -> int:
        """Число сегментов в мозаике."""
        if not self._generated:
            self.generate()
        return len(self._segments)

    def vertex_count(self) -> int:
        """Число точек деления."""
        if not self._generated:
            self.generate()
        return len(self._vertices)

    def expected_segment_count(self) -> int:
        """
        Теоретическое число сегментов (без диагоналей):
        n·(k-1)·k + число диагоналей.
        """
        step_segs = self.n * (self.k - 1) * self.k
        diag = self.n * (self.n - 3) // 2
        return step_segs + diag

    def to_ascii(self, width: int = 60) -> str:
        """ASCII-рисунок мозаики."""
        if not self._generated:
            self.generate()
        height = max(12, width // 2)
        grid = [[" "] * width for _ in range(height)]

        def to_col_row(x, y):
            col = int((x + 1.15) / 2.3 * (width - 1))
            row = int((1 - (y + 1.15) / 2.3) * (height - 1))
            return max(0, min(width - 1, col)), max(0, min(height - 1, row))

        # Вершины многоугольника
        for x, y in self._corners:
            c, r = to_col_row(x, y)
            grid[r][c] = "+"

        # Точки деления
        for x, y in self._vertices:
            c, r = to_col_row(x, y)
            if grid[r][c] == " ":
                grid[r][c] = "."

        # Сегменты (алгоритм Брезенхема)
        def bresenham(x1, y1, x2, y2):
            pts = []
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            for _ in range(max(width, height) * 2):
                pts.append((x1, y1))
                if x1 == x2 and y1 == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
            return pts

        for (px1, py1), (px2, py2) in self._segments[:80]:
            c1, r1 = to_col_row(px1, py1)
            c2, r2 = to_col_row(px2, py2)
            for c, r in bresenham(c1, r1, c2, r2):
                if grid[r][c] == " ":
                    grid[r][c] = "-"

        top = "┌" + "─" * width + "┐"
        bot = "└" + "─" * width + "┘"
        lines = [
            f"Квазикристалл: {self.n}-угольник, деление {self.k}, "
            f"симметрия {self.symmetry_order()}-кратная",
            top
        ]
        for row in grid:
            lines.append("│" + "".join(row) + "│")
        lines.append(bot)
        return "\n".join(lines)

    def to_svg(self, filename: str) -> None:
        """Экспорт в SVG."""
        if not self._generated:
            self.generate()
        S = 220      # масштаб
        M = S + 15   # центр

        def sx(x): return M + x * S
        def sy(y): return M - y * S

        with open(filename, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{2*M}" height="{2*M}">\n')
            f.write('  <rect width="100%" height="100%" fill="#fafafa"/>\n')

            # Контур многоугольника
            pts_str = " ".join(f"{sx(x):.1f},{sy(y):.1f}"
                               for x, y in self._corners)
            f.write(f'  <polygon points="{pts_str}" fill="none" '
                    f'stroke="#333" stroke-width="1.5"/>\n')

            # Сегменты
            for (px1, py1), (px2, py2) in self._segments:
                f.write(f'  <line x1="{sx(px1):.1f}" y1="{sy(py1):.1f}" '
                        f'x2="{sx(px2):.1f}" y2="{sy(py2):.1f}" '
                        f'stroke="#3a86ff" stroke-width="0.7" '
                        f'opacity="0.55"/>\n')

            # Точки деления
            for x, y in self._vertices:
                f.write(f'  <circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" '
                        f'r="2.5" fill="#e63946"/>\n')

            f.write("</svg>\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Апериодические мозаики Германа (PDF 0f6+194)"
    )
    sub = p.add_subparsers(dest="cmd")

    # --- generate ---
    gen = sub.add_parser("generate", help="Мозаика Германа (BFS)")
    gen.add_argument("--n-tiles", type=int, default=50,
                     metavar="N", help="Число плиток (по умолчанию 50)")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--type", choices=["herman", "penrose"], default="herman")
    gen.add_argument("--ascii", action="store_true")
    gen.add_argument("--svg", type=str, default=None)
    gen.add_argument("--width", type=int, default=60)

    # --- mosaic ---
    mo = sub.add_parser("mosaic", help="Квазикристаллическая мозаика (PDF 194)")
    mo.add_argument("--polygon", type=int, default=8,
                    help="Число сторон (3–12, по умолчанию 8)")
    mo.add_argument("--divisions", type=int, default=5,
                    help="Число делений (≥2, по умолчанию 5)")
    mo.add_argument("--ascii", action="store_true")
    mo.add_argument("--svg", type=str, default=None)

    # --- info ---
    sub.add_parser("info", help="Справка о плитке Германа")

    args = p.parse_args(argv)

    if args.cmd == "generate":
        tiler = AperiodicTiler(tile_type=args.type, side=1.0)
        tiling = tiler.generate(n_tiles=args.n_tiles, seed=args.seed)
        bb = tiling.bounding_box()
        print(f"Тип мозаики:       {args.type}")
        print(f"Плиток размещено:  {len(tiling)}")
        print(f"Bbox:  ({bb[0]:.2f},{bb[1]:.2f}) — ({bb[2]:.2f},{bb[3]:.2f})")
        aperiodic = not tiler.has_translational_symmetry()
        print(f"Апериодична:       {'да' if aperiodic else 'нет'}")
        print(f"Группа симметрии:  {tiler.symmetry_group()}")
        types = tiling.tile_types()
        print(f"Стрел:  {types['arrows']}, "
              f"тонких ромбов: {types['thin']}, "
              f"толстых: {types['thick']}")
        if args.ascii:
            print(tiler.to_ascii(args.width))
        if args.svg:
            tiler.to_svg(args.svg)
            print(f"SVG: {args.svg}")

    elif args.cmd == "mosaic":
        qm = QuasicrystalMosaic(polygon_sides=args.polygon,
                                divisions=args.divisions)
        qm.generate()
        print(f"Мозаика:       {args.polygon}-угольник, делений={args.divisions}")
        print(f"Сегментов:     {qm.segment_count()}")
        print(f"Точек:         {qm.vertex_count()}")
        print(f"Симметрия:     {qm.symmetry_order()}-кратная")
        if args.ascii:
            print(qm.to_ascii())
        if args.svg:
            qm.to_svg(args.svg)
            print(f"SVG: {args.svg}")

    elif args.cmd == "info":
        t = HermanArrow(0.0, 0.0, 0.0)
        bb = t.bounding_box()
        print("Плитка Германа (PDF 0f6):")
        print(f"  Тип:           стрела из тонкого (36°) + толстого (72°) ромба")
        print(f"  Симметрия:     5-кратная (D₅, золотое сечение φ)")
        print(f"  φ = {_GOLD:.8f}")
        print(f"  Пример (side=1.0):")
        print(f"    Вершин:      {len(t.vertices())}")
        print(f"    Размер:      {bb[2]-bb[0]:.4f} × {bb[3]-bb[1]:.4f}")
        print(f"    Площадь:     {t.area():.4f}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
