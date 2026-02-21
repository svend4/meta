"""
kub_glyphs.py — Визуализация 64 глифов Q6 как КУБ числа 4 (4×4×4)

Три координаты КУБа (x,y,z) ∈ {0,1,2,3}:
  h = x + 4*y + 16*z

Цвет по расстоянию от центра куба (1.5, 1.5, 1.5).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from libs.hexcore.hexcore import render, yang_count as yang


# ANSI цвета — 6 цветов по yang (слои куба)
YANG_COLORS = {
    0: "\033[90m",   # тёмно-серый  (yang=0: угол)
    1: "\033[34m",   # синий       (yang=1: ребро)
    2: "\033[36m",   # голубой     (yang=2: грань)
    3: "\033[32m",   # зелёный     (yang=3: экватор)
    4: "\033[33m",   # жёлтый      (yang=4: грань)
    5: "\033[35m",   # пурпурный   (yang=5: ребро)
    6: "\033[37m",   # светло-серый(yang=6: угол)
}
RESET = "\033[0m"

# Символы для yang-уровней
YANG_SYMBOLS = {0: "○", 1: "◦", 2: "·", 3: "■", 4: "◉", 5: "●", 6: "◆"}


def h_to_xyz(h: int) -> tuple:
    """Перевод номера глифа h (0..63) в 3D-координаты куба 4×4×4."""
    x = h & 3          # биты 0-1
    y = (h >> 2) & 3   # биты 2-3
    z = (h >> 4) & 3   # биты 4-5
    return (x, y, z)


def xyz_to_h(x: int, y: int, z: int) -> int:
    """Обратное преобразование: (x,y,z) → h."""
    return x + 4*y + 16*z


def distance_from_center(h: int) -> float:
    """Расстояние вершины h от центра куба (1.5, 1.5, 1.5)."""
    x, y, z = h_to_xyz(h)
    return ((x-1.5)**2 + (y-1.5)**2 + (z-1.5)**2) ** 0.5


def render_cube_layer(z: int, use_color: bool = True) -> str:
    """Один слой куба 4×4 при фиксированном z."""
    lines = [f"  Слой z={z}:  (h = {16*z}..{16*z+15})"]
    header = "       x=0       x=1       x=2       x=3"
    lines.append(header)
    for y in range(4):
        row_parts = []
        for x in range(4):
            h = xyz_to_h(x, y, z)
            y_val = yang(h)
            sym = YANG_SYMBOLS[y_val]
            color = YANG_COLORS[y_val] if use_color else ""
            rst = RESET if use_color else ""
            row_parts.append(f" {color}{sym}{h:2d}(y{y_val}){rst}")
        lines.append(f"  y={y}: " + "  ".join(row_parts))
    return "\n".join(lines)


def render_full_cube(use_color: bool = True) -> str:
    """Полный куб 4×4×4: все 64 глифа по слоям."""
    lines = [
        "=" * 60,
        "64 глифа Q6 = КУБ числа 4 (4×4×4 сетка)",
        "Кодировка: h = x + 4*y + 16*z,  x,y,z ∈ {0,1,2,3}",
        "Обозначения: символ + номер(yang)",
        "=" * 60,
    ]
    for z in range(4):
        lines.append("")
        lines.append(render_cube_layer(z, use_color))
    lines.extend([
        "",
        "Yang-уровни (число единичных бит):",
        "  yang=0 ○:  1 глиф  — вершина (0,0,0)",
        "  yang=1 ◦:  6 глифов — рёбра куба",
        "  yang=2 ·: 15 глифов — грани куба",
        "  yang=3 ■: 20 глифов — экватор (максимум)",
        "  yang=4 ◉: 15 глифов — грани (симм. yang=2)",
        "  yang=5 ●:  6 глифов — рёбра (симм. yang=1)",
        "  yang=6 ◆:  1 глиф  — вершина (3,3,3)",
        "  Итого: 1+6+15+20+15+6+1 = 64 = 4³ ✓",
    ])
    return "\n".join(lines)


def render_yang_shell(yang_level: int, use_color: bool = True) -> str:
    """Все глифы с заданным yang-уровнем — одна «оболочка» куба."""
    glyphs = [h for h in range(64) if yang(h) == yang_level]
    color = YANG_COLORS.get(yang_level, "") if use_color else ""
    rst = RESET if use_color else ""
    names = {0: "угол", 1: "рёбра", 2: "грани", 3: "экватор",
             4: "грани", 5: "рёбра", 6: "угол"}
    lines = [
        f"Yang={yang_level} ({names.get(yang_level,'')}):",
        f"  Число глифов: {len(glyphs)}",
        "  Глифы: " + " ".join(
            f"{color}{h:2d}{rst}" for h in glyphs
        ),
        "  3D-координаты:",
    ]
    for h in glyphs:
        x, y, z = h_to_xyz(h)
        d = distance_from_center(h)
        lines.append(f"    h={h:2d}: ({x},{y},{z})  dist={d:.3f}")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="kub_glyphs — 64 глифа Q6 как КУБ числа 4 (4×4×4)"
    )
    parser.add_argument("--cube", action="store_true",
                        help="Показать полный куб 4×4×4")
    parser.add_argument("--layer", type=int, metavar="Z",
                        help="Показать слой z=Z (0..3)")
    parser.add_argument("--yang", type=int, metavar="K",
                        help="Показать оболочку yang=K (0..6)")
    parser.add_argument("--xyz", type=int, metavar="H",
                        help="Показать 3D-координаты глифа H")
    parser.add_argument("--no-color", action="store_true",
                        help="Без цвета")
    args = parser.parse_args()

    use_color = not args.no_color

    if args.cube:
        print(render_full_cube(use_color))
    elif args.layer is not None:
        print(render_cube_layer(args.layer, use_color))
    elif args.yang is not None:
        print(render_yang_shell(args.yang, use_color))
    elif args.xyz is not None:
        h = args.xyz
        x, y, z = h_to_xyz(h)
        d = distance_from_center(h)
        print(f"\nГлиф h={h}:")
        print(f"  Бинарный: {bin(h)[2:].zfill(6)}")
        print(f"  Yang:     {yang(h)}")
        print(f"  3D (x,y,z) в кубе 4×4×4: ({x},{y},{z})")
        print(f"  Расстояние от центра (1.5,1.5,1.5): {d:.4f}")
        print(f"  КУБ-адрес: {h} = {x} + 4×{y} + 16×{z}")
    else:
        print(render_full_cube(use_color))


if __name__ == "__main__":
    main()
