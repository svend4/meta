"""knn_glyphs — Машинное обучение на Q6 через глифы.

Каждый глиф (0..63) — вектор признаков в метрическом пространстве
Q6 = (Z₂)⁶ с расстоянием Хэмминга d(x,y) = popcount(x⊕y).

Методы:
  • K-медоиды: кластеризация Q6 без евклидовой геометрии
    Медоид = элемент кластера с мин. суммой расстояний внутри кластера
  • K-ближайших соседей: классификатор по голосованию
  • Марковская цепь: случайное блуждание по Q6 (стационарное = равномерное)
  • Спектральное вложение: проекция Q6 → R² через лапласиан графа

Визуализация:
  kmedoids [k]     — k-медоидная кластеризация Q6
  knn <labels>     — k-NN для раскраски Q6 по образцу
  markov [steps]   — траектория случайного блуждания на Q6
  embed            — спектральное вложение Q6 → ASCII-плоскость

Команды CLI:
  kmedoids  [--k 4] [--seed 42]
  knn       [--k 3] [--labels yang|antipodal|random]
  markov    [--steps 20] [--start 0]
  embed
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexlearn.hexlearn import (
    KMedoids, KNN, MarkovChain,
    hamming_distance_matrix, medoid,
    yang_labeled_dataset, binary_yang_dataset, cluster_dataset,
    spectral_embed,
)
from libs.hexcore.hexcore import yang_count, antipode
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Цвета кластеров (8 штук)
# ---------------------------------------------------------------------------

_CLUSTER_COLORS = [
    '\033[38;5;27m',    # синий
    '\033[38;5;82m',    # зелёный
    '\033[38;5;196m',   # красный
    '\033[38;5;208m',   # оранжевый
    '\033[38;5;201m',   # пурпурный
    '\033[38;5;226m',   # жёлтый
    '\033[38;5;39m',    # голубой
    '\033[38;5;238m',   # серый
]
_CLUSTER_BG = [
    '\033[48;5;27m',
    '\033[48;5;82m',
    '\033[48;5;196m',
    '\033[48;5;208m',
    '\033[48;5;201m',
    '\033[48;5;226m',
    '\033[48;5;39m',
    '\033[48;5;238m',
]


# ---------------------------------------------------------------------------
# 1. K-медоиды
# ---------------------------------------------------------------------------

def render_kmedoids(k: int = 4, seed: int = 42, color: bool = True) -> str:
    """
    8×8 сетка: глифы раскрашены по принадлежности кластеру.
    Медоиды — ярко выделены.
    """
    km = KMedoids(k=k)
    km.fit(list(range(64)))
    labels = km.labels_
    medoids = km.medoids_

    # Для каждой вершины — метка кластера
    vertex_label = {h: labels[h] for h in range(64)}
    medoids_set = set(medoids)

    # Внутрикластерная дисперсия
    def intra_var(cluster_id):
        members = [h for h in range(64) if vertex_label[h] == cluster_id]
        m = medoids[cluster_id]
        return sum(bin(h ^ m).count('1') for h in members) / len(members)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  K-медоиды на Q6   k={k}   seed={seed}')
    lines.append(f'  Расстояние Хэмминга   медоиды выделены ярко')
    lines.append('═' * 66)

    for ci in range(k):
        members = [h for h in range(64) if vertex_label[h] == ci]
        m = medoids[ci]
        disp = intra_var(ci)
        if color:
            c = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
            lines.append(f'  {c}Кластер {ci}: медоид={m:02d} ({format(m,"06b")})  '
                         f'|C|={len(members)}  д̄={disp:.2f}{_RESET}')
        else:
            lines.append(f'  Кластер {ci}: медоид={m:02d}  |C|={len(members)}  д̄={disp:.2f}')
    lines.append('')

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            ci = vertex_label[h]
            rows3 = render_glyph(h)
            if color:
                is_medoid = (h in medoids_set)
                c = (_CLUSTER_BG[ci % len(_CLUSTER_BG)] + _BOLD) if is_medoid \
                    else _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            ci = vertex_label[h]
            is_m = h in medoids_set
            if color:
                c = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
                lbl.append(f'{c}{"M" if is_m else " "}{ci}{_RESET}')
            else:
                lbl.append(f'{"M" if is_m else " "}{ci}')
        lines.append('  ' + '    '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. K-ближайших соседей
# ---------------------------------------------------------------------------

def render_knn(k: int = 3, label_type: str = 'yang', color: bool = True) -> str:
    """
    8×8 сетка: каждый глиф h раскрашен по предсказанной метке kNN.

    Обучающий набор зависит от label_type:
      yang     — метка = yang_count(h) mod 2 (чётный/нечётный вес)
      antipodal — метка: h < 32 → 0, h ≥ 32 → 1 (упрощённо)
      yang3    — метка = yang_count(h) // 2 (3 класса)
    """
    # Построим обучающий набор
    if label_type == 'yang':
        dataset = [(h, yang_count(h) % 2) for h in range(64)]
        n_classes = 2
    elif label_type == 'antipodal':
        dataset = [(h, int(h >= 32)) for h in range(64)]
        n_classes = 2
    else:  # yang3
        dataset = [(h, min(2, yang_count(h) // 2)) for h in range(64)]
        n_classes = 3

    knn = KNN(k=k)
    knn.fit(dataset)
    predictions = {h: knn.predict(h) for h in range(64)}

    # Сколько правильных (= точность на обучающей выборке)
    correct = sum(1 for h, lbl in dataset if predictions[h] == lbl)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  K-ближайших соседей  k={k}  метки={label_type}')
    lines.append(f'  Точность на обуч. выборке: {correct}/64 = {correct/64:.3f}')
    lines.append('  Цвет = предсказанная метка')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            pred = predictions[h]
            true_lbl = dict(dataset)[h]
            rows3 = render_glyph(h)
            if color:
                wrong = (pred != true_lbl)
                ci = int(pred) % len(_CLUSTER_COLORS)
                c = (_CLUSTER_BG[ci] + _BOLD) if wrong else _CLUSTER_COLORS[ci]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl_row = []
        for col in range(8):
            h = row * 8 + col
            pred = predictions[h]
            true_lbl = dict(dataset)[h]
            wrong = (pred != true_lbl)
            if color:
                ci = int(pred) % len(_CLUSTER_COLORS)
                c = _CLUSTER_COLORS[ci]
                lbl_row.append(f'{c}p={pred}{"!" if wrong else " "}{_RESET}')
            else:
                lbl_row.append(f'p={pred}{"!" if wrong else " "}')
        lines.append('  ' + '  '.join(lbl_row))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Марковская цепь
# ---------------------------------------------------------------------------

def render_markov(steps: int = 20, start: int = 0, color: bool = True) -> str:
    """
    Траектория случайного блуждания по Q6: start → ... → конец.

    Каждый шаг: переход в случайного соседа (1-бит флип).
    Стационарное распределение = равномерное на Q6.
    """
    mc = MarkovChain()
    path = mc.simulate(start=start, steps=steps, seed=42)

    # Статистика посещений
    from collections import Counter
    visits = Counter(path)

    # Yang-распределение вдоль пути
    yang_path = [yang_count(h) for h in path]

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Случайное блуждание по Q6   steps={steps}   start={start}')
    lines.append(f'  Переходы: каждый шаг = случайный флип 1 бита (6 соседей)')
    lines.append(f'  Стационарное распределение = равномерное (1/64)')
    lines.append('═' * 64)

    # Мини-траектория: первые 8 шагов
    lines.append(f'\n  Первые {min(8, len(path))} шагов:')
    chunk = path[:8]
    glyphs_p = [render_glyph(h) for h in chunk]
    if color:
        glyphs_p = [
            [_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
            for h, g in zip(chunk, glyphs_p)
        ]
    for ri in range(3):
        lines.append('    ' + ' → '.join(g[ri] for g in glyphs_p))
    yang_str = ' → '.join(str(y) for y in yang_path[:8])
    lines.append(f'    yang: {yang_str}')

    # Карта посещений
    lines.append(f'\n  Карта посещений всех 64 вершин за {steps} шагов:')
    max_v = max(visits.values()) if visits else 1
    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            v = visits.get(h, 0)
            rows3 = render_glyph(h)
            if color:
                level = int(6 * v / max_v) if max_v > 0 else 0
                level = max(0, min(6, level))
                c = (_YANG_BG[level] + _BOLD) if v == max_v else _YANG_ANSI[level]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            v = visits.get(h, 0)
            if color:
                level = int(6 * v / max_v) if max_v > 0 else 0
                c = _YANG_ANSI[max(0, min(6, level))]
                lbl.append(f'{c}{v:2d}{_RESET}')
            else:
                lbl.append(f'{v:2d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    visited_count = len(visits)
    lines.append(f'  Посещено: {visited_count}/64 вершин за {steps} шагов')
    lines.append(f'  Начало={start}  Конец={path[-1]}  '
                 f'd(start,end)={bin(start ^ path[-1]).count("1")}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Спектральное вложение
# ---------------------------------------------------------------------------

def render_embed(color: bool = True) -> str:
    """
    Спектральное вложение Q6 → R²: проекция на 2 главных компоненты лапласиана.

    Координаты: v₂, v₃ — 2-й и 3-й собственные векторы лапласиана Q6.
    Вершины одного yang-уровня образуют группы.
    """
    coords_list = spectral_embed(list(range(64)), dim=2)

    # coords_list: список из 64 пар (x, y)
    xs = [coords_list[h][0] for h in range(64)]
    ys = [coords_list[h][1] for h in range(64)]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    W, H = 60, 20

    def to_grid(x, y):
        col = int((x - x_min) / (x_max - x_min + 1e-9) * (W - 1))
        row = int((1 - (y - y_min) / (y_max - y_min + 1e-9)) * (H - 1))
        return max(0, min(W-1, col)), max(0, min(H-1, row))

    # Сетка ASCII
    grid = [[' '] * W for _ in range(H)]
    grid_color = [[None] * W for _ in range(H)]
    for h in range(64):
        c, r = to_grid(xs[h], ys[h])  # noqa
        yc = yang_count(h)
        grid[r][c] = str(yc)
        grid_color[r][c] = yc

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append('  Спектральное вложение Q6 → R²')
    lines.append('  Оси: v₂, v₃ — собственные векторы лапласиана Q6')
    lines.append('  Символ = yang_count(h),  цвет = yang уровень')
    lines.append('═' * 64)
    lines.append('')

    for row in range(H):
        row_str = ''
        for col in range(W):
            ch = grid[row][col]
            if color and grid_color[row][col] is not None:
                yc = grid_color[row][col]
                row_str += _YANG_ANSI[yc] + ch + _RESET
            else:
                row_str += ch
        lines.append('  │' + row_str + '│')
    lines.append('  └' + '─' * W + '┘')

    # Легенда
    lines.append('')
    lines.append('  Легенда:')
    for yc in range(7):
        cnt = sum(1 for h in range(64) if yang_count(h) == yc)
        if color:
            c = _YANG_ANSI[yc]
            lines.append(f'  {c}  yang={yc}: C(6,{yc})={cnt} вершин{_RESET}')
        else:
            lines.append(f'    yang={yc}: {cnt} вершин')

    lines.append('')
    lines.append('  Граф Q6 6-регулярный: λ₁=6, λ₂=4 (кратность 6),')
    lines.append('  λ₃=2 (кратность 15), ..., λ₇=0 (кратность 20)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='knn_glyphs',
        description='Машинное обучение на Q6 через глифы гексаграмм',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('kmedoids', help='k-медоидная кластеризация Q6')
    s.add_argument('--k', type=int, default=4, help='число кластеров')
    s.add_argument('--seed', type=int, default=42)

    s = sub.add_parser('knn', help='k-NN классификация Q6')
    s.add_argument('--k', type=int, default=3)
    s.add_argument('--labels', default='yang',
                   choices=['yang', 'antipodal', 'yang3'])

    s = sub.add_parser('markov', help='случайное блуждание по Q6')
    s.add_argument('--steps', type=int, default=20)
    s.add_argument('--start', type=int, default=0)

    sub.add_parser('embed', help='спектральное вложение Q6 → R²')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'kmedoids':
        print(render_kmedoids(k=args.k, seed=args.seed, color=color))
    elif args.cmd == 'knn':
        print(render_knn(k=args.k, label_type=args.labels, color=color))
    elif args.cmd == 'markov':
        print(render_markov(steps=args.steps, start=args.start, color=color))
    elif args.cmd == 'embed':
        print(render_embed(color=color))


if __name__ == '__main__':
    main()
