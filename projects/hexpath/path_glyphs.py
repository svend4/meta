"""path_glyphs — Игра на Q6 и головоломки через глифы.

Q6 — игровая доска двух игроков: каждый ход = переворот одного бита.
Игрок A начинает в 0, игрок B — в 63, цели противоположны.
Захваченные вершины недоступны для противника.

Режимы игры:
  • PvP / PvAI: живой матч на Q6
  • AI minimax с alpha-beta (глубина настраивается)
  • Головоломки: достичь цели, обходя заблокированные вершины

Встроенные головоломки:
  «Через середину» — путь из 0 в 63, заблокированы {21, 42}
  «Нижняя триграмма» — путь в пределах yang≤3
  ... и другие

Визуализация:
  game     [--pa a --pb b]    — игровая доска: позиции, цели, соседи
  puzzle   [--idx n]          — головоломка: путь и заблокированные вершины
  ai       [--pa a --pb b]    — оценка позиций минимаксом
  paths    [--start s]        — кратчайшие пути из s до всех 64

Команды CLI:
  game    [--pa a --pb b]
  puzzle  [--idx n]
  ai      [--pa a --pb b --depth d]
  paths   [--start s]
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexpath.game import (
    new_game, best_move, GameState, Player, shortest_path,
)
from projects.hexpath.puzzle import (
    solve, generate_puzzle, BUILTIN_PUZZLES, Puzzle,
)
from libs.hexcore.hexcore import yang_count, neighbors, hamming
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_A_COLOR = '\033[38;5;39m'    # синий — игрок A
_B_COLOR = '\033[38;5;196m'   # красный — игрок B
_PATH_COLOR = '\033[38;5;82m'  # зелёный — путь
_BLOCK_COLOR = '\033[38;5;238m'  # серый — заблокировано


# ---------------------------------------------------------------------------
# 1. Игровая доска
# ---------------------------------------------------------------------------

def render_game(pos_a: int = 0, pos_b: int = 63, color: bool = True) -> str:
    """
    8×8 сетка: игровая доска с позициями двух игроков.

    Показывает позиции A, B, их цели, и соседей текущего хода.
    """
    state = new_game(pos_a=pos_a, pos_b=pos_b)
    target_a, target_b = state.target_a, state.target_b

    # Кратчайший путь A к цели
    path_a = shortest_path(pos_a, target_a)
    path_b = shortest_path(pos_b, target_b)

    # Соседи позиций
    nb_a = set(neighbors(pos_a))
    nb_b = set(neighbors(pos_b))

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Игровая доска Q6: два игрока движутся по рёбрам куба')
    lines.append(f'  A={pos_a:02d}({format(pos_a,"06b")}) → цель={target_a:02d}  '
                 f'B={pos_b:02d}({format(pos_b,"06b")}) → цель={target_b:02d}')
    lines.append(f'  Путь A: {" → ".join(str(h) for h in path_a)} '
                 f'({len(path_a)-1} ходов)')
    lines.append(f'  Путь B: {" → ".join(str(h) for h in path_b)} '
                 f'({len(path_b)-1} ходов)')
    lines.append('  A=синий  B=красный  цели=жирные  соседи=подчёркнуты')
    lines.append('═' * 66)

    path_a_set = set(path_a)
    path_b_set = set(path_b)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                if h == pos_a:
                    c = _YANG_BG[yang_count(h)] + _BOLD + _A_COLOR
                elif h == pos_b:
                    c = _YANG_BG[yang_count(h)] + _BOLD + _B_COLOR
                elif h == target_a:
                    c = _A_COLOR + _BOLD
                elif h == target_b:
                    c = _B_COLOR + _BOLD
                elif h in path_a_set:
                    c = _A_COLOR
                elif h in path_b_set:
                    c = _B_COLOR
                elif h in nb_a or h in nb_b:
                    c = _YANG_ANSI[yang_count(h)]
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            if h == pos_a:
                tag = ' A '
            elif h == pos_b:
                tag = ' B '
            elif h == target_a:
                tag = 'Tа '
            elif h == target_b:
                tag = 'Tb '
            elif h in path_a_set:
                tag = 'a  '
            elif h in path_b_set:
                tag = 'b  '
            else:
                tag = '   '
            if color:
                if 'A' in tag or tag == 'Tа ':
                    c = _A_COLOR
                elif 'B' in tag or 'b' in tag:
                    c = _B_COLOR
                else:
                    c = _YANG_ANSI[0]
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    lines.append(f'  Расстояние A до цели: {hamming(pos_a, target_a)}   '
                 f'Расстояние B до цели: {hamming(pos_b, target_b)}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Головоломка
# ---------------------------------------------------------------------------

def render_puzzle(idx: int = 0, color: bool = True) -> str:
    """
    8×8 сетка: головоломка hexpath.

    Показывает: старт, цель, заблокированные вершины и найденный путь.
    """
    if idx < 0 or idx >= len(BUILTIN_PUZZLES):
        idx = 0
    pz = BUILTIN_PUZZLES[idx]
    solution = solve(pz)

    solution_set = set(solution) if solution else set()
    blocked_set = set(pz.blocked)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Головоломка {idx}: «{pz.title}»')
    lines.append(f'  Старт: {pz.start:02d}   Цель: {pz.goal:02d}')
    lines.append(f'  Заблокировано: {sorted(blocked_set)}')
    if solution:
        lines.append(f'  Решение ({len(solution)-1} ходов): '
                     + ' → '.join(str(h) for h in solution))
    else:
        lines.append('  Решение: не найдено!')
    if pz.hint:
        lines.append(f'  Подсказка: {pz.hint}')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                if h == pz.start:
                    c = _YANG_BG[yc] + _BOLD + _A_COLOR
                elif h == pz.goal:
                    c = _YANG_BG[yc] + _BOLD + _B_COLOR
                elif h in blocked_set:
                    c = _BLOCK_COLOR
                elif h in solution_set:
                    c = _PATH_COLOR
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            if h == pz.start:
                tag = 'ST'
            elif h == pz.goal:
                tag = 'GO'
            elif h in blocked_set:
                tag = '██'
            elif h in solution_set:
                step = solution.index(h) if solution else 0
                tag = f's{step}'
            else:
                tag = '  '
            if color:
                if tag == 'ST':
                    c = _A_COLOR
                elif tag == 'GO':
                    c = _B_COLOR
                elif tag == '██':
                    c = _BLOCK_COLOR
                elif tag.startswith('s'):
                    c = _PATH_COLOR
                else:
                    c = _YANG_ANSI[0]
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Список всех встроенных головоломок
    lines.append('  Встроенные головоломки:')
    for i, pz2 in enumerate(BUILTIN_PUZZLES):
        sol2 = solve(pz2)
        n_steps = len(sol2) - 1 if sol2 else -1
        mark = '▶' if i == idx else ' '
        if color:
            c = _PATH_COLOR if i == idx else _YANG_ANSI[0]
            lines.append(f'  {c}{mark} [{i}] «{pz2.title}»  '
                         f'{pz2.start}→{pz2.goal}  блок={len(pz2.blocked)}'
                         f'  шагов={n_steps}{_RESET}')
        else:
            lines.append(f'  {mark} [{i}] «{pz2.title}»  '
                         f'{pz2.start}→{pz2.goal}  блок={len(pz2.blocked)}'
                         f'  шагов={n_steps}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. AI-оценка позиций
# ---------------------------------------------------------------------------

def render_ai(pos_a: int = 0, pos_b: int = 63, depth: int = 2, color: bool = True) -> str:
    """
    8×8 сетка: для каждого возможного хода A — показать оценку AI.

    Лучший ход выделен.
    """
    state = new_game(pos_a=pos_a, pos_b=pos_b)
    bm = best_move(state, depth=depth)

    # Соседи текущей позиции A — возможные ходы
    nb_a = neighbors(pos_a)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  AI minimax (глубина={depth}): оценка ходов для игрока A')
    lines.append(f'  A={pos_a:02d}  B={pos_b:02d}  цель_A={state.target_a:02d}  цель_B={state.target_b:02d}')
    lines.append(f'  Лучший ход A: {pos_a:02d} → {bm:02d}')
    lines.append('  Зелёный = лучший ход  Синий = возможный ход')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                if h == pos_a:
                    c = _YANG_BG[yc] + _BOLD + _A_COLOR
                elif h == pos_b:
                    c = _YANG_BG[yc] + _BOLD + _B_COLOR
                elif h == bm:
                    c = _PATH_COLOR + _BOLD
                elif h in nb_a:
                    c = _A_COLOR
                elif h == state.target_a:
                    c = _A_COLOR + _BOLD
                elif h == state.target_b:
                    c = _B_COLOR + _BOLD
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            if h == pos_a:
                tag = 'A'
            elif h == pos_b:
                tag = 'B'
            elif h == bm:
                tag = '★'
            elif h in nb_a:
                tag = '·'
            else:
                tag = ' '
            if color:
                if tag == 'A' or tag == '·':
                    c = _A_COLOR
                elif tag == 'B':
                    c = _B_COLOR
                elif tag == '★':
                    c = _PATH_COLOR
                else:
                    c = _YANG_ANSI[0]
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + '    '.join(lbl))
        lines.append('')

    lines.append(f'  Возможные ходы A: {nb_a}')
    lines.append(f'  Расстояние A до цели: {hamming(pos_a, state.target_a)}')
    lines.append(f'  Расстояние B до цели: {hamming(pos_b, state.target_b)}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Кратчайшие пути из start
# ---------------------------------------------------------------------------

def render_paths(start: int = 0, color: bool = True) -> str:
    """
    8×8 сетка: для каждого h — длина кратчайшего пути из start.

    Используется BFS (shortest_path) из игровой логики.
    """
    dists = [hamming(start, h) for h in range(64)]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Кратчайшие пути на Q6 из вершины {start}')
    lines.append(f'  Игровое расстояние = hamming(start, h) = число ходов')
    lines.append(f'  Максимум: {max(dists)} = антипод  {start ^ 63:02d}')
    lines.append('  Цвет = число ходов до h')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            d = dists[h]
            rows3 = render_glyph(h)
            if color:
                if h == start:
                    c = _YANG_BG[0] + _BOLD
                else:
                    c = _YANG_ANSI[d]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            d = dists[h]
            if color:
                c = _YANG_ANSI[d]
                lbl.append(f'{c}{d}{_RESET}')
            else:
                lbl.append(str(d))
        lines.append('  ' + '    '.join(lbl))
        lines.append('')

    # Пример нескольких путей
    targets = [start ^ 63, start ^ 7, start ^ 56, start ^ 21]
    lines.append('  Примеры кратчайших путей:')
    for t in targets:
        p = shortest_path(start, t)
        if color:
            c = _YANG_ANSI[len(p) - 1]
            lines.append(f'  {c}  {start:02d}→{t:02d}: {" → ".join(str(h) for h in p)}'
                         f'  ({len(p)-1} ходов){_RESET}')
        else:
            lines.append(f'    {start:02d}→{t:02d}: {" → ".join(str(h) for h in p)}'
                         f'  ({len(p)-1} ходов)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='path_glyphs',
        description='Игра hexpath на Q6 — доска и головоломки через глифы',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('game', help='игровая доска: позиции и пути игроков')
    s.add_argument('--pa', type=int, default=0)
    s.add_argument('--pb', type=int, default=63)

    s2 = sub.add_parser('puzzle', help='встроенная головоломка')
    s2.add_argument('--idx', type=int, default=0,
                    help=f'индекс головоломки (0..{len(BUILTIN_PUZZLES)-1})')

    s3 = sub.add_parser('ai', help='AI-оценка ходов (minimax)')
    s3.add_argument('--pa', type=int, default=0)
    s3.add_argument('--pb', type=int, default=63)
    s3.add_argument('--depth', type=int, default=2)

    s4 = sub.add_parser('paths', help='кратчайшие пути из стартовой вершины')
    s4.add_argument('--start', type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'game':
        print(render_game(pos_a=args.pa, pos_b=args.pb, color=color))
    elif args.cmd == 'puzzle':
        print(render_puzzle(idx=args.idx, color=color))
    elif args.cmd == 'ai':
        print(render_ai(pos_a=args.pa, pos_b=args.pb, depth=args.depth, color=color))
    elif args.cmd == 'paths':
        print(render_paths(start=args.start, color=color))


if __name__ == '__main__':
    main()
