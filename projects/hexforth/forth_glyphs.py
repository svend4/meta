"""forth_glyphs — HexForth: стековый язык на Q6 через глифы.

HexForth — минимальный стековый язык, где состояние = текущий глиф Q6.
Каждая инструкция FLIP-i перемещает по ребру Q6 (переворачивает бит i).
GOTO n — BFS-кратчайший путь к глифу n через Q6.

Структура языка:
  FLIP-0 .. FLIP-5   — ребро Q6 (перевернуть бит i)
  SET-i / CLR-i      — принудительно установить/сбросить бит i
  GOTO n             — BFS-путь к n (через FLIP-инструкции)
  ANTIPODE           — прыжок к антиподу (XOR 63)
  ASSERT-EQ n        — проверить состояние
  DEFINE w : ... ;   — определить слово

Трассировка GOTO h:
  GOTO h из начала 0 даёт путь по Q6 длины = hamming(0, h) = yang_count(h).

Визуализация:
  reach    [--start s]       — расстояния BFS от s до всех 64 глифов
  trace    [--start s --end e] — трассировка GOTO: путь s→e
  bits     [--start s]       — статистика использования битов для GOTO ко всем h
  words                      — эффект базовых слов на глиф 0

Команды CLI:
  reach    [--start s]
  trace    [--start s --end e]
  bits     [--start s]
  words
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexforth.interpreter import HexForth
from projects.hexforth.compiler import compile_to_ir, path_stats
from libs.hexcore.hexcore import yang_count, hamming, neighbors
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _goto_path(start: int, end: int) -> list[int]:
    """BFS-путь из start в end через Q6 (как делает GOTO)."""
    if start == end:
        return [start]
    from collections import deque
    visited = {start: None}
    q: deque[int] = deque([start])
    while q:
        cur = q.popleft()
        for nb in neighbors(cur):
            if nb not in visited:
                visited[nb] = cur
                if nb == end:
                    path = []
                    node = nb
                    while node is not None:
                        path.append(node)
                        node = visited[node]
                    return list(reversed(path))
                q.append(nb)
    return [start, end]


def _bit_usage(path: list[int]) -> dict[int, int]:
    """Посчитать, какие биты переворачиваются вдоль пути."""
    counts: dict[int, int] = {i: 0 for i in range(6)}
    for a, b in zip(path, path[1:]):
        diff = a ^ b
        for bit in range(6):
            if (diff >> bit) & 1:
                counts[bit] += 1
    return counts


# ---------------------------------------------------------------------------
# 1. Расстояния BFS (reach) от начала
# ---------------------------------------------------------------------------

def render_reach(start: int = 0, color: bool = True) -> str:
    """
    8×8 сетка: каждый глиф раскрашен по длине GOTO-пути из start.

    GOTO h из start пройдёт hamming(start, h) шагов (шагов по BFS в Q6).
    """
    dists = [hamming(start, h) for h in range(64)]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  HexForth GOTO: расстояния BFS из глифа {start}')
    lines.append(f'  Старт: {start} = {format(start, "06b")}   yang={yang_count(start)}')
    lines.append(f'  GOTO h требует hamming({start}, h) шагов = d(h) переворотов бит')
    lines.append('  Цвет = длина пути GOTO (= расстояние Хэмминга)')
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
                lbl.append(f'{c}d={d}{_RESET}')
            else:
                lbl.append(f'd={d}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Распределение расстояний
    import math
    from collections import Counter
    dist_cnt = Counter(dists)
    lines.append('  Распределение длин GOTO:')
    for d in range(7):
        cnt = dist_cnt.get(d, 0)
        binom = math.comb(6, d)
        if color:
            c = _YANG_ANSI[d]
            lines.append(f'  {c}  GOTO-длина {d}: {cnt:2d} глифов = C(6,{d})={binom}  '
                         f'(шагов для BFS из {start}){_RESET}')
        else:
            lines.append(f'    GOTO-длина {d}: {cnt:2d} глифов = C(6,{d})={binom}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Трассировка GOTO: путь s→e
# ---------------------------------------------------------------------------

def render_trace(start: int = 0, end: int = 42, color: bool = True) -> str:
    """
    Трассировка HexForth-программы 'GOTO end' из start.

    Показывает каждый промежуточный глиф и какой бит переворачивается.
    """
    path = _goto_path(start, end)
    bu = _bit_usage(path)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  HexForth: GOTO {end}  (от глифа {start})')
    lines.append(f'  Путь: {start} → ... → {end}   длина={len(path)-1} шагов')
    lines.append(f'  {format(start,"06b")} → {format(end,"06b")}   diff={format(start^end,"06b")}')
    lines.append('  Цвет = yang_count вдоль пути   жирный = конечный глиф')
    lines.append('═' * 66)

    # Показать путь
    path_glyphs = [render_glyph(h) for h in path]
    if color:
        path_glyphs = [
            [(_YANG_BG[yang_count(h)] + _BOLD if h == end else _YANG_ANSI[yang_count(h)])
             + r + _RESET
             for r in g]
            for h, g in zip(path, path_glyphs)
        ]

    # Вывод по группам ≤ 8 глифов в строку
    for chunk_start in range(0, len(path), 8):
        chunk = list(range(chunk_start, min(chunk_start + 8, len(path))))
        for ri in range(3):
            lines.append('  ' + ' → '.join(path_glyphs[i][ri] for i in chunk))
        # Подписи: номер глифа и перевёрнутый бит
        lbl_parts = []
        for i in chunk:
            h = path[i]
            if i < len(path) - 1:
                nxt = path[i + 1]
                diff = h ^ nxt
                flipped = next(b for b in range(6) if (diff >> b) & 1)
                lbl = f'{h:02d}[F{flipped}]'
            else:
                lbl = f'{h:02d}[end]'
            if color:
                c = _YANG_ANSI[yang_count(h)]
                lbl_parts.append(f'{c}{lbl}{_RESET}')
            else:
                lbl_parts.append(lbl)
        lines.append('  ' + ' → '.join(lbl_parts))
        lines.append('')

    # Статистика битов
    lines.append('  Использование битов (сколько раз каждый бит перевёрнут):')
    for bit in range(6):
        cnt = bu[bit]
        bar = '█' * cnt
        if color:
            c = _YANG_ANSI[bit + 1]
            lines.append(f'  {c}  бит {bit}: {cnt}x  {bar}{_RESET}')
        else:
            lines.append(f'    бит {bit}: {cnt}x  {bar}')

    # 8×8 карта: выделить путь
    path_set = set(path)
    lines.append('\n  8×8 карта пути на Q6:')
    for row in range(8):
        glyph_rows_list2 = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            in_path = h in path_set
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                if h == start:
                    c = _YANG_BG[0] + _BOLD
                elif h == end:
                    c = _YANG_BG[yc] + _BOLD
                elif in_path:
                    c = _YANG_ANSI[yc]
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list2[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list2[col][ri] for col in range(8)))  # type: ignore
        lbl2 = []
        for col in range(8):
            h = row * 8 + col
            in_path = h in path_set
            if color:
                c = _YANG_ANSI[yang_count(h)] if in_path else _YANG_ANSI[0]
                lbl2.append(f'{c}{"→" if in_path else " "}{_RESET}')
            else:
                lbl2.append('→' if in_path else ' ')
        lines.append('  ' + '    '.join(lbl2))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Битовая статистика GOTO из start
# ---------------------------------------------------------------------------

def render_bits(start: int = 0, color: bool = True) -> str:
    """
    8×8 сетка: для каждого глифа h — какой бит переворачивается последним
    в пути GOTO h из start.

    Показывает «структуру Q6» — какие биты нужны для достижения каждого глифа.
    """
    # Для каждого h: вся битовая статистика пути start→h
    bit_freq: list[dict[int, int]] = []
    for h in range(64):
        path = _goto_path(start, h)
        bit_freq.append(_bit_usage(path))

    # Наиболее используемый бит для каждого h
    dominant_bit = []
    for h in range(64):
        bf = bit_freq[h]
        if all(v == 0 for v in bf.values()):
            dominant_bit.append(-1)
        else:
            dominant_bit.append(max(bf, key=lambda b: bf[b]))

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  HexForth: статистика битов GOTO из глифа {start}')
    lines.append(f'  Для каждого h: dominant бит в пути GOTO h из {start}')
    lines.append('  Цвет = доминирующий бит (0..5)   Число = hamming(start,h)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            db = dominant_bit[h]
            rows3 = render_glyph(h)
            if color:
                if h == start:
                    c = _YANG_BG[0] + _BOLD
                elif db < 0:
                    c = _YANG_ANSI[0]
                else:
                    c = _YANG_ANSI[db + 1]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            db = dominant_bit[h]
            d = hamming(start, h)
            if color:
                c = _YANG_ANSI[db + 1] if db >= 0 else _YANG_ANSI[0]
                lbl.append(f'{c}b{db}d{d}{_RESET}')
            else:
                lbl.append(f'b{db}d{d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    # Суммарная битовая нагрузка
    total_bu = {bit: sum(bit_freq[h][bit] for h in range(64)) for bit in range(6)}
    total_flips = sum(total_bu.values())
    lines.append(f'  Суммарное использование битов (GOTO всех 64 глифов из {start}):')
    for bit in range(6):
        cnt = total_bu[bit]
        pct = cnt / total_flips * 100 if total_flips else 0
        bar = '█' * int(pct / 5)
        if color:
            c = _YANG_ANSI[bit + 1]
            lines.append(f'  {c}  бит {bit}: {cnt:3d} ({pct:.1f}%)  {bar}{_RESET}')
        else:
            lines.append(f'    бит {bit}: {cnt:3d} ({pct:.1f}%)  {bar}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Базовые слова HexForth на Q6
# ---------------------------------------------------------------------------

def render_words(color: bool = True) -> str:
    """
    Показать эффект базовых HexForth-слов на начальный глиф 0.

    FLIP-0..5: переместиться по 6 рёбрам Q6
    ANTIPODE: прыгнуть к антиподу (63)
    SET-i / CLR-i: установить/сбросить бит
    """
    test_start = 21  # интересный тест: 010101

    words_list = [
        ('FLIP-0', f'FLIP-0'),
        ('FLIP-1', f'FLIP-1'),
        ('FLIP-2', f'FLIP-2'),
        ('FLIP-3', f'FLIP-3'),
        ('FLIP-4', f'FLIP-4'),
        ('FLIP-5', f'FLIP-5'),
        ('ANTIPODE', 'ANTIPODE'),
        ('SET-0', 'SET-0'),
        ('SET-5', 'SET-5'),
        ('CLR-0', 'CLR-0'),
        ('CLR-5', 'CLR-5'),
        ('GOTO 0', 'GOTO 0'),
        ('GOTO 63', 'GOTO 63'),
        ('GOTO 42', 'GOTO 42'),
    ]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  HexForth: базовые слова на глифе {test_start} '
                 f'= {format(test_start,"06b")}')
    lines.append(f'  Начало: {test_start}   Результат каждого слова')
    lines.append('  Цвет = yang_count(результата)')
    lines.append('═' * 66)
    lines.append('')

    results = []
    for name, src in words_list:
        vm = HexForth(start=test_start)
        vm.run(src)
        results.append((name, test_start, vm.state))

    # Показать глифы результатов (до 8 в строку)
    for chunk_start in range(0, len(results), 7):
        chunk = results[chunk_start: chunk_start + 7]
        glyphs_r = [render_glyph(r) for _, _, r in chunk]
        if color:
            glyphs_r = [
                [_YANG_ANSI[yang_count(r)] + row + _RESET for row in g]
                for (_, _, r), g in zip(chunk, glyphs_r)
            ]
        for ri in range(3):
            lines.append('  ' + '  '.join(g[ri] for g in glyphs_r))
        for (name, s, r) in chunk:
            label = f'{name}→{r:02d}'
            if color:
                c = _YANG_ANSI[yang_count(r)]
                lines.append(f'  {c}{label}{_RESET}')
            else:
                lines.append(f'  {label}')
        lines.append('')

    # 8×8 карта: пометить глифы, достигаемые за 1 шаг от test_start
    reachable_1 = {test_start ^ (1 << b) for b in range(6)}
    reachable_2 = {h ^ (1 << b) for h in reachable_1 for b in range(6)}
    lines.append(f'  8×8 карта: достижимость из {test_start} за 1 и 2 шага:')

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                if h == test_start:
                    c = _YANG_BG[yc] + _BOLD
                elif h in reachable_1:
                    c = _YANG_ANSI[yc]
                elif h in reachable_2:
                    c = _YANG_ANSI[2]
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
            if h == test_start:
                tag = 'S'
            elif h in reachable_1:
                tag = '1'
            elif h in reachable_2:
                tag = '2'
            else:
                tag = ' '
            if color:
                c = _YANG_ANSI[yang_count(h)] if tag != ' ' else _YANG_ANSI[0]
                lbl.append(f'{c}{tag}{_RESET}')
            else:
                lbl.append(tag)
        lines.append('  ' + '    '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='forth_glyphs',
        description='HexForth — стековый язык на Q6 через глифы',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('reach', help='расстояния BFS (GOTO) от стартового глифа')
    s.add_argument('--start', type=int, default=0, metavar='S')

    s2 = sub.add_parser('trace', help='трассировка GOTO s→e')
    s2.add_argument('--start', type=int, default=0, metavar='S')
    s2.add_argument('--end',   type=int, default=42, metavar='E')

    s3 = sub.add_parser('bits', help='битовая статистика GOTO из стартового глифа')
    s3.add_argument('--start', type=int, default=0, metavar='S')

    sub.add_parser('words', help='базовые слова HexForth на Q6')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'reach':
        print(render_reach(start=args.start, color=color))
    elif args.cmd == 'trace':
        print(render_trace(start=args.start, end=args.end, color=color))
    elif args.cmd == 'bits':
        print(render_bits(start=args.start, color=color))
    elif args.cmd == 'words':
        print(render_words(color=color))


if __name__ == '__main__':
    main()
