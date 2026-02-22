"""gl6_glyphs — GL(6,2) и линейные преобразования Q6 через глифы.

Каждый глиф (0..63) — вектор v ∈ (GF(2))⁶.
Матрица M ∈ GL(6,2) задаёт линейное отображение v ↦ Mv.

GL(6,2) — группа обратимых 6×6 матриц над GF(2):
  |GL(6,2)| = Π_{k=0}^{5} (2⁶ − 2^k) = 20 158 709 760

Визуализация:
  • action  — действие матрицы M на все 64 вершины Q6 (v ↦ Mv)
  • kernel  — ядро матрицы (если M необратима): ker(M) как глифы
  • rows    — строки матрицы M как 6 глифов (базис образа)
  • orbits  — орбиты {v, Mv, M²v, ...} под повторным применением M

Команды CLI:
  action  [--seed s] [--mat random|hadamard|symplectic|identity]
  kernel  [--seed s]
  rows    [--seed s]
  orbits  [--seed s] [--start h]
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexmat.hexmat import (
    mat_vec_mul, mat_mul, mat_kernel, mat_image,
    random_invertible, gl6_order,
    mat_identity, mat_hadamard_gf2, symplectic_matrix,
    mat_from_rows, is_invertible, mat_det, mat_rank,
    mat_transpose,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _mat_to_int_rows(M) -> list[int]:
    """Строки матрицы M — уже целые (bitmask-представление GF(2)⁶)."""
    return list(M)


def _apply_mat(M, h: int) -> int:
    """Применить матрицу M к вершине h: возвращает Mh как целое."""
    return mat_vec_mul(M, h)


def _get_mat(name: str, seed: int = 42):
    """Получить матрицу по имени."""
    if name == 'random':
        return random_invertible(seed=seed)
    elif name == 'hadamard':
        return mat_hadamard_gf2()
    elif name == 'symplectic':
        return symplectic_matrix()
    elif name == 'identity':
        return mat_identity()
    raise ValueError(f'Unknown matrix: {name}')


# ---------------------------------------------------------------------------
# 1. Действие матрицы на Q6
# ---------------------------------------------------------------------------

def render_action(M, color: bool = True) -> str:
    """
    8×8 сетка: глиф v раскрашен по yang_count(Mv).

    Показывает, куда матрица M отправляет каждую вершину Q6.
    Обратимая матрица — биекция Q6 → Q6.
    """
    images = [_apply_mat(M, h) for h in range(64)]

    row_glyphs = _mat_to_int_rows(M)
    rk = mat_rank(M)
    det = mat_det(M)
    inv = is_invertible(M)

    # Число неподвижных точек
    fixed = sum(1 for h in range(64) if images[h] == h)

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Действие матрицы GL(6,2) на Q6: v ↦ Mv')
    lines.append(f'  rank={rk}  det={det}  '
                 f'{"обратима" if inv else "необратима!"}  '
                 f'неподвижных точек={fixed}')
    lines.append('  Строки M как глифы:  ' +
                 '  '.join(format(r, '06b') for r in row_glyphs))
    lines.append('  Цвет глифа v = yang_count(Mv)')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            mh = images[h]
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(mh)
                is_fixed = (mh == h)
                c = (_YANG_BG[yc] + _BOLD) if is_fixed else _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            mh = images[h]
            if color:
                yc = yang_count(mh)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}→{mh:02d}{_RESET}')
            else:
                lbl.append(f'→{mh:02d}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Yang-distribution: сколько v имеет yang_count(Mv)=k
    from collections import Counter
    yang_dist = Counter(yang_count(images[h]) for h in range(64))
    lines.append('  Распределение yang_count(Mv):')
    for k in range(7):
        cnt = yang_dist.get(k, 0)
        if color:
            lines.append(f'  {_YANG_ANSI[k]}  yang={k}: {cnt:2d} вершин{_RESET}')
        else:
            lines.append(f'    yang={k}: {cnt:2d} вершин')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Строки матрицы как глифы (базис образа)
# ---------------------------------------------------------------------------

def render_rows(M, color: bool = True) -> str:
    """
    6 строк матрицы M — 6 глифов.

    Каждая строка M[i] ∈ (GF(2))⁶ — это вектор, т.е. глиф.
    При обратимой M строки образуют базис (GF(2))⁶.
    """
    row_glyphs = _mat_to_int_rows(M)
    rk = mat_rank(M)
    det = mat_det(M)
    inv = is_invertible(M)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append('  Строки матрицы M ∈ GL(6,2) как глифы Q6')
    lines.append(f'  rank={rk}  det={det}  {"обратима" if inv else "необратима!"}')
    lines.append(f'  |GL(6,2)| = {gl6_order()}')
    lines.append('  Каждая строка M[i] = вектор базиса образа')
    lines.append('═' * 64)

    # Матрица в числовом виде
    lines.append('\n  Матрица M над GF(2):')
    for i, row_h in enumerate(row_glyphs):
        bits = format(row_h, '06b')
        if color:
            c = _YANG_ANSI[yang_count(row_h)]
            lines.append(f'  {c}  строка {i}: {bits}  (глиф {row_h:2d}){_RESET}')
        else:
            lines.append(f'    строка {i}: {bits}  (глиф {row_h:2d})')

    # 6 глифов строк в ряд
    lines.append('')
    glyphs6 = [render_glyph(h) for h in row_glyphs]
    if color:
        glyphs6 = [
            [_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
            for h, g in zip(row_glyphs, glyphs6)
        ]
    headers = ['  ' + f'M[{i}]={row_glyphs[i]:02d}' for i in range(6)]
    lines.append('  ' + '  '.join(f'M[{i}]' for i in range(6)))
    for ri in range(3):
        lines.append('  ' + '   '.join(g[ri] for g in glyphs6))
    lines.append('  ' + '  '.join(format(h, '06b') for h in row_glyphs))

    # Транспонированная матрица (столбцы)
    MT = mat_transpose(M)
    col_glyphs = _mat_to_int_rows(MT)
    lines.append('\n  Столбцы Mᵀ (= строки Mᵀ):')
    glyphs_t = [render_glyph(h) for h in col_glyphs]
    if color:
        glyphs_t = [
            [_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
            for h, g in zip(col_glyphs, glyphs_t)
        ]
    lines.append('  ' + '  '.join(f'col{i}' for i in range(6)))
    for ri in range(3):
        lines.append('  ' + '   '.join(g[ri] for g in glyphs_t))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Ядро матрицы
# ---------------------------------------------------------------------------

def render_kernel(M, color: bool = True) -> str:
    """
    Ядро матрицы M: ker(M) = {v : Mv = 0}.

    Для обратимой M: ker = {0}.
    Для вырожденной: подпространство размерности 6 − rank.
    """
    kernel_ints = mat_kernel(M)
    rk = mat_rank(M)
    dim_ker = 6 - rk

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Ядро матрицы M: ker(M) = {{v : Mv = 0}}')
    lines.append(f'  rank(M)={rk}   dim(ker)={dim_ker}   |ker|={2**dim_ker}')
    lines.append('═' * 64)

    if dim_ker == 0:
        lines.append('')
        lines.append('  ker(M) = {0}  — матрица обратима, ядро тривиально.')
        lines.append(f'  |GL(6,2)| = {gl6_order()}')
        return '\n'.join(lines)

    # Базис ядра
    lines.append(f'\n  Базис ядра ({len(kernel_ints)} вектора):')
    glyphs_k = [render_glyph(h) for h in kernel_ints]
    if color:
        glyphs_k = [
            [_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
            for h, g in zip(kernel_ints, glyphs_k)
        ]
    for ri in range(3):
        lines.append('  ' + '  '.join(g[ri] for g in glyphs_k))
    lines.append('  ' + '  '.join(format(h, '06b') for h in kernel_ints))

    # Все элементы ядра
    all_ker = []
    for mask in range(2 ** len(kernel_ints)):
        v = 0
        for bit in range(len(kernel_ints)):
            if (mask >> bit) & 1:
                v ^= kernel_ints[bit]
        all_ker.append(v)
    all_ker.sort()

    lines.append(f'\n  Все {len(all_ker)} элементов ker(M):')
    glyphs_all = [render_glyph(h) for h in all_ker]
    if color:
        glyphs_all = [
            [_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
            for h, g in zip(all_ker, glyphs_all)
        ]
    for ri in range(3):
        lines.append('  ' + '  '.join(g[ri] for g in glyphs_all))
    lines.append('  ' + '  '.join(f'{h:02d}' for h in all_ker))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Орбиты под повторным применением M
# ---------------------------------------------------------------------------

def render_orbits(M, start: int = 1, color: bool = True) -> str:
    """
    Орбита вершины start под итерациями M: start → Mˡstart → M²start → ...

    Для обратимой M орбита конечна (порядок элемента GL(6,2)).
    """
    # Находим все орбиты
    visited = [False] * 64
    orbits: list[list[int]] = []
    for h0 in range(64):
        if not visited[h0]:
            orb = []
            cur = h0
            while not visited[cur]:
                visited[cur] = True
                orb.append(cur)
                cur = _apply_mat(M, cur)
            orbits.append(orb)

    # Сортируем по размеру
    orbits.sort(key=lambda o: (-len(o), o[0]))

    rk = mat_rank(M)
    det = mat_det(M)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Орбиты Q6 под действием M: v → Mv → M²v → ...')
    lines.append(f'  rank={rk}  det={det}  '
                 f'Орбит: {len(orbits)}')
    lines.append('═' * 64)
    lines.append('')

    from collections import Counter
    size_count = Counter(len(o) for o in orbits)
    lines.append('  Размеры орбит:')
    for sz in sorted(size_count):
        cnt = size_count[sz]
        if color:
            c = _YANG_ANSI[min(6, sz // 4)]
            lines.append(f'  {c}  длина {sz}: {cnt} орбит(а) — итого {sz*cnt} вершин{_RESET}')
        else:
            lines.append(f'    длина {sz}: {cnt} орбит(а) — итого {sz*cnt} вершин')
    lines.append('')

    # Показываем первые 5 орбит
    for idx, orb in enumerate(orbits[:5]):
        lines.append(f'  Орбита {idx+1} (длина {len(orb)}):  '
                     + ' → '.join(f'{h:02d}' for h in orb[:8])
                     + (' → ...' if len(orb) > 8 else ''))
        glyphs_o = [render_glyph(h) for h in orb[:8]]
        if color:
            glyphs_o = [
                [_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
                for h, g in zip(orb[:8], glyphs_o)
            ]
        for ri in range(3):
            lines.append('    ' + ' → '.join(g[ri] for g in glyphs_o))
        lines.append('')

    if len(orbits) > 5:
        lines.append(f'  ... ещё {len(orbits)-5} орбит(ы) ...')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='gl6_glyphs',
        description='GL(6,2) и линейные преобразования Q6 через глифы',
    )
    p.add_argument('--no-color', action='store_true')
    p.add_argument('--mat', default='random',
                   choices=['random', 'hadamard', 'symplectic', 'identity'],
                   help='матрица M')
    p.add_argument('--seed', type=int, default=42)
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('action', help='действие M на все 64 вершины Q6')
    sub.add_parser('rows',   help='строки матрицы M как 6 глифов')
    sub.add_parser('kernel', help='ядро матрицы M')
    s = sub.add_parser('orbits', help='орбиты Q6 под повторным M')
    s.add_argument('--start', type=int, default=1)
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color
    M = _get_mat(args.mat, seed=args.seed)

    if args.cmd == 'action':
        print(render_action(M, color))
    elif args.cmd == 'rows':
        print(render_rows(M, color))
    elif args.cmd == 'kernel':
        print(render_kernel(M, color))
    elif args.cmd == 'orbits':
        print(render_orbits(M, start=args.start, color=color))


if __name__ == '__main__':
    main()
