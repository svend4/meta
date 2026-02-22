"""hexmatroot/matroot_glyphs.py — Q6 глифы через 2×2 матричные корни.

Каждый глиф h отображается на симметрическую 2×2 матрицу через пары битов:
    A(h) = [[a, b],   a = (h >> 4) & 3   # биты 5,4 → 0..3
             [b, c]]  b = (h >> 2) & 3   # биты 3,2 → 0..3
                      c =  h       & 3   # биты 1,0 → 0..3

det(A) = a·c − b²
tr(A)  = a + c  ∈ {0,...,6}
eigenvalues: λ = (tr ± √(tr²−4·det)) / 2

Квадратный корень A¹/² существует тогда и только тогда, когда оба собственных
значения ≥ 0, т.е. det ≥ 0 И tr ≥ 0 И tr²−4·det ≥ 0.

Идемпотентность A² = A ↔ tr = 1 И det = 0.
Для нашей семьи: tr = 1 → a+c = 1 → (a=0,c=1) или (a=1,c=0),
  det = 0 → b=0. Единственные идемпотенты:
    h=1  (биты 00_00_01 → a=0,b=0,c=1 → проектор на ось y)
    h=16 (биты 01_00_00 → a=1,b=0,c=0 → проектор на ось x)

Матрицы Паули: σ₁=[[0,1],[1,0]] ↔ h=4, σ₃=[[2,0],[0,0]] (нет)
               I = [[1,0],[0,1]] ↔ h=17

Визуализация (8×8, Gray-код Q6):
  roots  — есть ли квадратный корень A(h)¹/²  (Y/N)
  idem   — идемпотентность A(h)² = A(h)        (I/·)
  det    — знак det(A(h)): + / 0 / −
  trace  — значение tr(A(h)) ∈ 0..6

Команды CLI:
  roots
  idem
  det
  trace
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexmatroot.hexmatroot import MatrixAlgebra
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Матричное отображение h → A(h)
# ---------------------------------------------------------------------------

_MA = MatrixAlgebra()


def _matrix(h: int) -> list[list[float]]:
    """A(h) = симметрическая 2×2 матрица из трёх 2-битных полей h."""
    a = float((h >> 4) & 3)
    b = float((h >> 2) & 3)
    c = float(h & 3)
    return [[a, b], [b, c]]


def _det(A: list[list[float]]) -> float:
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def _tr(A: list[list[float]]) -> float:
    return A[0][0] + A[1][1]


def _has_sqrt(A: list[list[float]]) -> bool:
    """True если A имеет вещественный квадратный корень."""
    t = _tr(A)
    d = _det(A)
    disc = t * t - 4.0 * d
    if disc < -1e-12:
        return False           # комплексные собственные значения
    sqrt_disc = math.sqrt(max(disc, 0.0))
    lam1 = (t + sqrt_disc) / 2.0
    lam2 = (t - sqrt_disc) / 2.0
    return lam1 >= -1e-12 and lam2 >= -1e-12


def _is_idempotent(A: list[list[float]], tol: float = 1e-9) -> bool:
    """True если A² ≈ A."""
    t = _tr(A)
    d = _det(A)
    return abs(t - 1.0) < tol and abs(d) < tol


def _det_sign(A: list[list[float]]) -> str:
    d = _det(A)
    if abs(d) < 1e-9:
        return '0'
    return '+' if d > 0 else '−'


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

_GRAY3 = [i ^ (i >> 1) for i in range(8)]

# Цвета для ролей
_YES_COLOR  = '\033[38;5;82m'    # зелёный = да (есть корень / идемпот.)
_NO_COLOR   = '\033[38;5;196m'   # красный = нет
_ZERO_COLOR = '\033[38;5;226m'   # жёлтый = нулевой определитель
_POS_COLOR  = '\033[38;5;82m'    # зелёный = det > 0
_NEG_COLOR  = '\033[38;5;196m'   # красный = det < 0


def _header(title: str, subtitle: str = '') -> list[str]:
    lines = ['═' * 66, f'  {title}']
    if subtitle:
        lines.append(f'  {subtitle}')
    lines.append('═' * 66)
    col_hdr = '  '.join(format(g, '03b') for g in _GRAY3)
    lines.append(f'        {col_hdr}')
    lines.append('        ' + '─' * len(col_hdr))
    return lines


# ---------------------------------------------------------------------------
# 1. Квадратный корень
# ---------------------------------------------------------------------------

def render_roots(color: bool = True) -> str:
    """8×8 сетка: есть ли квадратный корень матрицы A(h).

    Y = √A(h) существует (оба собственных значения ≥ 0)
    N = √A(h) не существует в ℝ

    A(h) = [[a,b],[b,c]], a=(h>>4)&3, b=(h>>2)&3, c=h&3.
    Корень существует ↔ λ₁ ≥ 0 И λ₂ ≥ 0  (λ = (tr ± √(tr²−4det))/2).
    """
    lines = _header(
        'MatRoot: квадратный корень √A(h)',
        'Y=корень существует (зелён.)  N=нет (красн.)  A(h)=[[a,b],[b,c]] поэлементно',
    )

    yes_count = 0
    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            A = _matrix(h)
            has = _has_sqrt(A)
            if has:
                yes_count += 1
            sym = 'Y' if has else 'N'
            if color:
                c = _YES_COLOR if has else _NO_COLOR
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Существует √A(h)  : {yes_count}/64 глифов')
    lines.append(f'  Не существует     : {64 - yes_count}/64 глифов')
    lines.append('  Условие: tr(A)≥0 И det(A)≥0 И tr²−4·det≥0')
    lines.append('  A(h) = [[a,b],[b,c]], a=(h>>4)&3, b=(h>>2)&3, c=h&3')
    lines.append('  det(A) = a·c − b²  tr(A) = a + c')

    # Примеры
    lines.append('')
    lines.append('  Примеры:')
    examples = [
        (0,  False, 'A=[[0,0],[0,0]] вырожд.'),
        (1,  True,  'A=[[0,0],[0,1]] λ=(0,1)'),
        (17, True,  'A=[[1,0],[0,1]] = I,  λ=(1,1)'),
        (21, False, 'A=[[1,1],[1,1]] det=0 tr=2, λ=(0,2)'),
        (27, False, 'A=[[1,2],[2,3]] det<0'),
        (63, True,  'A=[[3,3],[3,3]] λ=(0,6)'),
    ]
    for h, exp, note in examples:
        A = _matrix(h)
        has = _has_sqrt(A)
        c   = _YES_COLOR if has else _NO_COLOR
        r   = _RESET
        sym = 'Y' if has else 'N'
        lines.append(
            f'    h={h:2d}: {(c if color else "")}{sym}{(r if color else "")}  {note}'
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Идемпотентность
# ---------------------------------------------------------------------------

def render_idem(color: bool = True) -> str:
    """8×8 сетка: идемпотентные матрицы A(h)² = A(h).

    I = A² = A  (только h=1 и h=16 — проекторы)
    · = A² ≠ A
    """
    lines = _header(
        'MatRoot: идемпотентность A(h)² = A(h)',
        'I=идемпотент (tr=1,det=0)  ·=прочие',
    )

    idem_list = []
    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            A = _matrix(h)
            is_idem = _is_idempotent(A)
            if is_idem:
                idem_list.append(h)
            sym = 'I' if is_idem else '·'
            if color:
                c = _YES_COLOR if is_idem else _YANG_ANSI[yang_count(h)]
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  Идемпотентных матриц: {len(idem_list)}/64')
    for h in idem_list:
        A = _matrix(h)
        a, b, c = A[0][0], A[0][1], A[1][1]
        lines.append(
            f'    h={h:2d} ({format(h,"06b")})'
            f'  A=[[{int(a)},{int(b)}],[{int(b)},{int(c)}]]'
            f'  tr={_tr(A):.0f}  det={_det(A):.0f}'
        )
    lines.append('')
    lines.append('  Теория: A²=A ↔ A — проектор (ортог. проецирование).')
    lines.append('  Для нашей семьи: tr=1 → a+c=1; det=0 → b=0.')
    lines.append('  h=1  → [[0,0],[0,1]] проектор на ось y')
    lines.append('  h=16 → [[1,0],[0,0]] проектор на ось x')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Знак определителя
# ---------------------------------------------------------------------------

def render_det(color: bool = True) -> str:
    """8×8 сетка: знак det(A(h)) = a·c − b².

    + = det > 0 (оба собств. знака одинаковы)
    0 = det = 0 (вырожденная, одно λ = 0)
    − = det < 0 (собств. значения разных знаков, √ не существует)
    """
    lines = _header(
        'MatRoot: знак det(A(h)) = a·c − b²',
        '+(зел.)=det>0  0(жёлт.)=det=0  −(красн.)=det<0',
    )

    counts = {'+': 0, '0': 0, '−': 0}
    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            A = _matrix(h)
            ds = _det_sign(A)
            counts[ds] = counts.get(ds, 0) + 1
            if color:
                c = {'+': _POS_COLOR, '0': _ZERO_COLOR, '−': _NEG_COLOR}[ds]
                cell = f'{c}{ds}{_RESET}'
            else:
                cell = ds
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    for sym, label in [('+', 'det > 0  (λ₁,λ₂ одного знака)'),
                       ('0', 'det = 0  (одно λ = 0, вырожд.)'),
                       ('−', 'det < 0  (λ₁,λ₂ разных знаков, √ ∉ ℝ)')]:
        c = {'+': _POS_COLOR, '0': _ZERO_COLOR, '−': _NEG_COLOR}[sym] if color else ''
        r = _RESET if color else ''
        lines.append(f'  {c}{sym}{r}: {counts.get(sym, 0):3d} глифов  — {label}')
    lines.append('')
    lines.append('  det(A) = a·c − b²,  a=(h>>4)&3, b=(h>>2)&3, c=h&3')
    lines.append('  det < 0 ↔ b² > a·c ↔ «смешанный» член доминирует')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Значение следа
# ---------------------------------------------------------------------------

def render_trace(color: bool = True) -> str:
    """8×8 сетка: tr(A(h)) = a + c ∈ {0,...,6}.

    Следствие: tr = yang_count применённый к старшему биту и двум младшим?
    Нет — tr(A(h)) = (h>>4)&3 + h&3, что независимо от битов 3,2.
    """
    lines = _header(
        'MatRoot: след tr(A(h)) = a + c  ∈ {0,...,6}',
        'a=(h>>4)&3  c=h&3  tr=a+c  (ян-цвет по tr)',
    )

    # Сгруппировать по trace
    tr_count: dict[int, int] = {}
    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            A = _matrix(h)
            t = int(round(_tr(A)))
            tr_count[t] = tr_count.get(t, 0) + 1
            sym = str(t)
            if color:
                # tr ∈ 0..6 → можно использовать _YANG_ANSI[t] напрямую
                c = _YANG_ANSI[t] if 0 <= t <= 6 else ''
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append('  Распределение tr(A(h)):')
    for t in range(7):
        c = _YANG_ANSI[t] if color else ''
        r = _RESET if color else ''
        cnt = tr_count.get(t, 0)
        lines.append(f'    tr={t}: {c}{cnt:3d} глифов{r}')
    lines.append('')
    lines.append('  tr(A) = a + c = (h>>4)&3 + h&3')
    lines.append('  Замечание: tr(A) не совпадает с yang_count(h)=popcount(h).')
    lines.append('  tr=0: A = [[0,b],[b,0]] антидиагональная.')
    lines.append('  tr=6: a=c=3, A = [[3,b],[b,3]] — максимальный след.')

    # Связь с MatrixAlgebra: cyclic group для h=17 (Identity-like)
    lines.append('')
    lines.append('  Матрица Паули σ₁ ↔ h=4: A=[[0,1],[1,0]] tr=0 det=-1')
    lines.append('  Единичная матрица I ↔ h=17: A=[[1,0],[0,1]] tr=2 det=1')
    A_I = _matrix(17)
    try:
        grp = _MA.cyclic_group_of_4(A_I)
        # grp = {'E': I, 'sqrt': ..., 'S': ..., 'S3': ...}
        for name, mat in grp.items():
            lines.append(
                f'    {name}: [[{mat[0][0]:.1f},{mat[0][1]:.1f}],'
                f'[{mat[1][0]:.1f},{mat[1][1]:.1f}]]'
            )
    except Exception:
        lines.append('    (cyclic_group_of_4 недоступна для I)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='matroot_glyphs',
        description='Q6 глифы через 2×2 матричные корни и идемпотенты.',
    )
    p.add_argument('--no-color', action='store_true', help='отключить ANSI-цвет')
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('roots', help='есть ли квадратный корень √A(h)')
    sub.add_parser('idem',  help='идемпотентность A(h)²=A(h)')
    sub.add_parser('det',   help='знак определителя det(A(h))')
    sub.add_parser('trace', help='след tr(A(h)) = a + c ∈ 0..6')
    args = p.parse_args(argv)
    color = not args.no_color

    dispatch = {
        'roots': render_roots,
        'idem':  render_idem,
        'det':   render_det,
        'trace': render_trace,
    }
    if args.cmd in dispatch:
        print(dispatch[args.cmd](color))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
