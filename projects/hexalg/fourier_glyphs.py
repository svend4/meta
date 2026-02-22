"""fourier_glyphs — Гармонический анализ на Q6 через глифы.

Q6 = ((Z₂)⁶, ⊕) — абелева группа.
Характеры: χ_u(h) = (−1)^{⟨u,h⟩} ∈ {−1, +1}  для u, h ∈ Q6.
Преобразование Уолша–Адамара (= группа Фурье на Z₂⁶):
    f̂(u) = Σ_{h=0}^{63} f(h) χ_u(h)

Обратное: f(h) = (1/64) Σ_u f̂(u) χ_u(h)

Тождество Парсеваля: Σ_h f(h)² = (1/64) Σ_u f̂(u)²

Граф Кэли: Cay(Q6, S) = граф с ребром (h, h⊕s) для s ∈ S.
Собственные значения = значения f̂(u) для индикатора множества S.

Визуализация:
  chars   — таблица характеров χ_u(h) для малого блока u, h
  wht     — WHT произвольной функции: f̂(u) для u=0..63
  convolve — свёртка двух функций: (f*g)(h) = Σ_x f(x)g(x⊕h)
  cayley   — граф Кэли и его спектр

Команды CLI:
  chars    [--rows r] [--cols c]
  wht      [--func yang|indicator|random]
  convolve [--f yang] [--g yang]
  cayley   [--gen n]   (n генераторов из хэмминговых соседей)
"""

from __future__ import annotations
import sys
import argparse
import math

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexalg.hexalg import (
    character, fourier_transform, inverse_fourier_transform,
    convolve, autocorrelation, cayley_eigenvalues,
    subgroup_generated, coset_decomposition,
    is_bent_function,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _popcount(x: int) -> int:
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def _func_from_name(name: str) -> list[float]:
    """Построить функцию f: Q6 → R по имени."""
    if name == 'yang':
        return [float(yang_count(h)) for h in range(64)]
    elif name == 'indicator':
        # Индикатор уровня yang=3 (20 вершин)
        return [1.0 if yang_count(h) == 3 else 0.0 for h in range(64)]
    elif name == 'random':
        import random
        rng = random.Random(42)
        return [float(rng.randint(0, 1)) for _ in range(64)]
    elif name == 'hamming':
        # d(h, 0) = yang_count(h)
        return [float(yang_count(h)) for h in range(64)]
    raise ValueError(f'Unknown func: {name!r}')


def _wht_color(val: float, v_max: float, color: bool) -> str:
    if not color or v_max == 0:
        return ''
    level = int(6 * abs(val) / abs(v_max))
    level = max(0, min(6, level))
    if val > 0:
        return _YANG_ANSI[level]
    else:
        return '\033[38;5;196m' if level > 3 else _YANG_ANSI[1]


# ---------------------------------------------------------------------------
# 1. Таблица характеров
# ---------------------------------------------------------------------------

def render_chars(n_rows: int = 8, n_cols: int = 8, color: bool = True) -> str:
    """
    Блок таблицы характеров χ_u(h) = (−1)^{⟨u,h⟩}.

    Строки = u = 0..n_rows−1,  Столбцы = h = 0..n_cols−1.
    Значения: +1 (янь) и −1 (инь).
    """
    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Таблица характеров χ_u(h) = (−1)^{⟨u,h⟩} на Q6')
    lines.append('  Строки u = 0..{},  столбцы h = 0..{}'.format(
        n_rows - 1, n_cols - 1))
    lines.append('  +1 = ян (чётный скалярный продукт),  −1 = инь (нечётный)')
    lines.append('═' * 66)
    lines.append('')

    # Заголовок: глифы h
    header_glyphs = [render_glyph(h) for h in range(n_cols)]
    if color:
        header_glyphs = [
            [_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
            for h, g in zip(range(n_cols), header_glyphs)
        ]
    lines.append('      h=')
    for ri in range(3):
        lines.append('         ' + '  '.join(g[ri] for g in header_glyphs))
    lines.append('  u↓    ' + '  '.join(f'{h:4d}' for h in range(n_cols)))
    lines.append('')

    for u in range(n_rows):
        vals = [character(u, h) for h in range(n_cols)]
        # Глиф u слева
        g_u = render_glyph(u)
        if color:
            g_u = [_YANG_ANSI[yang_count(u)] + r + _RESET for r in g_u]

        # Значения χ
        row_vals = []
        for h, v in enumerate(vals):
            if color:
                c = _YANG_ANSI[5] if v > 0 else _YANG_ANSI[1]
                row_vals.append(f'{c}{v:+3d}{_RESET}')
            else:
                row_vals.append(f'{v:+3d}')

        if u == 0:
            lines.append(f'  {g_u[0]}  u={u}: ' + '  '.join(row_vals))
            lines.append(f'  {g_u[1]}')
            lines.append(f'  {g_u[2]}')
        else:
            for ri in range(3):
                prefix = f'  {g_u[ri]}  u={u}: ' if ri == 1 else f'  {g_u[ri]}        '
                if ri == 1:
                    lines.append(prefix + '  '.join(row_vals))
                else:
                    lines.append(prefix)
        lines.append('')

    # Ортогональность
    lines.append('  Ортогональность: Σ_h χ_u(h)χ_v(h) = 64·[u=v]')
    # Проверим для u=0 и u=1
    dot01 = sum(character(0, h) * character(1, h) for h in range(64))
    dot11 = sum(character(1, h) * character(1, h) for h in range(64))
    lines.append(f'  Σ_h χ₀(h)·χ₁(h) = {dot01}   (= 0 ✓)')
    lines.append(f'  Σ_h χ₁(h)·χ₁(h) = {dot11}   (= 64 ✓)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. WHT произвольной функции
# ---------------------------------------------------------------------------

def render_wht(func_name: str = 'yang', color: bool = True) -> str:
    """
    8×8 сетка глифов u, раскрашенных по f̂(u) = WHT(f)(u).

    Цвет = абсолютное значение WHT-коэффициента.
    """
    f = _func_from_name(func_name)
    F = fourier_transform(f)  # list[float] длиной 64

    f_max = max(abs(v) for v in F)
    f_norm2 = sum(x ** 2 for x in f)
    parseval_check = sum(v ** 2 for v in F) / 64.0
    is_bent = is_bent_function(f)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  WHT-спектр функции f = {func_name!r}')
    lines.append(f'  ||f||² = {f_norm2:.2f}   (1/64)·||f̂||² = {parseval_check:.2f}')
    lines.append(f'  max|f̂| = {f_max:.2f}   '
                 f'{"BENT-функция!" if is_bent else "не bent"}')
    lines.append('  Цвет глифа u = |f̂(u)|')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            u = row * 8 + col
            val = F[u]
            rows3 = render_glyph(u)
            if color:
                is_max = (abs(val) == f_max)
                c = (_YANG_BG[5] + _BOLD) if is_max else _wht_color(val, f_max, color)
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            u = row * 8 + col
            val = F[u]
            if color:
                c = _wht_color(val, f_max, color)
                lbl.append(f'{c}{val:+6.1f}{_RESET}')
            else:
                lbl.append(f'{val:+6.1f}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    lines.append(f'  Тождество Парсеваля: Σf² = (1/64)Σf̂² = {f_norm2:.4f}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Свёртка
# ---------------------------------------------------------------------------

def render_convolve(f_name: str = 'yang', g_name: str = 'indicator',
                    color: bool = True) -> str:
    """
    Свёртка (f*g)(h) = Σ_x f(x)g(x⊕h) и её WHT-факторизация.

    По теореме о свёртке: WHT(f*g) = WHT(f)·WHT(g) (поточечно).
    """
    f = _func_from_name(f_name)
    g = _func_from_name(g_name)
    fg = convolve(f, g)

    fg_max = max(abs(v) for v in fg)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Свёртка (f*g)(h) = Σ_x f(x)·g(x⊕h)')
    lines.append(f'  f = {f_name!r}   g = {g_name!r}')
    lines.append(f'  max|f*g| = {fg_max:.2f}')
    lines.append('  Цвет глифа h = |(f*g)(h)|')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            val = fg[h]
            rows3 = render_glyph(h)
            if color:
                level = int(6 * abs(val) / (fg_max + 1e-9))
                level = max(0, min(6, level))
                c = _YANG_ANSI[level]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            val = fg[h]
            if color:
                level = int(6 * abs(val) / (fg_max + 1e-9))
                level = max(0, min(6, level))
                c = _YANG_ANSI[level]
                lbl.append(f'{c}{val:+6.1f}{_RESET}')
            else:
                lbl.append(f'{val:+6.1f}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    # Автокорреляция как специальный случай
    ac = autocorrelation(f)
    ac_max = max(abs(v) for v in ac)
    lines.append(f'  Автокорреляция f⋆f: max={ac_max:.2f}  (в 0: {ac[0]:.2f})')
    lines.append('  Теорема о свёртке: WHT(f*g) = WHT(f) · WHT(g)')
    lines.append('  (умножение поточечно, не матричное)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Граф Кэли и его спектр
# ---------------------------------------------------------------------------

def render_cayley(n_gen: int = 1, color: bool = True) -> str:
    """
    Граф Кэли Cay(Q6, S) для S = {2⁰, 2¹, ..., 2^{n_gen−1}}.

    При n_gen=6: полный Q6 (6-регулярный).
    Собственные значения = WHT(1_S)(u) = Σ_{s∈S} χ_u(s).

    Глифы раскрашены по λ(u) = собственному значению.
    """
    # Генераторы S = {1, 2, 4, ..., 2^{n_gen-1}}
    S = [1 << i for i in range(min(n_gen, 6))]
    f_S = [1.0 if h in S else 0.0 for h in range(64)]

    eigenvalues = cayley_eigenvalues(S)   # list[float] длиной 64
    ev_max = max(eigenvalues)
    ev_min = min(eigenvalues)

    # Связность: граф связен ↔ S порождает Q6
    gen_subgroup = subgroup_generated(S)
    connected = (len(gen_subgroup) == 64)
    degree = len(S)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Граф Кэли Cay(Q6, S) для S = {S}')
    lines.append(f'  |S|={degree}  (регулярный граф степени {degree})')
    lines.append(f'  Связен: {"да" if connected else "нет"}  '
                 f'λ_max={ev_max:.1f}  λ_min={ev_min:.1f}')
    lines.append('  Цвет глифа u = собственное значение λ(u)')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            u = row * 8 + col
            ev = eigenvalues[u]
            rows3 = render_glyph(u)
            if color:
                # Нормируем λ ∈ [ev_min, ev_max] → [0, 6]
                span = ev_max - ev_min or 1.0
                level = int(6 * (ev - ev_min) / span)
                level = max(0, min(6, level))
                is_top = (ev == ev_max)
                c = (_YANG_BG[level] + _BOLD) if is_top else _YANG_ANSI[level]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            u = row * 8 + col
            ev = eigenvalues[u]
            if color:
                span = ev_max - ev_min or 1.0
                level = int(6 * (ev - ev_min) / span)
                c = _YANG_ANSI[max(0, min(6, level))]
                lbl.append(f'{c}{ev:+5.1f}{_RESET}')
            else:
                lbl.append(f'{ev:+5.1f}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    # Уникальные собственные значения
    from collections import Counter
    ev_count = Counter(round(e, 6) for e in eigenvalues)
    lines.append('  Спектр графа (кратности):')
    for ev_val in sorted(ev_count, reverse=True):
        mult = ev_count[ev_val]
        if color:
            span = ev_max - ev_min or 1.0
            level = int(6 * (ev_val - ev_min) / span)
            c = _YANG_ANSI[max(0, min(6, level))]
            lines.append(f'  {c}  λ={ev_val:+.1f}  кратность={mult}{_RESET}')
        else:
            lines.append(f'    λ={ev_val:+.1f}  кратность={mult}')

    lines.append('')
    lines.append('  Полный Q6 (S=все 6 генераторов): λ_k = 6−2k  для k=0..6')
    lines.append('  Диаметральный граф (по рангу yang) = дистанционный граф')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='fourier_glyphs',
        description='Гармонический анализ на Q6 через глифы гексаграмм',
    )
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('chars', help='таблица характеров χ_u(h)')
    s.add_argument('--rows', type=int, default=8)
    s.add_argument('--cols', type=int, default=8)

    s = sub.add_parser('wht', help='WHT-спектр функции')
    s.add_argument('--func', default='yang',
                   choices=['yang', 'indicator', 'random', 'hamming'])

    s = sub.add_parser('convolve', help='свёртка двух функций')
    s.add_argument('--f', default='yang',
                   choices=['yang', 'indicator', 'random', 'hamming'])
    s.add_argument('--g', default='indicator',
                   choices=['yang', 'indicator', 'random', 'hamming'])

    s = sub.add_parser('cayley', help='граф Кэли и его спектр')
    s.add_argument('--gen', type=int, default=6,
                   help='число генераторов (1..6)')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'chars':
        print(render_chars(n_rows=args.rows, n_cols=args.cols, color=color))
    elif args.cmd == 'wht':
        print(render_wht(func_name=args.func, color=color))
    elif args.cmd == 'convolve':
        print(render_convolve(f_name=args.f, g_name=args.g, color=color))
    elif args.cmd == 'cayley':
        print(render_cayley(n_gen=args.gen, color=color))


if __name__ == '__main__':
    main()
