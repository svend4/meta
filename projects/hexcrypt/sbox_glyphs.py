"""sbox_glyphs — Криптографический анализ S-блоков через глифы Q6.

Каждый глиф (0..63) — 6-битное входное или выходное слово S-блока.
S-блок: биекция f: Q6 → Q6.

Ключевые таблицы криптанализа:
  DDT[a][b] = |{x : f(x⊕a) ⊕ f(x) = b}|   — дифференциальный криптанализ
  LAT[a][b] = Σ_x (−1)^{⟨a,x⟩⊕⟨b,f(x)⟩}    — линейный криптанализ

  Дифференциальная равномерность δ = max_{a≠0,b} DDT[a][b]
  Нелинейность nl = min_{u≠0} (64 − max_a|LAT[a][u]|) / 2

Идеальный S-блок:
  δ = 2 (APN — almost perfect nonlinear)  → не существует для n=6 (чётное n)
  nl = 28 (bent-подобная нелинейность)

Визуализация:
  ddt  <sbox>    — строка DDT[a=*]: все выходные разности для фикс. входной
  lat  <sbox>    — строка LAT[a=*]: все линейные смещения для фикс. маски
  map  <sbox>    — 8×8 карта: глиф x → глиф f(x), цвет = yang_count(f(x))
  cmp            — сравнение нескольких S-блоков по nl, δ, deg

Команды CLI:
  ddt  [--sbox identity|affine|random|complement] [--row a]
  lat  [--sbox ...]  [--col b]
  map  [--sbox ...]
  cmp
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexcrypt.hexcrypt import (
    SBox,
    identity_sbox, bit_reversal_sbox, affine_sbox,
    complement_sbox, random_sbox, yang_sort_sbox,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Фабрика S-блоков по имени
# ---------------------------------------------------------------------------

def _get_sbox(name: str) -> SBox:
    _map = {
        'identity':    identity_sbox,
        'bit_reversal': bit_reversal_sbox,
        'affine':      affine_sbox,
        'complement':  complement_sbox,
        'random':      random_sbox,
        'yang_sort':   yang_sort_sbox,
    }
    if name not in _map:
        raise ValueError(f'Unknown sbox: {name!r}. Choose: {list(_map)}')
    return _map[name]()


# Цвет по значению DDT/LAT
def _ddt_color(v: int, v_max: int, color: bool) -> str:
    if not color:
        return ''
    if v == 0:
        return _YANG_ANSI[0]
    if v == 64:
        return _YANG_ANSI[6] + _BOLD
    level = max(1, min(5, int(5 * v / v_max)))
    return _YANG_ANSI[level]


def _lat_color(v: int, color: bool) -> str:
    if not color:
        return ''
    if v == 0:
        return _YANG_ANSI[0]
    ab = abs(v)
    level = max(1, min(6, ab // 10))
    return _YANG_ANSI[level]


# ---------------------------------------------------------------------------
# 1. Карта S-блока: x → f(x)
# ---------------------------------------------------------------------------

def render_map(sb: SBox, color: bool = True) -> str:
    """
    8×8 сетка: глиф x раскрашен по yang_count(f(x)).

    Строки глифа x показывают сам x, цвет = «куда он отображается».
    Под каждым глифом — значение f(x) в двоичном виде.
    """
    table = sb.table()
    nl = sb.nonlinearity()
    du = sb.differential_uniformity()
    deg = sb.algebraic_degree()
    is_apn = sb.is_almost_perfect_nonlinear()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Карта S-блока: x ↦ f(x)')
    lines.append(f'  nl={nl}   δ={du}{"(APN!)" if is_apn else ""}   deg={deg}')
    lines.append('  Цвет глифа x = yang_count(f(x))')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            x = row * 8 + col
            fx = table[x]
            rows3 = render_glyph(x)
            if color:
                yc = yang_count(fx)
                c = _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            x = row * 8 + col
            fx = table[x]
            if color:
                yc = yang_count(fx)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}→{fx:02d}{_RESET}')
            else:
                lbl.append(f'→{fx:02d}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Сохранение yang-count: сколько x и f(x) имеют одинаковый yang
    same_yang = sum(1 for x in range(64) if yang_count(x) == yang_count(table[x]))
    lines.append(f'  yang_count(x) = yang_count(f(x)) для {same_yang}/64 элементов')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Строка DDT
# ---------------------------------------------------------------------------

def render_ddt(sb: SBox, row_a: int = 1, color: bool = True) -> str:
    """
    Строка DDT[row_a][*]: 64 глифа, раскрашенных по DDT[a][b].

    DDT[a][b] = число пар (x, x⊕a) с f(x⊕a)⊕f(x) = b.
    Максимум δ определяет устойчивость к дифференциальному криптанализу.
    """
    ddt = sb.difference_distribution_table()
    row = ddt[row_a]
    v_max = max(row)
    du = sb.differential_uniformity()

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  DDT[a={row_a}][b], b=0..63')
    lines.append(f'  Входная разность Δx={row_a} (={format(row_a,"06b")})')
    lines.append(f'  δ={du}   max в этой строке={v_max}   DDT[a,0]={row[0]}')
    lines.append('  Цвет: яркий=высокое значение (плохо для безопасности)')
    lines.append('═' * 64)

    for r in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            b = r * 8 + col
            val = row[b]
            rows3 = render_glyph(b)
            if color:
                is_max = (val == v_max and val > 0)
                c = _ddt_color(val, v_max, color)
                if is_max:
                    c = _YANG_BG[5] + _BOLD
                rows3 = [c + s + _RESET for s in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            b = r * 8 + col
            val = row[b]
            if color:
                c = _ddt_color(val, v_max, color)
                lbl.append(f'{c}{val:3d}{_RESET}')
            else:
                lbl.append(f'{val:3d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    nonzero_b = [b for b in range(64) if row[b] > 0]
    lines.append(f'  Ненулевых выходных разностей Δy: {len(nonzero_b)}/64')
    lines.append(f'  Сумма строки: {sum(row)} (= 64 всегда)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Строка LAT
# ---------------------------------------------------------------------------

def render_lat(sb: SBox, col_b: int = 1, color: bool = True) -> str:
    """
    Столбец LAT[*][col_b]: 64 глифа a, раскрашенных по |LAT[a][b]|.

    LAT[a][b] = корреляция линейного приближения ⟨a,x⟩ ≈ ⟨b,f(x)⟩.
    Максимальное смещение = max_a |LAT[a][b]| / 64 (чем меньше, тем лучше).
    """
    lat = sb.linear_approximation_table()
    col = [lat[a][col_b] for a in range(64)]
    v_max = max(abs(v) for v in col)
    nl = sb.nonlinearity()

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  LAT[a][b={col_b}], a=0..63   (выходная маска b={format(col_b,"06b")})')
    lines.append(f'  nl={nl}   max|LAT[*,{col_b}]|={v_max}   смещение={v_max}/64={v_max/64:.4f}')
    lines.append('  Цвет: яркий=высокое |смещение| (плохо)')
    lines.append('═' * 64)

    for r in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col_idx in range(8):
            a = r * 8 + col_idx
            val = col[a]
            rows3 = render_glyph(a)
            if color:
                is_max = (abs(val) == v_max and val != 0)
                if is_max:
                    c = _YANG_BG[5] + _BOLD
                elif val == 0:
                    c = _YANG_ANSI[0]
                else:
                    level = min(5, abs(val) * 5 // (v_max + 1) + 1)
                    c = _YANG_ANSI[level]
                rows3 = [c + s + _RESET for s in rows3]
            glyph_rows[col_idx] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[c][ri] for c in range(8)))  # type: ignore
        lbl = []
        for col_idx in range(8):
            a = r * 8 + col_idx
            val = col[a]
            if color:
                c = _lat_color(val, color)
                lbl.append(f'{c}{val:+4d}{_RESET}')
            else:
                lbl.append(f'{val:+4d}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Сравнительная таблица S-блоков
# ---------------------------------------------------------------------------

def render_comparison(color: bool = True) -> str:
    """
    Сравнить 6 S-блоков по nl, δ, deg, SAC.
    """
    named: list[tuple[str, SBox]] = [
        ('identity',    identity_sbox()),
        ('bit_reversal', bit_reversal_sbox()),
        ('affine',      affine_sbox()),
        ('complement',  complement_sbox()),
        ('yang_sort',   yang_sort_sbox()),
        ('random',      random_sbox(seed=42)),
    ]

    lines: list[str] = []
    lines.append('╔' + '═' * 62 + '╗')
    lines.append('║  Сравнение S-блоков на Q6' + ' ' * 36 + '║')
    lines.append('║  nl=нелинейность  δ=дифф.равномерность  deg=степень' + ' ' * 10 + '║')
    lines.append('╚' + '═' * 62 + '╝')
    lines.append('')
    lines.append(f'  {"Имя":15s}  {"nl":>5}  {"δ":>5}  {"deg":>5}  '
                 f'{"APN":>5}  {"Оценка"}')
    lines.append('  ' + '─' * 55)

    for name, sb in named:
        nl = sb.nonlinearity()
        du = sb.differential_uniformity()
        deg = sb.algebraic_degree()
        apn = sb.is_almost_perfect_nonlinear()

        # Криптографическая оценка
        score = nl * 2 - du - (6 - deg)
        if score > 40:
            grade = 'отлично'
        elif score > 25:
            grade = 'хорошо'
        elif score > 10:
            grade = 'средне'
        else:
            grade = 'слабо'

        if color:
            level = min(6, max(0, nl // 4))
            c = _YANG_ANSI[level]
            lines.append(
                f'  {c}{name:15s}  {nl:5d}  {du:5d}  {deg:5d}  '
                f'{"✓" if apn else "✗":>5}  {grade}{_RESET}'
            )
        else:
            lines.append(
                f'  {name:15s}  {nl:5d}  {du:5d}  {deg:5d}  '
                f'{"Y" if apn else "N":>5}  {grade}'
            )

    lines.append('')
    lines.append('  Идеальные параметры для n=6:')
    lines.append('    nl = 28 (bent-bound, достижима для n=6)')
    lines.append('    δ = 2 (APN) — недостижима для биекции при чётном n!')
    lines.append('    δ = 4 — лучшее возможное для биекции на 6 битах')
    lines.append('    deg = 5 (максимальная для биекции: ≤ n−1)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='sbox_glyphs',
        description='Криптографический анализ S-блоков через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    p.add_argument('--sbox', default='affine',
                   choices=['identity', 'bit_reversal', 'affine',
                            'complement', 'yang_sort', 'random'],
                   help='выбор S-блока')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('map', help='карта x→f(x): глифы раскрашены по f(x)')

    s = sub.add_parser('ddt', help='строка DDT для входной разности a')
    s.add_argument('--row', type=int, default=1, metavar='a',
                   help='входная разность a (1..63)')

    s = sub.add_parser('lat', help='столбец LAT для выходной маски b')
    s.add_argument('--col', type=int, default=1, metavar='b',
                   help='выходная маска b (1..63)')

    sub.add_parser('cmp', help='сравнение нескольких S-блоков')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color
    sb = _get_sbox(args.sbox)

    if args.cmd == 'map':
        print(render_map(sb, color))
    elif args.cmd == 'ddt':
        print(render_ddt(sb, row_a=args.row, color=color))
    elif args.cmd == 'lat':
        print(render_lat(sb, col_b=args.col, color=color))
    elif args.cmd == 'cmp':
        print(render_comparison(color))


if __name__ == '__main__':
    main()
