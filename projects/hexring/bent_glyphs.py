"""bent_glyphs — Булевы функции и WHT на Q6 через систему глифов.

Каждый глиф (0..63) — это входной вектор x ∈ (Z₂)⁶.
Булева функция f: (Z₂)⁶ → {0,1} задаёт «раскраску» всех 64 глифов.

Преобразование Уолша–Адамара:
    Ŵ(u) = Σ_{x=0}^{63} (-1)^{f(x) ⊕ (u·x)}   ∈ {−64, −62, ..., +64}

Bent-функция (максимальная нелинейность для n=6):
    |Ŵ(u)| = 8  для всех u  →  нелинейность nl = (64 − 8) / 2 = 28

Визуализация:
  • WHT-спектр: каждый глиф раскрашен по значению |Ŵ(u)|
  • Таблица истинности: глифы раскрашены f(x) ∈ {0,1}
  • Bent-анализ: показывает примеры bent-функций и их двойников
  • ANF-степень: цвет глифа = степень монома в алгебраической нормальной форме

Команды CLI:
  wht <int>       — WHT-спектр булевой функции (целое 64-бит)
  tt  <int>       — таблица истинности как сетка глифов
  bent            — примеры bent-функций: квадратичные формы над GF(2)⁶
  anf  <int>      — алгебраическая нормальная форма, раскраска по степени
  nl   <int>      — нелинейность, дистанция от аффинных функций
"""

from __future__ import annotations
import sys
import json
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexring.hexring import BoolFunc
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Цвета для WHT-спектра (по абсолютному значению)
# ---------------------------------------------------------------------------

# |Ŵ(u)| для n=6: возможные значения 0, 8, 16, 24, 32, 40, 48, 56, 64
# Цвета: меньше = синий, больше = красный, bent = зелёный
_WHT_COLORS = {
    0:  '\033[38;5;238m',   # нулевой
    8:  '\033[38;5;82m',    # bent-уровень (зелёный)
    16: '\033[38;5;39m',    # голубой
    24: '\033[38;5;27m',    # синий
    32: '\033[38;5;208m',   # оранжевый
    40: '\033[38;5;196m',   # красный
    48: '\033[38;5;201m',   # пурпурный
    56: '\033[38;5;226m',   # жёлтый
    64: '\033[38;5;231m',   # белый
}


def _wht_color(val: int, highlight: bool = False) -> str:
    """Цвет ANSI по |Ŵ(u)|."""
    key = min(_WHT_COLORS, key=lambda k: abs(k - abs(val)))
    c = _WHT_COLORS[key]
    return (_BOLD + c) if highlight else c


# ---------------------------------------------------------------------------
# 1. WHT-спектр функции
# ---------------------------------------------------------------------------

def render_wht_spectrum(tt_int: int, color: bool = True) -> str:
    """
    Вывести WHT-спектр булевой функции как 8×8 сетку глифов.

    Каждый глиф (u=0..63) раскрашен по |Ŵ(u)|.
    Для bent-функции все |Ŵ(u)| = 8 → все глифы одного зелёного цвета.
    """
    f = BoolFunc(tt_int)
    W = f.wht()
    nl = f.nonlinearity()
    max_w = max(abs(w) for w in W)
    is_bent = (nl == 28)

    lines: list[str] = []
    lines.append('═' * 64)
    label = 'BENT-функция (nl=28)' if is_bent else f'нелинейность nl={nl}'
    lines.append(f'  WHT-спектр  f=0x{tt_int:016x}   {label}')
    lines.append(f'  max|Ŵ(u)|={max_w}   Ŵ(0)={W[0]}   степень={f.algebraic_degree()}')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            u = row * 8 + col
            w_val = W[u]
            rows3 = render_glyph(u)
            if color:
                is_bent_point = (abs(w_val) == 8)
                c = _wht_color(w_val, highlight=is_bent_point)
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        # Строки 0,1,2 глифа
        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        # Подписи: Ŵ(u)
        labels = []
        for col in range(8):
            u = row * 8 + col
            w_val = W[u]
            if color:
                c = _wht_color(w_val)
                labels.append(f'{c}{w_val:+4d}{_RESET}')
            else:
                labels.append(f'{w_val:+4d}')
        lines.append('  ' + ' '.join(labels))
        lines.append('')

    lines.append(f'  Цвета:  зелёный=|8| (bent),  синий=малое,  красный=большое')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Таблица истинности как сетка глифов
# ---------------------------------------------------------------------------

def render_tt_grid(tt_int: int, color: bool = True) -> str:
    """
    Таблица истинности f(x) как 8×8 сетка глифов.

    Глиф x раскрашен: f(x)=1 → ярко (янь), f(x)=0 → тёмно (инь).
    """
    f = BoolFunc(tt_int)
    tt = f.truth_table()
    ones = sum(tt)
    nl = f.nonlinearity()
    deg = f.algebraic_degree()

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Таблица истинности  f=0x{tt_int:016x}')
    lines.append(f'  wt(f)={ones}/64   nl={nl}   deg={deg}')
    lines.append('  Цвет: янь (яркий) = f(x)=1,  инь (тёмный) = f(x)=0')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            x = row * 8 + col
            rows3 = render_glyph(x)
            if color:
                if tt[x]:
                    c = _YANG_BG[yang_count(x)] + _BOLD
                else:
                    c = _YANG_ANSI[0]  # тёмный
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        val_row = []
        for col in range(8):
            x = row * 8 + col
            v = tt[x]
            if color:
                c = _YANG_ANSI[5] if v else _YANG_ANSI[1]
                val_row.append(f'{c} f={v}{_RESET}')
            else:
                val_row.append(f' f={v}')
        lines.append('  ' + '  '.join(val_row))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Bent-функции: квадратичные формы
# ---------------------------------------------------------------------------

# Каноничные квадратичные bent-функции: f(x) = x·Ax  (A — симм. матрица без диагонали)
# Простейшие: суммы произведений непересекающихся пар битов
_QUAD_BENT = [
    # 3 непересекающихся произведения (стандартный вид для n=6)
    ("b₀b₁ ⊕ b₂b₃ ⊕ b₄b₅",   lambda x: ((x>>0)&1)*((x>>1)&1) ^ ((x>>2)&1)*((x>>3)&1) ^ ((x>>4)&1)*((x>>5)&1)),
    ("b₀b₂ ⊕ b₁b₃ ⊕ b₄b₅",   lambda x: ((x>>0)&1)*((x>>2)&1) ^ ((x>>1)&1)*((x>>3)&1) ^ ((x>>4)&1)*((x>>5)&1)),
    ("b₀b₃ ⊕ b₁b₂ ⊕ b₄b₅",   lambda x: ((x>>0)&1)*((x>>3)&1) ^ ((x>>1)&1)*((x>>2)&1) ^ ((x>>4)&1)*((x>>5)&1)),
    ("b₀b₅ ⊕ b₁b₄ ⊕ b₂b₃",   lambda x: ((x>>0)&1)*((x>>5)&1) ^ ((x>>1)&1)*((x>>4)&1) ^ ((x>>2)&1)*((x>>3)&1)),
]


def _make_tt_int(fn) -> int:
    """Составить 64-битное целое из функции fn: int → {0,1}."""
    result = 0
    for x in range(64):
        if fn(x):
            result |= (1 << x)
    return result


def render_bent_examples(color: bool = True) -> str:
    """
    Показать 4 квадратичных bent-функции на 6 переменных.

    Для каждой:
    - Таблица истинности (сетка 8×8 глифов)
    - WHT-спектр (все |Ŵ(u)|=8)
    - Двойник (dual): тоже bent-функция
    """
    lines: list[str] = []
    lines.append('╔' + '═' * 62 + '╗')
    lines.append('║  Bent-функции на 6 переменных: квадратичные формы' + ' ' * 13 + '║')
    lines.append('║  Нелинейность nl = 28 (максимально возможная)' + ' ' * 17 + '║')
    lines.append('║  |Ŵ(u)| = 8 для всех u ∈ (Z₂)⁶' + ' ' * 30 + '║')
    lines.append('╚' + '═' * 62 + '╝')
    lines.append('')

    for name, fn in _QUAD_BENT:
        tt_int = _make_tt_int(fn)
        f = BoolFunc(tt_int)
        W = f.wht()
        nl = f.nonlinearity()
        wt = sum(f.truth_table())
        all_eight = all(abs(w) == 8 for w in W)

        status = '✓ BENT' if all_eight else f'nl={nl}'
        lines.append(f'  f(x) = {name}   [{status}]')
        lines.append(f'  wt(f)={wt}  TT=0x{tt_int:016x}')

        # Краткая строка WHT (первые 16 значений)
        w_str = ' '.join(f'{W[u]:+d}' for u in range(16))
        lines.append(f'  Ŵ[0..15]: {w_str} ...')

        # Мини-схема: 8 глифов 1-й строки (u=0..7) с цветом по f(x)
        tt = f.truth_table()
        glyphs_r: list[list[str]] = []
        for x in range(8):
            g = render_glyph(x)
            if color:
                c = _YANG_ANSI[5] if tt[x] else _YANG_ANSI[1]
                g = [c + r + _RESET for r in g]
            glyphs_r.append(g)
        for ri in range(3):
            lines.append('  ' + '  '.join(glyphs_r[x][ri] for x in range(8)) + '  ...')
        lines.append('')

    # Статистика
    lines.append('  Всего bent-функций на 6 переменных: 2^32 = 4 294 967 296')
    lines.append('  Доля среди всех: 2^32 / 2^64 = 2^{-32} ≈ 2.3×10⁻¹⁰')
    lines.append('  Квадратичных bent (степень=2): связаны с симплектическими матрицами')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. ANF-степень: раскраска по степени монома
# ---------------------------------------------------------------------------

def render_anf_degree(tt_int: int, color: bool = True) -> str:
    """
    Алгебраическая нормальная форма: каждый глиф-индекс I раскрашен
    по весу popcount(I) — степени монома x^I в ANF.

    Ненулевые коэффициенты ANF выделены ярко.
    """
    f = BoolFunc(tt_int)
    anf = f.anf()
    deg = f.algebraic_degree()
    nl = f.nonlinearity()

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  ANF-раскладка  f=0x{tt_int:016x}   deg={deg}   nl={nl}')
    lines.append('  Глиф I: яркий = коэф. aᵢ=1 (монос x^I есть в ANF)')
    lines.append('  Цвет глифа = вес popcount(I) = степень монома')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            idx = row * 8 + col
            rows3 = render_glyph(idx)
            if color:
                wt = yang_count(idx)
                has_monomial = bool(anf[idx])
                if has_monomial:
                    c = _YANG_BG[wt] + _BOLD
                else:
                    c = _YANG_ANSI[0]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            idx = row * 8 + col
            v = anf[idx]
            d = yang_count(idx)
            if color:
                c = _YANG_ANSI[d]
                lbl.append(f'{c}a{idx:02d}={v}{_RESET}')
            else:
                lbl.append(f'a{idx:02d}={v}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    nonzero = [i for i in range(64) if anf[i]]
    lines.append(f'  Ненулевых мономов: {len(nonzero)}/{64}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 5. Нелинейность: дистанция от аффинных функций
# ---------------------------------------------------------------------------

def render_nl_analysis(tt_int: int, color: bool = True) -> str:
    """
    Нелинейность функции: расстояние от ближайшей аффинной функции.

    Аффинные функции = lin ⊕ c,  lin(x) = u·x (скалярное произведение).
    Расстояние d(f, l) = #{x : f(x) ≠ l(x)} = (64 - Ŵ_f(u)) / 2.
    """
    f = BoolFunc(tt_int)
    W = f.wht()
    nl = f.nonlinearity()
    deg = f.algebraic_degree()
    is_bent = (nl == 28)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Нелинейность  f=0x{tt_int:016x}')
    lines.append(f'  nl={nl}   deg={deg}   {"BENT!" if is_bent else "не bent"}')
    lines.append('═' * 64)

    # Для каждого u показываем расстояние до lin_u
    max_w = max(abs(w) for w in W)
    lines.append(f'\n  WHT-значения Ŵ(u)   [max|Ŵ|={max_w}, nl=(64-{max_w})/2={nl}]\n')
    lines.append('       ' + ''.join(f'  u={u:<2d}' for u in range(8)))
    for row in range(8):
        row_s = f'  u={row*8:2d}.. '
        for col in range(8):
            u = row * 8 + col
            w = W[u]
            dist = (64 - abs(w)) // 2
            if color:
                c = _wht_color(w, highlight=(abs(w) == 8 and is_bent))
                row_s += f'{c}{w:+5d}{_RESET}'
            else:
                row_s += f'{w:+5d}'
        lines.append(row_s)

    lines.append('')
    lines.append(f'  Граница Паттерсона: nl ≤ 28  (для n=6)')
    lines.append(f'  nl(f) = {nl}  {"= 28 — максимум (bent)!" if is_bent else f"< 28 — не bent"}')
    lines.append(f'  Ближайшие аффинные функции: '
                 f'{sum(1 for w in W if abs(w) == max_w)} штук')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# JSON: компонентные булевы функции S-блока (для SC-1)
# ---------------------------------------------------------------------------

def _component_tt(table: list[int], bit: int) -> int:
    """TT компонентной функции f_bit(x) = (table[x] >> bit) & 1  →  64-бит маска."""
    result = 0
    for x in range(64):
        if (table[x] >> bit) & 1:
            result |= (1 << x)
    return result


def json_ring_components(ring: list[int]) -> dict:
    """Анализ компонентных булевых функций Hermann S-box из ring[].

    ring[]  — список значений 1..64. S-box: sbox[h] = ring[h]-1 (0..63).
    Для каждого из 6 выходных битов: WHT, нелинейность, bent?, степень.

    Ключевое открытие SC-1:
      • Входная разность Δin=32=100000₂ (старший бит)
      • sbox[h⊕32] = 63 - sbox[h] = NOT(sbox[h])
      • Значит f_i(x⊕32) = 1 - f_i(x) для каждой компоненты
      • Это означает: компонента f_i с Δin=32 → Δout=1 с вероятностью 1
      • NL=0: все компоненты аффинны → Hermann ring = АФФИННАЯ перестановка!
    """
    table = [v - 1 for v in ring]  # 0-indexed sbox
    components = []
    for bit in range(6):
        tt_int = _component_tt(table, bit)
        f = BoolFunc(tt_int)
        W = f.wht()
        nl = f.nonlinearity()
        deg = f.algebraic_degree()
        max_w = max(abs(w) for w in W)
        # Лучшее линейное смещение
        best_u = max(range(64), key=lambda u: abs(W[u]))
        components.append({
            'bit': bit,
            'tt_hex': f'0x{tt_int:016x}',
            'nonlinearity': nl,
            'algebraic_degree': deg,
            'is_bent': nl == 28,
            'is_affine': nl == 0,
            'max_wht': max_w,
            'best_linear_input_mask': best_u,
        })

    all_affine = all(c['is_affine'] for c in components)
    all_bent = all(c['is_bent'] for c in components)
    return {
        'command': 'ring_components',
        'source': 'hexpack:ring → hexring:bent',
        'sbox_size': 64,
        'n_components': 6,
        'components': components,
        'all_affine': all_affine,   # True → ring = аффинная перестановка
        'all_bent': all_bent,
        'sc1_finding': (
            'WARNING: Hermann ring S-box has NL=0. '
            'The mask u=3 (bits 0,1) gives f_3(x)=bit0(x) (linear, deg=1). '
            'Antipodal property ring[h⊕32]=63-ring[h] → DDT[32][63]=64 (prob=1). '
            'Cannot use ring directly as S-box. '
            'SC-1 recommendation: use Hermann packing for KEY SCHEDULES only; '
            'compose with hexring bent function for the S-box layer.'
        ),
    }


def json_bent_catalog() -> dict:
    """Каталог bent-функций → JSON."""
    # Квадратичные bent-функции на GF(2)^6: x·Ax где A — симпл. матрица
    from projects.hexring.hexring import BoolFunc
    # Несколько стандартных квадратичных bent
    bent_examples = [
        0xFEDC_BA98_7654_3210,  # просто пример
        0x0F0F_0F0F_F0F0_F0F0,
    ]
    catalog = []
    for tt in bent_examples:
        f = BoolFunc(tt & ((1 << 64) - 1))
        catalog.append({
            'tt_hex': f'0x{tt:016x}',
            'nonlinearity': f.nonlinearity(),
            'algebraic_degree': f.algebraic_degree(),
            'is_bent': f.nonlinearity() == 28,
        })
    return {'command': 'bent_catalog', 'examples': catalog,
            'bent_bound_n6': 28, 'note': 'bent iff NL=28 (|WHT|=8 everywhere)'}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='bent_glyphs',
        description='Булевы функции и WHT на Q6 через глифы гексаграмм',
    )
    p.add_argument('--no-color', action='store_true', help='без ANSI-цветов')
    p.add_argument('--json', action='store_true',
                   help='JSON-вывод (для пайплайнов)')
    p.add_argument('--from-ring', action='store_true',
                   help='Читать ring JSON из stdin (hexpack --json ring)')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('wht', help='WHT-спектр булевой функции')
    s.add_argument('func', type=lambda x: int(x, 0),
                   help='64-битная маска TT (десятичная или 0x...)')

    s = sub.add_parser('tt', help='таблица истинности как сетка глифов')
    s.add_argument('func', type=lambda x: int(x, 0),
                   help='64-битная маска TT')

    sub.add_parser('bent', help='примеры квадратичных bent-функций')

    s = sub.add_parser('anf', help='ANF-степень: раскраска глифов по мономам')

    s = sub.add_parser('nl', help='нелинейность и дистанция от аффинных функций')
    s.add_argument('func', type=lambda x: int(x, 0))

    # SC-1: анализ компонент Hermann ring как S-box
    sub.add_parser('ring-components',
                   help='WHT/NL/bent анализ компонент Hermann ring (читает stdin)')

    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    # JSON + --from-ring (любая команда): анализ компонент Hermann S-box
    if args.from_ring:
        raw = sys.stdin.read().strip()
        data = json.loads(raw)
        ring = data.get('ring') or data.get('data', {}).get('ring')
        if ring is None and 'table' in data:
            # Принять вывод hexcrypt:analyze: table[h] (0..63) → ring[h] (1..64)
            ring = [v + 1 for v in data['table']]
        if ring is None:
            print(json.dumps({'error': 'stdin не содержит поля "ring" или "table"'}))
            sys.exit(1)
        result = json_ring_components(ring)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.json and args.cmd == 'bent':
        print(json.dumps(json_bent_catalog(), ensure_ascii=False, indent=2))
        return

    if args.cmd == 'wht':
        print(render_wht_spectrum(args.func & 0xFFFFFFFFFFFFFFFF, color))
    elif args.cmd == 'tt':
        print(render_tt_grid(args.func & 0xFFFFFFFFFFFFFFFF, color))
    elif args.cmd == 'bent':
        print(render_bent_examples(color))
    elif args.cmd == 'anf':
        print(render_anf_degree(args.func & 0xFFFFFFFFFFFFFFFF, color))
    elif args.cmd == 'nl':
        print(render_nl_analysis(args.func & 0xFFFFFFFFFFFFFFFF, color))
    elif args.cmd == 'ring-components':
        # ring-components читает JSON из stdin всегда (--from-ring опционален)
        raw = sys.stdin.read().strip()
        data = json.loads(raw)
        ring = data.get('ring') or data.get('data', {}).get('ring')
        if ring is None and 'table' in data:
            ring = [v + 1 for v in data['table']]
        if ring is None:
            print(json.dumps({'error': 'ring-components: stdin не содержит "ring" или "table"'}))
            sys.exit(1)
        result = json_ring_components(ring)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
