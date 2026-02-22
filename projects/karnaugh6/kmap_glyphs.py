"""kmap_glyphs — Карта Карно 6 переменных с визуализацией через глифы.

Карта Карно для 6 переменных — это ровно та же 8×8 сетка в коде Грея,
что и render_glyph_grid из hexvis. Каждая ячейка — глиф, показывающий
значение данного минтерма.

Ключевые наблюдения:
  • Глиф в ячейке = визуальное представление номера минтерма
  • Ян-счёт глифа = вес минтерма (число единичных переменных)
  • Прямоугольники минтермов = подкубы Q6 = паттерны схожих глифов
  • Простая импликанта = группа глифов с одинаковым «базовым» рисунком

Цвет ячеек:
  • Зелёный фон — минтерм (f=1)
  • Оранжевый фон — безразличный (don't care)
  • По умолчанию цвет по ян-счёту
"""

from __future__ import annotations
import json
import sys
from itertools import combinations

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, to_bits, SIZE
from projects.karnaugh6.minimize import (
    minimize, quine_mccluskey, Implicant,
    _GRAY3, _GRAY3_LABELS, _cell_index,
)
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ANSI цвета для статуса ячейки
_GREEN_BG   = '\033[42m'
_ORANGE_BG  = '\033[48;5;208m'
_BLUE_BG    = '\033[44m'
_WHITE_FG   = '\033[97m'
_DIM        = '\033[2m'


# ---------------------------------------------------------------------------
# Карта Карно с глифами
# ---------------------------------------------------------------------------

def render_kmap_glyphs(
    minterms: list[int],
    dont_cares: list[int] | None = None,
    essential: list[Implicant] | None = None,
    color: bool = True,
    title: str = '',
) -> str:
    """
    8×8 карта Карно, где каждая ячейка — 3-строчный глиф Q6.

    Раскраска:
      Зелёный фон   — минтерм (f=1)
      Оранжевый фон — безразличный (don't care, f=-)
      Тусклый цвет  — ноль (f=0)
    """
    if dont_cares is None:
        dont_cares = []

    m_set  = set(minterms)
    dc_set = set(dont_cares)

    lines: list[str] = []
    if title:
        lines.append(f'  {title}')
    lines.append(f'  f(x5,x4,x3,x2,x1,x0): '
                 f'{len(m_set)} минтерм(ов), {len(dc_set)} безразличных')
    lines.append(f'  Минтермы: {sorted(m_set)}')
    lines.append('')

    # Заголовок столбцов (x2x1x0)
    col_hdr = '         ' + '  '.join(f'{lb}' for lb in _GRAY3_LABELS)
    lines.append(col_hdr)
    lines.append('         ' + '─' * (len(_GRAY3) * 5 - 1))

    for row_idx, row_g in enumerate(_GRAY3):
        row_label = _GRAY3_LABELS[row_idx]
        row_data: list[tuple[int, str]] = []  # (h, status)
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            if h in m_set:
                status = '1'
            elif h in dc_set:
                status = '-'
            else:
                status = '0'
            row_data.append((h, status))

        glyphs = [render_glyph(h) for h, _ in row_data]

        for ri in range(3):
            parts: list[str] = []
            for (h, status), g in zip(row_data, glyphs):
                cell = g[ri]
                if color:
                    if status == '1':
                        cell = _GREEN_BG + _WHITE_FG + cell + _RESET
                    elif status == '-':
                        cell = _ORANGE_BG + _WHITE_FG + cell + _RESET
                    else:
                        cell = _DIM + _YANG_ANSI[yang_count(h)] + cell + _RESET
                elif status == '1':
                    cell = '[' + cell[1] + ']'
                elif status == '-':
                    cell = '(' + cell[1] + ')'
                parts.append(cell)
            prefix = f'  {row_label} │ ' if ri == 1 else '         │ '
            lines.append(prefix + '  '.join(parts))

        # Строка с номерами минтермов (только для 1 и -)
        num_parts = []
        for h, status in row_data:
            if status == '1':
                num_parts.append(f'{h:>2d}▶')
            elif status == '-':
                num_parts.append(f'{h:>2d}?')
            else:
                num_parts.append(f'   ')
        lines.append('         │ ' + ' '.join(num_parts))
        lines.append('         │')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Импликанты как раскрашенные регионы
# ---------------------------------------------------------------------------

def render_implicant_glyph(imp: Implicant, color: bool = True) -> str:
    """Импликанта как строка с глифами минтермов и её булевым выражением."""
    elems = sorted(imp.covered)
    glyphs = [render_glyph(h) for h in elems]
    expr = imp.to_expr()
    label = f'  ({expr})  [{len(elems)} минт.]  '
    pad = ' ' * len(label)
    lines: list[str] = []
    for ri in range(3):
        parts: list[str] = []
        for h, g in zip(elems, glyphs):
            cell = g[ri]
            if color:
                cell = _GREEN_BG + _WHITE_FG + cell + _RESET
            parts.append(cell)
        prefix = label if ri == 1 else pad
        lines.append(prefix + ' '.join(parts))
    return '\n'.join(lines)


def render_minimization(
    minterms: list[int],
    dont_cares: list[int] | None = None,
    color: bool = True,
) -> str:
    """Полный вывод: карта + минимизация + существенные импликанты."""
    if dont_cares is None:
        dont_cares = []

    result = minimize(minterms, dont_cares)
    pis = result['prime_implicants']
    ess = result['essential']
    expr = result['expression']

    lines: list[str] = []

    # 1. Карта Карно с глифами
    lines.append(render_kmap_glyphs(
        minterms, dont_cares, essential=ess, color=color,
        title='Карта Карно (зелёный = 1, оранж = don\'t care)',
    ))
    lines.append('')

    # 2. Результат минимизации
    lines.append('═' * 60)
    lines.append(f'  Минимальная ДНФ:')
    lines.append(f'  f = {expr}')
    lines.append(f'  Простых импликант: {len(pis)}')
    lines.append(f'  Существенных: {len(ess)}')
    lines.append('═' * 60)

    if ess:
        lines.append('\n  Существенные импликанты (глифы минтермов каждой):')
        for imp in ess:
            lines.append(render_implicant_glyph(imp, color=color))
            lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Тематические примеры
# ---------------------------------------------------------------------------

def example_yang_parity(color: bool = True) -> str:
    """
    Пример: f=1 iff ян-счёт чётный (чётное число единичных бит).
    Минтермы: все элементы с yang_count ∈ {0, 2, 4, 6}.
    """
    minterms = [h for h in range(64) if yang_count(h) % 2 == 0]
    lines = [
        '  Пример: f = 1 ⟺ чётное число единичных битов (yang ∈ {0,2,4,6})',
        render_kmap_glyphs(minterms, color=color),
        '',
        f'  Это линейная функция (чётность): f = x0 ⊕ x1 ⊕ x2 ⊕ x3 ⊕ x4 ⊕ x5',
    ]
    return '\n'.join(lines)


def example_rank_3(color: bool = True) -> str:
    """Пример: f=1 iff yang_count=3 (средний ранг решётки B₆, 20 элементов)."""
    minterms = [h for h in range(64) if yang_count(h) == 3]
    lines = [
        '  Пример: f = 1 ⟺ yang_count = 3  (20 минтермов, ранг 3 в B₆)',
        render_kmap_glyphs(minterms, color=color),
        '',
        f'  Минтермы образуют центральный слой решётки Буля B₆',
        f'  На карте Карно это 20 рассеянных глифов с весом 3',
    ]
    return '\n'.join(lines)


def example_triangle(color: bool = True) -> str:
    """Пример: f=1 iff глиф содержит треугольник K₃ (4 таких глифа)."""
    # Треугольники в K₄: {TL,TR,BL}={0,1,2}, {TL,TR,BR}={0,1,3},
    # {TL,BL,BR}={0,2,3}, {TR,BL,BR}={1,2,3}
    # Рёбра треугольника {0,1,2}: bit0=(TL,TR), bit2=(TL,BL), bit5=(TR,BL)
    tri_edges = [
        {0, 2, 5},   # треугольник TL-TR-BL  (биты 0,2,5)
        {0, 3, 4},   # треугольник TL-TR-BR  (биты 0,3,4) ← wait
        {1, 2, 4},   # треугольник TL-BL-BR  (биты 1,2,4)
        {1, 3, 5},   # треугольник TR-BL-BR  (биты 1,3,5)
    ]
    # bit4=(TL,BR), bit3=(TR,BR)
    # Triangle TL-TR-BR: edges TL-TR(0), TL-BR(4), TR-BR(3)
    tri_edges = [
        frozenset([0, 2, 5]),   # TL-TR(0), TL-BL(2), TR-BL(5)
        frozenset([0, 3, 4]),   # TL-TR(0), TR-BR(3), TL-BR(4)
        frozenset([1, 2, 4]),   # BL-BR(1), TL-BL(2), TL-BR(4)
        frozenset([1, 3, 5]),   # BL-BR(1), TR-BR(3), TR-BL(5)
    ]
    minterms = [h for h in range(64)
                if any(edges.issubset({b for b in range(6) if (h >> b) & 1})
                       for edges in tri_edges)]
    lines = [
        f'  Пример: f = 1 ⟺ глиф содержит треугольник K₃  ({len(minterms)} минтермов)',
        render_kmap_glyphs(minterms, color=color),
        '',
        f'  Каждый треугольник — подграф K₃ на 3 из 4 вершин квадрата',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# JSON-экспорт (для пайплайнов TSC-1)
# ---------------------------------------------------------------------------

def json_sbox_minimize(sbox_data: dict) -> dict:
    """
    Минимизировать все 6 компонентных функций S-блока.
    Также показывает линейную маску u, объясняющую NL=0.

    Входной формат: вывод hexcrypt:analyze или hexpack:ring (поля 'table' или 'ring').
    """
    # Принять как table (sbox 0-63) или ring (1-64)
    if 'table' in sbox_data:
        table = sbox_data['table']
    elif 'ring' in sbox_data:
        table = [v - 1 for v in sbox_data['ring']]
    elif 'data' in sbox_data and 'table' in sbox_data['data']:
        table = sbox_data['data']['table']
    else:
        return {'error': 'не найдены поля "table" или "ring" во входных данных'}

    components = []
    total_literals = 0

    for bit in range(6):
        minterms = [h for h in range(64) if (table[h] >> bit) & 1]
        result = minimize(minterms)

        ess = result['essential']
        n_imps = len(ess)
        n_lits = sum(imp.bits.count('0') + imp.bits.count('1') for imp in ess)
        total_literals += n_lits

        # Линейная ли функция? Признак: 1 существенная импликанта и 32 минтерма
        is_linear_est = (n_imps == 1 and len(minterms) == 32)

        components.append({
            'bit': bit,
            'n_minterms': len(minterms),
            'n_essential_implicants': n_imps,
            'total_literals': n_lits,
            'expression': result['expression'],
            'is_linear_est': is_linear_est,
            'implicants': [
                {
                    'pattern': imp.bits,
                    'literals': imp.bits.count('0') + imp.bits.count('1'),
                    'cube_size': imp.size(),
                    'covered': len(imp.covered),
                }
                for imp in ess
            ],
        })

    # Специальный анализ: линейная маска u=3 (биты 0+1)
    # f₀⊕f₁(h) должна быть линейной (= input bit 0) для Hermann ring
    combined_minterms = [h for h in range(64)
                         if ((table[h] & 1) ^ ((table[h] >> 1) & 1))]
    combined_result = minimize(combined_minterms)
    combined_ess = combined_result['essential']
    n_combined_lits = sum(
        imp.bits.count('0') + imp.bits.count('1') for imp in combined_ess
    )
    # Проверить: совпадают ли мнтермы с {h : bit0(h)=1} = нечётные числа
    odd_numbers = set(range(1, 64, 2))
    mask3_is_bit0 = set(combined_minterms) == odd_numbers

    n_linear = sum(1 for c in components if c['is_linear_est'])

    return {
        'command': 'sbox_minimize',
        'n_bits': 6,
        'total_literals': total_literals,
        'n_linear_components': n_linear,
        'components': components,
        'linear_mask_u3': {
            'description': 'f₀(x) XOR f₁(x) — комбинированная компонента с маской u=3',
            'n_minterms': len(combined_minterms),
            'n_essential_implicants': len(combined_ess),
            'total_literals': n_combined_lits,
            'expression': combined_result['expression'],
            'mask3_equals_bit0_input': mask3_is_bit0,
            'nl0_explanation': (
                'f₀⊕f₁(h) = bit0(h) (вход, не выход) → линейна → NL=0 для маски u=3'
                if mask3_is_bit0 else
                'f₀⊕f₁ ≠ bit0(h): другое правило, NL > 0 для маски u=3'
            ),
        },
        'table': table,  # передать дальше по пайплайну
    }


_KMAP_JSON_DISPATCH = {
    'sbox-minimize': lambda args, data: json_sbox_minimize(data),
}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='kmap_glyphs — Карта Карно 6 переменных с глифами Q6'
    )
    parser.add_argument('--json', action='store_true',
                        help='Машиночитаемый JSON-вывод (для пайплайнов)')
    parser.add_argument('--from-sbox', action='store_true',
                        help='Читать S-box JSON из stdin (hexcrypt:analyze или hexpack:ring)')
    sub = parser.add_subparsers(dest='cmd')

    # sbox-minimize — TSC-1 шаг 3: минимизация компонент S-блока
    sub.add_parser('sbox-minimize',
                   help='Минимизировать компонентные функции S-блока → JSON')

    p_min = sub.add_parser('minimize', help='Минимизировать функцию (минтермы)')
    p_min.add_argument('minterms', nargs='+', type=int)
    p_min.add_argument('--dc', nargs='*', type=int, default=[], metavar='DC',
                       help='Безразличные минтермы')

    p_map = sub.add_parser('map', help='Показать карту Карно с глифами')
    p_map.add_argument('minterms', nargs='+', type=int)
    p_map.add_argument('--dc', nargs='*', type=int, default=[], metavar='DC')

    sub.add_parser('yang_parity', help='Пример: чётность ян-счёта')
    sub.add_parser('rank3',       help='Пример: ян-счёт = 3')
    sub.add_parser('triangle',    help='Пример: содержит треугольник K₃')

    for p in sub.choices.values():
        p.add_argument('--no-color', action='store_true')

    args = parser.parse_args()
    color = not getattr(args, 'no_color', False)

    if args.cmd == 'sbox-minimize':
        if args.from_sbox:
            raw = sys.stdin.read().strip()
            sbox_data = json.loads(raw)
        else:
            # Demo: use Hermann ring by default
            import subprocess
            result_proc = subprocess.run(
                [sys.executable, '-m', 'projects.hexpack.pack_glyphs', '--json', 'ring'],
                capture_output=True, text=True,
            )
            sbox_data = json.loads(result_proc.stdout)
        result = json_sbox_minimize(sbox_data)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f'  S-блок: минимизация 6 компонентных функций')
            print()
            for c in result['components']:
                lin = '✓ линейна' if c['is_linear_est'] else f'{c["n_essential_implicants"]} импликант'
                print(f'  Бит {c["bit"]}: мнтермов={c["n_minterms"]}  '
                      f'лит={c["total_literals"]}  {lin}')
                print(f'    {c["expression"][:72]}')
            m3 = result['linear_mask_u3']
            print()
            print(f'  Маска u=3 (f₀⊕f₁): лит={m3["total_literals"]}  '
                  f'{"= bit0(h) → NL=0 !" if m3["mask3_equals_bit0_input"] else ""}')
            print(f'  {m3["expression"]}')
        sys.exit(0)

    if args.cmd == 'minimize':
        print(render_minimization(args.minterms, args.dc, color=color))

    elif args.cmd == 'map':
        print(render_kmap_glyphs(args.minterms, args.dc, color=color))

    elif args.cmd == 'yang_parity':
        print(example_yang_parity(color=color))

    elif args.cmd == 'rank3':
        print(example_rank_3(color=color))

    elif args.cmd == 'triangle':
        print(example_triangle(color=color))

    else:
        # Default: показать все три примера
        print('  Карта Карно 6 переменных с глифами Q6')
        print('  ─' * 35)
        print()
        print(example_yang_parity(color=color))
        print()
        print(example_rank_3(color=color))
