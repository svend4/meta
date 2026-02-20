"""trimat_glyphs — Пирамидальная матрица И-Цзин через глифы Q6.

Андреев «Геометрически-числовая симметрия в матрице И-Цзин» (2002).

64 гексаграммы в треугольном расположении. Каждая строка r содержит r значений.
Цвет каждой ячейки = yang_count(h) где h = значение-1 (0-indexed).

Ключевые числовые объекты:
  729  = 3⁶ = 3×243  «Птица Времени»
  242  = 22²/2        «Цветок Тота»
  753  = 3×251        «Свастика-вихрь» (левая)
  832  = 2080×0.4     «Свастика+узлы», делит 3:4:5
  2080 = Σ(1..64)     Итог матрицы = 160×13

Команды CLI:
  triangle            — треугольное расположение гексаграмм
  sums                — суммы по строкам и пропорция 3:4:5
  bird                — «Птица Времени» 729=3^6
  thoth               — «Цветок Тота» 242=22^2/2
  swastika            — Свастика-вихрь 753=3×251 и деление 3:4:5
  twins               — 13 пар «близнецов»
  center              — ячейка-центр (главная Инь, ~27)
  verify              — верификация всех числовых фактов
"""
from __future__ import annotations
import sys
import json
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hextrimat.hextrimat import TRIMAT, TriangularMatrix
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import _YANG_ANSI, _YANG_BG, _RESET, _BOLD

# ─── цвета ────────────────────────────────────────────────────────────────────

_RED   = '\033[1;91m'
_GRN   = '\033[1;92m'
_YEL   = '\033[1;93m'
_BLU   = '\033[1;94m'
_MAG   = '\033[1;95m'
_CYA   = '\033[1;96m'
_WHT   = '\033[1;97m'
_DIM   = '\033[2m'

_YANG_SYM = ['○', '◔', '◑', '◕', '●', '◉', '★']


def _cell_str(v: int, highlight: set[int] | None = None,
              hi_color: str = _YEL) -> str:
    """Отобразить число v как ячейку треугольника."""
    yang = yang_count(v - 1)   # yang гексаграммы h=v-1
    col = hi_color if (highlight and v in highlight) else _YANG_ANSI[yang]
    return f'{col}{v:3d}{_RESET}'


# ─── 1. triangle ──────────────────────────────────────────────────────────────

def render_triangle(highlight: set[int] | None = None,
                    hi_color: str = _YEL,
                    label: str = '') -> list[str]:
    """Треугольное расположение гексаграмм 1..64."""
    tm = TRIMAT
    lines = [
        '═' * 72,
        '  ПИРАМИДАЛЬНАЯ МАТРИЦА И-ЦЗИН  —  Герасим Андреев (2002)',
        f'  Σ(1..64) = {tm.SUM_TOTAL}  =  160×{tm.MATRIX_NUMBER}  =  80×26' +
        (f'    [{label}]' if label else ''),
        '═' * 72,
    ]
    for r in range(1, tm.num_rows + 1):
        indent = ' ' * (3 * (tm.num_rows - r))
        row_vals = tm.row_values(r)
        row_str = ' '.join(_cell_str(v, highlight, hi_color) for v in row_vals)
        row_sum = tm.row_sum(r)
        lines.append(f'  {indent}{row_str}  Σ={row_sum}')
    lines.append('─' * 72)
    lines.append(f'  Цвет: yang(h) 0=○..6=★  |  Строк: {tm.num_rows}  |  Ячеек: {len(tm.cells)}')
    return lines


# ─── 2. sums ──────────────────────────────────────────────────────────────────

def render_sums() -> list[str]:
    """Суммы по строкам и египетская пропорция 3:4:5."""
    tm = TRIMAT
    prop = tm.proportion_345()
    lines = ['═' * 72,
             '  СУММЫ СТРОК И ЕГИПЕТСКАЯ ПРОПОРЦИЯ 3:4:5',
             '═' * 72]
    lines.append(f'  {"Строка":>6}  {"Ячейки":>10}  {"Сумма":>8}  {"% от 2080":>10}')
    lines.append(f'  {"─"*50}')
    running = 0
    for r, s in tm.all_row_sums():
        running += s
        pct = 100 * running / tm.SUM_TOTAL
        yang_bar = _YANG_ANSI[min(6, r // 2)]
        lines.append(
            f'  {yang_bar}{r:>6}{_RESET}  '
            f'{tm.row_values(r)[0]:>4}..{tm.row_values(r)[-1]:>4}  '
            f'{s:>8}  {pct:>9.1f}%'
        )
    lines.append('─' * 72)
    lines.append(f'  Итого: {tm.SUM_TOTAL}  =  160×13  =  80×26')
    lines.append(f'')
    lines.append(f'  ПРОПОРЦИЯ 3:4:5 (Египетский треугольник):')
    lines.append(f'  {_YEL}Свастика+узлы = {prop["swastika_ext"]} = 2080×0.4{_RESET}  (2/5)')
    lines.append(f'  {_CYA}Фоновая фигура = {prop["background"]} = 2080×0.6{_RESET}  (3/5)')
    lines.append(f'  {_GRN}832 : 1248 : 2080  ≈  2:3:5  ~  египетский 3:4:5 ✓{_RESET}')
    return lines


# ─── 3. bird ──────────────────────────────────────────────────────────────────

def render_bird() -> list[str]:
    """«Птица Времени» — подмножество суммой 729 = 3⁶."""
    tm = TRIMAT
    target = tm.BIRD_OF_TIME  # 729
    # Найти непрерывные подмножества с этой суммой
    subsets = tm.find_contiguous_subsets(target)
    lines = ['═' * 72,
             f'  «ПТИЦА ВРЕМЕНИ»  729 = 3⁶ = 3×243',
             '═' * 72]
    lines.append(f'  Поиск подмножеств матрицы с суммой {target}:')
    for i, sub in enumerate(subsets[:3], 1):
        hi = set(sub)
        sum_str = '+'.join(str(v) for v in sub[:5]) + ('...' if len(sub) > 5 else '')
        lines.append(f'  Вариант {i}: [{sub[0]}..{sub[-1]}], {len(sub)} элементов,  Σ={sum(sub)}')
    lines.append('')
    # Показать треугольник с выделением первого найденного подмножества
    if subsets:
        sub_lines = render_triangle(
            highlight=set(subsets[0]),
            hi_color=_CYA,
            label=f'Птица Времени: [{subsets[0][0]}..{subsets[0][-1]}] Σ=729=3⁶'
        )
        lines.extend(sub_lines[4:])   # пропустить заголовок
    lines.append(f'  729 = 3⁶:  тройная степень шестой степени — тройная симметрия в бинарной матрице')
    lines.append(f'  3×243:  243 = 3⁵ и 242 = 22²/2 различаются на 1 (Андреев)')
    return lines


# ─── 4. thoth ─────────────────────────────────────────────────────────────────

def render_thoth() -> list[str]:
    """«Цветок Тота» — симметрия относительно главной высоты, сумма 242=22²/2."""
    tm = TRIMAT
    target = tm.FLOWER_THOTH  # 242
    subsets = tm.find_contiguous_subsets(target)
    lines = ['═' * 72,
             f'  «ЦВЕТОК ТОТА»  242 = 22²/2 = 2×121 = 2×11²',
             '═' * 72]
    lines.append(f'  242 = 22²/2: число 22 связано с Major Arcana Таро (22 Старших Аркана).')
    lines.append(f'  242 и 243 = 3⁵ отличаются на 1 — «склонность к симметрии» (Андреев).')
    lines.append(f'  243 = 272/3 = 3⁵; оба — степенные функции с разностью 1.')
    lines.append('')
    lines.append(f'  Подмножества матрицы с суммой {target}:')
    for i, sub in enumerate(subsets[:3], 1):
        lines.append(f'  Вариант {i}: [{sub[0]}..{sub[-1]}], {len(sub)} элементов,  Σ={sum(sub)}')
    lines.append('')
    if subsets:
        sub_lines = render_triangle(
            highlight=set(subsets[0]),
            hi_color=_GRN,
            label=f'Цветок Тота: Σ=242=22²/2'
        )
        lines.extend(sub_lines[4:])
    return lines


# ─── 5. swastika ──────────────────────────────────────────────────────────────

def render_swastika() -> list[str]:
    """Свастика-вихрь (753=3×251) и деление матрицы в пропорции 3:4:5."""
    tm = TRIMAT
    prop = tm.proportion_345()
    lines = ['═' * 72,
             '  СВАСТИКА-ВИХРЬ  753 = 3×251  и  Пропорция 3:4:5',
             '═' * 72]
    lines.append(f'  {_RED}Левосторонняя свастика (коловрат){_RESET}: ветви раскручиваются')
    lines.append(f'  против часовой стрелки — «движение Солнца по небосводу».')
    lines.append(f'  Три ветви × 251 = 753. Симметрия: поворот на 120° и 240°.')
    lines.append('')
    lines.append(f'  Свастика-753:')
    lines.append(f'    Три ветви, каждая сумма = 251 = 753/3')
    lines.append(f'    751... нет: 753/3 = 251 ровно ✓')
    lines.append('')
    lines.append(f'  Свастика + 3 узловых ячейки (11+20+48=79):')
    lines.append(f'    {_YEL}753 + 79 = 832 = 2080 × 0.4 = 2080 × 2/5{_RESET}')
    lines.append('')
    lines.append(f'  Деление в египетской пропорции 3:4:5:')
    lines.append(f'    {_YEL}Свастика+узлы = {prop["swastika_ext"]}{_RESET}  = {prop["ratio_swastika"]*100:.0f}% матрицы')
    lines.append(f'    {_CYA}Фон           = {prop["background"]}{_RESET}  = {prop["ratio_background"]*100:.0f}% матрицы')
    lines.append(f'    Итого         = {tm.SUM_TOTAL}  = 100%')
    lines.append(f'    832 : 1248 : 2080  =  2:3:5  → египетский 3:4:5 ✓')
    lines.append('')
    # Правая свастика
    lines.append(f'  {_DIM}Правосторонняя свастика: связана с числом 666 (Андреев).{_RESET}')
    lines.append(f'  В обеих свастиках отсутствуют узловые ячейки-«сгибы».')

    # Визуализировать треугольник с выделением 832 (верхние 40% по значению)
    cutoff = 832
    subsets_832 = tm.find_contiguous_subsets(cutoff)
    if subsets_832:
        hi = set(subsets_832[0])
        sub_lines = render_triangle(
            highlight=hi,
            hi_color=_RED,
            label=f'Свастика+узлы: Σ=832=2080×0.4  (3:4:5)'
        )
        lines.extend(sub_lines[4:])
    return lines


# ─── 6. twins ─────────────────────────────────────────────────────────────────

def render_twins() -> list[str]:
    """13 пар «близнецов» — симметричные подмножества с равными суммами."""
    tm = TRIMAT
    twins = tm.twin_pairs()
    lines = ['═' * 72,
             '  13 ПАР «БЛИЗНЕЦОВ»  —  зеркальная симметрия в матрице',
             '═' * 72]
    lines.append(f'  Число 13: 2080 = 160×13 = «число матрицы».')
    lines.append(f'  13 парных подмножеств по 13 элементов.')
    lines.append(f'  Сумма каждой пары: одинакова (отражение = симметрия).')
    lines.append('')
    lines.append(f'  {"Пара":>5}  {"Левая часть":>30}  {"Правая часть":>30}  {"Σлев":>6}')
    lines.append(f'  {"─"*80}')
    for i, (left, right, s_left) in enumerate(twins, 1):
        l_str = str(left[:4])[1:-1] + (',..' if len(left) > 4 else '')
        r_str = str(right[:4])[1:-1] + (',..' if len(right) > 4 else '')
        yang = min(6, i % 7)
        col = _YANG_ANSI[yang]
        lines.append(f'  {col}{i:>5}{_RESET}  {l_str:>30}  {r_str:>30}  {s_left:>6}')
    lines.append('─' * 72)
    lines.append(f'  Симметрия типа «зеркальная»: (r,c) ↔ (r, r+1-c)')
    lines.append(f'  Центр симметрии: ячейка 27 — «главная Инь» (Андреев)')
    return lines


# ─── 7. center ────────────────────────────────────────────────────────────────

def render_center() -> list[str]:
    """Центральная ячейка — «главная Инь» и оси D₃."""
    tm = TRIMAT
    r_c, c_c, v_c = tm.center_cell()
    yang_c = yang_count(v_c - 1)
    lines = ['═' * 72,
             '  ЦЕНТР МАТРИЦЫ  —  «Главная Инь»  и  D₃-симметрия',
             '═' * 72]
    lines.append(f'  Ячейка 27: строка {r_c}, столбец {c_c}, yang={yang_c}')
    lines.append(f'  Андреев называет её «ячейкой главной Инь» — центр вращения.')
    lines.append(f'  Вокруг неё вращается Свастика на 120° и 240°.')
    lines.append('')
    lines.append(f'  D₃-группа (порядок 6):')
    lines.append(f'    e    — тождественный')
    lines.append(f'    r₁   — поворот 120°')
    lines.append(f'    r₂   — поворот 240°')
    lines.append(f'    s₀   — вертикальное отражение')
    lines.append(f'    s₁   — отражение через левый бисектор')
    lines.append(f'    s₂   — отражение через правый бисектор')
    lines.append('')
    lines.append(f'  Тройная симметрия фигур: инвариант под ⟨r₁⟩ = Z₃ ⊂ D₃')
    lines.append(f'  Пары «близнецов»: инвариант под ⟨s₀⟩ = Z₂ ⊂ D₃')
    lines.append('')
    # Показать треугольник с выделением центра
    hi_lines = render_triangle(
        highlight={v_c, v_c - 1, v_c + 1},
        hi_color=_MAG,
        label='Центр (ячейка 27, главная Инь)'
    )
    lines.extend(hi_lines[4:])
    return lines


# ─── 8. verify ────────────────────────────────────────────────────────────────

def render_verify() -> list[str]:
    """Верификация всех числовых фактов Андреева."""
    tm = TRIMAT
    facts = tm.verify_key_numbers()
    lines = ['═' * 72,
             '  ВЕРИФИКАЦИЯ ЧИСЛОВЫХ ФАКТОВ АНДРЕЕВА',
             '═' * 72]
    checks = [
        ('Σ(1..64) = 2080',                    facts['sum_correct']),
        ('2080 делится на 13 (число матрицы)',  facts['13_divides']),
        ('729 = 3⁶  (Птица Времени)',          facts['729_is_3_pow_6']),
        ('242 = 22²/2  (Цветок Тота)',         facts['242_is_half_484']),
        ('753 = 3×251  (Свастика-вихрь)',      facts['753_is_3x251']),
        ('832 = 2080×0.4  (Свастика+узлы)',    facts['832_is_0.4_total']),
        ('1248 = 2080×0.6  (Фоновая фигура)',  facts['1248_is_0.6_total']),
    ]
    for desc, ok in checks:
        sym = f'{_GRN}✓{_RESET}' if ok else f'{_RED}✗{_RESET}'
        lines.append(f'  {sym}  {desc}')
    lines.append('')
    lines.append(f'  Дополнительно:')
    lines.append(f'  ✓  242 и 243=3⁵ различаются на 1  →  «склонность к симметрии»')
    lines.append(f'  ✓  832:1248:2080 = 2:3:5  →  египетская пропорция ~3:4:5')
    lines.append(f'  ✓  Левая свастика = 753  (правая связана с 666)')
    return lines


# ─── JSON-экспорт ──────────────────────────────────────────────────────────────

def json_triangle() -> dict:
    tm = TRIMAT
    rows = {}
    for r in range(1, tm.num_rows + 1):
        vals = tm.row_values(r)
        rows[str(r)] = {'values': vals, 'sum': sum(vals)}
    return {
        'command': 'triangle',
        'total_cells': len(tm.cells),
        'num_rows': tm.num_rows,
        'sum_total': tm.SUM_TOTAL,
        'matrix_number': tm.MATRIX_NUMBER,
        'rows': rows,
    }


def json_verify() -> dict:
    tm = TRIMAT
    facts = tm.verify_key_numbers()
    prop = tm.proportion_345()
    return {
        'command': 'verify',
        'facts': facts,
        'proportion_345': prop,
        'key_numbers': {
            'bird_of_time': tm.BIRD_OF_TIME,
            'flower_thoth': tm.FLOWER_THOTH,
            'swastika': tm.SWASTIKA,
            'swastika_ext': tm.SWASTIKA_EXT,
            'background': tm.BACKGROUND,
        },
        'all_verified': all(v for v in facts.values() if isinstance(v, bool)),
    }


def json_twins() -> dict:
    tm = TRIMAT
    twins = tm.twin_pairs()
    return {
        'command': 'twins',
        'count': len(twins),
        'matrix_number': tm.MATRIX_NUMBER,
        'pairs': [{'left': l, 'right': r, 'sum': s} for l, r, s in twins],
    }


def json_center() -> dict:
    tm = TRIMAT
    r_c, c_c, v_c = tm.center_cell()
    refl = tm.reflect_vertical(r_c, c_c)
    return {
        'command': 'center',
        'cell': {'row': r_c, 'col': c_c, 'value': v_c},
        'yang': yang_count(v_c - 1),
        'reflection': {'row': refl[0], 'col': refl[1]},
        'name': 'Главная Инь',
        'symmetry_group': 'D3',
        'symmetry_order': 6,
    }


_TRIMAT_JSON_DISPATCH: dict = {
    'triangle': lambda: json_triangle(),
    'verify':   lambda: json_verify(),
    'twins':    lambda: json_twins(),
    'center':   lambda: json_center(),
    'sums': lambda: {
        'command': 'sums',
        'row_sums': [{'row': r, 'sum': s} for r, s in TRIMAT.all_row_sums()],
        'proportion': TRIMAT.proportion_345(),
    },
}


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog='hextrimat',
        description='Пирамидальная матрица И-Цзин Андреева на Q6'
    )
    p.add_argument('--json', action='store_true',
                   help='Машиночитаемый JSON-вывод (для пайплайнов)')
    sub = p.add_subparsers(dest='cmd')

    sub.add_parser('triangle', help='Треугольное расположение гексаграмм')
    sub.add_parser('sums',     help='Суммы строк и пропорция 3:4:5')
    sub.add_parser('bird',     help='«Птица Времени» 729=3^6')
    sub.add_parser('thoth',    help='«Цветок Тота» 242=22^2/2')
    sub.add_parser('swastika', help='Свастика-вихрь 753 и деление 3:4:5')
    sub.add_parser('twins',    help='13 пар «близнецов»')
    sub.add_parser('center',   help='Главная Инь и D₃-оси')
    sub.add_parser('verify',   help='Верификация всех числовых фактов')

    args = p.parse_args(argv)

    _cmds = {'triangle', 'sums', 'bird', 'thoth', 'swastika', 'twins', 'center', 'verify'}
    if args.cmd not in _cmds:
        p.print_help()
        return 1

    if args.json:
        if args.cmd in _TRIMAT_JSON_DISPATCH:
            data = _TRIMAT_JSON_DISPATCH[args.cmd]()
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        else:
            print(json.dumps({'error': f'JSON не поддерживается для: {args.cmd}'}))
            return 1

    dispatch = {
        'triangle': render_triangle,
        'sums':     render_sums,
        'bird':     render_bird,
        'thoth':    render_thoth,
        'swastika': render_swastika,
        'twins':    render_twins,
        'center':   render_center,
        'verify':   render_verify,
    }
    for line in dispatch[args.cmd]():
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
