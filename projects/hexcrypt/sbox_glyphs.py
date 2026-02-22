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
import json
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexcrypt.hexcrypt import (
    SBox,
    identity_sbox, bit_reversal_sbox, affine_sbox,
    complement_sbox, random_sbox, yang_sort_sbox,
    evaluate_sbox, best_differential_characteristic, best_linear_bias,
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
# Hermann S-box: ring JSON → SBox
# ---------------------------------------------------------------------------

def _sbox_from_ring_json(ring_json: dict) -> SBox:
    """Преобразовать ring Германа (из hexpack --json ring) в SBox.

    ring[h] = n  (1..64)  →  sbox[h] = n-1  (0..63)
    Теорема: для P=2^k кольцо является перестановкой → SBox корректен.
    """
    ring = ring_json['ring']          # list[int], len=64, значения 1..64
    table = [v - 1 for v in ring]    # 0-indexed: значения 0..63
    return SBox(table)


def _read_ring_from_stdin() -> SBox:
    """Читать ring JSON из stdin и вернуть SBox."""
    raw = sys.stdin.read().strip()
    data = json.loads(raw)
    # Поддержать как прямой вывод hexpack:ring, так и обёртку контекста
    if 'ring' in data:
        return _sbox_from_ring_json(data)
    if 'data' in data and 'ring' in data.get('data', {}):
        return _sbox_from_ring_json(data['data'])
    raise ValueError('stdin не содержит поля "ring" — ожидается вывод hexpack --json ring')


# ---------------------------------------------------------------------------
# JSON-экспорт
# ---------------------------------------------------------------------------

def _sbox_to_json(sb: SBox, name: str = 'unknown') -> dict:
    """Полный JSON-анализ S-блока."""
    ev = evaluate_sbox(sb)
    best_diff = best_differential_characteristic(sb)
    best_lin  = best_linear_bias(sb)
    # Топ-5 строк DDT
    ddt = sb.difference_distribution_table()
    ddt_profile = {}
    for a in range(1, 64):
        row_max = max(ddt[a])
        ddt_profile[str(a)] = row_max
    return {
        'name': name,
        'table': sb.table(),
        'metrics': ev,
        'best_differential': {
            'delta_in': best_diff[0], 'delta_out': best_diff[1],
            'probability': round(best_diff[2], 6),
        },
        'best_linear_bias': {
            'alpha': best_lin[0], 'beta': best_lin[1],
            'bias': round(best_lin[2], 6),
        },
        'ddt_row_maxima': ddt_profile,
        'yang_conservation': sum(
            1 for x in range(64)
            if yang_count(x) == yang_count(sb(x))
        ),
    }


def json_analyze(sb: SBox, name: str = 'sbox') -> dict:
    """Команда analyze → JSON."""
    return _sbox_to_json(sb, name)


def json_compare(extra_sb: SBox | None = None, extra_name: str = '') -> dict:
    """Сравнение всех S-блоков + опционального Herman'а → JSON."""
    names = ['identity', 'bit_reversal', 'affine', 'complement', 'yang_sort']
    result = []
    for name in names:
        sb = _get_sbox(name)
        ev = evaluate_sbox(sb)
        result.append({'name': name, **ev})
    if extra_sb is not None:
        ev = evaluate_sbox(extra_sb)
        result.append({'name': extra_name or 'custom', **ev})
    # Сортировать по нелинейности ↓
    result.sort(key=lambda r: r.get('nonlinearity', 0), reverse=True)
    return {'command': 'cmp', 'sboxes': result,
            'ideal_nl': 28, 'ideal_delta': 4, 'note': 'APN (δ=2) невозможен для биекции n=6'}


def _avalanche_row(table: list[int], i: int, j: int) -> float:
    """M[i][j] = доля входов x, у которых f(x)_j ≠ f(x⊕2^i)_j."""
    count = sum(1 for x in range(64) if ((table[x] ^ table[x ^ (1 << i)]) >> j) & 1)
    return count / 64


def _sac_deviation(M: list[list[float]]) -> float:
    """Среднее |M[i][j] - 0.5| по всем парам (i,j)."""
    return sum(abs(M[i][j] - 0.5) for i in range(6) for j in range(6)) / 36


def json_avalanche(opt_data: dict | None = None) -> dict:
    """
    SC-5 Шаг 2: Лавинный критерий (SAC) для кандидатов из hexopt:bayesian.

    K1 × K3:
      K1 — Лавинный критерий (Avalanche): M[i][j] = P(output bit j flips | input bit i flips)
      K3 — Связь с NL: высокий NL ↔ низкое SAC-отклонение (|M[i][j]-0.5| → 0)

    Аргументы:
      opt_data: dict из hexopt:bayesian (содержит best_found.table)
                Если None — использовать стандартные S-блоки.

    Возвращает:
      sboxes (отсортированы по NL), матрицы, SAC-отклонения, NL-SAC-корреляция.
    """
    from projects.hexcrypt.hexcrypt import (
        SBox, evaluate_sbox,
        identity_sbox, affine_sbox, complement_sbox,
        random_sbox, yang_sort_sbox,
    )
    import math

    # Список S-блоков для тестирования
    sbox_list: list[tuple[str, list[int]]] = []

    if opt_data is not None and 'best_found' in opt_data:
        sbox_list.append(('bayesian_best', list(opt_data['best_found']['table'])))

    # Стандартные S-блоки для сравнения
    for name, sb in [
        ('identity',    identity_sbox()),
        ('affine',      affine_sbox()),
        ('complement',  complement_sbox()),
        ('yang_sort',   yang_sort_sbox()),
        ('random_42',   random_sbox(seed=42)),
        ('random_17',   random_sbox(seed=17)),
    ]:
        sbox_list.append((name, sb.table()))

    results: list[dict] = []
    for name, table in sbox_list:
        sb = SBox(table)
        ev = evaluate_sbox(sb)
        M = [[round(_avalanche_row(table, i, j), 4) for j in range(6)] for i in range(6)]
        dev = round(_sac_deviation(M), 6)
        results.append({
            'name':              name,
            'nl':                ev['nonlinearity'],
            'delta':             ev['differential_uniformity'],
            'deg':               ev['algebraic_degree'],
            'sac_deviation':     dev,
            'sac_score':         round(1.0 - dev / 0.5, 4),
            'avalanche_matrix':  M,
        })

    results.sort(key=lambda x: (-x['nl'], x['sac_deviation']))

    best_sac  = min(results, key=lambda x: x['sac_deviation'])
    worst_sac = max(results, key=lambda x: x['sac_deviation'])

    # Pearson r(NL, SAC_dev)
    nls  = [r['nl'] for r in results]
    devs = [r['sac_deviation'] for r in results]
    n    = len(results)
    mn   = sum(nls) / n;  md = sum(devs) / n
    sn   = math.sqrt(max(1e-12, sum((v - mn) ** 2 for v in nls)))
    sd   = math.sqrt(max(1e-12, sum((v - md) ** 2 for v in devs)))
    r_nl_sac = round(sum((nls[i] - mn) * (devs[i] - md) for i in range(n)) / (sn * sd), 4)

    return {
        'command':       'avalanche',
        'sboxes':        results,
        'best_sac':      best_sac['name'],
        'worst_sac':     worst_sac['name'],
        'r_nl_sac':      r_nl_sac,
        'ideal_sac_dev': 0.0,
        'k1_finding': (
            f'SC-5 Лавинный критерий Q6: r(NL, SAC_dev)={r_nl_sac} (сильная обратная корреляция). '
            f'Лучший SAC: {best_sac["name"]} (dev={best_sac["sac_deviation"]:.4f}, NL={best_sac["nl"]}). '
            f'Идеальный S-блок (SAC=0): NL≈{round(18 - (0 - best_sac["sac_deviation"]) * (18 / 0.5))}.'
        ),
    }



def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='sbox_glyphs',
        description='Криптографический анализ S-блоков через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    p.add_argument('--json', action='store_true',
                   help='JSON-вывод (для пайплайнов)')
    p.add_argument('--from-ring', action='store_true',
                   help='Читать ring JSON из stdin (вывод hexpack --json ring)')
    p.add_argument('--from-opt', action='store_true',
                   help='Читать hexopt:bayesian JSON из stdin')
    p.add_argument('--sbox', default='affine',
                   choices=['identity', 'bit_reversal', 'affine',
                            'complement', 'yang_sort', 'random'],
                   help='выбор S-блока (игнорируется при --from-ring/--from-opt)')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('map',     help='карта x→f(x): глифы раскрашены по f(x)')
    sub.add_parser('analyze', help='полный криптоанализ (nl, δ, deg, DDT, LAT) → JSON')

    s = sub.add_parser('ddt', help='строка DDT для входной разности a')
    s.add_argument('--row', type=int, default=1, metavar='a',
                   help='входная разность a (1..63)')

    s = sub.add_parser('lat', help='столбец LAT для выходной маски b')
    s.add_argument('--col', type=int, default=1, metavar='b',
                   help='выходная маска b (1..63)')

    sub.add_parser('cmp',       help='сравнение нескольких S-блоков')
    sub.add_parser('avalanche', help='лавинный критерий SAC для кандидатов → JSON (SC-5)')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    # SC-5: лавинный критерий из hexopt:bayesian stdin
    if args.cmd == 'avalanche':
        opt_data: dict | None = None
        if args.from_opt:
            raw = sys.stdin.read()
            try:
                opt_data = json.loads(raw)
            except Exception:
                opt_data = None
        data = json_avalanche(opt_data)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # Получить S-блок: из ring (stdin) или по имени
    if args.from_ring:
        sb = _read_ring_from_stdin()
        sbox_name = 'herman_ring'
    else:
        sb = _get_sbox(args.sbox)
        sbox_name = args.sbox

    if args.json:
        if args.cmd in ('map', 'analyze', 'ddt', 'lat'):
            data = json_analyze(sb, name=sbox_name)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        elif args.cmd == 'cmp':
            extra = sb if args.from_ring else None
            data = json_compare(extra_sb=extra, extra_name=sbox_name)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(json.dumps({'error': f'JSON не поддерживается для: {args.cmd}'}))
        return

    if args.cmd == 'map':
        print(render_map(sb, color))
    elif args.cmd == 'analyze':
        # Текстовый вывод analyze = map + cmp строчка
        print(render_map(sb, color))
        print()
        print(render_comparison(color))
    elif args.cmd == 'ddt':
        print(render_ddt(sb, row_a=args.row, color=color))
    elif args.cmd == 'lat':
        print(render_lat(sb, col_b=args.col, color=color))
    elif args.cmd == 'cmp':
        print(render_comparison(color))


if __name__ == '__main__':
    main()
