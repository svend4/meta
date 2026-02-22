"""syndrome_glyphs — Синдромное декодирование через глифы Q6.

Каждый глиф (0..63) — кодовое слово, вектор ошибки или синдром.

Двоичный линейный [n,k,d]-код C ⊆ (GF(2))^n:
  • n=6 (длина),  k (размерность),  d (мин. расстояние)
  • Кодовые слова: C = {Gᵀu : u ∈ GF(2)^k}  (2^k слов)
  • Синдром: s(y) = Hy mod 2   (H — проверочная матрица)
  • Смежные классы: Q6 = C ∪ (c₁⊕C) ∪ ... (2^{6-k} классов)
  • Лидер класса: вектор минимального веса в классе

Визуализация:
  cosets   — смежные классы кода: каждый класс своим цветом
  syndrome — синдромная таблица: глиф → синдром/лидер ошибки
  generator — строки порождающей матрицы как глифы
  codes    — сравнение стандартных кодов длины 6

Команды CLI:
  cosets    [--code repetition|parity|shortened_hamming|even_weight|dual_rep]
  syndrome  [--code ...]
  generator [--code ...]
  codes
"""

from __future__ import annotations
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexcode.hexcode import (
    BinaryCode,
    repetition_code, parity_check_code, shortened_hamming_code,
    even_weight_code, dual_repetition_code,
    singleton_bound, hamming_bound, plotkin_bound,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Фабрика кодов
# ---------------------------------------------------------------------------

def _get_code(name: str) -> BinaryCode:
    _map = {
        'repetition':        repetition_code,
        'parity':            parity_check_code,
        'shortened_hamming': shortened_hamming_code,
        'even_weight':       even_weight_code,
        'dual_rep':          dual_repetition_code,
    }
    return _map[name]()


# Набор цветов для смежных классов (по yang уровням + дополнительные)
_COSET_COLORS = [
    '\033[38;5;27m',    # синий
    '\033[38;5;82m',    # зелёный
    '\033[38;5;208m',   # оранжевый
    '\033[38;5;196m',   # красный
    '\033[38;5;201m',   # пурпурный
    '\033[38;5;226m',   # жёлтый
    '\033[38;5;39m',    # голубой
    '\033[38;5;238m',   # серый
]


# ---------------------------------------------------------------------------
# 1. Смежные классы
# ---------------------------------------------------------------------------

def render_cosets(code: BinaryCode, color: bool = True) -> str:
    """
    8×8 сетка: каждый глиф раскрашен по принадлежности смежному классу.

    Кодовые слова — в классе 0 (яркий). Остальные классы — разными цветами.
    """
    cosets_list = code.cosets()
    codewords_set = set(code.codewords())
    info = code.info()

    # Для каждой вершины — номер её класса
    vertex_coset = [0] * 64
    for ci, coset in enumerate(cosets_list):
        for h in coset:
            vertex_coset[h] = ci

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Смежные классы кода C  [{info["n"]},{info["k"]},{info["d"]}]')
    lines.append(f'  2^k={2**info["k"]} кодовых слов   '
                 f'2^{{n-k}}={len(cosets_list)} классов   '
                 f'd={info["d"]}   t={info["t"]}')
    lines.append('  Яркий = кодовое слово,  остальные цвета = смежные классы')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            ci = vertex_coset[h]
            rows3 = render_glyph(h)
            if color:
                is_cw = (h in codewords_set)
                yc = yang_count(h)
                if is_cw:
                    c = _YANG_BG[yc] + _BOLD
                else:
                    c = _COSET_COLORS[ci % len(_COSET_COLORS)]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            ci = vertex_coset[h]
            is_cw = h in codewords_set
            if color:
                c = _YANG_BG[yang_count(h)] + _BOLD if is_cw else _COSET_COLORS[ci % len(_COSET_COLORS)]
                lbl.append(f'{c}C{ci:d}{_RESET}')
            else:
                lbl.append(f'C{ci}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Лидеры классов
    lines.append('  Лидеры смежных классов (минимальный вес):')
    for ci, coset in enumerate(cosets_list[:8]):
        leader = min(coset, key=lambda h: bin(h).count('1'))
        w = bin(leader).count('1')
        if color:
            c = _COSET_COLORS[ci % len(_COSET_COLORS)]
            lines.append(f'  {c}  Класс {ci}: лидер={leader:02d}={format(leader,"06b")}'
                         f'  вес={w}  |класс|={len(coset)}{_RESET}')
        else:
            lines.append(f'    Класс {ci}: лидер={leader:02d}  вес={w}  '
                         f'|класс|={len(coset)}')
    if len(cosets_list) > 8:
        lines.append(f'  ... ещё {len(cosets_list)-8} классов ...')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Синдромная таблица
# ---------------------------------------------------------------------------

def render_syndrome(code: BinaryCode, color: bool = True) -> str:
    """
    8×8 сетка: каждый глиф h раскрашен по syndrome(h).

    syndrome(h) = Hh mod 2 → целое число (индекс синдрома).
    Для кодовых слов syndrome = 0.
    """
    codewords_set = set(code.codewords())
    cosets_list = code.cosets()
    info = code.info()

    # Синдром каждой вершины = номер её смежного класса
    vertex_coset = [0] * 64
    for ci, coset in enumerate(cosets_list):
        for h in coset:
            vertex_coset[h] = ci

    n_syndromes = len(cosets_list)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Синдромная таблица [{info["n"]},{info["k"]},{info["d"]}]')
    lines.append(f'  {n_syndromes} синдромов  (2^{{n-k}}={n_syndromes})')
    lines.append('  Глиф раскрашен по синдрому s(h) = номеру смежного класса')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            ci = vertex_coset[h]
            rows3 = render_glyph(h)
            if color:
                is_cw = (ci == 0)
                if is_cw:
                    c = _YANG_BG[yang_count(h)] + _BOLD
                else:
                    c = _COSET_COLORS[ci % len(_COSET_COLORS)]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            ci = vertex_coset[h]
            is_cw = (ci == 0)
            if color:
                c = _YANG_BG[yang_count(h)] + _BOLD if is_cw else _COSET_COLORS[ci % len(_COSET_COLORS)]
                lbl.append(f'{c}s={ci:2d}{_RESET}')
            else:
                lbl.append(f's={ci:2d}')
        lines.append('  ' + '  '.join(lbl))
        lines.append('')

    lines.append('  Декодирование по синдрому:')
    lines.append('  1. Получить y = x + e  (x ∈ C, e — ошибка)')
    lines.append('  2. Вычислить s = syndrome(y) → номер смежного класса')
    lines.append('  3. Лидер этого класса = оценка ошибки ê')
    lines.append('  4. Исправленное слово = y ⊕ ê')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Порождающая матрица как глифы
# ---------------------------------------------------------------------------

def render_generator(code: BinaryCode, color: bool = True) -> str:
    """
    Строки порождающей матрицы G как k глифов.
    Строки проверочной матрицы H как (n−k) глифов.
    """
    G = code.generator_matrix
    H = code.parity_check_matrix()
    info = code.info()
    codewords = code.codewords()

    def rows_to_ints(M) -> list[int]:
        result = []
        for row in M:
            v = 0
            for i, bit in enumerate(row):
                v |= (int(bit) << i)
            result.append(v)
        return result

    g_ints = rows_to_ints(G)
    h_ints = rows_to_ints(H)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Матрицы кода [{info["n"]},{info["k"]},{info["d"]}]')
    lines.append(f'  G: {info["k"]}×6   H: {6-info["k"]}×6')
    lines.append('═' * 64)

    # Строки G
    lines.append(f'\n  Порождающая матрица G ({info["k"]} строк = {info["k"]} глифа):')
    if g_ints:
        glyphs_g = [render_glyph(h) for h in g_ints]
        if color:
            glyphs_g = [[_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
                        for h, g in zip(g_ints, glyphs_g)]
        for ri in range(3):
            lines.append('  ' + '   '.join(g[ri] for g in glyphs_g))
        lines.append('  ' + '   '.join(format(h, '06b') for h in g_ints))

    # Строки H
    lines.append(f'\n  Проверочная матрица H ({6-info["k"]} строк = {6-info["k"]} глифа):')
    if h_ints:
        glyphs_h = [render_glyph(h) for h in h_ints]
        if color:
            glyphs_h = [[_YANG_ANSI[yang_count(h)] + r + _RESET for r in g]
                        for h, g in zip(h_ints, glyphs_h)]
        for ri in range(3):
            lines.append('  ' + '   '.join(g[ri] for g in glyphs_h))
        lines.append('  ' + '   '.join(format(h, '06b') for h in h_ints))

    # Кодовые слова
    lines.append(f'\n  Кодовые слова ({len(codewords)}):')
    glyphs_cw = [render_glyph(h) for h in codewords[:8]]
    if color:
        glyphs_cw = [[_YANG_BG[yang_count(h)] + _BOLD + r + _RESET for r in g]
                     for h, g in zip(codewords[:8], glyphs_cw)]
    for ri in range(3):
        lines.append('  ' + '  '.join(g[ri] for g in glyphs_cw))
    lines.append('  ' + '  '.join(
        (_YANG_ANSI[yang_count(h)] + format(h, '06b') + _RESET if color else format(h, '06b'))
        for h in codewords[:8]
    ))
    if len(codewords) > 8:
        lines.append(f'  ... ещё {len(codewords)-8} слов ...')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Сравнение стандартных кодов
# ---------------------------------------------------------------------------

def render_codes(color: bool = True) -> str:
    """
    Таблица: 5 стандартных кодов длины 6 с их параметрами.
    """
    named = [
        ('repetition',        repetition_code()),
        ('parity [6,5,2]',    parity_check_code()),
        ('shortened_ham',     shortened_hamming_code()),
        ('even_weight',       even_weight_code()),
        ('dual_rep [6,2,4]',  dual_repetition_code()),
    ]

    lines: list[str] = []
    lines.append('╔' + '═' * 64 + '╗')
    lines.append('║  Стандартные двоичные коды длины n=6' + ' ' * 28 + '║')
    lines.append('╚' + '═' * 64 + '╝')
    lines.append(f'\n  {"Код":20s}  {"[n,k,d]":10s}  {"t":>3}  '
                 f'{"R":>6}  {"Слов":>5}  {"Совершенный":>12}  {"MDS":>4}')
    lines.append('  ' + '─' * 70)

    for name, code in named:
        info = code.info()
        n, k, d = info['n'], info['k'], info['d']
        t = info['t']
        rate = info['rate']
        size = info['size']
        perfect = info['is_perfect']
        mds = info['is_mds']

        if color:
            c = _YANG_ANSI[min(6, k)]
            pfx = _BOLD if perfect else ''
            lines.append(
                f'  {c}{pfx}{name:20s}  [{n},{k},{d}]{" "*5}  '
                f'{t:>3}  {rate:>6.3f}  {size:>5}  '
                f'{"да" if perfect else "нет":>12}  '
                f'{"да" if mds else "нет":>4}{_RESET}'
            )
        else:
            lines.append(
                f'  {name:20s}  [{n},{k},{d}]{" "*5}  '
                f'{t:>3}  {rate:>6.3f}  {size:>5}  '
                f'{"да" if perfect else "нет":>12}  '
                f'{"да" if mds else "нет":>4}'
            )

    lines.append('')
    lines.append('  t = (d−1)/2 — число исправляемых ошибок')
    lines.append('  R = k/n — скорость кода')
    lines.append('  Граница Хэмминга: 2^k · V(6,t) ≤ 2^6')
    lines.append('  MDS-граница Синглтона: d ≤ n−k+1')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='syndrome_glyphs',
        description='Синдромное декодирование через глифы Q6',
    )
    p.add_argument('--no-color', action='store_true')
    p.add_argument('--code', default='shortened_hamming',
                   choices=['repetition', 'parity', 'shortened_hamming',
                            'even_weight', 'dual_rep'],
                   help='выбор кода')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('cosets',    help='смежные классы кода')
    sub.add_parser('syndrome',  help='синдромная таблица декодирования')
    sub.add_parser('generator', help='порождающая и проверочная матрицы')
    sub.add_parser('codes',     help='сравнение стандартных кодов n=6')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color
    code = _get_code(args.code)

    if args.cmd == 'cosets':
        print(render_cosets(code, color))
    elif args.cmd == 'syndrome':
        print(render_syndrome(code, color))
    elif args.cmd == 'generator':
        print(render_generator(code, color))
    elif args.cmd == 'codes':
        print(render_codes(color))


if __name__ == '__main__':
    main()
