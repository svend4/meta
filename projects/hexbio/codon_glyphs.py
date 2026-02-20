"""codon_glyphs — Алфавит кодонов: каждый из 64 кодонов как уникальный глиф.

Биохимическая кодировка (hexbio):
    A=00, C=01, G=10, U=11
    Кодон XYZ → 6-битный номер: биты [5:4]=X, [3:2]=Y, [1:0]=Z

Визуальный алфавит:
    • 64 кодона = 64 уникальных глифа
    • Однонуклеотидная мутация = изменение 2 соседних битов позиции
      (мутационное расстояние ≠ расстояние Хэмминга в Q6, но близко)
    • Синонимичные кодоны (одна аминокислота) образуют кластеры глифов
    • Ян-счёт глифа ≈ G+C содержание кодона (G=10, C=01 дают ровно 1 бит)

Применение:
    • Визуализация последовательностей ДНК/РНК как цепочки иконок
    • Быстрое распознавание стоп-кодонов, старт-кодонов
    • Анализ мутационных расстояний по «несхожести» глифов
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexbio.hexbio import (
    codon_to_int, int_to_codon, translate, synonymous_codons,
    mutation_distance, amino_acids, stop_codons,
)
from libs.hexcore.hexcore import yang_count, hamming
from projects.hexvis.hexvis import (
    render_glyph, render_glyph_grid,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Вспомогательные данные: полные имена аминокислот
# ---------------------------------------------------------------------------

_AA_FULLNAME: dict[str, str] = {
    'A': 'Ala (аланин)',      'R': 'Arg (аргинин)',
    'N': 'Asn (аспарагин)',   'D': 'Asp (аспартат)',
    'C': 'Cys (цистеин)',     'Q': 'Gln (глутамин)',
    'E': 'Glu (глутамат)',    'G': 'Gly (глицин)',
    'H': 'His (гистидин)',    'I': 'Ile (изолейцин)',
    'L': 'Leu (лейцин)',      'K': 'Lys (лизин)',
    'M': 'Met (метионин)',    'F': 'Phe (фенилаланин)',
    'P': 'Pro (пролин)',      'S': 'Ser (серин)',
    'T': 'Thr (треонин)',     'W': 'Trp (триптофан)',
    'Y': 'Tyr (тирозин)',     'V': 'Val (валин)',
    '*': 'СТОП-кодон',
}


def aa_fullname(aa: str) -> str:
    return _AA_FULLNAME.get(aa, aa)


# ---------------------------------------------------------------------------
# GC-содержание кодона через ян-счёт
# ---------------------------------------------------------------------------

def gc_count(n: int) -> int:
    """
    Число G+C нуклеотидов в кодоне n.
    G=10 (2 бита, 1 ян), C=01 (2 бита, 1 ян).
    Но A=00 (0 ян) и U=11 (2 ян) — нарушают линейность.
    Приближение: yang_count(n) ≈ 2 * GC, но только для G/C кодонов.
    """
    codon = int_to_codon(n)
    return sum(1 for nuc in codon if nuc in ('G', 'C'))


# ---------------------------------------------------------------------------
# Визуализация: все 64 кодона в сетке 8×8
# ---------------------------------------------------------------------------

def render_codon_grid(
    highlight_aa: str | None = None,
    color: bool = True,
    show_codon: bool = True,
) -> str:
    """
    8×8 сетка кодонов в кодировке Грея.
    Строки = биты [5:3] (нуклеотиды 1-2, Gray), столбцы = биты [2:0] (нуклеотид 3).
    Каждая клетка: глиф (3 строки) + кодон + аминокислота.

    highlight_aa: выделить глифы данной аминокислоты (напр. 'M' для метионина).
    """
    _GRAY3 = [i ^ (i >> 1) for i in range(8)]

    hl_set: set[int] = set()
    if highlight_aa:
        hl_set = set(synonymous_codons(highlight_aa))

    lines: list[str] = []
    title = f'  Алфавит кодонов'
    if highlight_aa:
        title += f'  (выделена: {highlight_aa} = {aa_fullname(highlight_aa)})'
    lines.append(title)

    # Заголовок столбцов: третий нуклеотид
    col_labels = []
    for col_g in _GRAY3:
        nuc3 = int_to_codon(col_g)[2]  # третья позиция (биты 1:0 = позиция Z)
        # Для 3-й позиции биты — [1:0], но в сетке col = биты [2:0]
        # Проще: взять любой кодон с этим col_g и смотреть все 3 нуклеотида
        pass
    col_header = '       ' + '   '.join(format(g, '03b') for g in _GRAY3)
    lines.append(col_header)
    lines.append('       ' + '─' * (len(_GRAY3) * 6 - 1))

    for row_g in _GRAY3:
        row_label = format(row_g, '03b')
        row_data = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            row_data.append(h)

        glyphs = [render_glyph(h) for h in row_data]

        # 3 строки глифа
        for ri in range(3):
            parts: list[str] = []
            for h, g in zip(row_data, glyphs):
                cell = g[ri]
                is_hl = h in hl_set
                if color:
                    yc = yang_count(h)
                    c = (_YANG_BG[yc] + _BOLD) if is_hl else _YANG_ANSI[yc]
                    cell = c + cell + _RESET
                elif is_hl:
                    cell = '[' + cell[1] + ']'
                parts.append(cell)
            prefix = f'  {row_label} │ ' if ri == 1 else '       │ '
            lines.append(prefix + '  '.join(parts))

        if show_codon:
            # Строка с кодонами
            codon_parts = []
            for h in row_data:
                codon = int_to_codon(h)
                aa = translate(h)
                is_hl = h in hl_set
                codon_str = f'{codon}'
                if is_hl and color:
                    codon_str = _BOLD + codon_str + _RESET
                codon_parts.append(f'{codon_str}')
            lines.append('       │ ' + '  '.join(codon_parts))

            # Строка с аминокислотами
            aa_parts = []
            for h in row_data:
                aa = translate(h)
                is_hl = h in hl_set
                aa_str = f' {aa} '
                if is_hl and color:
                    yc = yang_count(h)
                    aa_str = _YANG_BG[yc] + _BOLD + aa_str + _RESET
                aa_parts.append(aa_str)
            lines.append('       │ ' + '  '.join(aa_parts))

        lines.append('       │')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Визуализация по аминокислотам
# ---------------------------------------------------------------------------

def render_aa_glyphs(color: bool = True) -> str:
    """
    Все аминокислоты: для каждой показать все синонимичные кодоны как глифы.
    """
    all_aa = sorted(amino_acids()) + ['*']
    lines: list[str] = []
    lines.append('═' * 70)
    lines.append('  ГЕНЕТИЧЕСКИЙ КОД: аминокислоты и их кодоны (глифы)')
    lines.append('  Каждый глиф = уникальный кодон RNA')
    lines.append('  Один сегмент = одна пара бит (один нуклеотид у позиции)')
    lines.append('═' * 70)

    for aa in all_aa:
        if aa == '*':
            codons = stop_codons()
        else:
            codons = synonymous_codons(aa)
        n = len(codons)
        fullname = aa_fullname(aa)
        label = f'  {aa}  {fullname:<22s}  ({n} кодон{"а" if n in (2,3,4) else "ов" if n > 4 else ""})  '
        pad = ' ' * len(label)

        glyphs = [render_glyph(h) for h in codons]

        for ri in range(3):
            parts: list[str] = []
            for gi, h in enumerate(codons):
                cell = glyphs[gi][ri]
                if color:
                    yc = yang_count(h)
                    # Стоп-кодоны красным
                    c = _YANG_ANSI[5] if aa == '*' else _YANG_ANSI[yc]
                    cell = c + cell + _RESET
                parts.append(cell)
            prefix = label if ri == 1 else pad
            lines.append(prefix + ' '.join(parts))

        # Кодоны под глифами
        codon_row = pad + ' '.join(f'{int_to_codon(h):>3s}' for h in codons)
        lines.append(codon_row)
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Мутационные расстояния: соседи кодона
# ---------------------------------------------------------------------------

def render_codon_neighbors(n: int, color: bool = True) -> str:
    """
    Кодон n и все его однонуклеотидные мутации.
    Каждая позиция (X, Y, Z) может мутировать в 3 других нуклеотида.
    """
    from projects.hexbio.hexbio import point_mutations

    codon = int_to_codon(n)
    aa = translate(n)
    lines: list[str] = []
    lines.append(f'  Кодон {codon} → {aa} ({aa_fullname(aa)})  [глиф h={n}]')

    glyph = render_glyph(n)
    if color:
        yc = yang_count(n)
        colored = [_YANG_BG[yc] + _BOLD + row + _RESET for row in glyph]
    else:
        colored = glyph

    for row in colored:
        lines.append(f'    {row}')

    lines.append(f'\n  Однонуклеотидные мутации (9 соседей):')
    lines.append(f'  {"Позиция":<10} {"Кодон":<8} {"AA":<5} {"Глиф":<12} {"Тип мутации"}')
    lines.append('  ' + '─' * 55)

    mutations = point_mutations(n)
    # point_mutations returns flat list of mutant ints; reconstruct position info
    from projects.hexbio.hexbio import codon_nucleotides, _ALTERNATIVES, _NUC
    nuc = list(codon_nucleotides(n))
    mut_with_pos: list[tuple[int, int]] = []
    for pos in range(3):
        original = nuc[pos]
        for alt in _ALTERNATIVES[original]:
            mutated = nuc[:]
            mutated[pos] = alt
            m_int = (_NUC[mutated[0]] << 4) | (_NUC[mutated[1]] << 2) | _NUC[mutated[2]]
            mut_with_pos.append((pos, m_int))

    for pos, mutant_n in sorted(mut_with_pos, key=lambda x: (x[0], x[1])):
        m_codon = int_to_codon(mutant_n)
        m_aa = translate(mutant_n)
        m_glyph = render_glyph(mutant_n)
        ham = hamming(n, mutant_n)
        mut_type = 'синоним' if m_aa == aa else ('СТОП' if m_aa == '*' else f'→{m_aa}')
        pos_name = f'поз.{pos+1}({codon[pos]}→{m_codon[pos]})'
        glyph_str = m_glyph[1]  # средняя строка
        if color:
            yc = yang_count(mutant_n)
            c = _YANG_ANSI[5] if m_aa == '*' else _YANG_ANSI[yc]
            glyph_str = c + glyph_str + _RESET
        lines.append(
            f'  {pos_name:<12} {m_codon:<8} {m_aa:<5} {glyph_str:<12} '
            f'ΔHamming={ham}  {mut_type}'
        )

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Последовательность ДНК как цепочка глифов
# ---------------------------------------------------------------------------

def render_sequence(seq: str, color: bool = True, per_line: int = 10) -> str:
    """
    Отобразить последовательность нуклеотидов как цепочку глифов-кодонов.
    seq: строка из A,T,C,G (или U). Длина должна быть кратна 3.
    """
    seq = seq.upper().replace('T', 'U')
    if len(seq) % 3 != 0:
        seq = seq[:len(seq) - len(seq) % 3]  # обрезаем до кратности 3

    codons_list: list[tuple[int, str, str]] = []
    for i in range(0, len(seq), 3):
        codon_str = seq[i:i + 3]
        try:
            h = codon_to_int(codon_str)
            aa = translate(h)
            codons_list.append((h, codon_str, aa))
        except ValueError:
            continue

    lines: list[str] = []
    lines.append(f'  Последовательность ({len(codons_list)} кодонов, {len(seq)} нт):')
    lines.append(f'  {seq}')
    lines.append('')

    for chunk_start in range(0, len(codons_list), per_line):
        chunk = codons_list[chunk_start:chunk_start + per_line]
        glyphs = [render_glyph(h) for h, _, _ in chunk]

        # Номера позиций
        pos_row = '  ' + '  '.join(f'{chunk_start + i + 1:>4d}' for i in range(len(chunk)))
        lines.append(pos_row)

        # 3 строки глифов
        for ri in range(3):
            parts: list[str] = []
            for (h, codon_str, aa), g in zip(chunk, glyphs):
                cell = g[ri]
                if color:
                    yc = yang_count(h)
                    c = _YANG_ANSI[5] if aa == '*' else _YANG_ANSI[yc]
                    cell = c + cell + _RESET
                parts.append(cell)
            lines.append('  ' + '  '.join(parts))

        # Кодон + АК
        codon_row = '  ' + '  '.join(f'{cs:>4s}' for _, cs, _ in chunk)
        aa_row    = '  ' + '  '.join(f'  {aa:>1s} ' for _, _, aa in chunk)
        lines.append(codon_row)
        lines.append(aa_row)
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Статистика: GC-содержание и ян-счёт
# ---------------------------------------------------------------------------

def render_gc_yang_correlation(color: bool = True) -> str:
    """
    Корреляция между GC-содержанием кодона и ян-счётом его глифа.
    Показывает насколько «похожи» эти метрики.
    """
    lines: list[str] = []
    lines.append('  Связь GC-содержания и ян-счёта глифа:')
    lines.append(f'  {"GC":>3} {"Ян":>3} {"Кодон":>5} {"АК":>4} {"Глиф (средняя строка)"}')
    lines.append('  ' + '─' * 45)

    for h in range(64):
        gc = gc_count(h)
        yc = yang_count(h)
        codon = int_to_codon(h)
        aa = translate(h)
        glyph_mid = render_glyph(h)[1]
        if color:
            c = _YANG_ANSI[yc]
            glyph_mid = c + glyph_mid + _RESET
        lines.append(f'  {gc:>3}  {yc:>3}  {codon:>5}  {aa:>4}  {glyph_mid}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# JSON-экспорт (для пайплайнов SC-4)
# ---------------------------------------------------------------------------

def json_codon_map() -> dict:
    """
    Полная карта: 64 кодона → 64 гексаграммы → позиции в треугольнике Андреева.

    SC-4 шаг 1: K4 (биология) пересекается с K6 (И-Цзин).
    Ключевое открытие: transitions (A↔G, C↔U) = Q6-рёбра (1-бит),
    transversions (A↔U, C↔G) = Q6-прыжки на 2 бита.
    """
    from projects.hextrimat.hextrimat import TriangularMatrix
    trimat = TriangularMatrix()
    pos_map: dict[int, tuple[int, int]] = {v: (r, c) for r, c, v in trimat.cells}

    codons_list = []
    for h in range(64):
        codon_str = int_to_codon(h)
        aa = translate(h)
        v = h + 1
        row, col = pos_map.get(v, (0, 0))
        u_cnt = codon_str.count('U')
        gc_cnt = sum(1 for n in codon_str if n in ('G', 'C'))
        codons_list.append({
            'hexagram': h,
            'codon': codon_str,
            'amino_acid': aa,
            'yang_count': yang_count(h),
            'gc_count': gc_cnt,
            'u_count': u_cnt,
            'trimat_row': row,
            'trimat_col': col,
            'trimat_value': v,
        })

    # Группировка по аминокислотам
    aa_map: dict[str, dict] = {}
    for c in codons_list:
        aa = c['amino_acid']
        if aa not in aa_map:
            aa_map[aa] = {'codons': [], 'hexagrams': [], 'rows': []}
        aa_map[aa]['codons'].append(c['codon'])
        aa_map[aa]['hexagrams'].append(c['hexagram'])
        aa_map[aa]['rows'].append(c['trimat_row'])

    for aa, info in aa_map.items():
        rows = info['rows']
        info['cluster_size'] = len(info['codons'])
        info['row_min'] = min(rows)
        info['row_max'] = max(rows)
        info['row_span'] = max(rows) - min(rows)
        info['wobble_clustered'] = (info['row_span'] <= 1)

    stop_codons = [c for c in codons_list if c['amino_acid'] == '*']
    start_codon = next((c for c in codons_list if c['codon'] == 'AUG'), None)

    # Переходы: transitions (1-бит XOR) vs transversions (2-бит XOR)
    # Каждая позиция нуклеотида кодируется 2 битами: A=00,C=01,G=10,U=11
    # Transition A↔G: XOR 10 (1 бит) — Q6-ребро!
    # Transition C↔U: XOR 10 (1 бит) — Q6-ребро!
    # Transversion A↔C: XOR 01 (1 бит) — тоже Q6-ребро!
    # Transversion A↔U: XOR 11 (2 бита) — 2-шаговый путь в Q6!
    # Transversion C↔G: XOR 11 (2 бита) — 2-шаговый путь в Q6!
    # Transversion G↔U: XOR 01 (1 бит) — Q6-ребро!
    # → Только ПУРИНОВЫЕ транзиции (A↔G) и ПИРИМИДИНОВЫЕ (C↔U) + некоторые трансверсии
    # На уровне пары нуклеотидов: A↔U и C↔G = комплементарные пары = XOR 11 = 2 бита

    complementary_pairs = [('A', 'U'), ('G', 'C')]  # Watson-Crick base pairs
    xor_11_pairs = {frozenset(['A', 'U']), frozenset(['C', 'G'])}

    mutation_classes = {
        'q6_edge_1bit': [],   # 1-bit XOR per position → Q6-ребро
        'q6_jump_2bit': [],   # 2-bit XOR per position → Q6-прыжок
    }
    from projects.hexbio.hexbio import _NUC
    for n1 in ('A', 'C', 'G', 'U'):
        for n2 in ('A', 'C', 'G', 'U'):
            if n1 >= n2:
                continue
            xor_val = _NUC[n1] ^ _NUC[n2]
            bit_count = bin(xor_val).count('1')
            pair = f'{n1}↔{n2}'
            if bit_count == 1:
                mutation_classes['q6_edge_1bit'].append(pair)
            else:
                mutation_classes['q6_jump_2bit'].append(pair)

    return {
        'command': 'codon_map',
        'n_codons': 64,
        'n_amino_acids': len(aa_map),
        'codons': codons_list,
        'amino_acid_clusters': aa_map,
        'stop_codons': {
            'codons': [c['codon'] for c in stop_codons],
            'hexagrams': [c['hexagram'] for c in stop_codons],
            'trimat_rows': [c['trimat_row'] for c in stop_codons],
        },
        'start_codon': start_codon,
        'mutation_classes': mutation_classes,
        'sc4_insight': (
            'Transitions A↔G, C↔U: XOR=10 или 01 (1 бит) = Q6-рёбра. '
            'Transversions A↔U, C↔G: XOR=11 (2 бита) = 2-шаговые пути Q6. '
            'Watson-Crick комплементарность = 2-битный Q6-прыжок!'
        ),
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='codon_glyphs — Алфавит кодонов: 64 кодона как глифы Q6'
    )
    parser.add_argument('--json', action='store_true',
                        help='Машиночитаемый JSON-вывод (для пайплайнов)')
    sub = parser.add_subparsers(dest='cmd')

    # codon-map — SC-4 шаг 1: карта кодонов K4→K6
    sub.add_parser('codon-map',
                   help='Полная карта кодон→гексаграмма→триматрица → JSON')

    sub.add_parser('grid',    help='Сетка 8×8 всех кодонов с глифами')
    sub.add_parser('amino',   help='Глифы по аминокислотам')
    p_hl = sub.add_parser('highlight', help='Выделить кодоны одной аминокислоты')
    p_hl.add_argument('aa', help='Аминокислота (1 буква, напр. M, W, *)')
    p_nb = sub.add_parser('neighbors', help='Мутационные соседи кодона')
    p_nb.add_argument('codon', help='Кодон (напр. AUG) или число (0..63)')
    p_seq = sub.add_parser('seq', help='Последовательность нуклеотидов как глифы')
    p_seq.add_argument('sequence', help='Строка нуклеотидов (A,T,C,G,U)')
    p_seq.add_argument('--width', type=int, default=10)

    for p in sub.choices.values():
        p.add_argument('--no-color', action='store_true')

    args = parser.parse_args()
    color = not getattr(args, 'no_color', False)

    if args.cmd == 'codon-map':
        result = json_codon_map()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f'  Карта кодонов Q6 → И-Цзин (K4×K6):')
            mc = result['mutation_classes']
            print(f'  Q6-рёбра (1-бит): {mc["q6_edge_1bit"]}')
            print(f'  Q6-прыжки (2-бит): {mc["q6_jump_2bit"]}')
            print()
            print(f'  Аминокислоты ({result["n_amino_acids"]}):')
            for aa, cl in sorted(result['amino_acid_clusters'].items()):
                wc = '✓' if cl['wobble_clustered'] else '✗'
                print(f'    {aa}: {cl["codons"]}  строки {cl["row_min"]}-{cl["row_max"]}  wobble={wc}')
            stop = result['stop_codons']
            print(f'\n  Стоп-кодоны: {stop["codons"]} → гексаграммы {stop["hexagrams"]}')
            start = result['start_codon']
            if start:
                print(f'  Старт-кодон AUG → гексаграмма {start["hexagram"]}  строка {start["trimat_row"]}')
            print(f'\n  SC-4: {result["sc4_insight"]}')
        sys.exit(0)

    if args.cmd == 'grid' or args.cmd is None:
        print(render_codon_grid(color=color))

    elif args.cmd == 'amino':
        print(render_aa_glyphs(color=color))

    elif args.cmd == 'highlight':
        print(render_codon_grid(highlight_aa=args.aa.upper(), color=color))

    elif args.cmd == 'neighbors':
        try:
            h = int(args.codon)
        except ValueError:
            h = codon_to_int(args.codon.upper().replace('T', 'U'))
        print(render_codon_neighbors(h, color=color))

    elif args.cmd == 'seq':
        print(render_sequence(args.sequence, color=color, per_line=args.width))
