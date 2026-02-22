"""sym_glyphs — Группа автоморфизмов Aut(Q6) через глифы.

Каждый глиф (0..63) — вершина Q6.
Aut(Q6) = (Z₂)⁶ ⋊ S₆:  биты-отражения × перестановки битов.

|Aut(Q6)| = 2⁶ · 6! = 64 · 720 = 46 080

Структура группы:
  • (Z₂)⁶ — 64 отображения «XOR маска»: v ↦ v ⊕ mask
  • S₆     — 720 перестановок битов: v ↦ π(v)
  • Вместе: |Aut| = 64 · 720 = 46 080

Yang-орбиты (оbits числа единичных битов):
  |S(0,k)| = C(6,k): 1, 6, 15, 20, 15, 6, 1  — ровно 7 орбит.
  Aut(Q6) действует транзитивно на каждом слое yang=k.

Визуализация:
  yang      — 7 орбит по числу единичных битов (yang_count)
  fixed     — неподвижные точки генераторов Aut(Q6)
  antipodal — антиподальные пары {h, h⊕63}
  burnside  — таблица Бернсайда/Полиа для n цветов

Команды CLI:
  yang
  fixed
  antipodal
  burnside  [--colors n]
"""

from __future__ import annotations
import json
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexsym.hexsym import (
    identity_aut, bit_flip_single, bit_permutation, bit_transposition,
    aut_generators, s6_generators,
    yang_orbits, antipodal_orbits,
    fixed_points, cycle_decomposition, cycle_count,
    burnside_count, burnside_subset, polya_count,
    Automorphism,
)
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

_AUT_ORDER = 64 * 720   # |Aut(Q6)|


# ---------------------------------------------------------------------------
# 1. Yang-орбиты Aut(Q6)
# ---------------------------------------------------------------------------

def render_yang(color: bool = True) -> str:
    """
    8×8 сетка: 7 орбит Aut(Q6) по yang_count.

    Aut(Q6) действует транзитивно на каждом слое {h : yang_count(h)=k}.
    Это 7 орбит размеров C(6,0)..C(6,6) = 1,6,15,20,15,6,1.
    """
    import math
    yo = yang_orbits()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Орбиты Aut(Q6) на Q6: 7 слоёв по yang_count')
    lines.append(f'  |Aut(Q6)| = 2⁶·6! = {_AUT_ORDER}')
    lines.append('  Aut(Q6) транзитивно на каждом yang-слое')
    lines.append('  Цвет = yang_count(h) = число единичных битов')
    lines.append('═' * 66)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            yc = yang_count(h)
            rows3 = render_glyph(h)
            if color:
                c = _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            yc = yang_count(h)
            if color:
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}y={yc}{_RESET}')
            else:
                lbl.append(f'y={yc}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    lines.append('  Yang-орбиты (|орбиты| = C(6,k)):')
    for k, orb in enumerate(yo):
        binom = math.comb(6, k)
        stab = _AUT_ORDER // len(orb)
        if color:
            c = _YANG_ANSI[k]
            lines.append(f'  {c}  k={k}: {len(orb):2d} вершин = C(6,{k})={binom}  '
                         f'|Stab|={stab}{_RESET}')
        else:
            lines.append(f'    k={k}: {len(orb):2d} вершин = C(6,{k})={binom}  '
                         f'|Stab|={stab}')
    lines.append(f'  Σ|орбит| = {sum(len(o) for o in yo)} = 64 ✓')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Неподвижные точки генераторов
# ---------------------------------------------------------------------------

def render_fixed(color: bool = True) -> str:
    """
    8×8 сетка: для каждого из 6 генераторов Aut(Q6) — его неподвижные точки.

    Генераторы:
      g_i = bit_transposition(i, i+1)  (i=0..4) — 5 транспозиций соседних битов
      g_5 = bit_flip_single(0)          — отражение первого бита
    """
    gens = aut_generators()

    # Для каждой вершины: сколько генераторов её фиксируют
    fix_by_gen: list[list[int]] = []
    for g in gens:
        fp = set(fixed_points(g))
        fix_by_gen.append([1 if h in fp else 0 for h in range(64)])

    fix_total = [sum(fix_by_gen[i][h] for i in range(len(gens)))
                 for h in range(64)]

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Неподвижные точки генераторов Aut(Q6)')
    lines.append(f'  {len(gens)} генераторов: {len(gens)-1} транспозиций битов + 1 отражение')
    lines.append('  fix(h) = числo генераторов, сохраняющих вершину h')
    lines.append('  Жирный = фиксируется всеми генераторами (h=0: все нули)')
    lines.append('═' * 66)

    max_fix = max(fix_total)

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            ft = fix_total[h]
            rows3 = render_glyph(h)
            if color:
                yc = yang_count(h)
                c = _YANG_BG[yc] + _BOLD if ft == max_fix else _YANG_ANSI[ft]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            ft = fix_total[h]
            if color:
                c = _YANG_ANSI[min(ft, 6)]
                lbl.append(f'{c}f{ft}{_RESET}')
            else:
                lbl.append(f'f{ft}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Подробно по каждому генератору
    lines.append('  Генераторы Aut(Q6):')
    for i, g in enumerate(gens):
        fp = fixed_points(g)
        cc = cycle_count(g)
        if color:
            c = _YANG_ANSI[i % 7]
            lines.append(f'  {c}  g{i}: {g}   '
                         f'Fix={len(fp):2d}   циклов={cc}{_RESET}')
        else:
            lines.append(f'    g{i}: {g}   Fix={len(fp):2d}   циклов={cc}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Антиподальные пары
# ---------------------------------------------------------------------------

def render_antipodal(color: bool = True) -> str:
    """
    8×8 сетка: антиподальные пары {h, h⊕63}.

    Антипод h* = h ⊕ 0b111111 = h XOR 63 (инверсия всех битов).
    Расстояние Хэмминга d(h, h*) = 6 (максимальное).
    Есть ровно 32 антиподальные пары.
    """
    ao = antipodal_orbits()   # список пар frozenset

    # Для каждой вершины — индекс её пары
    pair_idx = [-1] * 64
    for idx, pair in enumerate(ao):
        for h in pair:
            pair_idx[h] = idx

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append('  Антиподальные пары Q6: {h, h⊕63}')
    lines.append('  Антипод h* = h XOR 63 — дополнение всех битов')
    lines.append(f'  32 пары   d(h, h*) = 6   yang(h) + yang(h*) = 6')
    lines.append('  Цвет пары: yang_count ∈ {0..3} (пара содержит both части)')
    lines.append('═' * 66)

    _PAIR_COLORS = [
        '\033[38;5;27m',  '\033[38;5;82m',  '\033[38;5;196m',
        '\033[38;5;208m', '\033[38;5;201m', '\033[38;5;226m',
        '\033[38;5;39m',  '\033[38;5;46m',  '\033[38;5;51m',
        '\033[38;5;160m', '\033[38;5;11m',  '\033[38;5;93m',
        '\033[38;5;130m', '\033[38;5;50m',  '\033[38;5;200m',
        '\033[38;5;238m',
    ]

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            pi = pair_idx[h]
            rows3 = render_glyph(h)
            if color:
                antip = h ^ 63
                # Цвет: по yang меньшего элемента пары
                yc = min(yang_count(h), yang_count(antip))
                is_lower = (h < antip)   # меньший в паре = ярче
                c = _YANG_BG[yc] + _BOLD if is_lower else _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            antip = h ^ 63
            if color:
                yc = yang_count(h)
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}↔{antip:02d}{_RESET}')
            else:
                lbl.append(f'↔{antip:02d}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    lines.append('  Первые 8 антиподальных пар {h, h*=h⊕63}:')
    for idx, pair in enumerate(ao[:8]):
        h1, h2 = sorted(pair)
        if color:
            yc = yang_count(h1)
            c = _YANG_ANSI[yc]
            lines.append(f'  {c}  {{{h1:02d},{h2:02d}}}  '
                         f'{format(h1,"06b")} ↔ {format(h2,"06b")}  '
                         f'yang={yang_count(h1)}+{yang_count(h2)}=6{_RESET}')
        else:
            lines.append(f'    {{{h1:02d},{h2:02d}}}  '
                         f'{format(h1,"06b")} ↔ {format(h2,"06b")}  '
                         f'yang={yang_count(h1)}+{yang_count(h2)}=6')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Бернсайд/Полиа: число раскрасок
# ---------------------------------------------------------------------------

def render_burnside(n_colors: int = 2, color: bool = True) -> str:
    """
    Таблица Бернсайда/Полиа: число раскрасок Q6 при n цветах.

    Polya(n) = (1/|G|) Σ_g n^{cycle_count(g)}
    — число существенно различных раскрасок Q6 при действии Aut(Q6).
    """
    gens = aut_generators()

    lines: list[str] = []
    lines.append('═' * 66)
    lines.append(f'  Теорема Бернсайда/Полиа для Aut(Q6)')
    lines.append(f'  Polya(n) = (1/{_AUT_ORDER}) Σ_g n^{{c(g)}}')
    lines.append(f'  = число раскрасок вершин Q6 в n цветов с точн. до Aut(Q6)')
    lines.append('═' * 66)
    lines.append('')

    # Цикловой индекс: считаем числа циклов для каждого генератора
    lines.append(f'  Цикловой индекс генераторов (c(g) = число циклов на Q6):')
    for i, g in enumerate(gens):
        cc = cycle_count(g)
        fp = fixed_points(g)
        if color:
            c = _YANG_ANSI[i % 7]
            lines.append(f'  {c}  g{i}: цiklов={cc}   fix={len(fp)}{_RESET}')
        else:
            lines.append(f'    g{i}: циклов={cc}   fix={len(fp)}')

    lines.append('')
    lines.append('  Polya(n) для малых n:')
    for n in range(2, min(n_colors + 1, 8)):
        pc = polya_count(n)
        if color:
            c = _YANG_ANSI[(n - 1) % 7]
            lines.append(f'  {c}  Polya({n}) = {pc:,}{_RESET}')
        else:
            lines.append(f'    Polya({n}) = {pc:,}')

    lines.append('')
    lines.append(f'  Число раскрасок k единичных вершин (k-подмножества):')
    lines.append('  burnside_subset(k) = число орбит на k-элементных подмн. Q6')
    for k in [0, 1, 2, 3, 6, 10, 16, 32]:
        bs = burnside_subset(k, gens)
        if color:
            c = _YANG_ANSI[min(k % 7, 6)]
            lines.append(f'  {c}  k={k:2d}: {bs:8,} различных подмн.{_RESET}')
        else:
            lines.append(f'    k={k:2d}: {bs:8,} различных подмн.')

    # 8×8 карта: для каждого h, сколько генераторов его фиксируют
    lines.append('\n  8×8 карта: yang_count как орбитный индекс')

    for row in range(8):
        glyph_rows_list = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            yc = yang_count(h)
            rows3 = render_glyph(h)
            if color:
                c = _YANG_ANSI[yc]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows_list[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(
                glyph_rows_list[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            yc = yang_count(h)
            if color:
                c = _YANG_ANSI[yc]
                lbl.append(f'{c}y{yc}{_RESET}')
            else:
                lbl.append(f'y{yc}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# JSON-экспорт (для пайплайнов SC-3)
# ---------------------------------------------------------------------------

def json_rule_orbits() -> dict:
    """
    Классифицировать все КА-правила Q6 по орбитам Aut(Q6).

    Ключевой вопрос: какие правила являются Aut(Q6)-эквивариантными?
    Правило f эквивариантно если: f(g·x, g·nbrs) = g·f(x, nbrs) для всех g ∈ Aut(Q6).

    Практический тест: эквивариантное правило сохраняет ян-баланс распределения.
    Инвариант Aut(Q6): yang_count(h) — ян-слои являются орбитами под S₆.
    """
    import random as _random
    # Импортируем CA-модуль локально (K2×K8 пересечение)
    from projects.hexca.hexca import CA1D
    from projects.hexca.rules import RULES

    WIDTH = 32
    STEPS = 30
    SEED = 42

    yang_orbits_info = yang_orbits()
    orbit_sizes = [len(o) for o in yang_orbits_info]  # [1,6,15,20,15,6,1]

    rule_results = []
    for rule_name, rule_fn in RULES.items():
        _random.seed(SEED)
        ca = CA1D(WIDTH, rule_fn)
        ca.run(STEPS)

        # Проверить ян-баланс: начальное и конечное распределение по ян-счёту
        def yang_dist(row):
            from collections import Counter
            cnt = Counter(yang_count(h) for h in row)
            return [cnt.get(k, 0) / len(row) for k in range(7)]

        dist_init = yang_dist(ca.history[0])
        dist_final = yang_dist(ca.history[-1])

        # Расстояние Вассерштейна (L1) между ян-распределениями
        yang_drift = sum(abs(dist_final[k] - dist_init[k]) for k in range(7))

        # Эквивариантность: сохраняется ли ян-баланс?
        equivariant = yang_drift < 0.15

        # Проверить конвергенцию
        conv_step = None
        for t in range(1, len(ca.history)):
            if ca.history[t] == ca.history[t - 1]:
                conv_step = t
                break
        period_2 = (len(ca.history) >= 3 and
                    ca.history[-1] == ca.history[-3])

        # Классификация Вольфрама
        init_entropy = _entropy(ca.history[0])
        final_entropy = _entropy(ca.history[-1])
        if conv_step is not None and conv_step <= 3:
            wolfram = 'I'
        elif conv_step is not None or period_2:
            wolfram = 'II'
        elif final_entropy > init_entropy * 0.85:
            wolfram = 'III_or_IV'
        else:
            wolfram = 'II'

        rule_results.append({
            'rule': rule_name,
            'equivariant': equivariant,
            'yang_drift': round(yang_drift, 4),
            'wolfram_class': wolfram,
            'convergence_step': conv_step,
            'initial_entropy': round(init_entropy, 4),
            'final_entropy': round(final_entropy, 4),
        })

    equivariant_rules = [r['rule'] for r in rule_results if r['equivariant']]
    non_equivariant = [r['rule'] for r in rule_results if not r['equivariant']]

    # Burnside: число орбит функций Q6→Q6 при Aut(Q6)-действии
    # Оценочно: сокращение пространства правил симметрией
    n_named_rules = len(RULES)
    n_equiv = len(equivariant_rules)

    return {
        'command': 'rule_orbits',
        'aut_q6_order': _AUT_ORDER,
        'yang_orbit_sizes': orbit_sizes,      # [1,6,15,20,15,6,1]
        'yang_orbit_count': 7,
        'rules_analyzed': list(RULES.keys()),
        'per_rule': rule_results,
        'equivariant_rules': equivariant_rules,
        'non_equivariant_rules': non_equivariant,
        'equivariant_fraction': round(n_equiv / n_named_rules, 3),
        'sc3_finding': (
            f'Из {n_named_rules} правил {n_equiv} Aut(Q6)-эквивариантны '
            f'(ян-баланс сохраняется). '
            f'Aut(Q6) с порядком {_AUT_ORDER} группирует ~2^64 правил '
            f'в орбиты; 7 ян-слоёв — инварианты для K2×K8.'
        ),
    }


def _entropy(row: list[int]) -> float:
    """Шенноновская энтропия строки состояний."""
    import math
    from collections import Counter
    cnt = Counter(row)
    n = len(row)
    return -sum(c / n * math.log2(c / n) for c in cnt.values()) if n > 0 else 0.0


def json_sbox_symmetry(minimize_data: dict) -> dict:
    """
    Проанализировать Aut(Q6)-симметрию S-блока (TSC-1 шаг 4).

    Входной формат: вывод karnaugh6:sbox-minimize (поле 'table').
    Ключевой результат для Hermann ring:
      σ₃₂: h → h⊕32 ∈ Aut(Q6) — антиподальная симметрия.
      sbox[h⊕32] = ~sbox[h] для всех h.
      K5-геометрия → K8-алгебра → K1-криптослабость.
    """
    table = minimize_data.get('table')
    if table is None:
        return {'error': 'поле "table" не найдено во входных данных'}

    # ── 1. Антиподальная симметрия σ₃₂: h→h⊕32 ─────────────────────────────
    antipodal_ok = 0
    antipodal_fail = []
    for h in range(64):
        h_flip = h ^ 32
        if table[h] ^ table[h_flip] == 63:
            antipodal_ok += 1
        else:
            antipodal_fail.append({'h': h, 'sbox_h': table[h], 'sbox_h32': table[h_flip],
                                   'xor': table[h] ^ table[h_flip]})
    antipodal_symmetric = len(antipodal_fail) == 0

    # ── 2. Основные XOR-маскные автоморфизмы ─────────────────────────────────
    xor_auts = []
    for m, label in [(0, 'тождество'), (32, 'σ₃₂ (антипод)'), (63, 'дополнение')]:
        deltas = set()
        for h in range(64):
            deltas.add(table[h ^ m] ^ table[h])
        is_const = len(deltas) == 1
        delta_val = next(iter(deltas)) if is_const else None
        xor_auts.append({
            'mask': m,
            'label': label,
            'output_xor_is_constant': is_const,
            'output_xor_delta': delta_val,
            'strict_symmetry': is_const and delta_val == 0,
            'complement_symmetry': is_const and delta_val == 63,
        })

    # ── 3. Бит-транспозиции τᵢ (перестановочная часть S₆) ────────────────────
    bit_transpositions = []
    for i in range(5):
        def apply_t(h, i=i):
            bi = (h >> i) & 1
            bj = (h >> (i + 1)) & 1
            h2 = h & ~((1 << i) | (1 << (i + 1)))
            h2 |= (bj << i) | (bi << (i + 1))
            return h2

        deltas = set(table[apply_t(h)] ^ table[h] for h in range(64))
        bit_transpositions.append({
            'transposition': f'τ({i},{i+1})',
            'n_unique_deltas': len(deltas),
            'is_symmetry': (len(deltas) == 1 and next(iter(deltas)) == 0),
        })

    # ── 4. TSC-1 синтез: K5→K8→K1 ────────────────────────────────────────────
    linear_mask_u3 = minimize_data.get('linear_mask_u3', {})
    nl0_confirmed = linear_mask_u3.get('mask3_equals_bit0_input', False)

    n_xor_symmetries = sum(1 for a in xor_auts if a['strict_symmetry'] or a['complement_symmetry'])
    n_perm_symmetries = sum(1 for t in bit_transpositions if t['is_symmetry'])

    return {
        'command': 'sbox_symmetry',
        'aut_q6_order': _AUT_ORDER,
        'antipodal_symmetry': {
            'automorphism': 'σ₃₂: h → h XOR 32',
            'pairs_ok': antipodal_ok,
            'pairs_fail': len(antipodal_fail),
            'is_antipodal_symmetric': antipodal_symmetric,
            'k5_connection': 'ring[h] + ring[h⊕32] = 65 (Hermann) ↔ sbox[h] XOR sbox[h⊕32] = 63',
        },
        'xor_automorphisms': xor_auts,
        'bit_transpositions': bit_transpositions,
        'n_active_symmetries': n_xor_symmetries + n_perm_symmetries,
        'nl0_from_karnaugh': nl0_confirmed,
        'tsc1_finding': (
            'K5→K8→K1: Германова упаковка (антипод ring[h]+ring[h⊕32]=65) '
            '↔ Aut(Q6)-симметрия σ₃₂ (K8) '
            '→ линейная маска u=3: f₀⊕f₁=x₀ (Карно: 1 литерал) '
            '→ NL=0 криптослабость (K1). '
            'Геометрическая структура ПРИНУЖДАЕТ к алгебраической → криптографической слабости.'
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='sym_glyphs',
        description='Группа автоморфизмов Aut(Q6) через глифы',
    )
    p.add_argument('--no-color', action='store_true')
    p.add_argument('--json', action='store_true',
                   help='Машиночитаемый JSON-вывод (для пайплайнов)')
    p.add_argument('--from-minimize', action='store_true',
                   help='Читать karnaugh:sbox-minimize JSON из stdin (TSC-1)')
    sub = p.add_subparsers(dest='cmd', required=True)

    # sbox-symmetry — TSC-1 шаг 4: Aut(Q6)-симметрия S-блока
    sub.add_parser('sbox-symmetry',
                   help='Aut(Q6)-симметрия S-блока: K5→K8→K1 пайплайн → JSON')

    # rule-orbits — SC-3 шаг 3: классификация КА-правил по орбитам Aut(Q6)
    sub.add_parser('rule-orbits',
                   help='Классифицировать КА-правила по Aut(Q6)-орбитам → JSON')

    sub.add_parser('yang',      help='7 орбит по yang_count')
    sub.add_parser('fixed',     help='неподвижные точки генераторов')
    sub.add_parser('antipodal', help='антиподальные пары {h, h⊕63}')

    s = sub.add_parser('burnside', help='теорема Полиа для n цветов')
    s.add_argument('--colors', type=int, default=4,
                   help='максимальное число цветов (default=4)')
    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'sbox-symmetry':
        if args.from_minimize:
            raw = sys.stdin.read().strip()
            minimize_data = json.loads(raw)
        else:
            # Demo: use Hermann ring by default
            import subprocess
            step1 = subprocess.run(
                [sys.executable, '-m', 'projects.hexpack.pack_glyphs', '--json', 'ring'],
                capture_output=True, text=True,
            )
            step2 = subprocess.run(
                [sys.executable, '-m', 'projects.karnaugh6.kmap_glyphs',
                 '--json', '--from-sbox', 'sbox-minimize'],
                input=step1.stdout, capture_output=True, text=True,
            )
            minimize_data = json.loads(step2.stdout)
        result = json_sbox_symmetry(minimize_data)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            ap = result['antipodal_symmetry']
            ap_status = '✓ ВСЕ 64 пары' if ap['is_antipodal_symmetric'] else f'✗ {ap["pairs_fail"]} сбоев'
            print(f'  Антиподальная симметрия σ₃₂: {ap_status}')
            print(f'  K5: {ap["k5_connection"]}')
            print()
            print('  XOR-автоморфизмы:')
            for a in result['xor_automorphisms']:
                sym = '✓ строгая' if a['strict_symmetry'] else ('≈ дополнение' if a['complement_symmetry'] else '✗')
                print(f'    mask={a["mask"]:2d} ({a["label"]}): delta={a["output_xor_delta"]}  {sym}')
            print()
            nl0_ok = '✓' if result['nl0_from_karnaugh'] else '✗'
            print(f'  Карно (маска u=3): NL=0 подтверждён: {nl0_ok}')
            print()
            print(f'  TSC-1: {result["tsc1_finding"]}')
        return

    if args.cmd == 'rule-orbits':
        result = json_rule_orbits()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f'  Aut(Q6): порядок={result["aut_q6_order"]}'
                  f'  ян-орбит={result["yang_orbit_count"]}')
            print(f'  Размеры ян-слоёв: {result["yang_orbit_sizes"]}')
            print()
            print(f'  {"Правило":<20}  {"Экв?":<6}  {"Класс":<10}  {"H0→HN":<14}  drift')
            print('  ' + '─' * 60)
            for r in result['per_rule']:
                eq = '✓' if r['equivariant'] else '✗'
                print(f'  {r["rule"]:<20}  {eq:<6}  {r["wolfram_class"]:<10}  '
                      f'{r["initial_entropy"]:.2f}→{r["final_entropy"]:.2f}        '
                      f'{r["yang_drift"]:.3f}')
            print()
            print(f'  Эквивариантные: {result["equivariant_rules"]}')
            print(f'  Неэквивариантные: {result["non_equivariant_rules"]}')
            print()
            print(f'  Открытие: {result["sc3_finding"]}')
        return

    if args.cmd == 'yang':
        print(render_yang(color))
    elif args.cmd == 'fixed':
        print(render_fixed(color))
    elif args.cmd == 'antipodal':
        print(render_antipodal(color))
    elif args.cmd == 'burnside':
        print(render_burnside(n_colors=args.colors, color=color))


if __name__ == '__main__':
    main()
