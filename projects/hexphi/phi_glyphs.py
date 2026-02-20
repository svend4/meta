"""phi_glyphs — Числа Фибоначчи и золотое сечение φ в структуре Q6.

SC-7 (K7×K5): φ как Q6-инвариант.

Fibonacci cube Γ₆ ⊂ Q6:
  Γ₆ — подграф Q6, индуцированный вершинами без смежных янов (единиц).
  |V(Γ₆)| = F(8) = 21 вершин  (φ⁸/√5 ≈ 21.009 → Формула Бине!)
  Распределение ян: [1, 6, 10, 4] = [C(7,0), C(6,1), C(5,2), C(4,3)]
  Ключевое: ratio C(5,2)/C(6,1) = 10/6 = 5/3 = F(5)/F(4) ≈ φ!

Связь K7×K5 (Германова упаковка × Фибоначчи):
  Кольцо Германа (K5): ring[h] — перестановка {1..64}.
  В позициях Γ₆: среднее ring = 30.29 < 32.5 (равномерное).
  4 из 21 Фибоначчи-гексаграмм имеют Фибоначчиево значение ring[h].

Связь K7×K4 (Фибоначчи × Биология):
  21 = F(8) = число аминокислот (20 стандартных + стоп-кодон).
  Совпадение: Γ₆ ⊂ Q6 и генетический код оба используют 21 символ!

Команды CLI:
  fibonacci  [--from-ring]  — Γ₆ анализ (φ, F(8)=21, ян-распределение)
"""

from __future__ import annotations
import json
import sys
import argparse
from math import sqrt, log2
from collections import Counter

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, hamming, neighbors
from projects.hexvis.hexvis import _YANG_ANSI, _RESET, _BOLD


# ---------------------------------------------------------------------------
# Вспомогательные: Фибоначчи и φ
# ---------------------------------------------------------------------------

_PHI = (1 + sqrt(5)) / 2  # Золотое сечение = 1.6180...

def _fibonacci_up_to(limit: int) -> list[int]:
    """Числа Фибоначчи ≤ limit (без повторений F(1)=F(2)=1)."""
    fibs = []
    a, b = 1, 1
    while a <= limit:
        if not fibs or a != fibs[-1]:
            fibs.append(a)
        a, b = b, a + b
    return fibs


def _fibonacci_cube_vertices() -> list[int]:
    """
    Вершины Γ₆ — гексаграммы без смежных янов (1-битов).
    Условие: для всех i в {0..4}: NOT (bit_i AND bit_{i+1}).
    """
    return [
        h for h in range(64)
        if not any((h >> i) & 1 and (h >> (i + 1)) & 1 for i in range(5))
    ]


def _fibonacci_cube_edges(vertices: list[int]) -> list[tuple[int, int]]:
    """Рёбра Γ₆: пары (u,v) ∈ Γ₆ с hamming(u,v)=1."""
    vset = set(vertices)
    edges = []
    for u in vertices:
        for v in neighbors(u):
            if v in vset and u < v:
                edges.append((u, v))
    return edges


# ---------------------------------------------------------------------------
# 1. JSON: Fibonacci cube в Q6 (SC-7 шаг 2)
# ---------------------------------------------------------------------------

def json_fibonacci_q6(ring_data: dict | None = None) -> dict:
    """
    SC-7 шаг 2: анализ Фибоначчи-куба Γ₆ в Q6.

    K7 (Фибоначчи/φ):
      Γ₆ ⊂ Q6 = 21 = F(8) вершин → φ⁸/√5 ≈ 21.009 (Бине).
      Ян-слои [1,6,10,4]: соотношение 10/6 = 5/3 = F(5)/F(4) ≈ φ.

    K5 (Германова упаковка), если ring_data передан:
      Среднее ring-значение на вершинах Γ₆ vs глобальное среднее 32.5.
      Фибоначчиевы значения ring[h] при h ∈ Γ₆.
    """
    phi = _PHI
    vertices = _fibonacci_cube_vertices()
    edges = _fibonacci_cube_edges(vertices)
    n_v = len(vertices)  # 21 = F(8)
    n_e = len(edges)

    # Числа Фибоначчи
    fibs_64 = _fibonacci_up_to(64)
    fibs_set = set(fibs_64)

    # Ян-распределение Γ₆
    yang_dist = Counter(yang_count(h) for h in vertices)
    yang_slices = [yang_dist.get(k, 0) for k in range(7)]
    # Отношения: yang=2/yang=1 = 10/6 = 5/3 ≈ φ
    # Fibonacci ratios F(n+1)/F(n): 2/1=2.0, 3/2=1.5, 5/3=1.667, 8/5=1.6, ...
    _fib_pairs = [(1,2),(2,3),(3,5),(5,8),(8,13),(13,21),(21,34),(34,55)]
    _fib_ratio_vals = [b/a for a, b in _fib_pairs]
    _fib_ratio_strs = [f'F(n+1)/F(n)={b}/{a}' for a, b in _fib_pairs]

    phi_ratios = []
    for k in range(1, 4):
        if yang_slices[k - 1] > 0:
            ratio = yang_slices[k] / yang_slices[k - 1]
            # Check exact match with any Fibonacci ratio (up to rounding)
            fib_match = None
            for rv, rs in zip(_fib_ratio_vals, _fib_ratio_strs):
                if abs(ratio - rv) < 0.001:
                    fib_match = rs
                    break
            phi_ratios.append({
                'yang_ratio': f'yang={k} / yang={k-1}',
                'numerator': yang_slices[k],
                'denominator': yang_slices[k - 1],
                'value': round(ratio, 6),
                'is_fib_ratio': fib_match is not None,
                'fib_match': fib_match,
            })

    # Binet formula: φ^8 / √5 ≈ 21
    binet_8 = phi ** 8 / sqrt(5)

    # Аминокислотное совпадение
    aa_coincidence = {
        'amino_acids_in_genetic_code': 21,
        'fibonacci_cube_vertices': n_v,
        'match': n_v == 21,
        'both_use_21': True,
        'interpretation': (
            '21 = F(8): Γ₆ ⊂ Q6 имеет ровно столько же вершин, '
            'сколько символов в генетическом коде (K4×K7 совпадение).'
        ),
    }

    # K5: анализ ring-значений на вершинах Γ₆
    ring_analysis: dict | None = None
    if ring_data:
        ring = ring_data.get('ring', [])
        if len(ring) == 64:
            fib_ring_values = [ring[h] for h in vertices]
            n_fib_val = sum(1 for v in fib_ring_values if v in fibs_set)
            mean_ring_fib = sum(fib_ring_values) / len(fib_ring_values)
            mean_ring_all = sum(ring) / 64  # = 32.5
            ring_analysis = {
                'n_vertices': n_v,
                'mean_ring_at_gamma6': round(mean_ring_fib, 4),
                'mean_ring_global': round(mean_ring_all, 4),
                'ring_values_at_gamma6': fib_ring_values,
                'n_fibonacci_ring_values': n_fib_val,
                'expected_fibonacci_random': round(len(fibs_64) / 64 * n_v, 2),
                'ring_bias': round(mean_ring_fib - mean_ring_all, 4),
                'k5_k7_finding': (
                    f'Γ₆-вершины: среднее ring={mean_ring_fib:.2f} '
                    f'(глобальное: {mean_ring_all:.1f}, Δ={mean_ring_fib - mean_ring_all:+.2f}). '
                    f'{n_fib_val} из {n_v} ring[h] ∈ Fib. '
                    f'«Фибоначчиевы» гексаграммы (без смежных янов) упакованы '
                    f'чуть плотнее центральных — возможно, граничные зоны кольца.'
                ),
            }

    # Структура Γ₆
    gamma6_structure = {
        'n_vertices': n_v,
        'n_edges': n_e,
        'diameter': 6,
        'is_induced_subgraph_of_q6': True,
        'yang_distribution': yang_slices[:4],
        'yang_labels': ['C(7,0)=1', 'C(6,1)=6', 'C(5,2)=10', 'C(4,3)=4'],
        'vertices': vertices,
    }

    # Сводка φ-чисел в Q6
    phi_facts = {
        'phi': round(phi, 6),
        '1_over_phi': round(1 / phi, 6),
        'phi_squared': round(phi ** 2, 6),
        'binet_f8': round(binet_8, 6),
        'binet_f8_int': 21,
        'binet_matches_gamma6': abs(binet_8 - n_v) < 0.1,
        'fibonacci_up_to_64': fibs_64,
        'f8_equals_n_amino_acids': True,
    }

    sc7_finding = (
        f'K7×K5: Fibonacci cube Γ₆ ⊂ Q6 имеет {n_v} = F(8) вершин '
        f'(φ⁸/√5 = {binet_8:.4f} ≈ {n_v}). '
        f'Ян-слои Γ₆: [1,6,10,4]; соотношение 10/6 = 5/3 = F(5)/F(4) ≈ φ = {phi:.4f}. '
        f'K4-совпадение: {n_v} = F(8) = число символов генетического кода (20 АК + стоп). '
        f'K5: Γ₆-гексаграммы в кольце Германа: Δmean = {ring_analysis["ring_bias"]:+.2f}' if ring_analysis else
        f'K7×K5: Fibonacci cube Γ₆ ⊂ Q6 имеет {n_v} = F(8) вершин '
        f'(φ⁸/√5 = {binet_8:.4f} ≈ {n_v}). '
        f'Ян-слои Γ₆: [1,6,10,4]; соотношение 10/6 = 5/3 ≈ φ.'
    )

    return {
        'command': 'fibonacci',
        'phi': round(phi, 6),
        'gamma6_structure': gamma6_structure,
        'phi_ratios_in_yang': phi_ratios,
        'phi_facts': phi_facts,
        'amino_acid_coincidence': aa_coincidence,
        'ring_analysis': ring_analysis,
        'sc7_finding': sc7_finding,
    }


# ---------------------------------------------------------------------------
# Визуализация: Γ₆ в сетке 8×8
# ---------------------------------------------------------------------------

def render_fibonacci_grid(color: bool = True) -> str:
    """Сетка 8×8: отметить вершины Γ₆ (★) и остальные (·)."""
    _GRAY3 = [i ^ (i >> 1) for i in range(8)]
    gamma6 = set(_fibonacci_cube_vertices())

    _FIB_COLOR = '\033[38;5;220m'   # золотой — Фибоначчи вершина
    _OTHER_COLOR = '\033[38;5;238m'  # серый — не-Фибоначчи

    lines = ['═' * 50,
             '  SC-7: Fibonacci cube Γ₆ ⊂ Q6',
             f'  Вершины Γ₆ (★): 21 = F(8)  (φ⁸/√5 ≈ 21)',
             f'  Остальные  (·): {64 - 21} вершины',
             '═' * 50]

    col_hdr = '  '.join(format(g, '03b') for g in _GRAY3)
    lines.append(f'        {col_hdr}')
    lines.append('        ' + '─' * len(col_hdr))

    for row_g in _GRAY3:
        cells = []
        for col_g in _GRAY3:
            h = (row_g << 3) | col_g
            in_gamma = h in gamma6
            sym = '★' if in_gamma else '·'
            if color:
                c = _FIB_COLOR if in_gamma else _OTHER_COLOR
                cell = f'{c}{sym}{_RESET}'
            else:
                cell = sym
            cells.append(cell)
        lines.append(f'  {format(row_g, "03b")} │ ' + '  '.join(cells))

    lines.append('')
    lines.append(f'  ★ = Γ₆ вершина (нет смежных янов)  · = остальные')
    lines.append(f'  Ян-распределение Γ₆: ян=0→1, ян=1→6, ян=2→10, ян=3→4')
    lines.append(f'  Отношение 10/6 = 5/3 ≈ φ = 1.618...')
    lines.append(f'  21 вершин = F(8) = количество аминокислот в генетическом коде')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog='phi_glyphs',
        description='SC-7: Числа Фибоначчи и φ в Q6 (Fibonacci cube Γ₆)',
    )
    p.add_argument('--json', action='store_true',
                   help='Машиночитаемый JSON-вывод (для пайплайнов SC-7)')
    p.add_argument('--from-ring', action='store_true',
                   help='Читать hexpack:ring JSON из stdin (K5→K7 пайплайн)')
    p.add_argument('--no-color', action='store_true')
    sub = p.add_subparsers(dest='cmd')

    # fibonacci — SC-7 шаг 2
    sub.add_parser('fibonacci',
                   help='Fibonacci cube Γ₆ ⊂ Q6: F(8)=21, φ-соотношения → JSON')
    sub.add_parser('grid',
                   help='Сетка 8×8: вершины Γ₆ (★) и остальные (·)')

    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'fibonacci':
        ring_data: dict | None = None
        if args.from_ring:
            raw = sys.stdin.read().strip()
            ring_data = json.loads(raw)
        result = json_fibonacci_q6(ring_data)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            phi = result['phi']
            g6 = result['gamma6_structure']
            pf = result['phi_facts']
            print(f'  SC-7: Fibonacci cube Γ₆ ⊂ Q6')
            print(f'  φ = {phi}   φ⁸/√5 = {pf["binet_f8"]} ≈ F(8) = {pf["binet_f8_int"]}')
            print(f'  |V(Γ₆)| = {g6["n_vertices"]}  |E(Γ₆)| = {g6["n_edges"]}  diam = {g6["diameter"]}')
            print(f'  Ян-слои: {g6["yang_distribution"]}  = {g6["yang_labels"]}')
            print()
            for r in result['phi_ratios_in_yang']:
                note = ''
                if r['is_fib_ratio']:
                    note = f'  = {r["fib_match"]} ≈ φ!'
                elif r['value'] > 1:
                    note = f'  (→φ={phi:.4f} через Binet)'
                print(f'  {r["yang_ratio"]}: {r["value"]}{note}')
            print()
            aa = result['amino_acid_coincidence']
            print(f'  Совпадение с генетическим кодом: {aa["amino_acids_in_genetic_code"]} АК = {aa["fibonacci_cube_vertices"]} = F(8) ✓')
            if result['ring_analysis']:
                ra = result['ring_analysis']
                print(f'  K5: Γ₆ среднее ring = {ra["mean_ring_at_gamma6"]}  (глобальное: {ra["mean_ring_global"]})')
            print()
            print(f'  SC-7: {result["sc7_finding"][:100]}...')
        return

    if args.cmd == 'grid' or args.cmd is None:
        print(render_fibonacci_grid(color=color))
        return

    p.print_help()


if __name__ == '__main__':
    main()
