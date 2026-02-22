"""code_glyphs — SC-2: Платиновые S-блоки через теорию кодирования Q6.

K1 (Криптографический) × K8 (Алгебраический)

КЛЮЧЕВОЕ ОТКРЫТИЕ SC-2 (K1 × K8):
  Каждый S-блок f:Q6→Q6 определяет «граф-код» C_f ⊂ GF(2)^12:
    C_f = { (x ‖ f(x)) : x ∈ Q6 }   — 64 кодовых слова длины 12

  K8-теорема (антиплатиновый барьер):
    [12,6,7]-код (MDS, Singleton) НЕДОСТИЖИМ для биекций Q6.
    Доказательство: вес(x)=1 → нужно вес(f(x))≥6=63 для 6 различных x.
    Противоречие биективности. QED.

  Следствие: максимально достижимо d=6.
  Complement sbox f(x) = NOT(x) = 63-x — ЕДИНСТВЕННЫЙ [12,6,6]-экви-дистантный:
    ∀x≠0: вес(x)+вес(NOT(x)) = вес(x)+(6-вес(x)) = 6.

  K1-крипто: NL=0 (линейный!) — слабый. Трейдофф K1×K8:
    NL=18, d=2  (random_42)    — крипто-сильный, код-слабый
    NL= 0, d=6  (complement)   — код-оптимальный, крипто-слабый
    «Платиновый» Q6 S-блок = баланс обоих критериев (не достигнут стандартными)

Пайплайн SC-2:
  hexcrypt:sbox → karnaugh6:sbox-minimize --from-sbox → hexcode:sbox-code --from-minimize

Использование:
  python -m projects.hexcode.code_glyphs --json --from-minimize sbox-code
  python -m projects.hexcode.code_glyphs sbox-code
"""

from __future__ import annotations
import json
import math
import sys
import argparse
from collections import Counter

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, SIZE
from projects.hexvis.hexvis import _YANG_ANSI, _RESET, _BOLD


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_SBOX_NAMES = ['identity', 'bit_reversal', 'affine', 'complement', 'yang_sort', 'random_42']
_SINGLETON_BOUND_12_6 = 7   # d ≤ n - k + 1 = 12 - 6 + 1

# Веса для "платинового" рейтинга
_W_NL    = 0.5   # нелинейность (K1)
_W_DELTA = 0.3   # дифф. равномерность: меньше = лучше → инвертируем
_W_DIST  = 0.2   # дистанция граф-кода (K8)

_NL_MAX   = 24.0  # теоретический максимум NL для n=6 (bent function bound)
_DELTA_ID = 64.0  # дельта идентичного S-блока (максимум)


# ---------------------------------------------------------------------------
# Граф-код S-блока
# ---------------------------------------------------------------------------

def _graph_code_min_distance(table: list[int]) -> int:
    """
    Минимальное расстояние Хэмминга в граф-коде C_f = {(x‖f(x))}.
    Кодовое слово: w(x) = 12-битное число (x в битах 0-5, f(x) в битах 6-11).
    Вес кодового слова = weight(x) + weight(f(x)) (расстояние Хэмминга от 0).
    """
    min_d = 12
    for x in range(64):
        if x == 0 and table[0] == 0:
            continue  # нулевое кодовое слово пропускаем
        cw = x | (table[x] << 6)
        if cw == 0:
            continue
        w = bin(cw).count('1')
        if w < min_d:
            min_d = w
    return min_d


def _graph_code_weight_distribution(table: list[int]) -> dict[int, int]:
    """Распределение весов кодовых слов граф-кода."""
    dist: Counter[int] = Counter()
    for x in range(64):
        cw = x | (table[x] << 6)
        w = bin(cw).count('1')
        dist[w] += 1
    return dict(sorted(dist.items()))


def _is_equidistant(wd: dict[int, int]) -> bool:
    """Эквидистантный код: все ненулевые слова имеют одинаковый вес."""
    non_zero_weights = [w for w, cnt in wd.items() if w > 0]
    return len(set(non_zero_weights)) == 1


def _platinum_score(nl: int, delta: int, code_d: int) -> float:
    """
    Взвешенная «платиновая» оценка S-блока:
      score = 0.5 * nl/24 + 0.3 * (1 - delta/64) + 0.2 * d/6
    """
    s_nl    = nl / _NL_MAX
    s_delta = 1.0 - delta / _DELTA_ID
    s_dist  = code_d / 6.0   # 6 = максимально достижимое
    return _W_NL * s_nl + _W_DELTA * s_delta + _W_DIST * s_dist


# ---------------------------------------------------------------------------
# Основная JSON-функция SC-2
# ---------------------------------------------------------------------------

def json_sbox_code(minimize_data: dict | None = None) -> dict:
    """
    SC-2: Анализ S-блоков как граф-кодов [12,6,d].

    K1 × K8:
      K1 — криптографические свойства (NL, delta, deg, complexity)
      K8 — кодово-теоретический анализ (min distance, equidistant, MDS bound)

    Аргументы:
      minimize_data: dict из karnaugh6:sbox-minimize (содержит 'table')
                     Если None — использовать affine sbox по умолчанию.

    Возвращает:
      dict с comprehensive_analysis, platinum_ranking, k8_theorem.
    """
    # ── Загрузить все S-блоки ──────────────────────────────────────────────
    from projects.hexcrypt.hexcrypt import (
        identity_sbox, bit_reversal_sbox, affine_sbox,
        complement_sbox, random_sbox, yang_sort_sbox, evaluate_sbox,
    )
    from projects.hexcrypt.sbox_glyphs import json_analyze
    from projects.karnaugh6.kmap_glyphs import json_sbox_minimize

    sbox_factories = {
        'identity':    identity_sbox,
        'bit_reversal': bit_reversal_sbox,
        'affine':      affine_sbox,
        'complement':  complement_sbox,
        'yang_sort':   yang_sort_sbox,
        'random_42':   lambda: random_sbox(seed=42),
    }

    # ── Определить «текущий» S-блок из minimize_data ──────────────────────
    input_table: list[int] | None = None
    input_name = 'affine'
    input_literals = None
    if minimize_data is not None:
        if 'table' in minimize_data:
            input_table = minimize_data['table']
            input_literals = minimize_data.get('total_literals')
            # Попробовать угадать имя по литералам
            if input_literals == 6:
                input_name = 'linear-6lit'
            else:
                input_name = 'from-minimize'

    # ── Анализ всех стандартных S-блоков ──────────────────────────────────
    all_analyses: list[dict] = []

    for name, factory in sbox_factories.items():
        sb = factory()
        table = sb.table()
        ev = evaluate_sbox(sb)
        nl    = ev['nonlinearity']
        delta = ev['differential_uniformity']
        deg   = ev['algebraic_degree']

        # Граф-код
        min_d = _graph_code_min_distance(table)
        wd    = _graph_code_weight_distribution(table)
        equi  = _is_equidistant(wd)

        # Karnaugh сложность
        min_result = json_sbox_minimize(json_analyze(sb, name))
        lits = min_result['total_literals']
        n_linear_comps = min_result['n_linear_components']

        # Платиновая оценка
        score = _platinum_score(nl, delta, min_d)

        all_analyses.append({
            'name':                 name,
            'k1_crypto': {
                'nonlinearity':             nl,
                'differential_uniformity':  delta,
                'algebraic_degree':         deg,
                'is_apn':                   ev.get('is_apn', False),
            },
            'k8_code': {
                'graph_code_params': f'[12,6,{min_d}]',
                'min_distance':      min_d,
                'weight_distribution': wd,
                'is_equidistant':    equi,
                'achieves_mds':      min_d == _SINGLETON_BOUND_12_6,
            },
            'k1_circuit': {
                'total_literals':      lits,
                'n_linear_components': n_linear_comps,
            },
            'platinum_score': round(score, 4),
        })

    # Сортировка по платиновой оценке
    all_analyses.sort(key=lambda x: x['platinum_score'], reverse=True)

    # ── Специальный анализ текущего S-блока из minimize_data ──────────────
    current_analysis: dict | None = None
    if input_table is not None:
        min_d_cur = _graph_code_min_distance(input_table)
        wd_cur    = _graph_code_weight_distribution(input_table)
        current_analysis = {
            'name':         input_name,
            'total_literals': input_literals,
            'graph_code_params': f'[12,6,{min_d_cur}]',
            'min_distance': min_d_cur,
            'weight_distribution': wd_cur,
            'is_equidistant': _is_equidistant(wd_cur),
        }

    # ── K8-теорема (антиплатиновый барьер) ────────────────────────────────
    k8_theorem = {
        'statement': (
            'MDS [12,6,7] НЕВОЗМОЖЕН для биекций Q6: '
            '6 весо-1 входов x требовали бы f(x)=63 (весо-6) → противоречие биективности'
        ),
        'singleton_bound': _SINGLETON_BOUND_12_6,
        'max_achievable_d': 6,
        'achiever': 'complement (f(x)=NOT(x)=63-x)',
        'equidistant_proof': (
            'Complement: вес(x)+вес(NOT(x)) = вес(x)+(6-вес(x)) = 6 для всех x≠0. '
            'Все 63 ненулевых кодовых слова имеют вес 6 → эквидистантный [12,6,6].'
        ),
        'mds_impossibility_proof': (
            'Для d=7 нужно: ∀x≠0: вес(x)+вес(f(x)) ≥ 7. '
            'При вес(x)=1: нужно вес(f(x))≥6 → f(x)=63. '
            '6 различных весо-1 входов → f(x)=63 для 6 разных x. '
            'Нарушает биективность (63 — единственное весо-6 слово в Q6). QED.'
        ),
    }

    # ── Находка K1×K8 ─────────────────────────────────────────────────────
    best   = all_analyses[0]
    worst  = all_analyses[-1]
    complement_entry = next(a for a in all_analyses if a['name'] == 'complement')
    random_entry     = next(a for a in all_analyses if a['name'] == 'random_42')

    k1_k8_finding = (
        f"SC-2 K1×K8: граф-коды S-блоков Q6. "
        f"Complement → эквидист. [12,6,6] (макс. d, NL=0); "
        f"random_42 → [12,6,2] (лучший крипто NL={random_entry['k1_crypto']['nonlinearity']}, бедный код). "
        f"MDS d=7 математически невозможен для Q6-биекций. "
        f"«Платиновый» рейтинг: {best['name']} (score={best['platinum_score']:.4f})."
    )

    return {
        'command':            'sbox_code',
        'analysis':           all_analyses,
        'current_input':      current_analysis,
        'best_platinum':  {
            'name':           best['name'],
            'score':          best['platinum_score'],
            'nl':             best['k1_crypto']['nonlinearity'],
            'delta':          best['k1_crypto']['differential_uniformity'],
            'code_d':         best['k8_code']['min_distance'],
        },
        'complement_highlight': {
            'name':           'complement',
            'nl':             complement_entry['k1_crypto']['nonlinearity'],
            'code_d':         complement_entry['k8_code']['min_distance'],
            'is_equidistant': complement_entry['k8_code']['is_equidistant'],
            'literals':       complement_entry['k1_circuit']['total_literals'],
        },
        'k8_theorem':         k8_theorem,
        'singleton_bound':    _SINGLETON_BOUND_12_6,
        'max_achievable_d':   6,
        'k1_k8_finding':      k1_k8_finding,
        'sc_id':              'SC-2',
        'clusters':           ['K1', 'K8'],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DISPATCH: dict = {
    'sbox-code': lambda args, from_data: json_sbox_code(from_data),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog='python -m projects.hexcode.code_glyphs',
        description='SC-2: Граф-коды S-блоков Q6 (K1×K8)',
    )
    ap.add_argument('--json', action='store_true',
                    help='Вывести результат как JSON')
    ap.add_argument('--from-minimize', action='store_true',
                    help='Читать karnaugh6:sbox-minimize JSON из stdin')
    ap.add_argument('--no-color', action='store_true')

    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('sbox-code', help='Анализ S-блоков как граф-кодов [12,6,d]')

    args = ap.parse_args(argv)
    color = not args.no_color

    from_data: dict | None = None
    if args.from_minimize:
        raw = sys.stdin.read()
        try:
            from_data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f'Ошибка: не удалось разобрать stdin JSON: {e}', file=sys.stderr)
            sys.exit(1)

    cmd = args.cmd or 'sbox-code'
    if cmd not in _DISPATCH:
        ap.print_help()
        return

    result = _DISPATCH[cmd](args, from_data)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _render_human(result, color)


def _render_human(data: dict, color: bool = True) -> None:
    """Человекочитаемый вывод SC-2."""
    _R = '\033[38;5;196m' if color else ''
    _G = '\033[38;5;46m'  if color else ''
    _Y = '\033[38;5;226m' if color else ''
    _B = '\033[38;5;27m'  if color else ''
    _RST = _RESET if color else ''

    print()
    print('  SC-2: Платиновые S-блоки — Граф-коды Q6')
    print('  K1 (крипто: NL, δ) × K8 (код: [12,6,d])')
    print()

    # Таблица
    hdr = f"  {'S-блок':<14} {'NL':>4} {'δ':>4} {'deg':>4}  {'[n,k,d]':^10} {'equi':^5} {'lits':>5}  {'score':>7}"
    sep = '  ' + '─' * 62
    print(hdr)
    print(sep)
    for a in data['analysis']:
        nc = a['k1_crypto']
        kc = a['k8_code']
        ci = a['k1_circuit']
        equi = 'ДА' if kc['is_equidistant'] else 'нет'
        d = kc['min_distance']
        if d == 6:
            c_d = _G
        elif d <= 2:
            c_d = _R
        else:
            c_d = _Y
        params_str = f"{kc['graph_code_params']:^10}"  # pad first
        colored_params = f"{c_d}{params_str}{_RST}"    # then color
        line = (
            f"  {a['name']:<14} {nc['nonlinearity']:>4} {nc['differential_uniformity']:>4} "
            f"{nc['algebraic_degree']:>4}  "
            f"{colored_params} {equi:^5} {ci['total_literals']:>5}  "
            f"{a['platinum_score']:>7.4f}"
        )
        print(line)
    print(sep)
    print()

    # K8-теорема
    th = data['k8_theorem']
    print('  K8-теорема (антиплатиновый барьер):')
    print(f'    Singleton bound [12,6,{th["singleton_bound"]}]: НЕДОСТИЖИМ для Q6-биекций')
    print(f'    Причина: 6 весо-1 входов → f(x)=63 → нарушение биективности')
    print(f'    Максимально достижимо: d={th["max_achievable_d"]} (complement sbox)')
    print()

    # Complement highlight
    ch = data['complement_highlight']
    print(f'  Complement S-блок f(x)=NOT(x)=63-x — Q6-оптимум:')
    print(f'    [12,6,{ch["code_d"]}]-код, эквидистантный: {ch["is_equidistant"]}')
    print(f'    ∀x≠0: вес(x)+вес(f(x))=6 (= n/2×2 = максимально достижимо)')
    print(f'    Сложность: {ch["literals"]} литералов (минимум!), NL={ch["nl"]} (линейный)')
    print()

    # Платиновый рейтинг
    bp = data['best_platinum']
    print(f'  Платиновый рейтинг (NL×d×δ): {bp["name"]} (score={bp["score"]:.4f})')
    print(f'    NL={bp["nl"]}, delta={bp["delta"]}, code_d={bp["code_d"]}')
    print()
    print(f'  K1×K8-синтез:')
    print(f'  {data["k1_k8_finding"]}')
    print()


if __name__ == '__main__':
    main()
