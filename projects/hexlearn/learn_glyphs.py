"""learn_glyphs — TSC-2: AutoML-крипто на Q6.

K3 (Интеллектуальный) × K2 (Динамический) × K1 (Криптографический)

КЛЮЧЕВОЕ ОТКРЫТИЕ TSC-2:
  CA-правила Q6 — естественные кандидаты для криптографических key schedule.
  ML-ранжирование по трём K1-критериям выявляет:
    • Лучшие (K1): диффузионные правила (δH>0, нет сходимости) — xor_rule, random_walk
    • Средние:     статические правила (δH≈0) — identity, conway
    • Худшие:      сходящиеся правила (δH<0) — majority_vote (→ ян=3), cyclic

  K3-вклад: взвешенная ML-оценка = 0.35·H_fin + 0.35·drift + 0.15·balance + 0.15·nocvg
  K2-вклад: CA-динамика (entropy drift, convergence, period detection)
  K1-вклад: крипто-критерии (высокая H, балансированность, непредсказуемость)

Пайплайн:
  hexca:all-rules → hexlearn:ca-rank --from-rules

Использование:
  python -m projects.hexlearn.learn_glyphs --json --from-rules ca-rank
  python -m projects.hexlearn.learn_glyphs ca-rank
"""

from __future__ import annotations
import json
import math
import sys
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import yang_count, SIZE
from projects.hexvis.hexvis import (
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_MAX_ENTROPY = math.log2(64)   # 6.0 бит — теоретический максимум для Q6

# Весовые коэффициенты для K1-оценки (сумма = 1.0)
_W_ENTROPY       = 0.35   # нормированная финальная энтропия
_W_DRIFT         = 0.35   # диффузия = положительный дрейф
_W_BALANCE       = 0.15   # ян-баланс (ближе к 3.0 = лучше)
_W_NONCONVERGE   = 0.15   # отсутствие сходимости к неподвижной точке

_CRYPTO_RANK_LABELS = {
    'excellent':  'Отлично  — диффузионный, непредсказуемый',
    'good':       'Хорошо   — умеренная диффузия',
    'neutral':    'Нейтрально — статический, предсказуемый',
    'poor':       'Слабо    — сходящийся, предсказуемый',
}

# ANSI-цвета для ранга
_RANK_COLORS = {
    'excellent': '\033[38;5;46m',   # ярко-зелёный
    'good':      '\033[38;5;118m',  # зелёный
    'neutral':   '\033[38;5;226m',  # жёлтый
    'poor':      '\033[38;5;196m',  # красный
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _normalize(value: float, lo: float, hi: float) -> float:
    """Линейная нормализация value в [0, 1]."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _drift_score(drift: float) -> float:
    """
    Оценка дрейфа энтропии для крипто (K1):
      drift > +0.5  → 1.0 (отличная диффузия)
      drift ≈ 0     → 0.5 (нейтрально)
      drift < -0.5  → 0.0 (сходимость = плохо)
    Сигмоидный сдвиг: score = 0.5 + 0.5 * tanh(drift * 2)
    """
    return 0.5 + 0.5 * math.tanh(drift * 2.0)


def _balance_score(yang_mean: float) -> float:
    """
    Оценка ян-баланса (K1): ближе к 3.0 → 1.0, крайности → 0.0.
    score = 1 - |yang_mean - 3.0| / 3.0
    """
    return max(0.0, 1.0 - abs(yang_mean - 3.0) / 3.0)


def _crypto_rank_label(score: float) -> str:
    """Строковый ранг по общей оценке."""
    if score >= 0.75:
        return 'excellent'
    elif score >= 0.55:
        return 'good'
    elif score >= 0.35:
        return 'neutral'
    else:
        return 'poor'


def _k1_interpretation(rule: str, score: float, drift: float) -> str:
    """K1-интерпретация правила для key schedule."""
    rank = _crypto_rank_label(score)
    if rank == 'excellent':
        return f'{rule}: идеальный компонент KS — высокая H, диффузия δH={drift:+.3f}'
    elif rank == 'good':
        return f'{rule}: хороший компонент KS — умеренная диффузия δH={drift:+.3f}'
    elif rank == 'neutral':
        return f'{rule}: нейтральный — статичная энтропия (δH≈0), предсказуем'
    else:
        return f'{rule}: слабый компонент KS — сходимость δH={drift:+.3f}, аттрактор'


# ---------------------------------------------------------------------------
# Основная JSON-функция TSC-2
# ---------------------------------------------------------------------------

def json_ca_crypto_rank(ca_all_data: dict | None = None) -> dict:
    """
    TSC-2: ML-ранжирование CA-правил Q6 как компонентов key schedule.

    K3 × K2 × K1:
      K3 — взвешенная ML-оценка по 4 крипто-признакам
      K2 — CA-динамика: entropy drift, convergence, period
      K1 — крипто-критерии: H_final, диффузия, ян-баланс, непредсказуемость

    Аргументы:
      ca_all_data: dict из hexca:all-rules (--json all-rules)
                   Если None — запустить собственный hexca.

    Возвращает:
      dict с ranked (список), best/worst, leaderboard, K3-синтез.
    """
    # ── Источник данных ────────────────────────────────────────────────────
    if ca_all_data is None:
        from projects.hexca.ca_glyphs import json_all_rules
        ca_all_data = json_all_rules()

    rules_raw = ca_all_data.get('rules', [])
    width  = ca_all_data.get('width', 20)
    steps  = ca_all_data.get('steps', 20)
    seed   = ca_all_data.get('seed', 42)

    # ── Вычислить оценки ───────────────────────────────────────────────────
    scored: list[dict] = []
    for r in rules_raw:
        rule_name     = r['rule']
        h_init        = r['initial_entropy']
        h_fin         = r['final_entropy']
        drift         = h_fin - h_init
        yang_fin      = r.get('final_yang_mean', 3.0)
        has_converged = r.get('convergence_step') is not None

        # Четыре признака
        s_entropy     = _normalize(h_fin, 0.0, _MAX_ENTROPY)   # H_fin / 6.0
        s_drift       = _drift_score(drift)
        s_balance     = _balance_score(yang_fin)
        s_nonconverge = 0.0 if has_converged else 1.0

        # Взвешенная K3-оценка
        total = (
            _W_ENTROPY     * s_entropy +
            _W_DRIFT       * s_drift   +
            _W_BALANCE     * s_balance +
            _W_NONCONVERGE * s_nonconverge
        )

        rank_label = _crypto_rank_label(total)

        scored.append({
            'rule':              rule_name,
            'crypto_score':      round(total, 4),
            'rank':              rank_label,
            'features': {
                'entropy_final':       round(h_fin, 4),
                'entropy_initial':     round(h_init, 4),
                'entropy_drift':       round(drift, 4),
                'yang_mean_final':     round(yang_fin, 4),
                'has_convergence':     has_converged,
                'convergence_step':    r.get('convergence_step'),
                'period_1':            r.get('period_1', False),
            },
            'component_scores': {
                'entropy':      round(s_entropy, 4),
                'drift':        round(s_drift, 4),
                'balance':      round(s_balance, 4),
                'nonconverge':  round(s_nonconverge, 4),
            },
            'k1_interpretation': _k1_interpretation(rule_name, total, drift),
        })

    # Сортировка по убыванию оценки
    scored.sort(key=lambda x: x['crypto_score'], reverse=True)

    # ── Кластеры ───────────────────────────────────────────────────────────
    excellent = [r for r in scored if r['rank'] == 'excellent']
    good      = [r for r in scored if r['rank'] == 'good']
    neutral   = [r for r in scored if r['rank'] == 'neutral']
    poor      = [r for r in scored if r['rank'] == 'poor']

    # ── K3-синтез ──────────────────────────────────────────────────────────
    best_rule  = scored[0]
    worst_rule = scored[-1]
    n_diffusive   = len([r for r in scored if r['features']['entropy_drift'] > 0.05])
    n_convergent  = len([r for r in scored if r['features']['entropy_drift'] < -0.05])
    n_static      = len(scored) - n_diffusive - n_convergent

    # Оптимальный key schedule = топ-3 правила
    top3 = [r['rule'] for r in scored[:3]]
    top3_mean_H = sum(
        r['features']['entropy_final'] for r in scored[:3]
    ) / 3

    # ── Находки SC-7 (Binet) аналогия для крипто ──────────────────────────
    k3_finding = (
        f"TSC-2 K3×K2×K1: ML-ранжирование выявило {n_diffusive} диффузионных правил "
        f"(лучших для KS), {n_static} статических (нейтральных), "
        f"{n_convergent} сходящихся (слабых). "
        f"Оптимальный KS: {' → '.join(top3)} (ср. H_fin={top3_mean_H:.3f} бит)."
    )

    weights_used = {
        'entropy':     _W_ENTROPY,
        'drift':       _W_DRIFT,
        'balance':     _W_BALANCE,
        'nonconverge': _W_NONCONVERGE,
    }

    return {
        'command':        'ca_rank',
        'ca_params':      {'width': width, 'steps': steps, 'seed': seed},
        'n_rules':        len(scored),
        'weights':        weights_used,
        'ranked':         scored,
        'leaderboard': {
            'excellent':  [r['rule'] for r in excellent],
            'good':       [r['rule'] for r in good],
            'neutral':    [r['rule'] for r in neutral],
            'poor':       [r['rule'] for r in poor],
        },
        'best':  {
            'rule':  best_rule['rule'],
            'score': best_rule['crypto_score'],
            'rank':  best_rule['rank'],
        },
        'worst': {
            'rule':  worst_rule['rule'],
            'score': worst_rule['crypto_score'],
            'rank':  worst_rule['rank'],
        },
        'optimal_key_schedule': top3,
        'optimal_ks_mean_entropy': round(top3_mean_H, 4),
        'class_counts': {
            'diffusive':   n_diffusive,
            'static':      n_static,
            'convergent':  n_convergent,
        },
        'k3_finding': k3_finding,
        'sc_id': 'TSC-2',
        'clusters': ['K3', 'K2', 'K1'],
    }


# ---------------------------------------------------------------------------
# Визуализация таблицы ранжирования
# ---------------------------------------------------------------------------

def render_crypto_rank_table(data: dict, color: bool = True) -> str:
    """Отрендерить таблицу ранжирования CA-правил для терминала."""
    lines = []
    ranked = data['ranked']

    hdr = f"  {'#':<3} {'Правило':<22} {'Оценка':>7} {'H_fin':>7} {'δH':>8} {'Ян.':>6}  {'Ранг'}"
    sep = '  ' + '─' * 68
    lines.append(hdr)
    lines.append(sep)

    for i, r in enumerate(ranked):
        f = r['features']
        rank_lbl = r['rank']
        c_start = _RANK_COLORS.get(rank_lbl, '') if color else ''
        c_end   = _RESET if color else ''
        line = (
            f"  {i+1:<3} {r['rule']:<22} {r['crypto_score']:>7.4f} "
            f"{f['entropy_final']:>7.4f} {f['entropy_drift']:>+8.4f} "
            f"{f['yang_mean_final']:>6.2f}  "
            f"{c_start}{rank_lbl}{c_end}"
        )
        lines.append(line)

    lines.append(sep)
    best  = data['best']
    worst = data['worst']
    lines.append(f"  Лучшее:  {best['rule']} (оценка={best['score']:.4f})")
    lines.append(f"  Худшее:  {worst['rule']} (оценка={worst['score']:.4f})")
    lines.append(f"  Оптим. KS: {' → '.join(data['optimal_key_schedule'])}")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DISPATCH: dict = {
    'ca-rank': lambda args, from_data: json_ca_crypto_rank(from_data),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog='python -m projects.hexlearn.learn_glyphs',
        description='TSC-2: ML-ранжирование CA-правил (K3×K2×K1)',
    )
    ap.add_argument('--json', action='store_true',
                    help='Вывести результат как JSON')
    ap.add_argument('--from-rules', action='store_true',
                    help='Читать hexca:all-rules JSON из stdin')
    ap.add_argument('--no-color', action='store_true',
                    help='Отключить цвет')

    sub = ap.add_subparsers(dest='cmd')

    # ca-rank
    p_cr = sub.add_parser('ca-rank',
                           help='ML-ранжирование CA-правил по крипто-пригодности')
    p_cr.add_argument('--width', type=int, default=20,
                      help='Ширина CA (если не --from-rules)')
    p_cr.add_argument('--steps', type=int, default=20,
                      help='Шагов CA (если не --from-rules)')
    p_cr.add_argument('--seed', type=int, default=42,
                      help='Сид генератора')

    args = ap.parse_args(argv)
    color = not args.no_color

    # Читать данные из stdin (если --from-rules)
    from_data: dict | None = None
    if args.from_rules:
        raw = sys.stdin.read()
        try:
            from_data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f'Ошибка: не удалось разобрать stdin JSON: {e}', file=sys.stderr)
            sys.exit(1)

    cmd = args.cmd or 'ca-rank'

    if cmd not in _DISPATCH:
        ap.print_help()
        return

    result = _DISPATCH[cmd](args, from_data)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # Человекочитаемый вывод
        data = result
        print()
        print('  TSC-2: AutoML-крипто — ML-ранжирование CA Q6')
        print('  K3 (ML-оценка) × K2 (CA-динамика) × K1 (крипто)')
        print()
        print(render_crypto_rank_table(data, color=color))
        print()
        # Классы
        lb = data['leaderboard']
        print('  Классы (K1-пригодность для key schedule):')
        if lb['excellent']:
            print(f'    Отлично:    {", ".join(lb["excellent"])}')
        if lb['good']:
            print(f'    Хорошо:     {", ".join(lb["good"])}')
        if lb['neutral']:
            print(f'    Нейтрально: {", ".join(lb["neutral"])}')
        if lb['poor']:
            print(f'    Слабо:      {", ".join(lb["poor"])}')
        print()
        print(f'  K3-синтез: {data["k3_finding"]}')
        print()


if __name__ == '__main__':
    main()
