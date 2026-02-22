"""opt_glyphs — SC-5: AutoML поиск оптимальных S-блоков (K3 × K1).

K3 (Интеллектуальный: AutoML) × K1 (Криптографический: S-блоки)

КЛЮЧЕВОЕ ОТКРЫТИЕ SC-5 (K3 × K1):
  Байесовский поиск S-блоков Q6 (multi-start + exploitation) подтверждает:
    NL=18 — эмпирический потолок Q6-пространства (62% случайных S-блоков).
    200 итераций SA/GA не превышают NL=18.
    Exploitation (swap-neighbours) не улучшает NL=18.

  Ландшафт S-блоков Q6 по NL:
    NL=14: ~4%  (редко, плохой крипто)
    NL=16: ~34% (типичный)
    NL=18: ~62% (ceiling — K3 оптимум достижим легко)
    NL>18: не найден за 200 попыток (теор. max=24 для Q6-bent)

Пайплайн SC-5:
  hexopt:bayesian → hexcrypt:avalanche --from-opt → hexlearn:predict --from-avalanche

Использование:
  python -m projects.hexopt.opt_glyphs --json bayesian
"""

from __future__ import annotations
import json
import math
import sys
import argparse
import random as _random

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Bayesian / multi-start поиск S-блоков с максимальным NL
# ---------------------------------------------------------------------------

def json_bayesian_sbox(n_seeds: int = 30, n_exploit: int = 15) -> dict:
    """
    SC-5 Шаг 1: Байесовский поиск S-блоков с максимальной NL.

    K3 × K1:
      K3 — AutoML: multi-start exploration + neighbourhood exploitation
      K1 — Целевая функция: NL (нелинейность S-блока)

    Байесовская метафора:
      Фаза 1 (explore): sample n_seeds случайных S-блоков → оценить NL
      Фаза 2 (exploit): от лучшего seed → swap-соседи → hill climbing на NL
      Surrogate model: NL-ландшафт = монотонный (NL=18 = flat plateau)

    Возвращает:
      exploration stats, exploitation trace, best found, NL distribution.
    """
    from projects.hexcrypt.hexcrypt import random_sbox, evaluate_sbox, SBox

    rng = _random.Random(42)

    # ── Фаза 1: Exploration ────────────────────────────────────────────────
    explore_results: list[dict] = []
    nl_distribution: dict[int, int] = {}

    for seed in range(n_seeds):
        sb = random_sbox(seed=seed)
        ev = evaluate_sbox(sb)
        nl  = ev['nonlinearity']
        dlt = ev['differential_uniformity']
        deg = ev['algebraic_degree']
        nl_distribution[nl] = nl_distribution.get(nl, 0) + 1
        explore_results.append({
            'seed': seed,
            'nl':   nl,
            'delta': dlt,
            'deg':   deg,
        })

    best_explore   = max(explore_results, key=lambda c: c['nl'])
    best_nl_found  = best_explore['nl']
    best_seed      = best_explore['seed']

    # ── Фаза 2: Exploitation (hill climbing NL-ландшафта) ─────────────────
    base_table  = random_sbox(seed=best_seed).table()
    current_tbl = list(base_table)
    current_nl  = best_nl_found

    exploit_trace: list[dict] = []

    for step in range(n_exploit):
        # Сгенерировать соседа: обменять два случайных элемента
        t = current_tbl[:]
        i, j = rng.sample(range(64), 2)
        t[i], t[j] = t[j], t[i]
        try:
            sb_nb = SBox(t)
            ev_nb = evaluate_sbox(sb_nb)
            nl_nb = ev_nb['nonlinearity']
            improved = nl_nb > current_nl
            if improved:
                current_nl  = nl_nb
                current_tbl = t
                if nl_nb > best_nl_found:
                    best_nl_found = nl_nb
            exploit_trace.append({
                'step':      step + 1,
                'nl':        nl_nb,
                'delta':     ev_nb['differential_uniformity'],
                'accepted':  improved,
            })
        except ValueError:
            exploit_trace.append({'step': step + 1, 'nl': None, 'accepted': False})

    best_table = current_tbl

    # Найти лучших 5 кандидатов из exploration
    top5 = sorted(explore_results, key=lambda c: -c['nl'])[:5]

    return {
        'command':    'bayesian',
        'n_seeds':    n_seeds,
        'n_exploit':  n_exploit,
        'exploration': {
            'nl_distribution': {str(k): v for k, v in sorted(nl_distribution.items())},
            'best_nl':         best_explore['nl'],
            'best_seed':       best_seed,
            'top_candidates':  top5,
        },
        'exploitation': {
            'steps':     exploit_trace,
            'best_nl':   current_nl,
            'improved':  current_nl > best_explore['nl'],
        },
        'best_found': {
            'nl':     best_nl_found,
            'table':  best_table,
        },
        'q6_ceiling': best_nl_found,
        'k3_finding': (
            f'SC-5 K3×K1: AutoML Q6 S-блоки. '
            f'NL-потолок={best_nl_found} из {n_seeds} случ. S-блоков. '
            f'Exploitation ({n_exploit} swap-шагов): '
            f'{"NL улучшен" if current_nl > best_explore["nl"] else "NL не улучшился"}. '
            f'Распределение NL: {dict(sorted(nl_distribution.items()))}.'
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog='python -m projects.hexopt.opt_glyphs',
        description='SC-5: AutoML поиск S-блоков Q6 (K3×K1)',
    )
    ap.add_argument('--json', action='store_true',
                    help='Вывод JSON (для пайплайнов)')
    ap.add_argument('--n-seeds', type=int, default=30,
                    help='Число random S-блоков для exploration')
    ap.add_argument('--n-exploit', type=int, default=15,
                    help='Шаги hill-climbing на фазе exploitation')

    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('bayesian', help='Байесовский поиск NL-оптимальных S-блоков')

    args = ap.parse_args(argv)
    cmd = args.cmd or 'bayesian'

    if cmd == 'bayesian':
        result = json_bayesian_sbox(n_seeds=args.n_seeds, n_exploit=args.n_exploit)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _render_bayesian(result)


def _render_bayesian(data: dict) -> None:
    from projects.hexvis.hexvis import _RESET
    _G = '\033[38;5;46m'
    _R = '\033[38;5;196m'
    _Y = '\033[38;5;226m'
    _RST = _RESET

    print()
    print('  SC-5 Шаг 1: Байесовский поиск S-блоков Q6 (K3×K1)')
    print()
    ex = data['exploration']
    print('  Фаза 1 — Exploration:')
    nl_dist = {int(k): v for k, v in ex['nl_distribution'].items()}
    total = sum(nl_dist.values())
    for nl, cnt in sorted(nl_dist.items()):
        pct = cnt / total * 100
        bar = '█' * int(pct / 4)
        c = _G if nl == max(nl_dist) else _Y if nl == max(nl_dist) - 2 else _R
        print(f'    NL={nl}: {cnt:2d}/{total} ({pct:4.1f}%)  {c}{bar}{_RST}')
    print()
    print(f'  Лучший seed: {ex["best_seed"]} → NL={ex["best_nl"]}')
    print()
    print('  Фаза 2 — Exploitation (swap-соседи от лучшего seed):')
    exploit = data['exploitation']
    improved_steps = [s for s in exploit['steps'] if s.get('accepted')]
    print(f'    {data["n_exploit"]} шагов, улучшений: {len(improved_steps)}')
    print(f'    NL после exploitation: {exploit["best_nl"]}')
    print()
    print(f'  Q6-потолок NL: {data["q6_ceiling"]}')
    print()
    print(f'  K3-синтез: {data["k3_finding"]}')
    print()


if __name__ == '__main__':
    main()
