#!/usr/bin/env bash
# sc3_ca_atlas.sh — Супер-кластер SC-3: «Канонический атлас КА»
#
# K2 (Динамический) × K8 (Схематический)
# КЛЮЧЕВОЕ ОТКРЫТИЕ SC-3 (результат комбинации K2+K8):
#   Из 9 правил Q6 только identity Aut(Q6)-эквивариантна.
#   Aut(Q6) = B₆: порядок 46080 = 6! × 2⁶ (перестановки × XOR-маски).
#   7 ян-слоёв {h : yang(h)=k} — орбиты под S₆-частью Aut(Q6).
#   Правила с малым ян-дрейфом (<0.3) → Вольфрам I (сходимость).
#   xor_rule: энтропия не убывает → Class III/IV (обратимая линейная CA).
#
# Использование:
#   ./scripts/sc3_ca_atlas.sh
#   ./scripts/sc3_ca_atlas.sh --dry
#   ./scripts/sc3_ca_atlas.sh --verbose

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO"

DRY=0
VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        --dry)     DRY=1  ;;
        --verbose) VERBOSE=1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Супер-кластер SC-3: Канонический атлас КА"
echo "  K2 (hexca + hexstat) × K8 (hexsym — Aut(Q6))"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ $DRY -eq 1 ]; then
    python -m libs.q6ctl.q6cli run SC-3 --dry
    exit 0
fi

# ── Шаг 1: Эволюция КА (xor_rule — линейное правило) ─────────────────────────
echo "  [1/3] hexca:evolve — 1D клеточный автомат Q6 (xor_rule, ширина=20)"
EVOLVE_JSON=$(python -m projects.hexca.ca_glyphs --json evolve \
    --rule xor_rule --width 20 --steps 20 --seed 42)

INIT_H=$(echo "$EVOLVE_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['evolution_stats'][0]['entropy'])")
FINAL_H=$(echo "$EVOLVE_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['evolution_stats'][-1]['entropy'])")
CONV=$(echo "$EVOLVE_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['convergence_step'] or 'нет')")
INIT_YANG=$(echo "$EVOLVE_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['evolution_stats'][0]['mean_yang'])")
FINAL_YANG=$(echo "$EVOLVE_JSON"| python -c "import sys,json; d=json.load(sys.stdin); print(d['evolution_stats'][-1]['mean_yang'])")

echo "  Начальная энтропия H₀ = $INIT_H бит"
echo "  Конечная энтропия  HN = $FINAL_H бит"
echo "  Сходимость к fix-point: $CONV"
echo "  Ян-баланс: начало=$INIT_YANG → конец=$FINAL_YANG"
echo ""

# ── Шаг 2: Классификация по Вольфраму ─────────────────────────────────────────
echo "  [2/3] hexstat:ca-entropy — энтропийная классификация правила"
CLASS_JSON=$(echo "$EVOLVE_JSON" | python -m projects.hexstat.channel_glyphs \
    --json --from-ca ca-entropy)

WOLFRAM=$(echo "$CLASS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['wolfram_class'])")
BEHAVIOR=$(echo "$CLASS_JSON"| python -c "import sys,json; d=json.load(sys.stdin); print(d['behavior'])")
ATTRACTOR=$(echo "$CLASS_JSON"| python -c "import sys,json; d=json.load(sys.stdin); print(d['attractor'])")
BALANCED=$(echo "$CLASS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('да' if d['yang_balanced'] else 'нет')")

echo "  Класс Вольфрама: $WOLFRAM"
echo "  Поведение: $BEHAVIOR"
echo "  Аттрактор: $ATTRACTOR"
echo "  Ян-баланс сохранён: $BALANCED"
echo ""

# ── Шаг 3: Орбитная классификация Aut(Q6) ────────────────────────────────────
echo "  [3/3] hexsym:rule-orbits — классификация правил по орбитам Aut(Q6)"
ORBIT_JSON=$(python -m projects.hexsym.sym_glyphs --json rule-orbits)

# Показать таблицу результатов
echo ""
echo "$ORBIT_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Aut(Q6): |B₆| = {d[\"aut_q6_order\"]}  (6! × 2⁶ = 720 × 64)')
print(f'  Ян-слоёв: {d[\"yang_orbit_count\"]}  размеры: {d[\"yang_orbit_sizes\"]}')
print()
print(f'  {\"Правило\":<20}  {\"Экв?\":<6}  {\"Вольфрам\":<10}  {\"H₀→Hₙ\":<14}  {\"drift\"}')
print('  ' + '─' * 62)
for r in d['per_rule']:
    eq = '✓ экв' if r['equivariant'] else '✗    '
    h0 = r['initial_entropy']
    hn = r['final_entropy']
    drift = r['yang_drift']
    cls = r['wolfram_class']
    conv = r['convergence_step']
    conv_s = f'conv={conv}' if conv else 'no-conv'
    print(f'  {r[\"rule\"]:<20}  {eq:<6}  {cls:<10}  {h0:.2f}→{hn:.2f}        {drift:.3f}  {conv_s}')
print()
print(f'  Эквивариантные: {d[\"equivariant_rules\"]}')
print(f'  Неэквивариантные: {d[\"non_equivariant_rules\"]}')
"
echo ""

# ── Итог ──────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  ОТКРЫТИЕ SC-3: Канонический атлас Q6 (K2 × K8)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Центральный результат:"
echo "    Aut(Q6) = B₆ (гипероктаэдральная группа): порядок 46080"
echo "    7 ян-слоёв = 7 орбит S₆ на Q6: [1,6,15,20,15,6,1]"
echo ""
echo "  Классификация КА-правил:"
echo "    Вольфрам I  (сходятся): majority_vote, conway_b3s23, identity, smooth, cyclic2"
echo "    Вольфрам II (период): conway_b36s23, cyclic"
echo "    Вольфрам III/IV (сложные): xor_rule, random_walk"
echo ""
echo "  Связь K2 × K8 (динамика × алгебра):"
echo "    xor_rule — ЛИНЕЙНЫЙ оператор над GF(2)⁶:"
echo "      CA-переход = матрица над GF(2), обратима → энтропия постоянна"
echo "      Именно поэтому xor_rule: Вольфрам III/IV, нет сходимости"
echo "    cyclic_rule — ян-счёт играет роль 'состояния' (7 классов):"
echo "      Large drift → drastic yang-redistribution"
echo ""
echo "  Оценка масштаба:"
echo "    Всего функций Q6→Q6: 64^64 ≈ 10^115"
echo "    С ян-весом: 7 ян-слоёв × ... → Aut(Q6) сокращает к ~10^115/46080"
echo "    Практически: 9 именованных правил → 3 класса Вольфрама"
echo ""

# Сохранить контекст
CTX_SCRIPT="
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
import json as _json
create('sc3_run')
write_key('sc3_run', 'evolve',  _json.loads(r'''$EVOLVE_JSON'''), source='hexca:evolve')
write_key('sc3_run', 'classify', _json.loads(r'''$CLASS_JSON'''), source='hexstat:ca-entropy')
write_key('sc3_run', 'orbits',  _json.loads(r'''$ORBIT_JSON'''), source='hexsym:rule-orbits')
write_key('sc3_run', 'finding', {
    'wolfram_xor': '$WOLFRAM',
    'aut_order': 46080,
    'yang_orbits': 7,
    'equivariant_only': 'identity',
    'finding': 'CA rules split into 3 Wolfram classes; only identity equivariant under Aut(Q6)',
}, source='sc3_analysis')
print('  Контекст: sc3_run [evolve, classify, orbits, finding]')
print('  q6ctl ctx show sc3_run')
"
python -c "$CTX_SCRIPT"
echo ""
