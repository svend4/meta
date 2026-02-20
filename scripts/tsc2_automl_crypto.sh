#!/usr/bin/env bash
# tsc2_automl_crypto.sh — Тройной супер-кластер TSC-2: «AutoML-крипто»
#
# K3 (Интеллектуальный) × K2 (Динамический) × K1 (Криптографический)
#
# КЛЮЧЕВОЕ ОТКРЫТИЕ TSC-2 (K3 × K2 × K1):
#   ML-ранжирование CA-правил Q6 по крипто-пригодности (key schedule):
#
#   K3 (ML): взвешенная оценка = 0.35·H_fin + 0.35·drift + 0.15·баланс + 0.15·нет_сходим.
#   K2 (CA): entropy drift δH < 0 (сходимость) / ≈0 (стаз) / > 0 (диффузия)
#   K1 (крипто): диффузионные правила (δH>0) лучшие для key schedule
#
#   Топ-3 key schedule: xor_rule → random_walk → cyclic2
#   Худший компонент:   cyclic (δH=-2.59, аттрактор ян=5)
#
# Пайплайн (2 шага):
#   hexca:all-rules   → CA-динамика всех 9 правил Q6 (K2)
#   hexlearn:ca-rank  → ML-ранжирование по крипто-критериям (K3 × K1)
#
# Использование:
#   ./scripts/tsc2_automl_crypto.sh
#   ./scripts/tsc2_automl_crypto.sh --dry

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
export PYTHONPATH="$REPO"

DRY=0
for arg in "$@"; do
    case "$arg" in
        --dry) DRY=1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Тройной супер-кластер TSC-2: AutoML-крипто"
echo "  K3 (ML-ранжирование) × K2 (CA-динамика) × K1 (ключевое расписание)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ $DRY -eq 1 ]; then
    python -m libs.q6ctl.q6cli run TSC-2 --dry
    exit 0
fi

# ── Шаг 1: CA-динамика всех правил (K2) ───────────────────────────────────────
echo "  [1/2] hexca:all-rules — энтропийная динамика 9 CA-правил Q6 (K2)"
ALL_JSON=$(python -m projects.hexca.ca_glyphs --json all-rules)

echo "$ALL_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Параметры CA: ширина={d[\"width\"]} ячеек, шагов={d[\"steps\"]}, сид={d[\"seed\"]}')
print()
print(f'  {\"Правило\":<22} {\"H_нач\":>7} {\"H_фин\":>7} {\"δH\":>8}  Класс')
print(f'  {\"─\"*60}')
for r in sorted(d['rules'], key=lambda x: x['final_entropy']-x['initial_entropy'], reverse=True):
    drift = r['final_entropy'] - r['initial_entropy']
    if drift > 0.05:
        cls = 'диффузия  ↑ (K1: хорошо)'
    elif drift < -0.05:
        cls = 'сходимость ↓ (K1: плохо)'
    else:
        cls = 'стаз      = (K1: нейтр.)'
    print(f'  {r[\"rule\"]:<22} {r[\"initial_entropy\"]:>7.4f} {r[\"final_entropy\"]:>7.4f} {drift:>+8.4f}  {cls}')
"
echo ""

# ── Шаг 2: ML-ранжирование (K3 × K1) ─────────────────────────────────────────
echo "  [2/2] hexlearn:ca-rank — ML-ранжирование по крипто-пригодности (K3 × K1)"
RANK_JSON=$(echo "$ALL_JSON" | python -m projects.hexlearn.learn_glyphs \
    --json --from-rules ca-rank)

BEST_RULE=$(echo "$RANK_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['best']['rule'])")
BEST_SCR=$(echo "$RANK_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['best']['score'])")
WORST_RULE=$(echo "$RANK_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['worst']['rule'])")
WORST_SCR=$(echo "$RANK_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['worst']['score'])")
OPT_KS=$(echo "$RANK_JSON"     | python -c "import sys,json; d=json.load(sys.stdin); print(' → '.join(d['optimal_key_schedule']))")
OPT_H=$(echo "$RANK_JSON"      | python -c "import sys,json; d=json.load(sys.stdin); print(d['optimal_ks_mean_entropy'])")

echo ""
echo "  ML-таблица ранжирования (K3-оценка по K2+K1 признакам):"
echo "$RANK_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  {\"#\":<3} {\"Правило\":<22} {\"Оценка\":>7} {\"H_fin\":>7} {\"δH\":>8} {\"Ян.\":>6}  Ранг')
print(f'  {\"─\"*65}')
for i, r in enumerate(d['ranked']):
    f = r['features']
    print(f'  {i+1:<3} {r[\"rule\"]:<22} {r[\"crypto_score\"]:>7.4f} '
          f'{f[\"entropy_final\"]:>7.4f} {f[\"entropy_drift\"]:>+8.4f} '
          f'{f[\"yang_mean_final\"]:>6.2f}  {r[\"rank\"]}')
"
echo ""

echo "  Классы (K1-пригодность для key schedule):"
echo "$RANK_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
lb = d['leaderboard']
labels = [('excellent', 'Отлично   '), ('good', 'Хорошо    '), ('neutral', 'Нейтрально'), ('poor', 'Слабо     ')]
for rank, label in labels:
    rules = lb.get(rank, [])
    if rules:
        joined = ', '.join(rules)
        print(f'    {label}: {joined}')
"
echo ""

# ── Итог TSC-2 ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  ОТКРЫТИЕ TSC-2: AutoML-крипто (K3 * K2 * K1)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  K2 (hexca): 9 CA-правил Q6 → 3 класса по δH:"
echo "     диффузионные (δH>0): xor_rule, random_walk, cyclic2"
echo "     статические  (δH≈0): identity, conway_b3s23, conway_b36s23"
echo "     сходящиеся   (δH<0): smooth, majority_vote, cyclic"
echo ""
echo "  K3 (hexlearn): ML-взвешенная оценка (4 признака):"
echo "     0.35 × H_fin   + 0.35 × δH-диффузия"
echo "     0.15 × ян-баланс + 0.15 × без-сходимости"
echo ""
echo "  K1 (крипто): Лучший key schedule Q6:"
echo "     Лучшее: $BEST_RULE (score=$BEST_SCR)"
echo "     Худшее: $WORST_RULE (score=$WORST_SCR)"
echo "     Оптимальный KS: $OPT_KS"
echo "     Средняя H_fin оптим. KS: $OPT_H бит"
echo ""
echo "  K3-синтез (находка):"
echo "$RANK_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  {d[\"k3_finding\"][:90]}...')
"
echo ""

# Сохранить контекст
CTX_SCRIPT="
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
create('tsc2_run')
write_key('tsc2_run', 'all_rules', json.loads(r'''$ALL_JSON'''), source='hexca:all-rules')
write_key('tsc2_run', 'ca_rank',   json.loads(r'''$RANK_JSON'''), source='hexlearn:ca-rank')
write_key('tsc2_run', 'finding', {
    'best_rule': '$BEST_RULE',
    'best_score': $BEST_SCR,
    'worst_rule': '$WORST_RULE',
    'worst_score': $WORST_SCR,
    'optimal_key_schedule': json.loads(r'''$RANK_JSON''')['optimal_key_schedule'],
    'optimal_ks_mean_entropy': $OPT_H,
    'theorem': 'TSC-2 K3xK2xK1: ML-rank reveals xor_rule best KS (diffusive, H_fin=4.02); cyclic worst (convergent, attractor yang=5)',
}, source='tsc2_synthesis')
print('  Контекст: tsc2_run [all_rules, ca_rank, finding]')
print('  q6ctl ctx show tsc2_run')
"
python -c "$CTX_SCRIPT"
echo ""
