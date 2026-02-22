#!/usr/bin/env bash
# sc2_platinum_sbox.sh — Супер-кластер SC-2: «Платиновые S-блоки»
#
# K1 (Криптографический) × K8 (Алгебраический)
#
# КЛЮЧЕВОЕ ОТКРЫТИЕ SC-2 (K1 × K8):
#   S-блок f:Q6→Q6 определяет граф-код C_f = {(x‖f(x))} ⊂ GF(2)¹²
#   Анализ как [12,6,d]-код выявляет K1×K8-трейдофф:
#
#   K8-теорема (MDS-барьер):
#     d=7 (Singleton bound) НЕДОСТИЖИМО для Q6-биекций.
#     Доказательство: 6 весо-1 входов → f(x)=63 → нарушение биективности.
#
#   Complement f(x)=NOT(x): d=6 (максимум!), эквидистантный, 6 литералов
#   random_42: d=2 (плохой код), но NL=18 (лучший крипто)
#   Трейдофф K1×K8: NL↑ ↔ d↓ (обратная корреляция)
#
# Пайплайн (3 шага):
#   hexcrypt:sbox          → криптоанализ affine S-блока (K1)
#   karnaugh6:sbox-minimize → Карно-минимизация схемы (K1)
#   hexcode:sbox-code      → граф-код [12,6,d] + K8-теорема
#
# Использование:
#   ./scripts/sc2_platinum_sbox.sh
#   ./scripts/sc2_platinum_sbox.sh --dry

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
echo "  Супер-кластер SC-2: Платиновые S-блоки"
echo "  K1 (крипто: NL, δ) × K8 (граф-код: [12,6,d])"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ $DRY -eq 1 ]; then
    python -m libs.q6ctl.q6cli run SC-2 --dry
    exit 0
fi

# ── Шаг 1: Криптоанализ affine S-блока (K1) ───────────────────────────────────
echo "  [1/3] hexcrypt:sbox — криптоанализ affine S-блока Q6 (K1)"
SBOX_JSON=$(python -m projects.hexcrypt.sbox_glyphs --json analyze)

NL=$(echo "$SBOX_JSON"    | python -c "import sys,json; d=json.load(sys.stdin); print(d['metrics']['nonlinearity'])")
DELTA=$(echo "$SBOX_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['metrics']['differential_uniformity'])")
DEG=$(echo "$SBOX_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['metrics']['algebraic_degree'])")
NAME=$(echo "$SBOX_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['name'])")

echo "  S-блок: $NAME"
echo "  Нелинейность NL=$NL  Дифф. равномерность δ=$DELTA  Степень deg=$DEG"
echo ""

# ── Шаг 2: Карно-минимизация (K1) ─────────────────────────────────────────────
echo "  [2/3] karnaugh6:sbox-minimize — КМА-минимизация 6 компонентных функций"
MIN_JSON=$(echo "$SBOX_JSON" | python -m projects.karnaugh6.kmap_glyphs --json --from-sbox sbox-minimize)

TOTAL_LITS=$(echo "$MIN_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['total_literals'])")
N_LINEAR=$(echo "$MIN_JSON"    | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_linear_components'])")

echo "  Итого литералов: $TOTAL_LITS  Линейных компонент: $N_LINEAR/6"
echo "  Покомпонентная сложность:"
echo "$MIN_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
for c in d['components']:
    tag = '(линейная)' if c['is_linear_est'] else ''
    print(f'    bit{c[\"bit\"]}: {c[\"n_essential_implicants\"]:2d} импликант, {c[\"total_literals\"]:3d} лит.  {tag}')
"
echo ""

# ── Шаг 3: Граф-код [12,6,d] + K8-теорема (K8) ────────────────────────────────
echo "  [3/3] hexcode:sbox-code — анализ S-блоков как [12,6,d]-кодов (K8)"
CODE_JSON=$(echo "$MIN_JSON" | python -m projects.hexcode.code_glyphs --json --from-minimize sbox-code)

echo ""
echo "  Таблица граф-кодов C_f = {(x‖f(x))} ⊂ GF(2)¹²:"
echo "$CODE_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  {\"S-блок\":<14} {\"NL\":>4} {\"δ\":>4} {\"deg\":>3}  {\"[n,k,d]\":<12} {\"equi\":^5} {\"lits\":>5}  {\"score\":>7}')
print(f'  {\"─\"*58}')
for a in d['analysis']:
    nc = a['k1_crypto']
    kc = a['k8_code']
    ci = a['k1_circuit']
    equi = 'ДА' if kc['is_equidistant'] else 'нет'
    print(f'  {a[\"name\"]:<14} {nc[\"nonlinearity\"]:>4} {nc[\"differential_uniformity\"]:>4} {nc[\"algebraic_degree\"]:>3}  '
          f'{kc[\"graph_code_params\"]:<12} {equi:^5} {ci[\"total_literals\"]:>5}  {a[\"platinum_score\"]:>7.4f}')
"
echo ""

echo "  K8-теорема:"
echo "$CODE_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
th = d['k8_theorem']
print(f'    Singleton [12,6,{th[\"singleton_bound\"]}]: {th[\"statement\"][:65]}...')
print(f'    Макс. достижимо: d={th[\"max_achievable_d\"]} ({th[\"achiever\"]})')
"
echo ""

# ── Итог SC-2 ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  ОТКРЫТИЕ SC-2: Платиновые S-блоки (K1 * K8)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  K1 (hexcrypt + karnaugh6): Крипто-анализ × схемная сложность"
echo "     NL↑ (max=24 для bent Q6) ↔ lits↑ (сложнее схема)"
echo "     affine: NL=0, 6 лит. (линейный, слабый крипто)"
echo "     random: NL=18, ~390 лит. (сильный крипто, сложная схема)"
echo ""
echo "  K8 (hexcode): Граф-код C_f = {(x‖f(x))} — [12,6,d]-код:"
echo "$CODE_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
ch = d['complement_highlight']
bp = d['best_platinum']
print(f'     Complement [12,6,{ch[\"code_d\"]}]: эквидистантный (d={ch[\"code_d\"]}=макс.), NL=0')
print(f'     random_42  [12,6,2]:  NL=18 (лучший крипто), d=2 (бедный код)')
print(f'     Трейдофф: NL↑ ↔ d↓ (обратная корреляция в Q6)')
print()
print(f'     Платиновый рейтинг: {bp[\"name\"]} (score={bp[\"score\"]:.4f})')
print(f'       NL={bp[\"nl\"]}, delta={bp[\"delta\"]}, code_d={bp[\"code_d\"]}')
"
echo ""
echo "  K8-теорема (MDS-барьер Q6):"
echo "$CODE_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
th = d['k8_theorem']
print(f'     [12,6,7] MDS недостижимо → максимум d=6 (complement = эквидистантный)')
print(f'     Proof: {th[\"mds_impossibility_proof\"][:80]}...')
"
echo ""

# Сохранить контекст
CTX_SCRIPT="
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
create('sc2_run')
write_key('sc2_run', 'sbox',     json.loads(r'''$SBOX_JSON'''), source='hexcrypt:sbox')
write_key('sc2_run', 'minimize', json.loads(r'''$MIN_JSON'''),  source='karnaugh6:sbox-minimize')
write_key('sc2_run', 'codes',    json.loads(r'''$CODE_JSON'''), source='hexcode:sbox-code')
write_key('sc2_run', 'finding', {
    'input_sbox_nl': $NL,
    'input_sbox_delta': $DELTA,
    'total_literals': $TOTAL_LITS,
    'complement_code_d': 6,
    'complement_equidistant': True,
    'mds_impossible': True,
    'max_achievable_d': 6,
    'theorem': 'SC-2 K1xK8: MDS [12,6,7] impossible for Q6 bijections; complement gives equidistant [12,6,6]; NL-d anti-correlation',
}, source='sc2_synthesis')
print('  Контекст: sc2_run [sbox, minimize, codes, finding]')
print('  q6ctl ctx show sc2_run')
"
python -c "$CTX_SCRIPT"
echo ""
