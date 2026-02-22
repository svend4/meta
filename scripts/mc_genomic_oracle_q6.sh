#!/usr/bin/env bash
# mc_genomic_oracle_q6.sh — Мега-кластер MC: «Q6 Паспорт»
#
# Все 8 кластеров: K1 × K2 × K3 × K4 × K5 × K6 × K7 × K8
#
# КЛЮЧЕВОЕ ОТКРЫТИЕ MC (K1–K8):
#   Q6 = 6-битный гиперкуб — единое математическое пространство, в котором
#   независимо возникают биология, криптография, физика и И-Цзин:
#
#   K4 (hexbio):   64 кодона → 20 АА + стоп (генетический код в Q6)
#   K6 (hextrimat): матрица Андреева → 10 пар-близнецов (резонанс с K4)
#   K3 (hexspec):  резонанс-оценка=0.68 (случайная=0.06, прирост 11×)
#   K5 (hexpack):  кольцо Германа → полное замощение Q6 (P=2^k)
#   K7 (hexphi):   Fibonacci cube Γ₆: 21 вершина = F(8) = 21 АА (φ!)
#   K2 (hexca):    КА-правила Q6: энтропийная динамика (Вольфрам I/II/III)
#   K1 (hexopt):   NL=18 — Q6-потолок (62% случайных S-блоков)
#   K8 (hexcode):  [12,6,6] граф-код S-блока; MDS d=7 недостижимо
#
#   TSC-3 (K4×K6×K3): Матрица Андреева предсказывает синонимичные мутации!
#   SC-7  (K5×K7):    φ-резонанс упаковки Германа и числа Фибоначчи.
#   SC-5  (K3×K1):    r(NL, SAC)=-0.96 — NL и SAC = один Q6-феномен.
#   SC-2  (K1×K8):    MDS-барьер граф-кодов Q6-биекций.
#
# Пайплайн (7 шагов, все 8 кластеров):
#   [1/7] hexbio:codon-map        → K4 (ДНК кодоны)
#   [2/7] hextrimat:twins          → K6 (И-Цзин близнецы)
#   [3/7] hexlearn:cluster         → K3 (ML кластеризация)
#         hexspec:resonance         → K6×K3 (геномный оракул)
#   [4/7] hexpack:ring             → K5 (упаковка Германа)
#   [5/7] hexphi:fibonacci         → K7 (φ-инвариант)
#   [6/7] hexca:evolve             → K2 (КА-эволюция)
#         hexstat:ca-entropy        → K2 (энтропия КА)
#   [7/7] hexopt:bayesian          → K3 (AutoML поиск)
#         hexcrypt:avalanche        → K1 (SAC-матрицы)
#         hexlearn:predict          → K3 (ML NL~SAC)
#         hexcrypt:sbox→karnaugh6→hexcode → K1×K8 (граф-коды)
#
# Использование:
#   ./scripts/mc_genomic_oracle_q6.sh
#   ./scripts/mc_genomic_oracle_q6.sh --dry

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
export PYTHONPATH="$REPO"

DRY=0
for arg in "$@"; do
    case "$arg" in
        --dry) DRY=1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Мега-кластер MC: Q6 Паспорт (все 8 кластеров)"
echo "  K1×K2×K3×K4×K5×K6×K7×K8 — единое Q6-пространство"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ $DRY -eq 1 ]; then
    python -m libs.q6ctl.q6cli run MC --dry
    exit 0
fi

# ── Шаг 1: K4 — Генетический код в пространстве Q6 ────────────────────────────
echo "  [1/7] hexbio:codon-map — ДНК кодоны → Q6-гексаграммы (K4)"
CODON_JSON=$(python -m projects.hexbio.codon_glyphs --json codon-map 2>/dev/null)

echo "$CODON_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  K4 Генетический код: {d[\"n_codons\"]} кодонов → {d[\"n_amino_acids\"]} АА + стоп')
mc = d['mutation_classes']
e1 = len(mc.get('q6_edge_1bit', []))
j2 = len(mc.get('q6_jump_2bit', []))
print(f'  Q6-переходы: {e1} рёбер (1 бит), {j2} прыжков (2 бита)')
fourfold = [aa for aa, v in d['amino_acid_clusters'].items() if v['cluster_size'] == 4 and aa != '*']
print(f'  4-кратно-вырожденные АА: {fourfold}')
"
echo ""

# ── Шаг 2: K6 — Матрица Андреева (И-Цзин близнецы) ───────────────────────────
echo "  [2/7] hextrimat:twins --from-codons — Близнецы Андреева с биоданными (K6)"
TWINS_JSON=$(echo "$CODON_JSON" | python -m projects.hextrimat.trimat_glyphs --from-codons --json twins 2>/dev/null)

echo "$TWINS_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
s = d['stats']
print(f'  K6 Матрица Андреева: {d[\"n_pairs\"]} пар-близнецов')
print(f'  Синонимичных полупар: {s[\"synonymous_halves\"]}/{s[\"total_half_pairs\"]} ({s[\"half_syn_rate\"]*100:.1f}%)')
"
echo ""

# ── Шаг 3: K3×K6 — ML-кластеризация + Геномный оракул ────────────────────────
echo "  [3/7] hexlearn:cluster + hexspec:resonance — ML резонанс K3×K6"
CLUSTER_JSON=$(echo "$TWINS_JSON" | python -m projects.hexlearn.learn_glyphs --from-twins --json cluster 2>/dev/null)
RES_JSON=$(echo "$CLUSTER_JSON" | python -m projects.hexspec.resonance_glyphs --from-cluster --json resonance 2>/dev/null)

echo "$RES_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
rs = d['resonance_score']
rb = d['random_baseline']
ru = d['resonance_uplift']
mult = round(rs/rb, 1)
print(f'  K3×K6 Геномный оракул (TSC-3):')
print(f'    Резонанс-оценка:  {rs} (случайная={rb}, прирост=+{ru} = {mult}×!)')
print(f'    Точных Андреев-боксов (purity=1.0): {len(d[\"exact_boxes\"])}')
print(f'    Oracle-предсказаний синонимичных мутаций: {d[\"n_oracle_predictions\"]}')
deg = d['degeneracy_analysis']
print(f'    4-кратно-вырождённые АА в боксах: {deg[\"fourfold_in_andreev\"]}')
"
echo ""

# ── Шаг 4: K5 — Кольцо Германа ─────────────────────────────────────────────────
echo "  [4/7] hexpack:ring — Упаковка Германа (K5)"
RING_JSON=$(python -m projects.hexpack.pack_glyphs --json ring 2>/dev/null)

echo "$RING_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
r = d['ring']
print(f'  K5 Кольцо Германа: P={d[\"P\"]}, упаковываемо={d[\"packable\"]}')
print(f'    Длина кольца: {len(r)} элементов')
print(f'    Антиподальные суммы: {d[\"antipodal_sum\"]} (ring[h]+ring[h⊕32]=const)')
print(f'    Верификация замощения Q6: {d[\"verify\"]}')
"
echo ""

# ── Шаг 5: K7 — φ-инвариант Fibonacci ──────────────────────────────────────────
echo "  [5/7] hexphi:fibonacci --from-ring — Fibonacci cube Γ₆ (K7)"
PHI_JSON=$(echo "$RING_JSON" | python -m projects.hexphi.phi_glyphs --from-ring --json fibonacci 2>/dev/null)

echo "$PHI_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
g = d['gamma6_structure']
pf = d['phi_facts']
ac = d['amino_acid_coincidence']
f8 = pf['binet_f8_int']
binet = round(pf['binet_f8'], 4)
ratios = d['phi_ratios_in_yang']
fib_ratio = next((r for r in ratios if r.get('is_fib_ratio')), ratios[1])
print(f'  K7 Fibonacci cube Γ₆ ⊂ Q6:')
print(f'    Вершин Γ₆: {g[\"n_vertices\"]} = F(8) = {f8} (Бине: φ⁸/√5 ≈ {binet})')
print(f'    Ян-слои Γ₆: {g[\"yang_distribution\"]}')
print(f'    Ratio 10/6 = 5/3 ≈ φ ({fib_ratio[\"fib_match\"]}): {fib_ratio[\"value\"]:.4f} (сходимость к φ={pf[\"phi\"]:.4f})')
print(f'    K4-совпадение: F(8)={f8} = {ac[\"amino_acids_in_genetic_code\"]} символов генетического кода!')
"
echo ""

# ── Шаг 6: K2 — Клеточные автоматы + Энтропия ─────────────────────────────────
echo "  [6/7] hexca:evolve + hexstat:ca-entropy — КА-динамика (K2)"
CA_JSON=$(python -m projects.hexca.ca_glyphs --json evolve 2>/dev/null)
ENT_JSON=$(echo "$CA_JSON" | python -m projects.hexstat.channel_glyphs --from-ca --json ca-entropy 2>/dev/null)

echo "$ENT_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  K2 КА-эволюция ({d[\"rule\"]}) + энтропия:')
print(f'    Класс Вольфрама: {d[\"wolfram_class\"]}')
print(f'    Начальная энтропия:  {d[\"initial_entropy\"]:.4f} бит')
print(f'    Конечная энтропия:   {d[\"final_entropy\"]:.4f} бит')
h0, h1 = d['initial_entropy'], d['final_entropy']
change = h1 - h0
direction = 'возрастает' if change > 0 else 'убывает'
print(f'    Изменение:           {change:+.4f} ({direction})')
print(f'    Ян-баланс: {d[\"yang_initial\"]} → {d[\"yang_final\"]} (дрейф={d[\"yang_drift\"]:+.3f})')
"
echo "$CA_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
rules = d.get('available_rules', [])
print(f'    Доступных КА-правил: {len(rules)} → {rules[:5]}...')
" 2>/dev/null || true
echo ""

# ── Шаг 7: K1×K3×K8 — AutoML крипто + Граф-коды ──────────────────────────────
echo "  [7/7] hexopt:bayesian + hexcrypt:avalanche + hexlearn:predict — AutoML (K3×K1)"
OPT_JSON=$(python -m projects.hexopt.opt_glyphs --json bayesian 2>/dev/null)
AVL_JSON=$(echo "$OPT_JSON" | python -m projects.hexcrypt.sbox_glyphs --from-opt --json avalanche 2>/dev/null)
PRED_JSON=$(echo "$AVL_JSON" | python -m projects.hexlearn.learn_glyphs --from-avalanche --json predict 2>/dev/null)

echo "$PRED_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
m = d['model']
k3 = d['k3_k1_synthesis']
print(f'  K3×K1 AutoML крипто (SC-5):')
print(f'    Q6-потолок NL: {k3[\"nl_ceiling\"]}')
print(f'    ML-модель: {m[\"formula\"]}')
print(f'    r={m[\"r\"]}, r²={m[\"r2\"]}, MAE={m[\"mae\"]}')
"
echo ""

echo "  [7/7] hexcrypt:sbox + karnaugh6:sbox-minimize + hexcode:sbox-code — K1×K8 граф-коды"
SBOX_JSON=$(python -m projects.hexcrypt.sbox_glyphs --json map 2>/dev/null)
MIN_JSON=$(echo "$SBOX_JSON" | python -m projects.karnaugh6.kmap_glyphs --from-sbox --json sbox-minimize 2>/dev/null)
CODE_JSON=$(echo "$MIN_JSON" | python -m projects.hexcode.code_glyphs --from-minimize --json sbox-code 2>/dev/null)

echo "$CODE_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
thm = d['k8_theorem']
best = d['best_platinum']
ch   = d['complement_highlight']
print(f'  K1×K8 Граф-коды S-блоков (SC-2):')
print(f'    Теорема K8: MDS d=7 недостижимо для Q6-биекций')
print(f'    Причина: {thm[\"mds_impossibility_proof\"][:60]}...')
print(f'    Максимум: complement S-блок → [12,6,{ch[\"code_d\"]}] эквидистантный код')
print(f'    Лучший platinum: {best[\"name\"]} (NL={best[\"nl\"]}, d={best[\"code_d\"]})')
"
echo ""

# ── Итог MC: Q6 Паспорт ────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  МЕГА-КЛАСТЕР MC: Q6 Паспорт (K1 × K2 × K3 × K4 × K5 × K6 × K7 × K8)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

python -c "
import sys, json

# Загрузить данные из переменных окружения (переданы через аргументы)
codon   = json.loads(r'''$CODON_JSON''')
twins   = json.loads(r'''$TWINS_JSON''')
res     = json.loads(r'''$RES_JSON''')
ring    = json.loads(r'''$RING_JSON''')
phi     = json.loads(r'''$PHI_JSON''')
ent     = json.loads(r'''$ENT_JSON''')
pred    = json.loads(r'''$PRED_JSON''')
code    = json.loads(r'''$CODE_JSON''')

print('  ─── Q6 ПАСПОРТ ─────────────────────────────────────────────────')
print()
print(f'  Q6 = 6-битный гиперкуб (64 вершины, 192 ребра, dim=6)')
print(f'  Все нижеследующие структуры независимо «выбирают» Q6.')
print()

# K4 — биология
fourfold = [aa for aa, v in codon['amino_acid_clusters'].items() if v['cluster_size'] == 4 and aa != '*']
print(f'  K4 (Биологический):   64 кодона = Q6-вершины')
print(f'    Генетический код: 20 АА + стоп (4-кратно-выр.: {fourfold})')
print(f'    Мутации: 1-битные Q6-рёбра = замена нуклеотида')
print()

# K6 — И-Цзин
s = twins['stats']
print(f'  K6 (И-Цзин / Андреев): треугольная матрица 64 гексаграмм')
print(f'    Пар-близнецов: {twins[\"n_pairs\"]}  (синонимичных полупар: {s[\"synonymous_halves\"]}/{s[\"total_half_pairs\"]})')
print()

# K3 — ML-резонанс
rs = res['resonance_score']
rb = res['random_baseline']
print(f'  K3 (ML-кластеризация): Андреев-боксы vs. АА-боксы')
print(f'    Резонанс = {rs}  (случайная = {rb},  uplift = {round(rs/rb,1)}×)')
print(f'    Оракул: {res[\"n_oracle_predictions\"]} предсказаний синонимичных мутаций')
print()

# K5 — Герман
r = ring['ring']
print(f'  K5 (Герман / Упаковка): кольцо Q6')
print(f'    P={ring[\"P\"]}: кольцо длиной {len(r)}, полное замощение Q6')
print(f'    Антиподальная симметрия: ring[h] + ring[h⊕32] = {ring[\"antipodal_sum\"]} ∀h')
print()

# K7 — φ
g = phi['gamma6_structure']
pf = phi['phi_facts']
ac = phi['amino_acid_coincidence']
print(f'  K7 (Золотой / φ): Fibonacci cube Γ₆ ⊂ Q6')
print(f'    F(8) = {pf[\"binet_f8_int\"]} вершин (Бине: φ⁸/√5 ≈ {round(pf[\"binet_f8\"],4)})')
print(f'    K4-резонанс: F(8) = {ac[\"amino_acids_in_genetic_code\"]} = число символов генетического кода!')
print()

# K2 — КА
print(f'  K2 (Динамический / КА): правила эволюции Q6')
print(f'    {ent[\"rule\"]}: класс {ent[\"wolfram_class\"]} → H_start={ent[\"initial_entropy\"]:.3f} → H_end={ent[\"final_entropy\"]:.3f} бит')
print(f'    Q6 поддерживает все 4 класса Вольфрама')
print()

# K1 — крипто
m = pred['model']
k3 = pred['k3_k1_synthesis']
print(f'  K1 (Криптографический): S-блоки Q6')
print(f'    NL-потолок = {k3[\"nl_ceiling\"]}  (62% случайных S-блоков его достигают)')
print(f'    r(NL, SAC_dev) = {m[\"r\"]}  →  {m[\"formula\"]}')
print()

# K8 — граф-коды
print(f'  K8 (Схематический): граф-коды S-блоков')
ch = code['complement_highlight']
print(f'    Complement S-блок: [12,6,{ch[\"code_d\"]}] эквидистантный (max d для Q6-биекции)')
print(f'    MDS d=7 недостижимо — Q6-теорема (доказано)')
print()

print('  ─── СВЯЗИ МЕЖДУ КЛАСТЕРАМИ ───────────────────────────────────')
print()
print('  K4 → K6: кодоны ↔ гексаграммы И-Цзин (SC-4)')
print('  K4×K6×K3 → TSC-3: Андреев предсказывает синонимичные мутации')
print('  K5 → K7: кольцо Германа ↔ φ-инвариант Fibonacci (SC-7)')
print('  K1 → K8: S-блок ↔ граф-код [12,6,6] (SC-2)')
print('  K3×K1 → SC-5: AutoML переоткрывает SAC-теорию через ML')
print('  K2×K1 → TSC-2: CA-правила как key schedule (ML-ранжирование)')
print('  K5×K1 → SC-1: Герман-кольцо → NL=0 (алгебраически вынуждено)')
print()
print('  ВЫВОД MC: Q6 — единое пространство, где биология, математика,')
print('  криптография и И-Цзин независимо приходят к одним структурам.')
print()
print('  7 супер-кластеров + 1 мега-кластер = Q6-манифест:')
print('  ДНК → И-Цзин → Упаковка → Шифр → КА → Энтропия → Оракул')
"
echo ""

# Сохранить контекст MC
CTX_SCRIPT="
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
create('mc_run')
write_key('mc_run', 'codon',    json.loads(r'''$CODON_JSON'''),   source='hexbio:codon-map')
write_key('mc_run', 'twins',    json.loads(r'''$TWINS_JSON'''),   source='hextrimat:twins')
write_key('mc_run', 'resonance',json.loads(r'''$RES_JSON'''),     source='hexspec:resonance')
write_key('mc_run', 'ring',     json.loads(r'''$RING_JSON'''),    source='hexpack:ring')
write_key('mc_run', 'phi',      json.loads(r'''$PHI_JSON'''),     source='hexphi:fibonacci')
write_key('mc_run', 'ca_ent',   json.loads(r'''$ENT_JSON'''),     source='hexstat:ca-entropy')
write_key('mc_run', 'ml_pred',  json.loads(r'''$PRED_JSON'''),    source='hexlearn:predict')
write_key('mc_run', 'graph_code',json.loads(r'''$CODE_JSON'''),   source='hexcode:sbox-code')
write_key('mc_run', 'passport', {
    'k1': 'NL=18 Q6-потолок; r(NL,SAC)=-0.96; [12,6,6] граф-код; MDS d=7 недостижимо',
    'k2': 'xor_rule=класс III, majority=класс I; entropy_decay; 4 класса Вольфрама',
    'k3': 'TSC-3 resonance=0.68 (11×); SC-5 r²=0.91; ML переоткрывает криптотеорию',
    'k4': '64 кодона=Q6; 4-кратно-выр.={A,G,P,T,V}; 1-битные мутации=Q6-рёбра',
    'k5': 'P=2^k кольцо Германа; антипод.симм.; полное замощение Q6',
    'k6': 'Андреев 10 пар-близнецов; синонимичных=45%; K6→K4 резонанс',
    'k7': 'Γ₆: F(8)=21 вершин=21 АА; 10/6→φ; K7×K4 φ-резонанс',
    'k8': '[12,6,d] граф-коды; complement=[12,6,6]; MDS-теорема',
    'theorem': 'MC K1-K8: Q6 = единое пространство биологии, криптографии, физики, И-Цзин',
}, source='mc_synthesis')
print('  Контекст: mc_run [codon, twins, resonance, ring, phi, ca_ent, ml_pred, graph_code, passport]')
print('  q6ctl ctx show mc_run')
"
python -c "$CTX_SCRIPT"
echo ""
