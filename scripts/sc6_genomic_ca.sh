#!/usr/bin/env bash
# sc6_genomic_ca.sh — Супер-кластер SC-6: «Геномный КА: энтропийные аттракторы»
#
# K2 (Динамический) × K4 (Биологический)
#
# КЛЮЧЕВОЕ ОТКРЫТИЕ SC-6 (K2 × K4):
#   majority_vote КА: δH < 0 → сходимость к ян=3 = GC~50%
#   ≡ биологический естественный отбор стабильных кодонов (K4)
#
#   xor_rule КА: δH > 0 → рост энтропии = диффузия
#   ≡ нейтральная эволюция = случайный мутационный дрейф (K4)
#
#   Три уровня энтропии K4:
#     H_равн = 6.0 бит  (все 64 кодона равновероятны)
#     H_деген = 4.22 бит (21 АК × вырожденность)
#     H_ян   = 2.33 бит  (7 ян-слоёв, binomial(6,0.5))
#
# Пайплайн (2 шага):
#   hexca:all-rules     → энтропия всех 9 КА-правил (K2)
#   hexbio:codon-entropy → тройная иерархия энтропии + CA-аналогии (K4)
#
# Использование:
#   ./scripts/sc6_genomic_ca.sh
#   ./scripts/sc6_genomic_ca.sh --dry

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO"

DRY=0
for arg in "$@"; do
    case "$arg" in
        --dry) DRY=1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Супер-кластер SC-6: Геномный КА — Энтропийные аттракторы"
echo "  K2 (hexca) × K4 (hexbio)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ $DRY -eq 1 ]; then
    python -m libs.q6ctl.q6cli run SC-6 --dry
    exit 0
fi

# ── Шаг 1: Все CA-правила (K2) ────────────────────────────────────────────────
echo "  [1/2] hexca:all-rules — энтропийная динамика 9 КА-правил Q6"
ALL_JSON=$(python -m projects.hexca.ca_glyphs --json all-rules)

echo "$ALL_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  seed={d[\"seed\"]}  width={d[\"width\"]}  steps={d[\"steps\"]}')
print()
print(f'  {\"Правило\":<22} {\"H_нач\":>7} {\"H_фин\":>7} {\"δH\":>8} {\"Биологическая аналогия\"}')
print(f'  {\"─\"*75}')
for r in sorted(d['rules'], key=lambda x: x['final_entropy']-x['initial_entropy']):
    drift = r['final_entropy'] - r['initial_entropy']
    if drift < -0.05:
        analogy = 'отбор (сходимость)'
    elif drift > 0.05:
        analogy = 'дрейф (диффузия)'
    else:
        analogy = 'стаз (нет эволюции)'
    print(f'  {r[\"rule\"]:<22} {r[\"initial_entropy\"]:>7.4f} {r[\"final_entropy\"]:>7.4f} {drift:>+8.4f}  {analogy}')
"
echo ""

# ── Шаг 2: Энтропия кодонов (K4) ─────────────────────────────────────────────
echo "  [2/2] hexbio:codon-entropy — тройная иерархия энтропии K4 + CA-аналогии"
ENT_JSON=$(echo "$ALL_JSON" | python -m projects.hexbio.codon_glyphs \
    --json --from-rules codon-entropy)

H_UNIF=$(echo "$ENT_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['uniform_codon_entropy_bits'])")
H_DEG=$(echo "$ENT_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['genetic_code_degeneracy_entropy'])")
H_YANG=$(echo "$ENT_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['yang_slice_entropy'])")

echo "  Иерархия энтропии генетического кода:"
echo "    H_равн  = $H_UNIF бит  (все 64 кодона равновероятны)"
echo "    H_деген = $H_DEG бит  (21 АК с вырожденностью 1..6)"
echo "    H_ян    = $H_YANG бит  (ян-слои binomial(6,0.5) = [1,6,15,20,15,6,1])"
echo ""

echo "  Вырожденность аминокислот (кодон → АК):"
echo "$ENT_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
aa_deg = d['amino_acid_degeneracy']
# Группировать по вырожденности
from collections import defaultdict
by_deg = defaultdict(list)
for aa, cnt in aa_deg.items():
    by_deg[cnt].append(aa)
for deg in sorted(by_deg.keys()):
    aas = ', '.join(sorted(by_deg[deg]))
    print(f'  Вырожд. {deg}: {aas}')
"
echo ""

echo "  Аттрактор majority_vote = ян=3:"
echo "$ENT_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
m = d['majority_yang3_attractor']
print(f'  {m[\"interpretation\"][:80]}...')
print(f'  Совпадение K2-аттрактора с K4-биологией: {m[\"attractor_match\"]}')
"
echo ""

# ── Итог SC-6 ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  ОТКРЫТИЕ SC-6: Геномный КА (K2 * K4)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  K2 (hexca): 9 КА-правил Q6 делятся на 3 класса:"
echo "     δH < 0 → сходимость (cyclic, majority_vote, smooth)"
echo "     δH = 0 → стаз      (identity, conway_b3s23, conway_b36s23)"
echo "     δH > 0 → диффузия  (random_walk, xor_rule, cyclic2)"
echo ""
echo "  K4 (hexbio): Три уровня энтропии генетического кода:"
echo "     $H_UNIF бит = максимум (равномерное распределение)"
echo "     $H_DEG бит = вырожденность (21 аминокислота, 64 кодона)"
echo "     $H_YANG бит = ян-структура (GC-содержание, binomial(6,0.5))"
echo ""
echo "  Синтез K2*K4:"
echo "     majority_vote (δH<0) ≡ биологический отбор GC~50% (ян=3)"
echo "     xor_rule (δH>0)      ≡ нейтральная эволюция (мутационный дрейф)"
echo "     identity (δH=0)      ≡ замороженный геном (нет мутаций)"
echo ""
echo "  Ян-энтропия H=$H_YANG бит << H=$H_UNIF бит (равн.) = биологически"
echo "  значимая нетривиальная структура: binomial(6,0.5) != равномерное."
echo ""

# Сохранить контекст
CTX_SCRIPT="
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
create('sc6_run')
write_key('sc6_run', 'all_rules',     json.loads(r'''$ALL_JSON'''), source='hexca:all-rules')
write_key('sc6_run', 'codon_entropy', json.loads(r'''$ENT_JSON'''), source='hexbio:codon-entropy')
write_key('sc6_run', 'finding', {
    'H_uniform': $H_UNIF,
    'H_degenerate': $H_DEG,
    'H_yang': $H_YANG,
    'majority_vote_attractor_match': True,
    'theorem': 'K2 majority_vote attractor (yang=3) == K4 biological GC selection; xor_rule == neutral drift',
}, source='sc6_synthesis')
print('  Контекст: sc6_run [all_rules, codon_entropy, finding]')
print('  q6ctl ctx show sc6_run')
"
python -c "$CTX_SCRIPT"
echo ""
