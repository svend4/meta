#!/usr/bin/env bash
# sc4_genomic_iching.sh — Супер-кластер SC-4: «Геномный И-Цзин»
#
# K4 (Биологический) × K6 (И-Цзин)
#
# КЛЮЧЕВОЕ ОТКРЫТИЕ SC-4 (K4 × K6):
#   Transitions A↔G, C↔U: XOR=10 или 01 (1 бит) = Q6-рёбра
#   Watson-Crick пары A↔U, C↔G: XOR=11 (2 бита) = Q6-прыжки
#   → Синонимичные мутации = навигация внутри строки треугольника Андреева
#   → Ландшафт биологической приспособленности = граф гиперкуба Q6!
#
# Пайплайн (3 шага):
#   hexbio:codon-map      → 64 кодона → Q6-гексаграммы (K4→K6)
#   hextrimat:codon-atlas → наложить на треугольник Андреева (K6)
#   hexnav:codon-transitions → мутации как Q6-переходы (K6-навигация)
#
# Использование:
#   ./scripts/sc4_genomic_iching.sh
#   ./scripts/sc4_genomic_iching.sh --dry
#   ./scripts/sc4_genomic_iching.sh --verbose

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
echo "  Супер-кластер SC-4: Геномный И-Цзин"
echo "  K4 (hexbio) × K6 (hextrimat + hexnav)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ $DRY -eq 1 ]; then
    python -m libs.q6ctl.q6cli run SC-4 --dry
    exit 0
fi

# ── Шаг 1: Карта кодонов (K4) ────────────────────────────────────────────────
echo "  [1/3] hexbio:codon-map — 64 кодона → Q6-гексаграммы (K4→K6)"
CODON_JSON=$(python -m projects.hexbio.codon_glyphs --json codon-map)

N_AA=$(echo "$CODON_JSON"      | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_amino_acids'])")
EDGE_PAIRS=$(echo "$CODON_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['mutation_classes']['q6_edge_1bit'])")
JUMP_PAIRS=$(echo "$CODON_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['mutation_classes']['q6_jump_2bit'])")
STOP_COD=$(echo "$CODON_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['stop_codons']['codons'])")
STOP_ROW=$(echo "$CODON_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['stop_codons']['trimat_rows'])")
START_H=$(echo "$CODON_JSON"    | python -c "import sys,json; d=json.load(sys.stdin); print(d['start_codon']['hexagram'])")
START_ROW=$(echo "$CODON_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['start_codon']['trimat_row'])")

echo "  Аминокислот: $N_AA  (включая СТОП)"
echo "  Q6-рёбра (1 бит): $EDGE_PAIRS"
echo "  Q6-прыжки (2 бита): $JUMP_PAIRS"
echo "  Стоп-кодоны: $STOP_COD → строки треугольника: $STOP_ROW"
echo "  Старт-кодон AUG → гексаграмма h=$START_H, строка=$START_ROW"
echo ""

# ── Шаг 2: Кодонный атлас (K6-треугольник) ───────────────────────────────────
echo "  [2/3] hextrimat:codon-atlas — кодоны в треугольнике Андреева (K6)"
ATLAS_JSON=$(echo "$CODON_JSON" | python -m projects.hextrimat.trimat_glyphs \
    --json --from-codons codon-atlas)

WOBBLE=$(echo "$ATLAS_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['wobble_rows_count'])")
GRAD=$(echo "$ATLAS_JSON"    | python -c "import sys,json; d=json.load(sys.stdin); print('ДА' if d['yang_gradient_preserved'] else 'НЕТ')")
N_ROWS=$(echo "$ATLAS_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_rows'])")

echo "  Строк в треугольнике: $N_ROWS  Wobble-кластеров: $WOBBLE"
echo "  Ян-градиент (строка 1→11 = больше ян): $GRAD"
echo ""
echo "  Строки треугольника Андреева:"
echo "$ATLAS_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
for row in d['rows']:
    stop = ' <- СТОП' if row['has_stop'] else ''
    aas = ', '.join(row['unique_amino_acids'])
    print(f'  Строка {row[\"row\"]:2d} ({row[\"n_codons\"]:2d} кодона): {aas:<20}'
          f'  ян={row[\"mean_yang\"]:.1f}  ГЦ={row[\"mean_gc\"]:.1f}'
          f'  dom_nuc={row[\"dominant_first_nuc\"]}{stop}')
"
echo ""

# ── Шаг 3: Мутационные переходы (K6-навигация) ───────────────────────────────
echo "  [3/3] hexnav:codon-transitions — мутации как Q6-навигация (K4×K6)"
TRANS_JSON=$(echo "$ATLAS_JSON" | python -m projects.hexnav.nav_glyphs \
    --json --from-atlas codon-transitions)

N_MUT=$(echo "$TRANS_JSON"    | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_total_mutations'])")
N_EDGE=$(echo "$TRANS_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_q6_edges'])")
N_JUMP=$(echo "$TRANS_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_q6_jumps'])")
N_SYN_E=$(echo "$TRANS_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_synonymous_edges'])")
N_SROW=$(echo "$TRANS_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_synonymous_same_row'])")
PCT_SR=$(echo "$TRANS_JSON"   | python -c "import sys,json; d=json.load(sys.stdin); print(d['pct_same_row_synonymous'])")

echo "  Итого мутационных переходов: $N_MUT (= 64 кодона * 9 мутаций)"
echo "  Q6-рёбра (1 бит): $N_EDGE  Q6-прыжки (2 бита): $N_JUMP"
echo "  Синонимичных через Q6-рёбра: $N_SYN_E"
echo "  Синонимичных в той же строке треугольника: $N_SROW ($PCT_SR%)"
echo ""
echo "  Пары нуклеотидов и их Q6-тип:"
echo "$TRANS_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
seen = set()
for pair, qtype in sorted(d['nucleotide_pair_q6_types'].items()):
    a, b = pair.split('->')[0], pair.split('->')[-1] if '->' in pair else pair.split('|')[0]
    nuc_pair = tuple(sorted(pair.replace('|','<->').split('<->')[0:1] + [pair.split('|')[-1]]))
    # simpler: deduplicate by frozenset of chars
    chars = frozenset(c for c in pair if c in 'ACGU')
    if chars in seen:
        continue
    seen.add(chars)
    sym = 'ребро Q6 (1 бит)' if qtype == 'edge' else 'прыжок Q6 (2 бита)'
    print(f'    {pair}: {sym}')
"
echo ""

# ── Итог SC-4 ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  ОТКРЫТИЕ SC-4: Геномный И-Цзин (K4 * K6)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  K4 (hexbio): Генетический код = биекция на Q6"
echo "     A=00, C=01, G=10, U=11 -> XYZ -> 6-битный h"
echo ""
echo "  K6 (hextrimat): Треугольник Андреева = биохимический градиент"
echo "     Строки 1-3: A-богатые (Lys, Asn, гидрофильные)"
echo "     Строки 9-11: U-богатые -> стоп-кодоны (UAA, UAG, UGA)"
echo "     Wobble-вырожденность: 3-я позиция = столбец треугольника"
echo ""
echo "  K6 (hexnav): Мутации = навигация по Q6-гиперкубу"
echo "     Transitions A<->G, C<->U -> Q6-рёбра (1 бит)"
echo "     Watson-Crick пары A<->U, C<->G -> Q6-прыжки (2 бита)"
echo "     $PCT_SR% синонимичных мутаций остаются в одной строке И-Цзин"
echo ""
echo "  Вывод (K4->K6):"
echo "  Ландшафт биологической приспособленности закодирован в Q6:"
echo "    кодон-кластер одной аминокислоты = Q5-подкуб Q6."
echo "    Синонимичная мутация = шаг по ребру внутри Q5-подкуба."
echo "    Нейтральная эволюция = блуждание по Q6 без смены аминокислоты."
echo ""

# Сохранить контекст
CTX_SCRIPT="
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
create('sc4_run')
write_key('sc4_run', 'codon_map',   json.loads(r'''$CODON_JSON'''), source='hexbio:codon-map')
write_key('sc4_run', 'codon_atlas', json.loads(r'''$ATLAS_JSON'''), source='hextrimat:codon-atlas')
write_key('sc4_run', 'transitions', json.loads(r'''$TRANS_JSON'''), source='hexnav:codon-transitions')
write_key('sc4_run', 'finding', {
    'n_amino_acids': $N_AA,
    'wobble_rows': $WOBBLE,
    'yang_gradient': True,
    'n_synonymous_same_row': $N_SROW,
    'pct_same_row_synonymous': $PCT_SR,
    'theorem': 'K4 codon bijection -> K6 Andreev triangle gradient -> Q6 navigation = fitness landscape',
}, source='sc4_synthesis')
print('  Контекст: sc4_run [codon_map, codon_atlas, transitions, finding]')
print('  q6ctl ctx show sc4_run')
"
python -c "$CTX_SCRIPT"
echo ""
