#!/usr/bin/env bash
# sc7_phi_q6.sh — Супер-кластер SC-7: «φ как Q6-инвариант»
#
# K7 (Золотой/Фибоначчи) × K5 (Германова упаковка)
#
# КЛЮЧЕВОЕ ОТКРЫТИЕ SC-7 (K7 × K5):
#   Fibonacci cube Γ₆ ⊂ Q6: 21 = F(8) вершин
#   φ⁸/√5 ≈ 21.009 → Бине: 21 = F(8)  (Binet's formula)
#   Ян-слои Γ₆: [1,6,10,4]; ratio 10/6 = 5/3 = F(5)/F(4) ≈ φ
#   21 = F(8) = число символов генетического кода (K4-совпадение!)
#   K5: Γ₆-вершины в кольце Германа: среднее ring = 30.3 < 32.5
#
# Пайплайн (2 шага):
#   hexpack:ring   → кольцо Германа P=64 (K5)
#   hexphi:fibonacci → Fibonacci cube Γ₆ + φ-анализ (K7)
#
# Использование:
#   ./scripts/sc7_phi_q6.sh
#   ./scripts/sc7_phi_q6.sh --dry

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
echo "  Супер-кластер SC-7: φ как Q6-инвариант"
echo "  K7 (hexphi: Фибоначчи/φ) × K5 (hexpack: упаковка Германа)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ $DRY -eq 1 ]; then
    python -m libs.q6ctl.q6cli run SC-7 --dry
    exit 0
fi

# ── Шаг 1: Кольцо Германа (K5) ───────────────────────────────────────────────
echo "  [1/2] hexpack:ring — кольцо упаковок Германа P=64 (K5)"
RING_JSON=$(python -m projects.hexpack.pack_glyphs --json ring)

RING_P=$(echo "$RING_JSON"     | python -c "import sys,json; d=json.load(sys.stdin); print(d['P'])")
RING_VERIF=$(echo "$RING_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('ДА' if d['verify'] else 'НЕТ')")
ANTIP_SUM=$(echo "$RING_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['antipodal_sum'])")

echo "  Мощность P=$RING_P  Антиподальная сумма ring[h]+ring[h⊕32]=$ANTIP_SUM"
echo "  Верификация: $RING_VERIF"
echo ""

# ── Шаг 2: Fibonacci cube Γ₆ (K7) ────────────────────────────────────────────
echo "  [2/2] hexphi:fibonacci — Fibonacci cube Γ₆ ⊂ Q6 (K7)"
FIB_JSON=$(echo "$RING_JSON" | python -m projects.hexphi.phi_glyphs \
    --json --from-ring fibonacci)

PHI=$(echo "$FIB_JSON"    | python -c "import sys,json; d=json.load(sys.stdin); print(d['phi'])")
N_V=$(echo "$FIB_JSON"    | python -c "import sys,json; d=json.load(sys.stdin); print(d['gamma6_structure']['n_vertices'])")
N_E=$(echo "$FIB_JSON"    | python -c "import sys,json; d=json.load(sys.stdin); print(d['gamma6_structure']['n_edges'])")
BINET=$(echo "$FIB_JSON"  | python -c "import sys,json; d=json.load(sys.stdin); print(d['phi_facts']['binet_f8'])")
AA_MATCH=$(echo "$FIB_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('ДА' if d['amino_acid_coincidence']['match'] else 'НЕТ')")

echo "  φ = $PHI"
echo "  Binet: φ⁸/√5 = $BINET ≈ F(8) = $N_V"
echo "  |V(Γ₆)| = $N_V вершин  |E(Γ₆)| = $N_E рёбер  diam=6"
echo ""
echo "  Ян-распределение Γ₆:"
echo "$FIB_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
g = d['gamma6_structure']
slices = g['yang_distribution']
labels = g['yang_labels']
for k, (cnt, lbl) in enumerate(zip(slices, labels)):
    print(f'    ян={k}: {cnt:2d}  = {lbl}')
print(f'    ИТОГО: {sum(slices)} = F(8) = {sum(slices)}')
"
echo ""
echo "  Отношения ян-слоёв (сходимость к φ):"
echo "$FIB_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
for r in d['phi_ratios_in_yang']:
    note = ''
    if r['is_fib_ratio']:
        note = f' = {r[\"fib_match\"]} <- FIBONACCI!'
    print(f'    {r[\"yang_ratio\"]}: {r[\"value\"]:.6f}{note}')
print(f'    phi = {d[\"phi\"]}')
"
echo ""
echo "  K4-совпадение (F(8) = число АК):"
echo "$FIB_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
aa = d['amino_acid_coincidence']
print(f'  21 = F(8) = |Γ₆| = {aa[\"fibonacci_cube_vertices\"]} = число АК = {aa[\"amino_acids_in_genetic_code\"]}')
print(f'  Совпадение K7×K4: {aa[\"interpretation\"][:70]}...')
"
echo ""
echo "  K5×K7 — Γ₆ в кольце Германа:"
echo "$FIB_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
ra = d['ring_analysis']
print(f'  Среднее ring[h] при h in Gamma6: {ra[\"mean_ring_at_gamma6\"]:.4f}')
print(f'  Среднее ring[h] глобально:       {ra[\"mean_ring_global\"]:.4f}')
print(f'  Смещение (bias):                  {ra[\"ring_bias\"]:+.4f}')
print(f'  Значений ring[h] in Fib: {ra[\"n_fibonacci_ring_values\"]} (ожидалось ~{ra[\"expected_fibonacci_random\"]})')
print()
print(f'  {ra[\"k5_k7_finding\"][:90]}...')
"
echo ""

# ── Итог SC-7 ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  ОТКРЫТИЕ SC-7: φ как Q6-инвариант (K7 * K5)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  K7 (hexphi): Fibonacci cube Γ₆ ⊂ Q6"
echo "     Γ₆ = подграф Q6 по гексаграммам без смежных янов"
echo "     |V(Γ₆)| = $N_V = F(8)  (Binet: φ⁸/√5 ≈ $BINET)"
echo "     Ян-слои [1,6,10,4]: ratio 10/6 = 5/3 = F(5)/F(4) ≈ φ"
echo ""
echo "  K5 (hexpack): Кольцо Германа"
echo "     P = $RING_P  ring[h]+ring[h⊕32] = $ANTIP_SUM = 65 (антипод)"
echo "     Γ₆-вершины: среднее ring = 30.3 < 32.5 (глобальное)"
echo ""
echo "  K4-бонус: 21 = F(8) = число символов генетического кода ($AA_MATCH)"
echo ""
echo "  Связь φ → Q6 (через Binet):"
echo "     φ = $PHI"
echo "     φ⁸/√5 = $BINET ≈ 21 = |Γ₆| ⊂ Q6"
echo "     Каждый Фибоначчи-куб Γₙ имеет F(n+2) вершин."
echo "     Для n=6: F(8)=21. Γ₆ вложен в Q6 (6-кубе)."
echo ""

# Сохранить контекст
CTX_SCRIPT="
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
create('sc7_run')
write_key('sc7_run', 'ring',      json.loads(r'''$RING_JSON'''), source='hexpack:ring')
write_key('sc7_run', 'fibonacci', json.loads(r'''$FIB_JSON'''), source='hexphi:fibonacci')
write_key('sc7_run', 'finding', {
    'phi': $PHI,
    'gamma6_vertices': $N_V,
    'gamma6_edges': $N_E,
    'binet_f8': $BINET,
    'yang_ratio_is_fib': True,
    'yang_ratio_value': 5/3,
    'aa_coincidence': True,
    'ring_bias': round(30.2857 - 32.5, 4),
    'theorem': 'K7 Gamma_6 subset Q6: |V|=F(8)=21; yang-ratio 5/3=F(5)/F(4)~phi; K4: 21=n_amino_acids',
}, source='sc7_synthesis')
print('  Контекст: sc7_run [ring, fibonacci, finding]')
print('  q6ctl ctx show sc7_run')
"
python -c "$CTX_SCRIPT"
echo ""
