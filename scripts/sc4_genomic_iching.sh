#!/usr/bin/env bash
# sc4_genomic_iching.sh — Супер-кластер SC-4: «Геномный И-Цзин»
#
# K4 (Биологический) × K6 (И-Цзин)
# Цель: ДНК-мутации как переходы между гексаграммами матрицы Андреева
#
# Пайплайн:
#   1. hextrimat:triangle  → треугольная матрица И-Цзин (JSON)
#   2. hextrimat:verify    → верификация числовых фактов
#   3. [hexbio:codon]      → (TODO) кодоны → гексаграммы
#   4. [hexnav:transition] → (TODO) переходы между гексаграммами
#
# Использование:
#   ./scripts/sc4_genomic_iching.sh
#   ./scripts/sc4_genomic_iching.sh --show-matrix

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO"

SHOW_MATRIX=0
for arg in "$@"; do
    case "$arg" in
        --show-matrix) SHOW_MATRIX=1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Супер-кластер SC-4: Геномный И-Цзин"
echo "  K4 (Биологический) × K6 (И-Цзин / hextrimat)"
echo "  Результат: кодон → гексаграмма → позиция в матрице Андреева"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# ── Шаг 1: Матрица Андреева (JSON) ────────────────────────────────────────────
echo "  [1/3] hextrimat:triangle — Пирамидальная матрица И-Цзин"
echo "        64 гексаграммы в треугольном расположении (11 строк)"
echo ""

TRIANGLE_JSON=$(python -m projects.hextrimat.trimat_glyphs --json triangle)

# Статистика
python -c "
import sys, json
d = json.loads('''$TRIANGLE_JSON''')
print(f'  Ячеек: {d[\"total_cells\"]}   Строк: {d[\"num_rows\"]}')
print(f'  Σ(1..64) = {d[\"sum_total\"]} = 160×{d[\"matrix_number\"]}')
for r in ['1', '4', '7', '11']:
    if r in d['rows']:
        row = d['rows'][r]
        print(f'  Строка {r:>2}: {row[\"values\"]}  Σ={row[\"sum\"]}')
"
echo ""

# ── Шаг 2: Верификация числовых фактов ────────────────────────────────────────
echo "  [2/3] hextrimat:verify — Верификация числовых фактов Андреева"
VERIFY_JSON=$(python -m projects.hextrimat.trimat_glyphs --json verify)

python -c "
import sys, json
d = json.loads('''$VERIFY_JSON''')
print(f'  Все факты верны: {\"✓\" if d[\"all_verified\"] else \"✗\"}')
for k, v in d['key_numbers'].items():
    print(f'    {k:<20} = {v}')
"
echo ""

# ── Шаг 3: DNA → Hexagram mapping ─────────────────────────────────────────────
echo "  [3/3] Отображение: кодоны → позиции матрицы"
echo "  (Демонстрационная логика без hexbio)"
echo ""
python -c "
import sys, json
sys.path.insert(0, '$REPO')
from projects.hextrimat.hextrimat import TRIMAT

# 64 кодона (4^3) ↔ 64 гексаграммы
codons = []
bases = ['U', 'C', 'A', 'G']
for b1 in bases:
    for b2 in bases:
        for b3 in bases:
            codons.append(f'{b1}{b2}{b3}')

print(f'  64 кодона ↔ 64 гексаграммы в матрице Андреева:')
print(f'  {'Кодон':<8}  {'№ гексаграммы':<16}  {'Строка':<8}  {'Столбец':<8}  {'Сумма строки'}')
print(f'  {\"─\"*60}')

# Показать первые 12 и последние 4
samples = list(range(12)) + list(range(60, 64))
for i in samples:
    codon = codons[i]
    hexagram_n = i + 1
    r, c, v = TRIMAT.cells[i]
    row_sum = TRIMAT.row_sum(r)
    print(f'  {codon:<8}  {'гекс '+str(hexagram_n):<16}  стр.{r:<5}  поз.{c:<5}  Σ={row_sum}')
    if i == 11:
        print(f'  ...')
"
echo ""

# ── Итог ───────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  SC-4 завершён."
echo ""
echo "  Биологическая интерпретация:"
echo "  • Каждый кодон ДНК (UUU..GGG) → одна гексаграмма И-Цзин"
echo "  • Позиция в матрице Андреева (строка, столбец) — структурный адрес"
echo "  • Число 13 = «число матрицы»: 2080 = 160×13 (13 пар близнецов)"
echo "  • Птица Времени 729=3⁶: тройная симметрия генетического кода"
echo ""
echo "  Следующий шаг: передать данные в hexspec:resonance для"
echo "  поиска резонансных структур (TSC-3 = SC-4 + hexlearn)"
echo "══════════════════════════════════════════════════════════════════════"

# Сохранить контекст
python -c "
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
triangle = json.loads('''$TRIANGLE_JSON''')
verify = json.loads('''$VERIFY_JSON''')
create('sc4_run')
write_key('sc4_run', 'triangle', triangle, source='hextrimat:triangle')
write_key('sc4_run', 'verify', verify, source='hextrimat:verify')
print('  Контекст сохранён: sc4_run [ключи: triangle, verify]')
print('  Просмотр: python -m libs.q6ctl.q6cli ctx show sc4_run')
"

if [ $SHOW_MATRIX -eq 1 ]; then
    echo ""
    echo "  [Полная треугольная матрица]"
    python -m projects.hextrimat.trimat_glyphs triangle
fi
