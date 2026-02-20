#!/usr/bin/env bash
# q6_demo.sh — Демонстрация оркестратора q6ctl и JSON-пайпов Q6.
#
# Показывает три типа взаимодействия:
#   1. Прямой CLI-вызов отдельного модуля
#   2. JSON-экспорт и обработка в Python
#   3. q6ctl как единая точка входа
#
# Использование:
#   ./scripts/q6_demo.sh
#   ./scripts/q6_demo.sh --all    # показать все кластеры

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO"

SHOW_ALL=0
for arg in "$@"; do
    case "$arg" in
        --all) SHOW_ALL=1 ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Q6 ORCHESTRATOR — Демонстрация интерфейса            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── Тип 1: Прямой CLI ──────────────────────────────────────────────────────────
echo "━━━ Тип 1: Прямой CLI-вызов ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  \$ python -m projects.hexpack.pack_glyphs ring"
echo "  (краткий вывод):"
{ python -m projects.hexpack.pack_glyphs ring 2>/dev/null || true; } | head -10
echo "  ..."
echo ""

echo "  \$ python -m projects.hextrimat.trimat_glyphs verify"
echo "  (первые строки):"
{ python -m projects.hextrimat.trimat_glyphs verify 2>/dev/null || true; } | head -12
echo ""

# ── Тип 2: JSON-пайп ──────────────────────────────────────────────────────────
echo "━━━ Тип 2: JSON-пайп между модулями ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  \$ python -m projects.hexpack.pack_glyphs --json ring | python -c 'import sys,json; ..."
echo ""

# hexpack ring → extract exceptional_start → use in hexpack fixpoint
RING_JSON=$(python -m projects.hexpack.pack_glyphs --json ring)
EXC=$(echo "$RING_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['exceptional_start'])")
echo "  hexpack:ring → exceptional_start = $EXC"
echo "  hexpack:fixpoint --start $EXC  (ожидаем: 0 фиксированных точек)"
python -m projects.hexpack.pack_glyphs fixpoint --start "$EXC" | grep -E "(Нет|фикс|Старт)" | head -5
echo ""

# hextrimat verify → all_verified check
VERIFY=$(python -m projects.hextrimat.trimat_glyphs --json verify)
echo "$VERIFY" | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  hextrimat:verify → all_verified = {d[\"all_verified\"]}')
print(f'  Ключевые числа Андреева:')
for k, v in d['key_numbers'].items():
    print(f'    {k:<20} = {v}')
"
echo ""

# ── Тип 3: q6ctl оркестратор ──────────────────────────────────────────────────
echo "━━━ Тип 3: q6ctl — единая точка входа ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  \$ python -m libs.q6ctl.q6cli list modules  (фрагмент K5/K6):"
{ python -m libs.q6ctl.q6cli list modules 2>&1 || true; } | grep -A20 "K5\|K6" | head -25
echo ""

echo "  \$ python -m libs.q6ctl.q6cli info hextrimat:"
python -m libs.q6ctl.q6cli info hextrimat
echo ""

echo "  \$ python -m libs.q6ctl.q6cli run SC-4 --dry:"
python -m libs.q6ctl.q6cli run SC-4 --dry
echo ""

# ── Тип 4: Контекст (shared state) ────────────────────────────────────────────
echo "━━━ Тип 4: Общий контекст для передачи данных ━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  \$ python -m libs.q6ctl.q6cli ctx new demo"
python -m libs.q6ctl.q6cli ctx new demo

# Сохранить ring в контекст
python -c "
import sys, json
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key
ring = json.loads('''$RING_JSON''')
write_key('demo', 'ring', ring, source='hexpack:ring')
verify = json.loads('''$VERIFY''')
write_key('demo', 'verify_trimat', verify, source='hextrimat:verify')
"

echo ""
echo "  \$ python -m libs.q6ctl.q6cli ctx show demo:"
python -m libs.q6ctl.q6cli ctx show demo
echo ""

echo "  \$ python -m libs.q6ctl.q6cli ctx get demo ring  (фрагмент):"
python -m libs.q6ctl.q6cli ctx get demo ring | head -10
echo "  ..."
echo ""

# ── Итог ───────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Сводка интерфейсов Q6:                                        ║"
echo "║                                                                  ║"
echo "║  Tier 1 — Прямой:  python -m projects.<mod>.<glyphs> <cmd>     ║"
echo "║  Tier 2 — JSON:    ... --json <cmd> | python -c 'import json..'║"
echo "║  Tier 3 — Оркестр: python -m libs.q6ctl.q6cli run <SC-N>       ║"
echo "║  Tier 4 — Контекст: q6ctl ctx new/show/get <session>           ║"
echo "║                                                                  ║"
echo "║  Скрипты пайплайнов:                                            ║"
echo "║    scripts/sc1_herman_cipher.sh    — K5×K1                     ║"
echo "║    scripts/sc4_genomic_iching.sh   — K4×K6                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

if [ $SHOW_ALL -eq 1 ]; then
    echo ""
    echo "  Полный список кластеров:"
    python -m libs.q6ctl.q6cli list clusters
fi
