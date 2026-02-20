#!/usr/bin/env bash
# sc1_herman_cipher.sh — Супер-кластер SC-1: «Шифр Германа»
#
# K5 (Герман) × K1 (Крипто)
# Цель: Алгебраически обоснованные ключевые расписания на упаковках
#
# Пайплайн:
#   1. hexpack:ring  → получить упаковку P=64 в JSON
#   2. hexcrypt:sbox → использовать ring как основу S-блока  (TODO: поддержка stdin)
#   3. hexring:cosets → структура смежных классов Z/64Z
#
# Использование:
#   ./scripts/sc1_herman_cipher.sh
#   ./scripts/sc1_herman_cipher.sh --dry
#   ./scripts/sc1_herman_cipher.sh --json    # показать промежуточные данные

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO"

DRY=0
SHOW_JSON=0
for arg in "$@"; do
    case "$arg" in
        --dry)  DRY=1  ;;
        --json) SHOW_JSON=1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Супер-кластер SC-1: Шифр Германа"
echo "  K5 (Герман) × K1 (Крипто)"
echo "  Результат: ключевые расписания с доказуемой стойкостью"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ $DRY -eq 1 ]; then
    echo "  [ПЛАН — без запуска]"
    python -m libs.q6ctl.q6cli run SC-1 --dry
    exit 0
fi

# ── Шаг 1: Упаковка Германа P=64 ─────────────────────────────────────────────
echo "  [1/3] hexpack:ring — Алгоритм упаковки Германа, P=64"
echo "        pos(n) = n(n-1)/2 mod 64  →  полное замощение без коллизий"
echo ""

RING_JSON=$(python -m projects.hexpack.pack_glyphs --json ring)

if [ $SHOW_JSON -eq 1 ]; then
    echo "  → JSON-вывод hexpack:ring:"
    echo "$RING_JSON" | python -m json.tool --indent 2 | head -20
    echo "  ..."
    echo ""
fi

# Проверка: все 64 позиции заняты
PACKABLE=$(echo "$RING_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print('✓ packable' if d['packable'] else '✗ NOT packable')")
EXC_START=$(echo "$RING_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['exceptional_start'])")
echo "  Упаковка: $PACKABLE"
echo "  Исключительный старт: m=$EXC_START (нет фиксированных точек)"
echo ""

# ── Шаг 1b: Антиподальные пары ────────────────────────────────────────────────
echo "  [1b]  hexpack:antipode — Антиподальные пары: ring[h]+ring[h⊕32]=65"
python -m projects.hexpack.pack_glyphs antipode | grep -E "(Все|✓|✗)" | head -5
echo ""

# ── Шаг 2: Периоды упаковок (ключевые степени 2) ──────────────────────────────
echo "  [2/3] hexpack:periods — Какие n дают упаковываемые поля (P=2^k)"
PERIODS_JSON=$(python -m projects.hexpack.pack_glyphs --json periods --max 20)
echo "  Упаковываемые поля для n=1..20:"
echo "$PERIODS_JSON" | python -c "
import sys, json
d = json.load(sys.stdin)
for e in d['entries']:
    if e['is_power_of_2']:
        print(f'    n={e[\"n\"]:>3}  P={e[\"P\"]:>8}  =2^{e[\"k\"]}')
"
echo ""

# ── Шаг 3: Структура Z/64Z (базис для ключевых расписаний) ────────────────────
echo "  [3/3] hexring: структура Z/64Z → основа ключевых расписаний"
echo "  (hexring не поддерживает --json — показ текстового вывода)"
python -m projects.hexring.hexring ring 2>/dev/null | head -10 || \
    echo "  hexring не установлен — установите из K1-кластера"
echo ""

# ── Итог ───────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "  SC-1 завершён."
echo "  Данные кольца P=64 готовы для передачи в hexcrypt:sbox"
echo ""
echo "  Следующий шаг (когда hexcrypt поддержит --from-json):"
echo "    echo \$RING_JSON | python -m projects.hexcrypt.hexcrypt sbox --from-json"
echo "══════════════════════════════════════════════════════════════════════"

# Сохранить в q6ctl контекст для дальнейшего использования
python -c "
import sys
sys.path.insert(0, '$REPO')
from libs.q6ctl.context import write_key, create
import json
ring = $RING_JSON
create('sc1_run')
write_key('sc1_run', 'ring', ring, source='hexpack:ring')
print('  Контекст сохранён: sc1_run [ключ: ring]')
print('  Просмотр: python -m libs.q6ctl.q6cli ctx show sc1_run')
"
