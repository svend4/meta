"""spin_glyphs — Модель Изинга на Q6 через систему глифов.

Каждый глиф (0..63) — конфигурация 6 спинов:
  σᵢ = 2·bitᵢ − 1 ∈ {−1, +1}  (инь = −1, янь = +1)

Гамильтониан периодической цепочки Изинга:
  H(h) = −J Σᵢ σᵢσᵢ₊₁  (B=0 по умолчанию)

Распределение Больцмана:
  P(h|β) = exp(−β E(h)) / Z(β),  Z(β) = Σ_{h=0}^{63} exp(−β E(h))

Визуализация:
  • energy — 64 глифа, раскрашенных по энергии E(h) ∈ {−6,−4,...,+6}
  • gibbs  — 64 глифа, раскрашенных по весу P(h|β) при данном β
  • phase  — кривая намагниченности ⟨M⟩(β) и теплоёмкости C(β)
  • domain — «доменные стенки»: глифы по числу переходов ↑↓ в спиновой цепочке

Команды CLI:
  energy              — ландшафт энергий всех 64 конфигураций
  gibbs  <beta>       — распределение Больцмана при температуре 1/β
  phase               — фазовая кривая: ⟨M⟩ и C как функции β
  domain              — доменные стенки в спиновых конфигурациях
"""

from __future__ import annotations
import sys
import math
import argparse

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from projects.hexphys.hexphys import IsingChain, ising_spins
from libs.hexcore.hexcore import yang_count
from projects.hexvis.hexvis import (
    render_glyph,
    _YANG_ANSI, _YANG_BG, _RESET, _BOLD,
)

# ---------------------------------------------------------------------------
# Цвета энергии: от минимума (−6, ферромагнитный) до максимума (+6)
# ---------------------------------------------------------------------------

# E(h) для J=1, B=0: чётные значения от -6 до +6 → 7 уровней
_E_COLORS = {
    -6: '\033[38;5;27m',    # глубокий синий (ферромагнитный минимум)
    -4: '\033[38;5;39m',    # голубой
    -2: '\033[38;5;82m',    # зелёный
     0: '\033[38;5;238m',   # серый (нейтрально)
     2: '\033[38;5;208m',   # оранжевый
     4: '\033[38;5;196m',   # красный
     6: '\033[38;5;226m',   # жёлтый (антиферромагнитный максимум)
}


def _energy_color(e: float, highlight: bool = False) -> str:
    key = min(_E_COLORS, key=lambda k: abs(k - e))
    c = _E_COLORS[key]
    return (_BOLD + c) if highlight else c


def _domain_walls(h: int) -> int:
    """Число переходов σᵢ ≠ σᵢ₊₁ в замкнутой цепочке."""
    sigma = ising_spins(h)
    n = 6
    return sum(1 for i in range(n) if sigma[i] != sigma[(i + 1) % n])


# ---------------------------------------------------------------------------
# 1. Ландшафт энергий
# ---------------------------------------------------------------------------

def render_energy_landscape(J: float = 1.0, B: float = 0.0,
                             color: bool = True) -> str:
    """
    8×8 сетка глифов — все 64 конфигурации, раскрашены по E(h).

    Синие глифы = ферромагнитные (все спины одинаковы, E = −6).
    Жёлтые глифы = антиферромагнитные (чередование, E = +6).
    """
    chain = IsingChain(J=J, B=B, periodic=True)
    energies = [chain.energy(h) for h in range(64)]
    e_min = min(energies)
    e_max = max(energies)

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Ландшафт энергий Изинга  J={J:+.2f}  B={B:+.2f}')
    lines.append(f'  E_min={e_min:.1f} (ферромагнитный)   E_max={e_max:.1f} (АФМ)')
    lines.append('  Синий=низкая E,  серый=нейтральная,  жёлтый=высокая E')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            e = energies[h]
            rows3 = render_glyph(h)
            if color:
                is_extreme = (e == e_min or e == e_max)
                c = _energy_color(e, highlight=is_extreme)
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            e = energies[h]
            if color:
                c = _energy_color(e)
                lbl.append(f'{c}E={e:+.0f}{_RESET}')
            else:
                lbl.append(f'E={e:+.0f}')
        lines.append('  ' + '   '.join(lbl))
        lines.append('')

    # Гистограмма по уровням энергии
    from collections import Counter
    hist = Counter(int(e) for e in energies)
    lines.append('  Вырождение по уровням энергии:')
    for e_level in sorted(hist):
        count = hist[e_level]
        if color:
            c = _energy_color(e_level)
            lines.append(f'    {c}E={e_level:+3d}: {count:2d} конфигураций{_RESET}')
        else:
            lines.append(f'    E={e_level:+3d}: {count:2d} конфигураций')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 2. Распределение Больцмана
# ---------------------------------------------------------------------------

def render_gibbs(beta: float, J: float = 1.0, B: float = 0.0,
                 color: bool = True) -> str:
    """
    8×8 сетка глифов, раскрашенных по вероятности P(h|β).

    При β→∞: только основное состояние (ярко).
    При β→0: равномерное распределение.
    При β<0: «инвертированная температура» — антиферромагнитный порядок.
    """
    chain = IsingChain(J=J, B=B, periodic=True)
    weights = [math.exp(-beta * chain.energy(h)) for h in range(64)]
    Z = sum(weights)
    probs = [w / Z for w in weights]
    p_max = max(probs)
    p_min = min(probs)

    mean_E = sum(chain.energy(h) * probs[h] for h in range(64))
    mean_M = sum(((bin(h).count('1') - 3) / 3.0) * probs[h] for h in range(64))

    T_str = f'T=1/β={1/beta:.3f}' if abs(beta) > 1e-9 else 'T=∞'
    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Распределение Больцмана  β={beta:.3f}  ({T_str})')
    lines.append(f'  J={J:+.2f}  B={B:+.2f}')
    lines.append(f'  ⟨E⟩={mean_E:.3f}   ⟨M⟩={mean_M:.3f}   Z={Z:.4f}')
    lines.append('  Цвет: яркий=высокая вероятность,  тёмный=низкая')
    lines.append('═' * 64)

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            p = probs[h]
            rows3 = render_glyph(h)
            if color:
                # Нормируем на [0,6] для выбора цвета
                if p_max > p_min:
                    level = int(6 * (p - p_min) / (p_max - p_min))
                else:
                    level = 3
                level = max(0, min(6, level))
                is_max = (p == p_max)
                c = (_YANG_BG[level] + _BOLD) if is_max else _YANG_ANSI[level]
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            p = probs[h]
            if color:
                lvl = int(6 * (p - p_min) / (p_max - p_min + 1e-15))
                lvl = max(0, min(6, lvl))
                c = _YANG_ANSI[lvl]
                lbl.append(f'{c}{p:.3f}{_RESET}')
            else:
                lbl.append(f'{p:.3f}')
        lines.append('  ' + ' '.join(lbl))
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Фазовая кривая
# ---------------------------------------------------------------------------

def render_phase_curve(J: float = 1.0, color: bool = True) -> str:
    """
    ASCII-кривые ⟨M⟩(β) и C(β) для β = 0 .. 3.

    1D цепочка Изинга без поля: нет фазового перехода при конечном β.
    Но поведение ⟨M⟩ и C показывает характерное насыщение.
    """
    chain = IsingChain(J=J, B=0.0, periodic=True)

    betas = [i * 0.1 for i in range(31)]  # 0.0 .. 3.0
    Ms = [chain.magnetization(b) for b in betas]
    Cs = [chain.heat_capacity(b) if b > 0 else 0.0 for b in betas]

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Фазовая кривая модели Изинга  J={J:+.2f}  (B=0)')
    lines.append('  1D цепочка не имеет фазового перехода при T>0')
    lines.append('═' * 64)

    # Намагниченность — ASCII-график
    height = 12
    width = 31
    lines.append(f'\n  ⟨M⟩(β)  от β=0 (T=∞) до β=3 (T=0.33)\n')

    m_max = max(abs(m) for m in Ms) or 1.0
    grid = [[' '] * width for _ in range(height)]
    for i, m in enumerate(Ms):
        row = int((height - 1) * (1 - (m + m_max) / (2 * m_max)))
        row = max(0, min(height - 1, row))
        grid[row][i] = '●' if color else '*'

    for ri, row in enumerate(grid):
        m_val = m_max * (1 - 2 * ri / (height - 1))
        if color:
            lvl = int(6 * (m_val + m_max) / (2 * m_max))
            c = _YANG_ANSI[max(0, min(6, lvl))]
            lines.append(f'  {c}{m_val:+.2f}{_RESET} │ ' + ''.join(row))
        else:
            lines.append(f'  {m_val:+.2f} │ ' + ''.join(row))
    lines.append('         └' + '─' * width)
    lines.append('           β=0' + ' ' * (width - 8) + 'β=3.0')

    # Теплоёмкость
    lines.append(f'\n  C(β)  теплоёмкость\n')
    c_max = max(Cs) or 1.0
    grid2 = [[' '] * width for _ in range(height)]
    for i, cv in enumerate(Cs):
        row = int((height - 1) * (1 - cv / c_max))
        row = max(0, min(height - 1, row))
        grid2[row][i] = '●' if color else '*'

    for ri, row in enumerate(grid2):
        cv_val = c_max * (1 - ri / (height - 1))
        if color:
            lvl = int(6 * (1 - ri / (height - 1)))
            c = _YANG_ANSI[max(0, min(6, lvl))]
            lines.append(f'  {c}{cv_val:.4f}{_RESET} │ ' + ''.join(row))
        else:
            lines.append(f'  {cv_val:.4f} │ ' + ''.join(row))
    lines.append('          └' + '─' * width)
    lines.append('            β=0' + ' ' * (width - 8) + 'β=3.0')

    # Аналитические значения
    lines.append('')
    lines.append('  β=0.0:  ⟨M⟩={:.4f}  C={:.4f}'.format(Ms[0], Cs[0]))
    lines.append('  β=1.0:  ⟨M⟩={:.4f}  C={:.4f}'.format(Ms[10], Cs[10]))
    lines.append('  β=2.0:  ⟨M⟩={:.4f}  C={:.4f}'.format(Ms[20], Cs[20]))
    lines.append('  β=3.0:  ⟨M⟩={:.4f}  C={:.4f}'.format(Ms[30], Cs[30]))
    lines.append('')
    lines.append('  Корреляционная длина ξ(β=1) = {:.3f}'.format(
        chain.correlation_length(1.0)))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4. Доменные стенки
# ---------------------------------------------------------------------------

def render_domain_walls(J: float = 1.0, color: bool = True) -> str:
    """
    8×8 сетка глифов, раскрашенных по числу доменных стенок.

    Доменная стенка = граница σᵢ≠σᵢ₊₁.
    Возможные значения: 0 (ферромагнетик), 2, 4, 6 (антиферромагнетик).
    """
    walls = [_domain_walls(h) for h in range(64)]

    # Цвета по числу стенок (0, 2, 4, 6)
    _W_COLORS = {
        0: _YANG_ANSI[1],   # синий: 0 стенок
        2: _YANG_ANSI[2],   # голубой: 2 стенки
        4: _YANG_ANSI[4],   # оранжевый: 4 стенки
        6: _YANG_ANSI[6],   # жёлтый: 6 стенок
    }

    lines: list[str] = []
    lines.append('═' * 64)
    lines.append(f'  Доменные стенки в 6-спиновой цепочке Изинга (J={J:+.2f})')
    lines.append('  Синий=0 стенок (FM),  жёлтый=6 стенок (AFM)')
    lines.append('  Нечётное число стенок невозможно (периодические ГУ)')
    lines.append('═' * 64)

    # Считаем вырождение
    from collections import Counter
    count = Counter(walls)
    for w_val in sorted(count):
        c = _W_COLORS.get(w_val, _YANG_ANSI[3]) if color else ''
        r = _RESET if color else ''
        lines.append(f'  {c}{w_val} стенки: {count[w_val]:2d} конфигураций{r}')
    lines.append('')

    for row in range(8):
        glyph_rows = [None] * 8  # type: ignore
        for col in range(8):
            h = row * 8 + col
            w = walls[h]
            rows3 = render_glyph(h)
            if color:
                c = _W_COLORS.get(w, _YANG_ANSI[3])
                rows3 = [c + r + _RESET for r in rows3]
            glyph_rows[col] = rows3  # type: ignore

        for ri in range(3):
            lines.append('  ' + '  '.join(glyph_rows[col][ri] for col in range(8)))  # type: ignore
        lbl = []
        for col in range(8):
            h = row * 8 + col
            w = walls[h]
            if color:
                c = _W_COLORS.get(w, _YANG_ANSI[3])
                lbl.append(f'{c}w={w}{_RESET}')
            else:
                lbl.append(f'w={w}')
        lines.append('  ' + '    '.join(lbl))
        lines.append('')

    # Связь с энергией
    chain = IsingChain(J=J, B=0.0, periodic=True)
    lines.append('  E(h) = J·(6 − 2·walls(h))  при J>0:')
    lines.append(f'    0 стенок → E = {chain.energy(0):+.1f}   (основное состояние)')
    lines.append(f'    2 стенки → E = {chain.energy(3):+.1f}')
    lines.append(f'    4 стенки → E = {chain.energy(5):+.1f}')
    lines.append(f'    6 стенок → E = {chain.energy(21):+.1f}  (антиферромагнитное)')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='spin_glyphs',
        description='Модель Изинга на Q6 через глифы гексаграмм',
    )
    p.add_argument('--no-color', action='store_true', help='без ANSI-цветов')
    p.add_argument('-J', type=float, default=1.0,
                   help='константа связи (default=1.0)')
    p.add_argument('-B', type=float, default=0.0,
                   help='внешнее поле (default=0.0)')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('energy', help='ландшафт энергий всех 64 конфигураций')

    s = sub.add_parser('gibbs', help='распределение Больцмана при температуре 1/β')
    s.add_argument('beta', type=float, help='обратная температура β')

    sub.add_parser('phase', help='кривые ⟨M⟩(β) и C(β)')

    sub.add_parser('domain', help='доменные стенки в спиновых конфигурациях')

    return p


def main(argv: list[str] | None = None) -> None:
    p = _make_parser()
    args = p.parse_args(argv)
    color = not args.no_color

    if args.cmd == 'energy':
        print(render_energy_landscape(J=args.J, B=args.B, color=color))
    elif args.cmd == 'gibbs':
        print(render_gibbs(beta=args.beta, J=args.J, B=args.B, color=color))
    elif args.cmd == 'phase':
        print(render_phase_curve(J=args.J, color=color))
    elif args.cmd == 'domain':
        print(render_domain_walls(J=args.J, color=color))


if __name__ == '__main__':
    main()
