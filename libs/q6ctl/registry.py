"""registry.py — База данных всех модулей, кластеров и супер-кластеров Q6.

Структура:
  MODULES  — 41 модуль: {name: ModuleInfo}
  CLUSTERS — 8 кластеров: {id: ClusterInfo}
  SUPERCLUSTERS — 11 супер-кластеров: {id: SuperClusterInfo}

Каждый модуль знает:
  - где он живёт (путь к Python-модулю)
  - какие CLI-команды он поддерживает
  - поддерживает ли --json вывод
  - к каким кластерам принадлежит
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModuleInfo:
    name: str                     # краткое имя: 'hexpack'
    path: str                     # путь запуска: 'projects.hexpack.pack_glyphs'
    commands: list[str]           # доступные команды CLI
    cluster: str                  # ID кластера: 'K5'
    description: str              # одна строка
    json_ready: bool = False      # поддерживает --json


@dataclass
class ClusterInfo:
    id: str                       # 'K1'
    name: str                     # 'Cryptographic'
    modules: list[str]            # имена модулей
    description: str
    emergent: str                 # что возникает при совместной работе


@dataclass
class SuperClusterInfo:
    id: str                       # 'SC-1'
    name: str
    cluster_ids: list[str]
    description: str
    pipeline: list[str]           # пример pipeline: ['hexpack:ring', 'hexcrypt:sbox']
    emergent: str


# ─── Модули ───────────────────────────────────────────────────────────────────

MODULES: dict[str, ModuleInfo] = {
    # K1 — Криптографический кластер
    'hexcrypt': ModuleInfo(
        name='hexcrypt', path='projects.hexcrypt.sbox_glyphs',
        commands=['map', 'analyze', 'ddt', 'lat', 'cmp'],
        cluster='K1', json_ready=True,
        description='S-блоки и криптоанализ Q6 (NL, DDT, LAT, deg)',
    ),
    'hexring': ModuleInfo(
        name='hexring', path='projects.hexring.bent_glyphs',
        commands=['wht', 'tt', 'bent', 'anf', 'nl', 'ring-components'],
        cluster='K1', json_ready=True,
        description='Булевы функции WHT/bent/ANF на Q6',
    ),
    'hexgf': ModuleInfo(
        name='hexgf', path='projects.hexgf.hexgf',
        commands=['field', 'mult', 'poly'],
        cluster='K1', json_ready=False,
        description='Поле Галуа GF(2^6) — умножение и полиномы',
    ),
    'hexcode': ModuleInfo(
        name='hexcode', path='projects.hexcode.hexcode',
        commands=['encode', 'decode', 'distance'],
        cluster='K1', json_ready=False,
        description='Коды с исправлением ошибок на Q6',
    ),
    'karnaugh6': ModuleInfo(
        name='karnaugh6', path='projects.karnaugh6.kmap_glyphs',
        commands=['sbox-minimize', 'minimize', 'map', 'yang_parity', 'rank3', 'triangle'],
        cluster='K1', json_ready=True,
        description='Карты Карно 6 переменных + Q-M минимизация компонент S-блока',
    ),

    # K2 — Динамический кластер (CA/физика)
    'hexca': ModuleInfo(
        name='hexca', path='projects.hexca.ca_glyphs',
        commands=['evolve', 'all-rules', 'compact', 'history', 'diff', 'frames', 'stats'],
        cluster='K2', json_ready=True,
        description='Клеточные автоматы на гиперкубе Q6 (9 правил, JSON-эволюция)',
    ),
    'hexphys': ModuleInfo(
        name='hexphys', path='projects.hexphys.hexphys',
        commands=['lattice', 'energy', 'phase'],
        cluster='K2', json_ready=False,
        description='Физические модели на решётке Q6',
    ),
    'hexstat': ModuleInfo(
        name='hexstat', path='projects.hexstat.channel_glyphs',
        commands=['ca-entropy', 'bsc', 'kl', 'mutual', 'entropy'],
        cluster='K2', json_ready=True,
        description='Теория информации Q6: BSC, KL, ёмкость, классификация КА',
    ),
    'hexsym': ModuleInfo(
        name='hexsym', path='projects.hexsym.sym_glyphs',
        commands=['rule-orbits', 'yang', 'fixed', 'antipodal', 'burnside'],
        cluster='K2', json_ready=True,
        description='Aut(Q6)=B₆ симметрия: орбиты, Бёрнсайд, классификация КА',
    ),

    # K3 — Интеллектуальный кластер
    'hexlearn': ModuleInfo(
        name='hexlearn', path='projects.hexlearn.hexlearn',
        commands=['train', 'predict', 'cluster'],
        cluster='K3', json_ready=False,
        description='Машинное обучение на Q6-признаках',
    ),
    'hexopt': ModuleInfo(
        name='hexopt', path='projects.hexopt.hexopt',
        commands=['optimize', 'bayesian', 'pareto'],
        cluster='K3', json_ready=False,
        description='Байесовская оптимизация Q6-функций',
    ),
    'hexnet': ModuleInfo(
        name='hexnet', path='projects.hexnet.hexnet',
        commands=['graph', 'centrality', 'flow'],
        cluster='K3', json_ready=False,
        description='Сетевой анализ Q6-графов',
    ),

    # K4 — Биологический кластер
    'hexbio': ModuleInfo(
        name='hexbio', path='projects.hexbio.codon_glyphs',
        commands=['codon-map', 'codon-entropy', 'grid', 'amino', 'highlight', 'neighbors', 'seq'],
        cluster='K4', json_ready=True,
        description='Генетический код Q6: 64 кодона → гексаграммы, энтропия (K4×K6×K2)',
    ),
    'hexdim': ModuleInfo(
        name='hexdim', path='projects.hexdim.hexdim',
        commands=['fractal', 'dim', 'scaling'],
        cluster='K4', json_ready=False,
        description='Фрактальная размерность Q6-структур',
    ),
    'hexpath': ModuleInfo(
        name='hexpath', path='projects.hexpath.hexpath',
        commands=['path', 'hamiltonian', 'eulerian'],
        cluster='K4', json_ready=False,
        description='Гамильтоновы и Эйлеровы пути в Q6',
    ),

    # K5 — Кластер Германа
    'hexpack': ModuleInfo(
        name='hexpack', path='projects.hexpack.pack_glyphs',
        commands=['ring', 'antipode', 'fixpoint', 'packable', 'magic', 'periods'],
        cluster='K5', json_ready=True,
        description='Алгоритм упаковки Германа: P=2^k → полное замощение',
    ),
    'hexpowerxy': ModuleInfo(
        name='hexpowerxy', path='projects.hexpowerxy.hexpowerxy',
        commands=['power', 'roots', 'cycles'],
        cluster='K5', json_ready=False,
        description='Степенные циклы в Q6 (x^y mod 64)',
    ),
    'hexellipse': ModuleInfo(
        name='hexellipse', path='projects.hexellipse.hexellipse',
        commands=['ellipse', 'foci', 'eccen'],
        cluster='K5', json_ready=False,
        description='Эллиптические орбиты и Q6-инварианты',
    ),
    'hexcrossrat': ModuleInfo(
        name='hexcrossrat', path='projects.hexcrossrat.hexcrossrat',
        commands=['crossratio', 'moebius', 'fixed'],
        cluster='K5', json_ready=False,
        description='Двойное отношение и группа Мёбиуса над Q6',
    ),

    # K6 — Кластер И-Цзин
    'hextrimat': ModuleInfo(
        name='hextrimat', path='projects.hextrimat.trimat_glyphs',
        commands=['codon-atlas', 'triangle', 'sums', 'bird', 'thoth', 'swastika', 'twins', 'center', 'verify'],
        cluster='K6', json_ready=True,
        description='Треугольная матрица И-Цзин Андреева: 64 гексаграммы + кодонный атлас',
    ),
    'hexnav': ModuleInfo(
        name='hexnav', path='projects.hexnav.nav_glyphs',
        commands=['codon-transitions', 'trigrams', 'layers', 'antipode', 'bits'],
        cluster='K6', json_ready=True,
        description='Навигация Q6: триграммы, BFS-слои, мутации как Q6-переходы',
    ),
    'hexspec': ModuleInfo(
        name='hexspec', path='projects.hexspec.hexspec',
        commands=['spectrum', 'resonance', 'harmonics'],
        cluster='K6', json_ready=False,
        description='Спектральный анализ Q6-последовательностей',
    ),
    'hexvis': ModuleInfo(
        name='hexvis', path='projects.hexvis.hexvis',
        commands=['glyph', 'grid', 'yang'],
        cluster='K6', json_ready=False,
        description='Визуализация гексаграмм Q6 (ANSI-глифы)',
    ),

    # K7 — Золотой кластер (φ)
    'hexgeom': ModuleInfo(
        name='hexgeom', path='projects.hexgeom.hexgeom',
        commands=['phi', 'pentagon', 'star'],
        cluster='K7', json_ready=False,
        description='Золотое сечение φ и геометрия Q6',
    ),
    'hexmobius': ModuleInfo(
        name='hexmobius', path='projects.hexmobius.hexmobius',
        commands=['strip', 'twist', 'topology'],
        cluster='K7', json_ready=False,
        description='Лента Мёбиуса и топология Q6',
    ),
    'hexscrew': ModuleInfo(
        name='hexscrew', path='projects.hexscrew.hexscrew',
        commands=['helix', 'pitch', 'chirality'],
        cluster='K7', json_ready=False,
        description='Спираль и хиральность в Q6',
    ),
    'hexphi': ModuleInfo(
        name='hexphi', path='projects.hexphi.hexphi',
        commands=['ratio', 'fibonacci', 'lucas'],
        cluster='K7', json_ready=False,
        description='Числа Фибоначчи и φ в структуре Q6',
    ),

    # K8 — Схематический кластер
    'hexmat': ModuleInfo(
        name='hexmat', path='projects.hexmat.hexmat',
        commands=['matrix', 'det', 'eigen'],
        cluster='K8', json_ready=False,
        description='Матричные операции над GF(2)',
    ),
    'hexalg': ModuleInfo(
        name='hexalg', path='projects.hexalg.hexalg',
        commands=['algebra', 'ideals', 'quotient'],
        cluster='K8', json_ready=False,
        description='Абстрактная алгебра: идеалы, фактор-кольца Q6',
    ),
    'hexlat': ModuleInfo(
        name='hexlat', path='projects.hexlat.hexlat',
        commands=['lattice', 'meet', 'join'],
        cluster='K8', json_ready=False,
        description='Решётки (Lattice) над подпространствами Q6',
    ),
    'hexperms': ModuleInfo(
        name='hexperms', path='projects.hexperms.hexperms',
        commands=['perms', 'cycle', 'sign'],
        cluster='K8', json_ready=False,
        description='Перестановки 64 элементов: цикловая структура',
    ),
    'hexuniqgrp': ModuleInfo(
        name='hexuniqgrp', path='projects.hexuniqgrp.hexuniqgrp',
        commands=['unique', 'generators', 'presentation'],
        cluster='K8', json_ready=False,
        description='Уникальные подгруппы Q6 и их генераторы',
    ),

    # Остальные (распределены по ближайшим кластерам)
    'hexbuffon': ModuleInfo(
        name='hexbuffon', path='projects.hexbuffon.hexbuffon',
        commands=['needle', 'pi', 'montecarlo'],
        cluster='K2', json_ready=False,
        description='Задача Бюффона и π через Q6-случайность',
    ),
    'hexcubenets': ModuleInfo(
        name='hexcubenets', path='projects.hexcubenets.hexcubenets',
        commands=['nets', 'unfold', 'hamming'],
        cluster='K8', json_ready=False,
        description='Развёртки гиперкуба Q6 (11 сетей куба)',
    ),
    'hexforth': ModuleInfo(
        name='hexforth', path='projects.hexforth.hexforth',
        commands=['eval', 'stack', 'words'],
        cluster='K3', json_ready=False,
        description='Стек-машина Forth на Q6-словах',
    ),
    'hexgraph': ModuleInfo(
        name='hexgraph', path='projects.hexgraph.hexgraph',
        commands=['plot', 'subgraph', 'color'],
        cluster='K4', json_ready=False,
        description='Графы Q6: раскраска, клики, независимые множества',
    ),
    'hexhept': ModuleInfo(
        name='hexhept', path='projects.hexhept.hexhept',
        commands=['heptagon', '7star', 'angle'],
        cluster='K7', json_ready=False,
        description='Правильный семиугольник и 7-лучевая звезда в Q6',
    ),
    'hexintermed': ModuleInfo(
        name='hexintermed', path='projects.hexintermed.hexintermed',
        commands=['mediant', 'stern', 'farey'],
        cluster='K7', json_ready=False,
        description='Дерево Штерна–Броко и Фарей-последовательности',
    ),
    'hexmat': ModuleInfo(
        name='hexmat', path='projects.hexmat.hexmat',
        commands=['matrix', 'rank', 'kernel'],
        cluster='K8', json_ready=False,
        description='Линейная алгебра над GF(2): ранг, ядро, образ',
    ),
    'hexmatroot': ModuleInfo(
        name='hexmatroot', path='projects.hexmatroot.hexmatroot',
        commands=['root', 'sqrt', 'modular'],
        cluster='K8', json_ready=False,
        description='Квадратные корни и n-е корни в кольцах Q6',
    ),
    'hexnumderiv': ModuleInfo(
        name='hexnumderiv', path='projects.hexnumderiv.hexnumderiv',
        commands=['deriv', 'newton', 'finite_diff'],
        cluster='K8', json_ready=False,
        description='Числовое дифференцирование функций Q6→Q6',
    ),
    'hexpolyenum': ModuleInfo(
        name='hexpolyenum', path='projects.hexpolyenum.hexpolyenum',
        commands=['poly', 'enumerate', 'burnside'],
        cluster='K8', json_ready=False,
        description='Перечисление полиномов над GF(2) лемма Бёрнсайда',
    ),
    'hextile': ModuleInfo(
        name='hextile', path='projects.hextile.hextile',
        commands=['tile', 'aperiodic', 'penrose'],
        cluster='K7', json_ready=False,
        description='Апериодическое мощение плоскости (Пенроуз) через Q6',
    ),
}


# ─── Кластеры ─────────────────────────────────────────────────────────────────

CLUSTERS: dict[str, ClusterInfo] = {
    'K1': ClusterInfo(
        id='K1', name='Криптографический',
        modules=['hexcrypt', 'hexring', 'hexgf', 'hexcode', 'karnaugh6'],
        description='Криптопримитивы: S-блоки, поля Галуа, коды',
        emergent='Доказуемо оптимальные S-блоки с алгебраическим обоснованием',
    ),
    'K2': ClusterInfo(
        id='K2', name='Динамический',
        modules=['hexca', 'hexphys', 'hexstat', 'hexsym', 'hexbuffon'],
        description='КА, физика, статистика, симметрия',
        emergent='Канонический атлас КА Q6 с теормодинамической классификацией',
    ),
    'K3': ClusterInfo(
        id='K3', name='Интеллектуальный',
        modules=['hexlearn', 'hexopt', 'hexnet', 'hexforth'],
        description='ML, оптимизация, графы, стек-вычисления',
        emergent='Байесовская оптимизация Q6-гиперпараметров',
    ),
    'K4': ClusterInfo(
        id='K4', name='Биологический',
        modules=['hexbio', 'hexdim', 'hexpath', 'hexgraph'],
        description='Генетика, фракталы, пути',
        emergent='Геномные фрактальные отпечатки через Q6-кодоны',
    ),
    'K5': ClusterInfo(
        id='K5', name='Германа',
        modules=['hexpack', 'hexpowerxy', 'hexellipse', 'hexcrossrat'],
        description='Теория упаковок Германа и алгебраическая геометрия',
        emergent='Полный каталог упаковываемых полей с φ-инвариантами',
    ),
    'K6': ClusterInfo(
        id='K6', name='И-Цзин',
        modules=['hextrimat', 'hexnav', 'hexspec', 'hexvis'],
        description='Треугольная матрица, навигация, спектр гексаграмм',
        emergent='Автомат И-Цзин: КА переходов между гексаграммами',
    ),
    'K7': ClusterInfo(
        id='K7', name='Золотой',
        modules=['hexgeom', 'hexmobius', 'hexscrew', 'hexphi',
                 'hexhept', 'hexintermed', 'hextile'],
        description='Золотое сечение, спирали, топология',
        emergent='φ как алгебраический инвариант Q6',
    ),
    'K8': ClusterInfo(
        id='K8', name='Схематический',
        modules=['hexmat', 'hexalg', 'hexlat', 'hexperms', 'hexuniqgrp',
                 'hexcubenets', 'hexmatroot', 'hexnumderiv', 'hexpolyenum'],
        description='Линейная алгебра, решётки, перестановки',
        emergent='Полная классификация подструктур Q6',
    ),
}


# ─── Супер-кластеры ───────────────────────────────────────────────────────────

SUPERCLUSTERS: dict[str, SuperClusterInfo] = {
    'SC-1': SuperClusterInfo(
        id='SC-1', name='Шифр Германа',
        cluster_ids=['K5', 'K1'],
        description='Упаковки Германа: ring→S-box анализ (NL, δ, WHT компонент)',
        pipeline=['hexpack:ring',
                  'hexcrypt:analyze --from-ring',
                  'hexring:ring-components'],
        emergent='Ring→S-box: NL=0 (u=3 линейна), δ=64 (антипод). Ring=ключевые расписания, не S-блок.',
    ),
    'SC-2': SuperClusterInfo(
        id='SC-2', name='Платиновые S-блоки',
        cluster_ids=['K1', 'K8'],
        description='Минимальные схемы S-блоков через карты Карно',
        pipeline=['hexcrypt:sbox', 'karnaugh6:minimize', 'hexcode:encode'],
        emergent='Минимизированные S-блоки с оптимальным расстоянием Хэмминга',
    ),
    'SC-3': SuperClusterInfo(
        id='SC-3', name='Канонический атлас КА',
        cluster_ids=['K2', 'K8'],
        description='Классификация Q6 КА: энтропийная динамика × Aut(Q6)-симметрия',
        pipeline=['hexca:evolve',
                  'hexstat:ca-entropy --from-ca',
                  'hexsym:rule-orbits'],
        emergent=(
            'K2×K8: только identity эквивариантна под Aut(Q6). '
            'Правила с малым ян-дрейфом → Вольфрам I (сходимость). '
            'Aut(Q6) порядка 46080 разбивает ~2^64 правил на орбиты; '
            '7 ян-слоёв — главные K2-инварианты.'
        ),
    ),
    'SC-4': SuperClusterInfo(
        id='SC-4', name='Геномный И-Цзин',
        cluster_ids=['K4', 'K6'],
        description='ДНК-мутации как переходы по Q6 гиперкубу в пространстве гексаграмм',
        pipeline=['hexbio:codon-map',
                  'hextrimat:codon-atlas --from-codons',
                  'hexnav:codon-transitions --from-atlas'],
        emergent=(
            'K4×K6: transitions A↔G, C↔U = Q6-рёбра (1 бит); '
            'Watson-Crick пары A↔U, C↔G = 2-битные Q6-прыжки. '
            'Синонимичные мутации (биологически нейтральные) = навигация по строке треугольника Андреева. '
            '65% синонимичных мутаций остаются в одной строке. '
            'Ландшафт биологической приспособленности = граф Q6: кодон-кластеры = Q5-подкубы.'
        ),
    ),
    'SC-5': SuperClusterInfo(
        id='SC-5', name='AutoML для криптографии',
        cluster_ids=['K3', 'K1'],
        description='Байесовский поиск оптимальных S-блоков',
        pipeline=['hexopt:bayesian', 'hexcrypt:avalanche', 'hexlearn:predict'],
        emergent='Автоматический синтез S-блоков с заданным лавинным критерием',
    ),
    'SC-6': SuperClusterInfo(
        id='SC-6', name='Геномный КА: энтропийные аттракторы',
        cluster_ids=['K2', 'K4'],
        description='КА-правила Q6 как операторы эволюции генетического кода',
        pipeline=['hexca:all-rules',
                  'hexbio:codon-entropy --from-rules'],
        emergent=(
            'K2×K4: три уровня энтропии кода: H_равн=6.0 > H_деген=4.22 > H_ян=2.33 бит. '
            'majority_vote (K2): δH<0 → аттрактор ян=3 = GC~50% (K4-биологический отбор). '
            'xor_rule (K2): δH>0 → нейтральный дрейф = случайные мутации. '
            'ВЫВОД: биологический отбор GC ≡ majority_vote-аттрактор; '
            'нейтральная эволюция ≡ xor_rule-диффузия. Обе динамики в Q6.'
        ),
    ),
    'SC-7': SuperClusterInfo(
        id='SC-7', name='φ как Q6-инвариант',
        cluster_ids=['K7', 'K5'],
        description='Золотое сечение в упаковках и эллиптических орбитах',
        pipeline=['hexgeom:phi', 'hexpack:periods', 'hexellipse:foci'],
        emergent='φ проявляется в периодах упаковок и эксцентриситете орбит Q6',
    ),
    'TSC-1': SuperClusterInfo(
        id='TSC-1', name='Шифр + Симметрия',
        cluster_ids=['K5', 'K1', 'K8'],
        description='Упаковки + Карно + Aut(Q6) = алгебраически объяснённая крипто-слабость',
        pipeline=['hexpack:ring',
                  'karnaugh6:sbox-minimize --from-sbox',
                  'hexsym:sbox-symmetry --from-minimize'],
        emergent=(
            'K5→K8→K1: Германова антиподальная упаковка (ring[h]+ring[h⊕32]=65) '
            '↔ σ₃₂∈Aut(Q6) (K8) — все 64 пары: sbox[h⊕32]⊕sbox[h]=63. '
            'Карно (K1): f₀⊕f₁=x₀ (1 литерал!). Вывод: геометрия К5 '
            'ПРИНУЖДАЕТ к NL=0 через алгебраическую симметрию K8.'
        ),
    ),
    'TSC-2': SuperClusterInfo(
        id='TSC-2', name='AutoML-крипто',
        cluster_ids=['K3', 'K2', 'K1'],
        description='ML + КА-динамика + крипто = самообучающийся шифр',
        pipeline=['hexlearn:train', 'hexca:attractor', 'hexcrypt:rounds'],
        emergent='КА-аттракторы как источник непредсказуемости для шифра',
    ),
    'TSC-3': SuperClusterInfo(
        id='TSC-3', name='Геномный оракул',
        cluster_ids=['K4', 'K6', 'K3'],
        description='Биология + И-Цзин + ML = геномный оракул',
        pipeline=['hexbio:codon', 'hextrimat:twins', 'hexlearn:cluster', 'hexspec:resonance'],
        emergent='Предсказание мутаций через резонансные структуры матрицы Андреева',
    ),
    'MC': SuperClusterInfo(
        id='MC', name='Геномный Оракул Q6 (мега)',
        cluster_ids=['K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8'],
        description='Все 8 кластеров: полная Q6-система',
        pipeline=[
            'hexbio:codon',       # ДНК → Q6
            'hextrimat:triangle', # Q6 → матрица И-Цзин
            'hexpack:ring',       # матрица → упаковка
            'hexcrypt:sbox',      # упаковка → S-блок
            'hexca:evolve',       # S-блок → КА
            'hexstat:entropy',    # КА → энтропия
            'hexlearn:predict',   # энтропия → ML-оценка
        ],
        emergent='7-шаговый конвейер: ДНК → И-Цзин → Упаковка → Шифр → КА → Энтропия → Оракул',
    ),
}


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def get_module(name: str) -> ModuleInfo | None:
    return MODULES.get(name)


def get_cluster(cid: str) -> ClusterInfo | None:
    return CLUSTERS.get(cid)


def get_supercluster(sid: str) -> SuperClusterInfo | None:
    return SUPERCLUSTERS.get(sid)


def modules_in_cluster(cid: str) -> list[ModuleInfo]:
    c = CLUSTERS.get(cid)
    if not c:
        return []
    return [MODULES[m] for m in c.modules if m in MODULES]


def cluster_of_module(name: str) -> ClusterInfo | None:
    m = MODULES.get(name)
    if not m:
        return None
    return CLUSTERS.get(m.cluster)


def all_module_names() -> list[str]:
    return sorted(MODULES.keys())


def all_cluster_ids() -> list[str]:
    return sorted(CLUSTERS.keys())


def all_supercluster_ids() -> list[str]:
    return sorted(SUPERCLUSTERS.keys())


def json_ready_modules() -> list[str]:
    return [name for name, m in MODULES.items() if m.json_ready]
