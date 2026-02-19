# meta — монорепо hex-проектов

Монорепо для экосистемы проектов на основе **граф Q6** (система гексаграмм).

64 гексаграммы образуют 6-мерный гиперкуб: каждый узел — гексаграмма,
каждое ребро — изменение одной черты. Эта структура является основой
для всех проектов репозитория.

---

## Структура

```
meta/
├── libs/
│   └── hexcore/       — общая библиотека графа Q6 (Python)
├── projects/
│   ├── hexnav/        — интерактивный навигатор по Q6
│   ├── hexca/         — клеточный автомат (64 состояния)
│   ├── hexpath/       — абстрактная стратегическая игра
│   ├── hexforth/      — стековый язык в духе Forth
│   ├── karnaugh6/     — карта Карно для 6 переменных
│   ├── hexspec/       — язык спецификации протоколов
│   ├── hexgraph/      — теория графов (подграфы, спектр, изоморфизм)
│   ├── hexvis/        — визуализация (ASCII, DOT, SVG)
│   ├── hexcode/       — двоичные линейные коды в Q6
│   ├── hexlearn/      — ML на Q6 (k-NN, k-медоиды, Байес, Марков)
│   ├── hexopt/        — оптимизация на Q6 (SA, GA, TS, LocalSearch)
│   ├── hexring/       — булевы функции (WHT, ANF, bent, RM-коды)
│   ├── hexsym/        — группа автоморфизмов Aut(Q6), орбиты, лемма Бернсайда
│   ├── hexnet/        — Q6 как коммуникационная сеть (маршрутизация, надёжность)
│   ├── hexcrypt/      — криптографические примитивы (S-блок, DDT, LAT, Фейстель)
│   ├── hexstat/       — теория информации и статистика на Q6
│   ├── hexgeom/       — метрическая геометрия (шары Хэмминга, Вороной, пакинг)
│   ├── hexdim/        — размерности: тессеракты Q4, куб Q3, Q6=Q3×Q3, псевдо-QR
│   ├── hexalg/        — гармонический анализ (WHT/Фурье, свёртка, граф Кэли, bent)
│   ├── hexphys/       — статистическая физика (цепочка Изинга, Монте-Карло, кванты)
│   ├── hexgf/         — поле Галуа GF(2^6): умножение, след, цикл. классы, BCH
│   ├── hexmat/        — линейная алгебра над GF(2): матрицы, GL(6,2), линейные коды
│   ├── hexbio/        — биоинформатика: генетический код как граф мутаций на Q6
│   └── hexlat/        — булева решётка B₆: Мёбиус, цепи, антицепи, многочлены
├── docs/
│   └── q6-math.md     — математические основы графа Q6
├── flower_shop.py     — пример CLI-приложения (не hex-проект)
└── README.md
```

---

## Проекты

| Проект | Область | Ключевые файлы | Статус |
|---|---|---|---|
| [hexnav](projects/hexnav/) | Инструмент / навигация | `hexnav.py` | реализован |
| [hexca](projects/hexca/) | Наука / симуляция | `hexca.py`, `rules.py`, `animate.py` | реализован |
| [hexpath](projects/hexpath/) | Игры | `game.py`, `cli.py`, `puzzle.py` | реализован |
| [hexforth](projects/hexforth/) | Языки программирования | `interpreter.py`, `compiler.py`, `verifier.py` | реализован |
| [karnaugh6](projects/karnaugh6/) | Образование / электроника | `minimize.py` | реализован |
| [hexspec](projects/hexspec/) | Формальные методы | `verifier.py`, `generator.py` | реализован |
| [hexgraph](projects/hexgraph/) | Теория графов | `hexgraph.py` | реализован |
| [hexvis](projects/hexvis/) | Визуализация | `hexvis.py` | реализован |
| [hexcode](projects/hexcode/) | Теория кодирования | `hexcode.py` | реализован |
| [hexlearn](projects/hexlearn/) | Машинное обучение | `hexlearn.py` | реализован |
| [hexopt](projects/hexopt/) | Оптимизация | `hexopt.py` | реализован |
| [hexring](projects/hexring/) | Булевы функции / кольца | `hexring.py` | реализован |
| [hexsym](projects/hexsym/) | Теория групп / симметрия | `hexsym.py` | реализован |
| [hexnet](projects/hexnet/) | Коммуникационные сети | `hexnet.py` | реализован |
| [hexcrypt](projects/hexcrypt/) | Симметричная криптография | `hexcrypt.py` | реализован |
| [hexstat](projects/hexstat/) | Теория информации / статистика | `hexstat.py` | реализован |
| [hexgeom](projects/hexgeom/) | Метрическая геометрия | `hexgeom.py` | реализован |
| [hexdim](projects/hexdim/) | Размерности и проекции | `hexdim.py` | реализован |
| [hexalg](projects/hexalg/) | Гармонический анализ / алгебра | `hexalg.py` | реализован |
| [hexphys](projects/hexphys/) | Статистическая физика | `hexphys.py` | реализован |
| [hexgf](projects/hexgf/) | Поле Галуа GF(2^6) | `hexgf.py` | реализован |
| [hexmat](projects/hexmat/) | Линейная алгебра / GF(2) | `hexmat.py` | реализован |
| [hexbio](projects/hexbio/) | Биоинформатика | `hexbio.py` | реализован |
| [hexlat](projects/hexlat/) | Булева решётка / poset | `hexlat.py` | реализован |

### Быстрый старт

```bash
# Навигатор по Q6 (интерактивный)
python3 projects/hexnav/hexnav.py 42

# Клеточный автомат (1D и 2D)
python3 projects/hexca/hexca.py --rule xor_rule --steps 12
python3 projects/hexca/hexca.py --mode 2d --rule conway_b3s23 --width 40 --height 15 --steps 5

# Стратегическая игра на Q6 (человек vs AI)
python3 projects/hexpath/cli.py

# Минимизатор булевых функций (6 переменных, Куайн–МакКласки)
python3 projects/karnaugh6/minimize.py 0 1 2 3 4 5 6 7 --table

# HexForth: запуск программы
python3 projects/hexforth/interpreter.py projects/hexforth/examples/hello.hf

# Головоломка на Q6 (однопользовательский режим)
python3 projects/hexpath/puzzle.py list
python3 projects/hexpath/puzzle.py play --id 0

# Граф-теоретический анализ Q6
python3 projects/hexgraph/hexgraph.py q6 --spectrum
python3 projects/hexgraph/hexgraph.py layer 3
python3 projects/hexgraph/hexgraph.py hamilton 0 1 3 2 6 7 5 4 --cycle

# Визуализация Q6
python3 projects/hexvis/hexvis.py grid --highlight 0 42 63
python3 projects/hexvis/hexvis.py auto 0 63 --grid
python3 projects/hexvis/hexvis.py hexagram 42
# HexForth: компиляция в Python
python3 projects/hexforth/compiler.py projects/hexforth/examples/hello.hf --target python

# Двоичные линейные коды в Q6
python3 projects/hexcode/hexcode.py standard
python3 projects/hexcode/hexcode.py decode hex312 "101010"
python3 projects/hexcode/hexcode.py bounds

# Машинное обучение на Q6
python3 projects/hexlearn/hexlearn.py kmeans --k 4
python3 projects/hexlearn/hexlearn.py markov --start 0 --steps 20 --mixing
python3 projects/hexlearn/hexlearn.py knn --k 3 --query 42
python3 projects/hexlearn/hexlearn.py bayes --query 42

# Оптимизация на Q6
python3 projects/hexopt/hexopt.py hexagram all
python3 projects/hexopt/hexopt.py subset dominating ls
python3 projects/hexopt/hexopt.py compare --trials 10

# Булевы функции на Q6
python3 projects/hexring/hexring.py info bent
python3 projects/hexring/hexring.py table
python3 projects/hexring/hexring.py wht bent
python3 projects/hexring/hexring.py rm 2 --encode $(python3 -c "print('1'*22)")
python3 projects/hexring/hexring.py bent --n 5

# HexSpec: верификация автомата
python3 projects/hexspec/verifier.py projects/hexspec/examples/tcp.json
# HexSpec: генерация тестовых сценариев
python3 projects/hexspec/generator.py projects/hexspec/examples/tcp.json --coverage all
python3 projects/hexspec/generator.py projects/hexspec/examples/tcp.json --format hexforth

# Группа автоморфизмов Q6 (Aut(Q6) = B₆ = S₆ ⋉ (Z₂)⁶)
python3 projects/hexsym/hexsym.py info
python3 projects/hexsym/hexsym.py orbits --group s6
python3 projects/hexsym/hexsym.py burnside --colors 2 --group s6
python3 projects/hexsym/hexsym.py subsets --k 3 --group s6
python3 projects/hexsym/hexsym.py edge-orbits

# Q6 как коммуникационная сеть
python3 projects/hexnet/hexnet.py route 0 63
python3 projects/hexnet/hexnet.py broadcast 0
python3 projects/hexnet/hexnet.py stats
python3 projects/hexnet/hexnet.py percolation --p 0.3
python3 projects/hexnet/hexnet.py traffic
python3 projects/hexnet/hexnet.py hamilton

# Криптографические примитивы на Q6
python3 projects/hexcrypt/hexcrypt.py info random
python3 projects/hexcrypt/hexcrypt.py table affine
python3 projects/hexcrypt/hexcrypt.py stream 42 64
python3 projects/hexcrypt/hexcrypt.py feistel demo
python3 projects/hexcrypt/hexcrypt.py search 16

# Статистика и теория информации на Q6
python3 projects/hexstat/hexstat.py info
python3 projects/hexstat/hexstat.py sample 2000
python3 projects/hexstat/hexstat.py walk 10000
python3 projects/hexstat/hexstat.py entropy
python3 projects/hexstat/hexstat.py test

# Метрическая геометрия на Q6
python3 projects/hexgeom/hexgeom.py ball 0 2
python3 projects/hexgeom/hexgeom.py voronoi
python3 projects/hexgeom/hexgeom.py interval 0 63
python3 projects/hexgeom/hexgeom.py packing
python3 projects/hexgeom/hexgeom.py bounds

# Q6 как 6D-гиперкуб: тессеракты, псевдо-QR, Q12
python3 projects/hexdim/hexdim.py info
python3 projects/hexdim/hexdim.py hexagram 42
python3 projects/hexdim/hexdim.py tesseracts
python3 projects/hexdim/hexdim.py grid trigram
python3 projects/hexdim/hexdim.py gray
python3 projects/hexdim/hexdim.py q12

# Гармонический анализ на Q6 (WHT = Фурье, свёртка, граф Кэли, bent-функции)
python3 projects/hexalg/hexalg.py characters
python3 projects/hexalg/hexalg.py spectrum
python3 projects/hexalg/hexalg.py convolution
python3 projects/hexalg/hexalg.py cayley 1 2 4
python3 projects/hexalg/hexalg.py subgroup [1,2,4]
python3 projects/hexalg/hexalg.py bent

# Статистическая физика на Q6 (цепочка Изинга, ян-газ, Метрополис, кванты)
python3 projects/hexphys/hexphys.py ising 1.0
python3 projects/hexphys/hexphys.py yang 1.0
python3 projects/hexphys/hexphys.py mcmc 1.0 1.0
python3 projects/hexphys/hexphys.py quantum
python3 projects/hexphys/hexphys.py correlator 1.0 1.0

# Поле Галуа GF(2^6) (умножение, след, циклотомические классы, BCH)
python3 projects/hexgf/hexgf.py info
python3 projects/hexgf/hexgf.py mul 7 3
python3 projects/hexgf/hexgf.py power
python3 projects/hexgf/hexgf.py cosets
python3 projects/hexgf/hexgf.py minpoly 2
python3 projects/hexgf/hexgf.py trace

# Линейная алгебра над GF(2) на Q6 (матрицы, GL(6,2), ранг, коды)
python3 projects/hexmat/hexmat.py info
python3 projects/hexmat/hexmat.py rank
python3 projects/hexmat/hexmat.py code

# Биоинформатика: генетический код на Q6 (кодоны = гексаграммы, граф мутаций)
python3 projects/hexbio/hexbio.py info
python3 projects/hexbio/hexbio.py codon AUG
python3 projects/hexbio/hexbio.py mutation UUU UUC
python3 projects/hexbio/hexbio.py graph
python3 projects/hexbio/hexbio.py wobble

# Булева решётка B₆ = Q6 как poset (частичный порядок, Мёбиус, антицепи)
python3 projects/hexlat/hexlat.py info
python3 projects/hexlat/hexlat.py interval 5 63
python3 projects/hexlat/hexlat.py mobius 0 63
python3 projects/hexlat/hexlat.py chains
python3 projects/hexlat/hexlat.py antichain
```

---

## Общая библиотека: hexcore

Все проекты используют `libs/hexcore` как общее ядро.
Она реализует граф Q6 без внешних зависимостей (только стандартная библиотека Python).

```python
# Пример использования из любого проекта
import sys
sys.path.insert(0, '../../')  # или настроить PYTHONPATH

from libs.hexcore.hexcore import neighbors, shortest_path, render

h = 42
print(f"Гексаграмма {h}:")
print(render(h))
print(f"Соседи: {neighbors(h)}")
print(f"Путь 0→42: {shortest_path(0, 42)}")
```

Запуск демо:

```bash
python3 libs/hexcore/hexcore.py
```

---

## Математика: что такое Q6

```
64 гексаграммы = 2^6 вершин 6-мерного гиперкуба Q6
Каждая вершина: 6-битное число 0..63
Каждое ребро:   изменение ровно 1 бита (расстояние Хэмминга = 1)
Степень каждой вершины: ровно 6
Диаметр графа: 6 (максимальный путь = переворот всех 6 черт)
```

Граф Q6 — это граф Кэли группы (Z₂)⁶. Он появляется в:
- теории кодирования (коды Хэмминга, коды Грея)
- квантовых вычислениях (ворота Паули-X на 6 кубитах)
- синтезе цифровых схем (карты Карно)
- теории автоматов и формальных языков

Подробнее: [docs/q6-math.md](docs/q6-math.md)

---

## С чего начать

1. Прочитать `libs/hexcore/README.md` — понять ядро
2. Запустить `python3 libs/hexcore/hexcore.py` — увидеть граф в действии
3. Запустить `python3 projects/hexnav/hexnav.py` — интерактивно исследовать Q6
4. Выбрать проект из таблицы выше и изучить его README

---

## Агент и модель

- Агент: Claude Code (Anthropic)
- Модель: claude-sonnet-4-6
- Назначение: разработка и сопровождение проектов монорепо
