# Roadmap: Новые проекты из 49 дополнительных PDF

**Дата:** 19 февраля 2026 г.
**Дополнение к:** `roadmap_implementations.md` (8 проектов из первых 33 PDF)
**Итого после добавления:** 15 проектов

---

## Проект 9: `hexpowerxy` — Уравнение X^Y = Y^X

**Источник:** PDF `4c5` — «Уравнение X^Y = Y^X (Первая теорема)»
**Область:** Математический анализ, нелинейные уравнения, золотое сечение
**Размер:** ~180 строк

### Алгоритм

```python
# Параметрическое решение (параметр t = Y/X):
X(t) = t ** (1 / (t - 1))
Y(t) = t ** (t / (t - 1))

# Свойство симметрии:
X(1/t) == Y(t)  # при замене t → 1/t переменные меняются

# «Золотые» уравнения (единственный корень φ = 1.618...):
# 1) x^(1/(x-1)) = (1 + 1/x)^x          → x = φ
# 2) x^(x/(x-1)) = (1 + 1/x)^(x+1)      → x = φ

# Теорема об исключении:
# Для всех X > 1, X ≠ e: ∃ Y ≠ X: X^Y = Y^X
# При X = e: нет другого решения
```

### Python API

```python
from hexpowerxy import PowerXY

pxy = PowerXY()

# Параметрическая кривая решений
curve = pxy.generate_curve(t_min=1.01, t_max=10.0, steps=1000)
# → список (X, Y) пар, всё X^Y = Y^X

# Для конкретного X найти Y
pxy.find_y(X=2.0)      # → Y = 4.0 (известное: 2^4 = 4^2)
pxy.find_y(X=3.0)      # → Y ≈ 2.478...
pxy.find_y(X=2.71828)  # → None (исключение: X=e)

# Проверка: X^Y == Y^X?
pxy.verify(X=2.0, Y=4.0)   # → True

# Золотые уравнения
pxy.golden_eq1(x=1.618)    # → |LHS - RHS| ≈ 0 (φ — корень)
pxy.find_golden_root()      # → φ = 1.6180339887...

# Анализ исключений
pxy.is_exception(X=2.71828) # → True (e — исключение)
pxy.exception_constant()    # → e = 2.71828...

# Визуализация кривой решений
pxy.plot_curve(ascii=True, width=60)  # ASCII-график (X, Y) кривой

# Таблица замечательных пар
pxy.notable_pairs()
# → [(2, 4), (e, e), (φ, φ²), (√2, 2), ...]
```

### CLI

```
python hexpowerxy.py --find-y 2.0
python hexpowerxy.py --curve --steps 100
python hexpowerxy.py --golden-root
python hexpowerxy.py --plot --width 60
python hexpowerxy.py --verify 2 4
```

### Почему уникально

- Wolfram Alpha и sympy решают числовые случаи, но нет:
  - библиотечного объекта `PowerXY` с кривой решений
  - встроенного анализа «исключения e» и связи с φ
  - генератора пар (X, Y) по параметру t

### Аудитория

Студенты матанализа, олимпиадники (задача «2^4 = 4^2, найти другие пары»), любители φ и e.

---

## Проект 10: `hexcrossrat` — Группа сечений R6

**Источник:** PDF `372` — «Замыкание сложного отношения»
**Область:** Проективная геометрия, теория групп, компьютерное зрение
**Размер:** ~280 строк

### Алгоритм

```python
# Двойное отношение 4 точек:
w = (A1,A3) * (A2,A4) / ((A3,A2) * (A1,A4))

# 6 значений при перестановках = группа R6:
r = [w, 1/w, 1-w, 1/(1-w), (w-1)/w, w/(w-1)]

# Правило умножения (r_i ∘ r_j = подстановка):
r3 ∘ r2 = 1/(1-(1-w)) = 1/w = r1

# Тождества:
∏r_i = 1    ΣR_i = 3

# R6 ≅ S3 (таблица Кэли 6×6)
# V4 = 4 перестановки {1234} с одним значением w
# S4 = V4 ⊗ R6  (V4 — нормальный делитель S4)
```

### Python API

```python
from hexcrossrat import CrossRatioGroup, cross_ratio

# Вычисление двойного отношения
w = cross_ratio(A1=1, A2=3, A3=2, A4=5)  # числа или точки

# Группа R6
r6 = CrossRatioGroup(w)

# Все 6 элементов
r6.elements()   # → [w, 1/w, 1-w, 1/(1-w), (w-1)/w, w/(w-1)]

# Умножение в группе
r6.multiply(0, 2)  # → индекс результата в списке
r6.compose(0, 2)   # → r3 ∘ r2 = r1

# Таблица Кэли
r6.cayley_table()  # → 6×6 таблица (numpy или красивый ASCII)

# Изоморфизм R6 ≅ S3
r6.isomorphism_to_s3()  # → биекция индексов R6 → S3

# Четырёхгруппа Клейна V4
r6.klein_four_group()  # → 4 перестановки {1234}, оставляющие w

# Разложение S4 = V4 ⊗ R6
r6.s4_decomposition()  # → описание смежных классов

# Тождества
r6.verify_product_identity()   # → ∏r_i = 1
r6.verify_sum_identity()       # → ΣR_i = 3

# Применение к проективным точкам
pts = [(0,1), (1,1), (2,1), (3,1)]  # 4 коллинеарные точки
w_proj = cross_ratio(*pts, projective=True)
r6_proj = CrossRatioGroup(w_proj)
```

### CLI

```
python hexcrossrat.py --cross-ratio 1 3 2 5
python hexcrossrat.py --r6-group 0.5
python hexcrossrat.py --cayley 0.5
python hexcrossrat.py --isomorphism 0.5
python hexcrossrat.py --s4-decomposition
```

### Почему уникально

SymPy и sage имеют `crossratio()`, но нет:
- класса `CrossRatioGroup` с операцией
- явной реализации `V4 ⊗ S3 = S4`
- связи с проективными преобразованиями в образовательной форме

Применение в компьютерном зрении (homography estimation использует cross-ratio как инвариант).

### Интеграция с Q6

В **hexgf**: дробно-линейные преобразования над GF(2⁶) — новый вид автоморфизмов Q6.

---

## Проект 11: `hexuniqgrp` — Уникальные группы

**Источник:** PDF `708` — «Уникальные группы»
**Область:** Теория конечных групп, комбинаторика
**Размер:** ~350 строк

### Алгоритм

```python
# Q(n) = число различных групп порядка n

# Уникальная группа: Q(n) = 1 и n — не простое
# Первая: G(15) = G(3) ⊗ G(5)

# Проверка уникальности через UVW-классификацию:
U = {n : n ≡ 1 mod 6}  # u_k = 6k - 5
V = {n : n ≡ 3 mod 6}  # кратные 3 нечётные
W = {n : n ≡ 5 mod 6}  # w_k = 6k - 1

# Таблица умножения классов:
# U*U=U, W*W=U, U*W=W, U*V=V, W*V=V, V*V=V
# → порядок из класса V → никогда не уникальный

# Формула хорд V (n точек, k непересекающихся хорд):
def chord_count(n, k):
    result = 1
    for i in range(1, k+1):
        result *= C(n - 2*(i-1), 2)
    return result // factorial(k)
# C(m,2) = m*(m-1)//2

# Список 35 уникальных порядков до 300:
UNIQUE_ORDERS = [15,33,35,51,65,69,77,85,87,91,95,115,119,123,133,
                 141,143,145,159,177,185,187,209,213,217,235,247,249,
                 255,259,265,267,287,295,299]
```

### Python API

```python
from hexuniqgrp import UniqueGroups

ug = UniqueGroups()

# Классификация числа по UVW
ug.classify_odd(15)     # → 'V'  (кратное 3)
ug.classify_odd(35)     # → 'W'  (35 ≡ 5 mod 6)
ug.classify_odd(7)      # → 'U'  (prime, or U-class)

# Проверка уникальности порядка
ug.is_unique_order(15)  # → True
ug.is_unique_order(9)   # → False
ug.is_unique_order(7)   # → False (простое)

# Список уникальных порядков до N
ug.unique_orders_up_to(300)   # → [15, 33, 35, ...]
ug.unique_orders_up_to(1000)  # → расширенный список

# Построение G(15) явно
g15 = ug.build_group(15)
g15.cayley_table()        # → 15×15 таблица Кэли
g15.subgroups()           # → G(3), G(5) как подгруппы
g15.factor_group('G5')    # → G(15)/G(5) ≅ G(3)
g15.element_orders()      # → {a0:1, a1:3, a2:3, a3:5, ..., a14:15}

# Формула хорд
ug.chord_count(n=6, k=3)  # → 15 (3 непересекающихся хорды на 6 точках)
ug.chord_count(n=6, k=2)  # → 15

# Соответствие G(15) ↔ расстановки хорд ↔ прикрепления многогранников
ug.chord_realization(15)        # → 15 расстановок хорд = G(15)
ug.polyhedron_realization(15)   # → 15 способов прикрепить {O,Г,M} к граням

# Визуализация решётки подгрупп
ug.print_subgroup_lattice(15)
```

### CLI

```
python hexuniqgrp.py --is-unique 15
python hexuniqgrp.py --list-up-to 300
python hexuniqgrp.py --build-group 15
python hexuniqgrp.py --cayley 15
python hexuniqgrp.py --chord-count 6 3
python hexuniqgrp.py --classify-odd 35
```

### Почему уникально

GAP, sage, sympy — могут вычислять `number_of_groups(n)`, но нет:
- понятия «уникальная группа» (Q(n)=1 при n непростом) как именованного объекта
- UVW-классификации нечётных чисел с теоремами умножения
- формулы хорд как реализации G(15)
- авторского списка порядков (для n > 1000 нужна проверка гипотезы)

### Аудитория

Студенты алгебры (изучают группы порядка 15, 21, 35), исследователи теории конечных групп.

### Интеграция с Q6

В **hexsym**: добавить `unique_group_check` для порядков подгрупп Aut(Q6).

---

## Проект 12: `hexbuffon` — Обобщённая формула Бюффона

**Источник:** PDF `dbf` — «Формула паркета»
**Область:** Геометрическая вероятность, симуляции
**Размер:** ~200 строк

### Алгоритм

```python
# Обобщение задачи Бюффона:
# Бросить отрезок длиной L на паркет из плиток
# W = ожидаемое число пересечений с линиями паркета

# Классическая формула Бюффона (одна система параллельных линий):
W_buffon = (2 * L) / (pi * a)

# Обобщение для произвольной плитки (U = периметр, F = площадь):
W = L * U / (pi * F)

# Частные случаи:
W_rect(a, b, L) = (L / pi) * (1/a + 1/b)   # прямоугольная плитка a×b
W_square(a, L) = 4*L / (pi * a)             # квадратная плитка
W_hex(r, L)    = 2*L*sqrt(3) / (pi * r)     # правильный шестиугольник

# Золотой прямоугольник (a × a·φ), карандаш L = a·e/2:
W_golden = phi * e / pi  ≈ 1.4008...
# (φ = 1.618..., e = 2.718..., π = 3.14159...)
```

### Python API

```python
from hexbuffon import BuffonParquet

bp = BuffonParquet()

# Обобщённая формула
bp.general_formula(L=1.5, perimeter=6.0, area=2.0)   # → W

# Прямоугольная плитка
bp.rectangular(a=2.0, b=3.0, needle=1.0)              # → W

# Квадратная (классическая доска Бюффона)
bp.square(a=2.0, needle=1.0)                           # → W = 1/(π) при L=a/2

# Шестиугольная плитка
bp.hexagonal(r=1.0, needle=0.5)                        # → W

# Золотой прямоугольник — формула φe/π
bp.golden_rectangle(a=1.0)   # → φ·e/π = 1.4008...
bp.golden_rectangle_verify()  # → аналитически: L=ae/2, a×aφ → W = φe/π

# Симуляция (Монте-Карло)
bp.simulate(tile='rectangle', a=2.0, b=3.0, needle=1.0, n=100000)
# → {'estimated_W': 0.847, 'exact_W': 0.849, 'error': 0.2%}

bp.simulate(tile='golden', a=1.0, n=100000)
# → {'estimated_W': 1.398, 'exact_W': 1.4008, 'error': 0.2%}

# Обратная задача: подобрать иглу для заданного W
bp.find_needle_length(target_W=1.0, tile='square', a=1.0)
# → L = π/4

# φ, e, π в одной формуле — визуализация
bp.plot_formula_relation()   # ASCII-схема: φ·e/π

# Для Q6 (шестиугольная решётка — как у hexbio)
bp.hexagonal_q6(r=1.0)      # → вероятность для Q6-подобной решётки
```

### CLI

```
python hexbuffon.py --rectangular 2 3 --needle 1
python hexbuffon.py --golden --a 1
python hexbuffon.py --simulate golden --n 100000
python hexbuffon.py --find-needle --target 1.0 --square --a 1
python hexbuffon.py --formula-analysis
```

### Почему уникально

scipy.stats не имеет `BuffonNeedle` как объекта. Формула `W = LU/(πF)`:
- обобщает Бюффона на произвольные плитки (малоизвестный результат)
- φe/π как точное значение — нет нигде в Python
- симуляция + аналитическая формула + обратная задача — полный пакет

### Аудитория

Студенты теории вероятностей, преподаватели (демонстрация π), исследователи геом. вероятности.

### Интеграция с Q6

В **hexstat**: добавить `buffon_on_q6()` — симуляция броска иглы на шестиугольную решётку Q6.

---

## Проект 13: `hexpolyenum` — Топология и перебор многогранников

**Источник:** PDF `677` + `87c` — «Формула Эйлера и топология» + «Экс-додекаэдр»
**Область:** Комбинаторная топология, геометрия многогранников
**Размер:** ~450 строк

### Алгоритм

```python
# Правильные сферические многогранники (Γ+B-P=2):
Γ = 4·C_B / (2(C_B+C_Γ) - C_B·C_Γ)
B = 4·C_Γ / (2(C_B+C_Γ) - C_B·C_Γ)
# Перебор: C_B, C_Γ = 3..N; оставить целые → 5 тел Платона

# Тороидальные (Γ+B-P=0):
2(C_B+C_Γ) = C_B·C_Γ → 3 решения: (4,4), (3,6), (6,3)

# Экс-додекаэдр:
D = B*(B-1)//2 - 2*P//Γ * (P - Γ)       # число диагоналей
V_ex = (a**3 / 2) * (4 + 3*PHI)          # объём (PHI = золотое сечение)
# Конструкция: додекаэдр → вырезать 4 ромба → вогнуть вершины
```

### Python API

```python
from hexpolyenum import PolyhedronEnumerator, ExDodecahedron

pe = PolyhedronEnumerator()

# Перебор правильных сферических многогранников
pe.enumerate_spherical(max_degree=10)
# → [Tetrahedron(3,3,4,6), Cube(4,3,8,12), Octahedron(3,4,6,8),
#    Dodecahedron(5,3,20,30), Icosahedron(3,5,12,30),
#    + dihedra, hosohedra...]

# Конкретный многогранник по (C_B, C_Γ)
pe.from_degrees(face_degree=5, vertex_degree=3)
# → Dodecahedron: B=20, Γ=12, P=30, χ=2

# Тороидальные многогранники
pe.enumerate_toroidal()
# → [Square(4,4,k,k), TriHex(3,6,2k,k), HexTri(6,3,k,2k)]

# Проверка формулы Эйлера
pe.check_euler(B=20, Gamma=12, P=30)  # → χ=2 ✓

# Число диагоналей многогранника
pe.diagonal_count(B=20, Gamma=12, P=30)  # → 100

# Экс-додекаэдр
ed = ExDodecahedron(a=1.0)
ed.euler()              # → χ = 32+24-54 = 2 ✓
ed.volume()             # → (a³/2)(4+3φ)
ed.diagonal_count()     # → вычисляется по формуле
ed.construction_steps() # → пошаговый алгоритм из додекаэдра
ed.vertices_3d()        # → 32 трёхмерных точки
ed.faces()              # → 24 грани (описание)
ed.to_obj("ex_dodec.obj")  # → экспорт в OBJ

# Сравнительная таблица
pe.compare_table()      # ASCII-таблица всех правильных многогранников
```

### CLI

```
python hexpolyenum.py --spherical
python hexpolyenum.py --toroidal
python hexpolyenum.py --from-degrees 5 3
python hexpolyenum.py --diagonals 20 12 30
python hexpolyenum.py --ex-dodecahedron --edge 1.0
python hexpolyenum.py --ex-dodec-export obj
```

### Почему уникально

sage, polymake — умеют работать с многогранниками, но нет:
- `enumerate_spherical()` как простого Python CLI с объяснением
- `ExDodecahedron` как именованного объекта
- авторской формулы числа диагоналей
- образовательного сравнения сферической и тороидальной топологий

### Аудитория

Студенты топологии и геометрии, 3D-художники, дизайнеры.

### Интеграция с Q6

В **hexhept**: расширить RP²-checker новыми формулами; добавить ExDodecahedron.

---

## Проект 14: `hexellipse` — Скрытые параметры эллипса

**Источник:** PDF `a08` — «Линия катастроф. Скрытые параметры эллипса»
**Область:** Аналитическая геометрия, дифференциальная геометрия
**Размер:** ~300 строк

### Алгоритм

```python
# Параметры эллипса (a = большая полуось, b = малая):
p1 = b**2 / a    # фокальный параметр (= радиус кривизны в ±a)
p2 = a**2 / b    # радиальный параметр (= радиус описанной окружности)

# Фундаментальное тождество:
assert a * b == c * p  # p = ab/c, c = sqrt(a²-b²)
# Также: p1 * p2 ≈ a*b (через осевой параметр p)

# Линия катастроф (эквидистанта при q = p1):
# Горизонтальное смещение:
dx_squared = q**2 * b**2 * x1**2 / (a**4 - a**2*x1**2 + b**2*x1**2)
# Уравнение эквидистанты:
y2 = sqrt(a**2 - x1**2) * (b/a - q*a / sqrt(a**4 - x1**2 * c**2))

# Задача вписанных окружностей → R = p2 = a²/b
```

### Python API

```python
from hexellipse import EllipseAnalysis

ea = EllipseAnalysis(a=5.0, b=3.0)

# Основные параметры
ea.focal_parameter()     # → p1 = b²/a = 1.8
ea.radial_parameter()    # → p2 = a²/b = 8.33
ea.axial_parameter()     # → p = ab/c = 3.75
ea.verify_identity()     # → a*b == c*p: 15.0 ≈ 15.0 ✓
ea.eccentricity()        # → e = c/a = 0.8

# Эквидистанта (параллельная кривая)
contour = ea.equidistant(q=1.0, n_points=200)  # → [(x, y), ...]

# Линия катастроф (особая эквидистанта при q = p1)
catastrophe = ea.catastrophe_curve(n_points=200)
catastrophe_params = ea.at_catastrophe()
# → {'q': 1.8, 'type': 'cusp', 'inflection_points': [...]}

# Задача вписанных окружностей
ea.inscribed_circle_radius()    # → R = p2 = a²/b
ea.inscribed_system_solution()  # → решение системы (4)-(5)

# Построение p1 и p2 геометрически
ea.focal_parameter_construction()   # → инструкция для построения p1
ea.radial_parameter_construction()  # → инструкция для построения p2

# Визуализация
ea.plot_ascii(width=60)           # ASCII: эллипс + эквидистанты
ea.plot_ascii_catastrophe(w=60)   # ASCII: линия катастроф
```

### CLI

```
python hexellipse.py --ellipse 5 3 --parameters
python hexellipse.py --equidistant --q 1.8
python hexellipse.py --catastrophe
python hexellipse.py --inscribed-circle
python hexellipse.py --plot --width 60
```

### Почему уникально

scipy, sympy работают с эллипсом, но нет:
- p₁ и p₂ как «скрытых параметров» с тождеством a·b = c·p₁ (комплексно)
- «линии катастроф» как отдельного концепта
- образовательного набора всех эллипсных параметров в одном месте

### Аудитория

Студенты аналитической геометрии, инженеры (оптика, орбиты), программисты CNC.

---

## Проект 15: `hexcubenets` — Развёртки куба

**Источник:** PDF `c60` — «О развёртках куба»
**Область:** Комбинаторика, теория графов, образование
**Размер:** ~250 строк

### Алгоритм

```python
# Куб: 6 граней, 12 рёбер
# Для развёртки: разрезать 7 из 12 рёбер (оставить остов — spanning tree)

# Всего способов выбрать 7 рёбер:
V_total = C(12, 7)   # = 792

# Граф смежности граней куба (6 вершин, 12 рёбер)
CUBE_FACE_GRAPH = {
    'top': ['front', 'right', 'back', 'left'],
    'bottom': ['front', 'right', 'back', 'left'],
    'front': ['top', 'right', 'bottom', 'left'],
    # ...
}

# Алгоритм: для каждого из 792 выборов из 12 рёбер:
# 1. Удалить 7 рёбер → оставшиеся 5 должны образовывать дерево (spanning tree)
# 2. Если да → развернуть грани → проверить наличие наложений
# 3. Если без наложений → валидная развёртка

# Классификация 11 развёрток:
# Симметричные (6): оси или зеркальная симметрия
# Асимметричные (5): нет симметрий

# Доказательство ровно 11:
# V_valid = 6*24 + 5*48 = 384
# V_invalid = 324 + 60 + 24 = 408
# 384 + 408 = 792 ✓
# Различных развёрток = 11 (с учётом симметрий куба)
```

### Python API

```python
from hexcubenets import CubeNets

cn = CubeNets()

# Перечисление всех 11 развёрток
nets = cn.enumerate_all()   # → список из 11 Net объектов

# Проверка конкретного разреза
cuts = ['top-front', 'top-back', 'top-right', 'front-right', 'back-right', 'bottom-left', 'front-left']
result = cn.is_valid_net(cuts)   # → True/False

# Развернуть конкретную развёртку
net = cn.get_net(index=0)   # → Net(faces=6, shape='cross')
net.to_ascii()              # → ASCII-рисунок развёртки
net.symmetry()              # → 'mirror' / 'central' / 'none'

# Проверка «ровно 11»
cn.prove_count()            # → V_valid=384, V_invalid=408, total=792 ✓, nets=11

# Статистика
cn.classify()               # → {'symmetric': 6, 'asymmetric': 5}

# Раскраска (для дидактики)
net.to_colored_ascii(colors={'top': 'R', 'bottom': 'G', ...})

# Обобщение на другие многогранники
cn.tetrahedron_nets()       # → 2 различных развёртки тетраэдра
cn.octahedron_nets()        # → 11 развёрток октаэдра (симметрия куба!)

# Гиперкуб Q4 → развёртки (264 штуки)
cn.hypercube_nets(dim=4)    # → 264 различных развёртки тессеракта
```

### CLI

```
python hexcubenets.py --enumerate
python hexcubenets.py --prove-count
python hexcubenets.py --net 0 --ascii
python hexcubenets.py --classify
python hexcubenets.py --generalize tetrahedron
python hexcubenets.py --hypercube 4
```

### Почему уникально

Развёртки куба — классическая задача, решённая в 1984 (Turney). Но нет:
- Python-библиотеки `CubeNets` с полной классификацией
- алгоритмического доказательства «ровно 11» через граф смежности
- обобщения на тессеракт (Q4) в Python
- интеграции с Q6 (развёртка Q6 как 6-мерного гиперкуба?)

**Особая ценность:** обобщение `hypercube_nets(dim=4)` — 264 развёртки тессеракта связывают этот проект с **hexdim**.

### Аудитория

Студенты CS (алгоритмы), учителя геометрии, 3D-дизайнеры, разработчики игр (UV-развёртки).

### Интеграция с Q6

В **hexdim**: `CubeNets.hypercube_nets(6)` → развёртки Q6 как физического 6-мерного куба.

---

## Итоговая таблица всех 15 проектов (8 старых + 7 новых)

| # | Проект | Источник | Строк | Уникальность | Тир |
|---|--------|----------|-------|--------------|-----|
| 1 | **hexnumderiv** | 6a4 | 250 | ∂n нет нигде | 1 |
| 2 | **hexintermed** | 25d | 300 | H-ряд нет в OEIS | 1 |
| 3 | **hexscrew** | e2c | 400 | ScrewGroup нет в sympy | 1 |
| 4 | **hexmatroot** | a67 | 350 | Аналит. √A-формулы | 2 |
| 5 | **hexperms** | 17f | 200 | Unranking + алгоритм | 1 |
| 6 | **hextile** | 0f6+194 | 500 | Плитка Германа | 2 |
| 7 | **hexhept** | 8dd | 400 | RP²-checker нет нигде | 2 |
| 8 | **hexmobius** | 65b | 450 | Классификация+OBJ | 3 |
| 9 | **hexpowerxy** | 4c5 | 180 | X^Y=Y^X + φ + e | 1 |
| 10 | **hexcrossrat** | 372 | 280 | CrossRatioGroup нет | 1 |
| 11 | **hexuniqgrp** | 708 | 350 | UniqueGroups нет | 2 |
| 12 | **hexbuffon** | dbf | 200 | φe/π нигде | 1 |
| 13 | **hexpolyenum** | 677+87c | 450 | ExDodecahedron нет | 2 |
| 14 | **hexellipse** | a08 | 300 | p₁,p₂+катастрофа | 2 |
| 15 | **hexcubenets** | c60 | 250 | Граф+гиперкуб | 1 |

**Итого:** ~4 860 строк кода, 15 новых Python-проектов, 0 существующих аналогов (полных)

---

## Приоритетная очередь реализации

### Tier 1 — Реализовать немедленно (простой алг. + широкая аудитория)
```
hexpowerxy    ← 180 строк, отличный учебный пример, связь φ и e
hexbuffon     ← 200 строк, φe/π — уникальная формула
hexcubenets   ← 250 строк, классика CS + обобщение на Q6
hexcrossrat   ← 280 строк, применение в CV + теория групп
hexnumderiv   ← 250 строк (из roadmap_implementations.md)
hexintermed   ← 300 строк
hexperms      ← 200 строк
```

### Tier 2 — Реализовать вторыми
```
hexuniqgrp    ← 350 строк, нужна верификация авторского списка
hexpolyenum   ← 450 строк, включает ExDodecahedron
hexellipse    ← 300 строк, нужна геом. визуализация
hexscrew      ← 400 строк (из roadmap_implementations.md)
hexhept       ← 400 строк
hexmatroot    ← 350 строк
```

### Tier 3 — Реализовать третьими
```
hextile       ← 500 строк, апериодичность трудно верифицировать
hexmobius     ← 450 строк, нужен 3D-рендеринг
```

---

## Не реализовывать (нет алгоритмического содержания)

| ID | Причина |
|----|---------|
| 077/2d1 | Спекулятивная физика «тонкого мира» |
| c0a | Философия без формул |
| c89 | Эссе о пшеничных кругах |
| 834 | «Цикл Пенроуза» — только схема |
| 6a1 | «Туннельная сфера» без вычислимых формул |
| 628 | Просто формулировка теоремы (1 стр.) |
| a1e | Дублирует 6a4 |

---

*Roadmap подготовлен: 19 февраля 2026 г. На основе анализа 49 новых PDF.*
