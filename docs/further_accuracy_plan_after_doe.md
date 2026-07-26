# План дальнейших действий для повышения точности детектирования колец

> Исторический D6-only план для исходной 24-image выборки. После аудита 2026-07-21
> его заменяет `ARTICLE_EXPERIMENT_PROTOCOL.md`: старые точки используются как
> baseline/search-range evidence, но не как validation или final test.

## 1. Входные данные анализа

Использован файл `cv experimental factors(3).xlsx`.

Основной анализ выполнен по сопоставимым строкам с одинаковым строгим критерием matching:

```text
max_match_cost = 1.25
max_center_distance_px = 8.0
```

Причина: строки с другим matching-критерием нельзя напрямую использовать для оценки одной response surface, потому что при изменении критерия matching меняются TP, FP, precision, recall и F1 без изменения алгоритма.

## 2. Текущее лучшее состояние

Лучшая точка по F1 среди сопоставимых экспериментов:

```text
exp31
d6_method = amplitude_merge_delta_bic
d6_merge_center_px = 6
d6_merge_radius_px = 6
d6_delta_bic_threshold = 7000
d6_amp_min = 0.0125

TP = 141
FP = 15
precision = 0.90
recall = 0.82
F1 = 0.86
```

Лучшая точка по precision:

```text
exp33
d6_merge_center_px = 7
d6_merge_radius_px = 7
d6_delta_bic_threshold = 7000
d6_amp_min = 0.015

precision = 0.92
recall = 0.79
F1 = 0.85
```

Лучшая точка по recall среди сопоставимых строгих строк:

```text
exp30
d6_merge_center_px = 2
d6_merge_radius_px = 2
d6_delta_bic_threshold = 10000
d6_amp_min = 0.01

precision = 0.73
recall = 0.83
F1 = 0.77
```

Вывод: текущая рабочая область D6 уже дает высокую precision. Основная цена дальнейшего увеличения precision — потеря recall. Простое ослабление pruning увеличивает recall незначительно, но быстро увеличивает FP.

## 3. Локальная линейная регрессия

Регрессия построена по локальным точкам `exp31–exp36`.

Использованы факторы:

```text
x1 = d6_merge_center_px - 6
x2 = log10(d6_delta_bic_threshold) - log10(7000)
x3 = (d6_amp_min - 0.0125) / 0.0025
```

Модель:

```text
metric = b0 + b1*x1 + b2*x2 + b3*x3
```

Полученные локальные коэффициенты:

| Метрика | b0 | b1: merge | b2: log10(delta_bic) | b3: amp_min step |
|---|---:|---:|---:|---:|
| precision | 89.72 | 0.08 | -4.71 | 2.14 |
| recall | 81.17 | -0.91 | -9.00 | -1.84 |
| F1 | 85.28 | -0.37 | -6.11 | -0.28 |

RMSE локальной линейной аппроксимации:

| Метрика | RMSE, процентных пунктов |
|---|---:|
| precision | 0.27 |
| recall | 0.52 |
| F1 | 0.49 |

Интерпретация коэффициентов:

1. Увеличение `d6_amp_min` повышает precision и снижает recall.
2. Увеличение `d6_delta_bic_threshold` снижает recall и F1 в локальной области.
3. Увеличение `merge_center_px` от 6 к 7 не улучшает F1; оно либо нейтрально, либо снижает recall.
4. Лучшее измеренное значение F1 уже находится около локального максимума для текущей семьи D6-правил.

## 4. Есть ли запас при тех же D1–D5

Оценка: малый запас остается, но он ограничен.

Текущий measured best:

```text
precision = 0.90
recall = 0.82
F1 = 0.86
```

Оценка достижимого уровня при тех же D1–D5 и той же семье D6:

```text
precision ceiling: 0.92–0.94
recall при precision >= 0.88: 0.82–0.84
F1 ceiling: 0.86–0.88
```

Вывод:

```text
Локальный максимум практически достигнут для текущих D1–D5 и текущих D6-признаков:
amplitude threshold + merge + delta_bic.
```

Дальнейший рост F1 выше `0.88` маловероятен без добавления новых признаков D6 или изменения D4/D5.

## 5. Ближайшие D6-only эксперименты

Эти эксперименты нужны не для широкого поиска, а для проверки оставшегося локального запаса.

### Эксперимент 37: перенос recall gain от exp36 в точку exp31

Гипотеза: снижение `delta_bic_threshold` с 7000 до 5000 при `merge=6` и `amp_min=0.0125` может увеличить recall без резкого роста FP.

```text
d6_merge_center_px = 6
d6_merge_radius_px = 6
d6_delta_bic_threshold = 5000
d6_amp_min = 0.0125
```

Команда:

```bash
python smbh_cv_run.py all \
  --config next_plan_configs/exp37_d6_recall_probe_delta5000.json \
  --out runs/exp37_d6_recall_probe_delta5000 \
  --clean \
  --workers-generate 2 \
  --workers-d1 4 \
  --workers-d2 4 \
  --workers-d3 2 \
  --workers-d4 4 \
  --workers-d5 3 \
  --workers-d6 4
```

Критерий успеха:

```text
F1 >= 0.87
precision >= 0.88
recall >= 0.83
```

### Эксперимент 38: контролируемое ослабление pruning

Гипотеза: одновременное уменьшение `delta_bic_threshold` и `amp_min` может вернуть часть TP. Риск: FP вырастет.

```text
d6_merge_center_px = 6
d6_merge_radius_px = 6
d6_delta_bic_threshold = 3000
d6_amp_min = 0.01125
```

Команда:

```bash
python smbh_cv_run.py all \
  --config next_plan_configs/exp38_d6_recall_edge_delta3000_amp01125.json \
  --out runs/exp38_d6_recall_edge_delta3000_amp01125 \
  --clean \
  --workers-generate 2 \
  --workers-d1 4 \
  --workers-d2 4 \
  --workers-d3 2 \
  --workers-d4 4 \
  --workers-d5 3 \
  --workers-d6 4
```

Критерий успеха:

```text
recall >= 0.84
precision >= 0.85
F1 >= 0.86
```

Если precision падает ниже 0.85, этот путь закрывается.

### Эксперимент 39: потолок precision

Гипотеза: `amp_min=0.0175` покажет максимальную precision текущего D6-класса. Этот эксперимент нужен для оценки trade-off, а не для максимизации F1.

```text
d6_merge_center_px = 7
d6_merge_radius_px = 7
d6_delta_bic_threshold = 7000
d6_amp_min = 0.0175
```

Команда:

```bash
python smbh_cv_run.py all \
  --config next_plan_configs/exp39_d6_precision_ceiling_amp0175.json \
  --out runs/exp39_d6_precision_ceiling_amp0175 \
  --clean \
  --workers-generate 2 \
  --workers-d1 4 \
  --workers-d2 4 \
  --workers-d3 2 \
  --workers-d4 4 \
  --workers-d5 3 \
  --workers-d6 4
```

Критерий интерпретации:

```text
Если precision > 0.93 и recall < 0.78, то высокий precision достигается только ценой потери TP.
Если precision не растет выше 0.92, то текущий precision ceiling уже достигнут.
```

## 6. Правило принятия решения после exp37–exp39

После трех запусков построить таблицу:

```text
exp, TP, FP, precision, recall, F1, mean_n_error
```

Принять решение:

1. Если `exp37` или `exp38` дает `F1 >= 0.87`, выполнить подтверждающий прогон на другой seed.
2. Если все точки дают `F1 <= 0.86`, считать текущий D6-класс локально исчерпанным.
3. Если `exp39` повышает precision, но F1 падает, использовать его только для режима high-precision, не как основной detector.

## 7. Валидация на больших выборках

После фиксации лучшей D6-конфигурации выполнить три уровня проверки.

### Уровень 1: simple_model, другой seed

Цель: проверить, что exp31 не переобучен на конкретный seed.

```text
dataset_mode = simple_model
n_images = 200
seed = 17
```

Проверяемые конфигурации:

```text
best_current = exp31
best_after_exp37_39, если он улучшит F1
precision_mode = exp33 или exp39, если требуется high-precision режим
```

Критерий прохождения:

```text
precision >= 0.88
recall >= 0.80
F1 >= 0.84
```

### Уровень 2: simple_model, больший диапазон числа колец

Цель: проверить устойчивость к плотным сценам.

```text
dataset_mode = simple_model
n_images = 200
n_true_min = 1
n_true_max = 15
seed = 23
```

Метрики считать отдельно по группам:

```text
N_true <= 4
5 <= N_true <= 9
N_true >= 10
```

Критерий прохождения:

```text
F1 >= 0.82 в каждой группе
```

Если F1 падает только при `N_true >= 10`, проблема находится в merge/deduplication.

### Уровень 3: degraded_model

Цель: проверить перенос на менее идеальные изображения.

```text
dataset_mode = degraded_model
n_images = 200
beam_fwhm_px > 0
noise_model = correlated или gaussian
partial_rings = true, если реализовано
overlapping_rings = true, если реализовано
seed = 31
```

Критерий прохождения:

```text
precision >= 0.80
recall >= 0.70
F1 >= 0.75
```

Для degraded_model не следует требовать тех же метрик, что на simple_model, пока D1–D5 не были специально оптимизированы под деградации.

## 8. Дальнейшая работа, если D6-only улучшение исчерпано

Если exp37–exp39 не улучшают F1, следующий рост надо искать не в новых порогах, а в новых признаках принятия решения в D6.

### 8.1. Edge support

Для каждого fitted ring вычислить:

```text
edge_support_i =
(1 / K) * sum_j max_{||p - p_j|| <= delta} binary_edge(p)
```

где

```text
p_j = (x_i + R_i cos(phi_j), y_i + R_i sin(phi_j))
```

Удалять кольцо, если:

```text
edge_support_i < tau_edge
```

Рекомендуемый старт:

```text
tau_edge = 0.20, 0.30, 0.40
delta = 2 px
```

### 8.2. Local ring-support gain

Для кольца задать маску:

```text
M_i(x,y) = 1, если |sqrt((x-x_i)^2 + (y-y_i)^2) - R_i| <= 2*sigma_i
```

Вычислить локальную пользу:

```text
local_gain_i =
sum_{(x,y) in M_i} [res_without_i(x,y)^2 - res_full(x,y)^2]
```

Удалять кольцо, если:

```text
local_gain_i < tau_gain
```

### 8.3. RHT provenance score after fit

Проверить, лежит ли fitted ring в области голосов RHT:

```text
hough_score_i = U_interp(x_i, y_i, R_i) / max(U)
```

Удалять кольцо, если:

```text
hough_score_i < tau_hough
```

### 8.4. Combined D6 ranker

Сформировать итоговый score:

```text
score_i =
w1 * rank(local_gain_i)
+ w2 * rank(edge_support_i)
+ w3 * rank(hough_score_i)
- w4 * rank(duplicate_penalty_i)
```

Начальные веса:

```text
w1 = 0.40
w2 = 0.30
w3 = 0.20
w4 = 0.10
```

Выбирать порог score по validation set.

Цель:

```text
precision >= 0.92
recall >= 0.82
F1 >= 0.87
```

## 9. Работа над D4/D5 после D6

Если D6-only и новые D6-признаки не дают рост F1, проверить ошибку fit-модели.

Основные гипотезы:

1. Ложные кольца выживают, потому что джет или фон не попадают в правильное положение.
2. Часть колец используется моделью для компенсации ошибки джета.
3. Совместный fit всех колец и джета создает локальные минимумы.

Проверки:

```text
1. Считать jet_center_error на synthetic truth.
2. Сравнить residual maps для TP и FP.
3. Проверить, коррелируют ли FP с ошибкой положения джета.
```

Если корреляция есть, внедрить:

```text
D4 jet_init_method = matched_filter_local_fit
D5 jet_prefit = true
D5 alternating_fit = jet -> rings -> joint
```

## 10. Итоговый план

Последовательность действий:

1. Выполнить exp37–exp39.
2. Если F1 улучшается, подтвердить лучшую точку на `n_images = 200`, `seed = 17`.
3. Если F1 не улучшается, зафиксировать exp31 как лучший D6-threshold baseline.
4. Добавить D6-признаки `edge_support`, `local_gain`, `hough_score`.
5. Построить combined D6-ranker.
6. Провести validation на:
   - simple_model, другой seed;
   - simple_model, расширенный диапазон числа колец;
   - degraded_model.
7. Если FP остаются, анализировать связь FP с ошибками jet/background в D4/D5.
8. После стабилизации метрик фиксировать baseline для статьи и отдельно показывать ограничения на degraded_model.

## 11. Главный вывод

При текущих D1–D5 и текущем D6-классе `amplitude + merge + delta_bic` запас малый. Локальный максимум практически достигнут.

Ожидаемый дальнейший прирост от одних порогов D6:

```text
F1: +0.00 ... +0.02
precision: +0.00 ... +0.02 при потере recall
recall: +0.01 ... +0.02 при риске роста FP
```

Ожидаемый прирост от новых D6-признаков:

```text
F1: +0.02 ... +0.05
precision: +0.02 ... +0.05
recall: сохранить около 0.80 ... 0.83
```

Ожидаемый прирост от исправления jet initialization и D5, если FP связаны с ошибкой джета:

```text
F1: +0.02 ... +0.06
precision: +0.03 ... +0.08
recall: без существенного падения
```
