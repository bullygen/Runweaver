# Команды эксперимента article_v2

## Исходные данные

article_v2 использует тот же неизменяемый набор `datasets/article_v1`. Новый набор не создается. Описание находится в `docs/DATASET_ARTICLE_V1.md`.

Перед запуском:

```bash
cd "/home/bullygen/Documents/[Work] ASC SMBH CV ver 2"
export MPLCONFIGDIR=/tmp/smbh-mpl
mkdir -p "$MPLCONFIGDIR"

.venv/bin/python smbh_cv_dataset.py validate \
  --plan experiments/article_v1/dataset_plan.json \
  --out datasets/article_v1
```

Исполнитель:

```text
smbh_cv_experiments_v2.py
```

План:

```text
experiments/article_v2/search_plan.json
```

Результаты:

```text
experiment_runs/article_v2/<этап>/<конфигурация>/algorithm_seed_<зерно>/
```

Повтор той же команды продолжает незавершенные ступени. Параметр `--restart` удаляет выход выбранных прогонов и начинает их заново. После начала этапа не применять `plan --force`, иначе состав конфигураций может измениться.

## 0. Проверка запуска

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase smoke

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase smoke
```

## 1. Широкий поиск D1–D3

Создаются 96 точек последовательности Соболя. Исходная конфигурация рассчитывается отдельно как контроль и не допускается к продвижению:

\[
N=96+1.
\]

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase d1d3_global

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase d1d3_global
```

Упорядочение можно пересчитать без повторения D1–D3:

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  summarize --phase d1d3_global
```

Здесь сравниваются семейства обработки D1, способы голосования D2, способы выделения максимумов D3 и их численные параметры. `d2_votes_fraction` не меняется. Число голосов задается непосредственно параметром `d2_max_votes`.

## 2. Адаптивное уточнение D1–D3

На основании 20 лучших точек первого этапа строятся 64 новые точки. Для численных параметров применяется распределение около лучших точек, для дискретных вариантов — частоты вариантов с единичной добавкой. Доля равномерного исследования равна 0.20.

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase d1d3_adaptive

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase d1d3_adaptive
```

## 3. Проверка 12 конфигураций D1–D3

Из объединения широкого и адаптивного поиска выбираются 12 различных конфигураций. Каждая проходит 240 изображений при зернах 239 и 991.

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase d1d3_promotion

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase d1d3_promotion
```

## 4. Локальная резкость D1–D3

Вокруг трех лучших конфигураций численные координаты сдвигаются на

\[
h\in\{0.02,\;0.05,\;0.10\}
\]

от ширины соответствующего промежутка поиска. Для каждого допустимого параметра рассчитываются

\[
S_i(h)=\frac{|Q(x+h e_i)-Q(x-h e_i)|}{2h},
\]

\[
C_i(h)=\frac{|Q(x+h e_i)-2Q(x)+Q(x-h e_i)|}{h^2}.
\]

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase d1d3_local

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase d1d3_local
```

Результаты:

```text
experiment_runs/article_v2/d1d3_local/local_sensitivity.json
experiment_runs/article_v2/d1d3_local/local_sensitivity.csv
```

Двенадцать лучших локальных точек повторно проверяются на 240 изображениях и двух зернах:

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase d1d3_confirm

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase d1d3_confirm

.venv/bin/python smbh_cv_experiments_v2.py \
  freeze \
  --phase d1d3_confirm \
  --out experiments/article_v2/frozen_d1d3.json
```

Заморозка разрешена при

\[
R_{D3}\ge 0.75,\qquad
L_{D3,0}\le 5,\qquad
M_{D3}\le 0.15,
\]

где \(L_{D3,0}\) — среднее число предложений D3 на пустом изображении, \(M_{D3}\) — доля непустых изображений без предложений.

Для исследования причины провала допуска можно явно добавить `--allow-gate-failure`. Такая конфигурация помечается как диагностическая и не подтверждает качество.

## 5. Настройка D4–D5

D1–D3 не повторяются. Их сохраненные каталоги подключаются из прогона `frozen_d1d3.json`.

D4 читает изображение и выход D3. D5 читает изображение и выход D4. Файл `truth.json` эти ступени не открывают. После D6 модуль статистики читает разметку и упорядочивает полностью готовые прогоны.

Сначала рассчитываются 48 точек на 120 изображениях:

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase d4d5_global

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase d4d5_global
```

Затем 8 лучших точек проходят 240 изображений и два зерна:

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase d4d5_promotion

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase d4d5_promotion

.venv/bin/python smbh_cv_experiments_v2.py \
  freeze \
  --phase d4d5_promotion \
  --out experiments/article_v2/frozen_d1d5.json
```

Меняются `d4_strip_half_width`, `d4_radial_window` и `d5_f_scale`.

## 6. Плотный поиск D6

D1–D5 не повторяются. Для 256 точек D6 используются сохраненные выходы `frozen_d1d5.json`.

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase d6_global

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase d6_global
```

Меняются порог амплитуды, порог приращения БИК и расстояние слияния. Конфигурации разделяются на слои Парето по точности, полноте, нижней границе \(F_1\), ложным объектам на пустом изображении и изменчивости между зернами.

Непрерывная оценка имеет вид

\[
Q=F_{1,\;0.025}^{\min}
-0.25s(F_1)-0.05L_0
-\Delta_P-\Delta_R-0.10\Delta_{L_0}.
\]

Пороги допуска учитываются отдельно:

\[
P\ge0.80,\qquad R\ge0.70,\qquad L_0\le0.50.
\]

## 7. Отбор

Три лучших допустимых конфигурации D6 проходят все 300 изображений отбора. Команда не выполнится, если допуск прошли менее трех конфигураций разработки.

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  plan --phase validation

.venv/bin/python smbh_cv_experiments_v2.py \
  run --phase validation

.venv/bin/python smbh_cv_experiments_v2.py \
  freeze \
  --phase validation \
  --out experiments/article_v2/frozen_config.json
```

После этой команды параметры не меняются.

## 8. Итоговое испытание

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  run \
  --phase test \
  --frozen experiments/article_v2/frozen_config.json
```

Испытательная часть открывается один раз. Ее результат не используется для подбора.

## 9. Проверка переноса

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  run \
  --phase verification \
  --frozen experiments/article_v2/frozen_config.json
```

## 10. Решение о качестве

```bash
ARTICLE_CANDIDATE_ID="$(
  .venv/bin/python -c \
    'import json; from pathlib import Path; print(json.loads(Path("experiments/article_v2/frozen_config.json").read_text())["candidate_id"])'
)"

.venv/bin/python smbh_cv_quality.py \
  --test "experiment_runs/article_v2/test/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --verification "experiment_runs/article_v2/verification/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --validation "experiment_runs/article_v2/validation/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --thresholds experiments/article_v2/quality_gate.json \
  --out experiment_runs/article_v2/article_quality_decision.json
```

## Продолжение после остановки

Повторить ту же команду `run`. Исполнитель проверит выход каждого изображения и запустит только отсутствующие вычисления.

Для повторного упорядочения:

```bash
.venv/bin/python smbh_cv_experiments_v2.py \
  summarize --phase d6_global
```

Для просмотра плана:

```bash
.venv/bin/python -m json.tool \
  experiment_runs/article_v2/d6_global/candidate_plan.json
```
