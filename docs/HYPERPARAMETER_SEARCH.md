# Fine-tuning D1--D6: автоматизированный план запусков

## Неподвижные гипотезы

В `experiments/article_v1/search_plan.json` заблокированы названия и семейства методов:

```text
D1 percentile_cc -> legacy_laplace_positive -> positive -> legacy_control_open
D2 sparse_rht_stratified, vote_weight=mean
D3 ph_union_find_sparse, threshold=bootstrap
D4 profile_harmonic
D5 scipy, soft_l1, ring_support
D6 amplitude_merge_delta_bic
```

Runner проверяет соответствие base config этим значениям и запрещает помещать их в search space. Меняются только численные гиперпараметры вокруг существующих математических методов.

## Что оптимизируется

24 Latin-hypercube кандидата из четырёх независимых starts плюс исходный exp31 baseline покрывают:

- D1 jet percentile и долю bad pixels;
- D2 vote fraction, minimum triangle area, sectors;
- D3 bootstrap quantile, NMS center/radius, maximum candidates;
- D4 strip half-width и radial window;
- D5 robust-loss scale;
- D6 amplitude, delta-BIC и coupled center/radius merge thresholds.

`d4_max_init_candidates` автоматически равен `d3_max_candidates`, а `d6_merge_radius_px` -- `d6_merge_center_px`. Это уменьшает effective dimension и предотвращает бессмысленные несовместимые конфигурации.

## Измеримая objective function

Для каждого algorithm seed statistics вычисляет image-cluster bootstrap CI по
article-primary matching `article_geometric_iou_v1`. Ranking score:

```text
score = min_seed(F1_CI95_low)
        - 0.25 * std_seed(F1)
        - 0.05 * max_seed(null_FPPI)
```

Если хотя бы на одном seed `precision < 0.80` или `recall < 0.70`, из score вычитается 1.0. Все слагаемые находятся в `statistics/summary.json`; скрытой экспертной оценки нет.

## Последовательность

### 0. Проверка плана

```bash
MPLCONFIGDIR=/tmp/smbh-mpl \
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json plan
```

Результат: 25 кандидатов в `experiment_runs/article_v1/candidate_plan.json`.

### 1. Screening

```bash
MPLCONFIGDIR=/tmp/smbh-mpl \
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run --phase screening
```

Все 25 кандидатов сравниваются на тех же 60 development images, seed алгоритма 239. Для экономии используется `d2_max_votes=120000`, `d5_max_nfev=250`. Это low-fidelity отбор, его метрики нельзя публиковать как итоговые.

Для проверки инфраструктуры сначала рекомендуется отдельная одно-image фаза,
которая не смешивается с каталогом screening:

```bash
MPLCONFIGDIR=/tmp/smbh-mpl \
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run --phase smoke --max-candidates 1
```

Повторная команда пропускает полностью готовые стадии D1--D6, на первой
незавершённой стадии запускает только изображения без соответствующего stage
output и затем так же проверяет следующие стадии. `run_signature.json` фиксирует
dataset hashes, image IDs, phase, seed и полный config; runner откажется смешивать
старые stage artifacts с изменённым планом. `--restart` удаляет только выбранный
run и запускает его заново; использовать этот флаг следует осознанно.

### 2. Promotion

После завершения screening:

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  summarize --phase screening

MPLCONFIGDIR=/tmp/smbh-mpl \
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run --phase promotion
```

Top-6 проверяются на 240 development images и двух algorithm seeds (239, 991) с `250000` votes и `600` D5 evaluations.

### 3. Validation и заморозка

```bash
MPLCONFIGDIR=/tmp/smbh-mpl \
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run --phase validation

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  freeze --phase validation \
  --out experiments/article_v1/frozen_config.json
```

Top-3 promotion-кандидата проверяются на всех 300 validation images и двух algorithm seeds. После создания `frozen_config.json` запрещено менять алгоритмические параметры на основании test/verification.

### 4. Final test

```bash
MPLCONFIGDIR=/tmp/smbh-mpl \
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run --phase test \
  --frozen experiments/article_v1/frozen_config.json
```

Основной article result берётся из predeclared `algorithm_seed_239`. Seeds 991 и 2027 являются sensitivity analysis stochastic RHT, а не тремя независимыми датасетами. Нельзя объединять их кольца как 1500 независимых изображений.

### 5. OOD verification

Только после test:

```bash
MPLCONFIGDIR=/tmp/smbh-mpl \
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run --phase verification \
  --frozen experiments/article_v1/frozen_config.json
```

Verification служит для границ применимости. Провал отдельного сценария сообщается как failure mode; параметры не подгоняются повторно.

## Как читать результаты

Для каждой комбинации создаются:

```text
experiment_runs/article_v1/<phase>/<candidate_id>/algorithm_seed_<seed>/
  config_used.json
  experiment_metadata.json
  statistics/summary.json
  statistics/per_image_metrics.csv
  statistics/factor_metrics.csv
  runtime/*_stage_summary.json
```

Ranking находится в `<phase>/ranking.json` и `ranking.csv`. Для каждой factor level в `factor_metrics.csv` записаны TP/FP/FN, precision/recall/F1 с bootstrap CI и null FPPI.

## Оценка времени

Одно-image smoke в отдельной low-budget фазе проверяет только запуск и из-за малого
числа кандидатов не оценивает стоимость D5. Сначала выполните один настоящий
screening candidate и по `runtime/d5_stage_summary.json` оцените wall time полной
серии. Worker plan задаётся в `experiments/article_v1/search_plan.json`; для
D1--D6 установлено по 12 workers. Стоимость D5 может меняться на порядки вместе
с числом D3-кандидатов.

Если вычислительный бюджет ограничен, допустимо остановиться после promotion и описать результат как development study. Недопустимо открывать test, выбрать по нему конфигурацию и затем называть тот же test независимым.
