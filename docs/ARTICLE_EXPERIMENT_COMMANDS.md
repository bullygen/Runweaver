# Команды запуска эксперимента `article_v1`

## Перед любым запуском

```bash
cd "/home/bullygen/Documents/[Work] ASC SMBH CV ver 2"
export MPLCONFIGDIR=/tmp/smbh-mpl
mkdir -p "$MPLCONFIGDIR"
```

## Один раз: создать `.venv`, если её нет

```bash
cd "/home/bullygen/Documents/[Work] ASC SMBH CV ver 2"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Только проверить существующий датасет

```bash
.venv/bin/python smbh_cv_dataset.py validate \
  --plan experiments/article_v1/dataset_plan.json \
  --out datasets/article_v1
```

## Сгенерировать датасет, если `datasets/article_v1` отсутствует

```bash
.venv/bin/python smbh_cv_dataset.py plan \
  --plan experiments/article_v1/dataset_plan.json \
  --split all \
  --scale full

.venv/bin/python smbh_cv_dataset.py generate \
  --plan experiments/article_v1/dataset_plan.json \
  --split all \
  --scale full \
  --out datasets/article_v1 \
  --workers 4
```

## Быстро проверить, что D1–D6 запускаются

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  plan

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase smoke \
  --max-candidates 1
```

## Перезапустить smoke с нуля

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase smoke \
  --max-candidates 1 \
  --restart
```

## Измерить время на одном настоящем screening-кандидате

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase screening \
  --max-candidates 1

.venv/bin/python -m json.tool \
  experiment_runs/article_v1/screening/baseline/algorithm_seed_239/runtime/d5_stage_summary.json
```

## Полный эксперимент: запускать строго сверху вниз

```bash
(
set -euo pipefail

.venv/bin/python smbh_cv_dataset.py validate \
  --plan experiments/article_v1/dataset_plan.json \
  --out datasets/article_v1

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  plan

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase screening

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase promotion

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase validation

test ! -e experiments/article_v1/frozen_config.json

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  freeze \
  --phase validation \
  --out experiments/article_v1/frozen_config.json

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase test \
  --frozen experiments/article_v1/frozen_config.json

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase verification \
  --frozen experiments/article_v1/frozen_config.json

ARTICLE_CANDIDATE_ID="$(
  .venv/bin/python -c \
    'import json; from pathlib import Path; print(json.loads(Path("experiments/article_v1/frozen_config.json").read_text())["candidate_id"])'
)"

.venv/bin/python smbh_cv_quality.py \
  --test "experiment_runs/article_v1/test/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --verification "experiment_runs/article_v1/verification/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --validation "experiment_runs/article_v1/validation/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --thresholds experiments/article_v1/quality_gate.json \
  --out experiment_runs/article_v1/article_quality_decision.json
)
```

## Продолжить screening после остановки

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase screening
```

## Продолжить promotion после остановки

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase promotion
```

## Продолжить validation после остановки

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase validation
```

## Продолжить test после остановки

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase test \
  --frozen experiments/article_v1/frozen_config.json
```

## Продолжить verification после остановки

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  run \
  --phase verification \
  --frozen experiments/article_v1/frozen_config.json
```

## Пересобрать ranking без повторного расчёта D1–D6

```bash
.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  summarize \
  --phase screening

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  summarize \
  --phase promotion

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  summarize \
  --phase validation
```

## Создать frozen-конфигурацию после готовой validation

```bash
test ! -e experiments/article_v1/frozen_config.json

.venv/bin/python smbh_cv_experiments.py \
  --spec experiments/article_v1/search_plan.json \
  freeze \
  --phase validation \
  --out experiments/article_v1/frozen_config.json
```

## Только запустить quality gate после готовых test и verification

```bash
ARTICLE_CANDIDATE_ID="$(
  .venv/bin/python -c \
    'import json; from pathlib import Path; print(json.loads(Path("experiments/article_v1/frozen_config.json").read_text())["candidate_id"])'
)"

.venv/bin/python smbh_cv_quality.py \
  --test "experiment_runs/article_v1/test/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --verification "experiment_runs/article_v1/verification/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --validation "experiment_runs/article_v1/validation/${ARTICLE_CANDIDATE_ID}/algorithm_seed_239/statistics/summary.json" \
  --thresholds experiments/article_v1/quality_gate.json \
  --out experiment_runs/article_v1/article_quality_decision.json
```

## Показать итог quality gate

```bash
.venv/bin/python -m json.tool \
  experiment_runs/article_v1/article_quality_decision.json
```

## Показать готовые результаты

```bash
find experiment_runs/article_v1 \
  -path '*/statistics/summary.json' \
  -print \
  | sort

find experiment_runs/article_v1 \
  -name ranking.csv \
  -print \
  | sort
```
