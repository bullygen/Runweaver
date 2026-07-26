# Воспроизводимый протокол экспериментов для статьи

## Короткий ответ

Старое значение около precision 0.90 / recall 0.82 получено после многократного выбора параметров на 24 идеальных изображениях и не является final test result. Новый протокол разделяет 1900 независимых сцен на development, validation, test и OOD verification, включает null-контроль и все запрошенные факторы. Реализован возобновляемый 25-кандидатный multi-start optimizer, factor-wise bootstrap metrics и заранее фиксированный quality gate.

Основные файлы:

- аудит старых результатов: `docs/EXPERIMENT_AUDIT_2026_07_21.md`;
- состав и генерация данных: `docs/DATASET_ARTICLE_V1.md`;
- порядок fine-tuning: `docs/HYPERPARAMETER_SEARCH.md`;
- формальное решение о качестве: `docs/QUALITY_DECISION_DRAFT.md`;
- dataset design: `experiments/article_v1/dataset_plan.json`;
- search design: `experiments/article_v1/search_plan.json`;
- predeclared thresholds: `experiments/article_v1/quality_gate.json`.

## Этапы исследования

| Этап | Данные | Разрешённое действие | Запрещённое действие |
|---|---|---|---|
| аудит | старые 24/100 images | сформировать гипотезы/search ranges | считать метрики итоговыми |
| screening | 60 development | широкий low-fidelity поиск | открывать test |
| promotion | 240 development × 2 seeds | проверить top-6 | менять семейства D1--D6 |
| validation | 300 × 2 seeds | выбрать top-1 из top-3, freeze | широкий новый поиск |
| test | 500 × 3 seeds | один раз оценить frozen config | tuning по результату |
| verification | 500 OOD × 3 seeds | определить robustness limits | ретроспективно менять gate |

## Что изменено в коде

Добавлен только внешний экспериментальный контур:

- факторный генератор и split manifest;
- image-domain uv/dirty-beam proxy, noise/background/dynamics/mismatch/null generators;
- leakage/content validation;
- multi-start Latin-hypercube runner с successive fidelity и resume;
- bootstrap CI, factor slices и null metrics;
- deterministic quality gate;
- тесты workflow.

Функции и названия основных математических методов D1--D6 не заменялись. Старый seven-parameter matching сохранён в полях `global_*` для сопоставимости с Excel. Для статьи добавлен отдельный `article_geometric_iou_v1`: normalized center error <= 0.15, normalized radius error <= 0.15 и annular IoU >= 0.10. Его поля `article_*` использует optimizer и quality gate. Это не позволяет другому radial profile стать FN только из-за A, sigma, B или phi. Исправлен и D3 diagnostic: он теперь оценивает только параметры, которые D3 действительно выдаёт.

## Что делать прямо сейчас

1. Проверить существующие данные:

   ```bash
   .venv/bin/python smbh_cv_dataset.py validate --out datasets/article_v1
   ```

2. Запустить один baseline screening smoke и измерить время:

   ```bash
   MPLCONFIGDIR=/tmp/smbh-mpl .venv/bin/python smbh_cv_experiments.py \
     --spec experiments/article_v1/search_plan.json \
     run --phase smoke --max-candidates 1
   ```

3. Если runtime приемлем, выполнить полный screening, затем строго команды из `HYPERPARAMETER_SEARCH.md`.
4. После freeze выполнить test один раз, затем verification.
5. Запустить quality gate и использовать его tier как основу формулировки результата.

## Обязательные таблицы и рисунки из получаемых файлов

- main table: global counts/metrics/CI на test и verification;
- robustness table: `factor_metrics.csv`;
- null table: FPPI и null image FPR по null type;
- parameter table: `parameter_metrics.csv` плюс quantiles из long CSV;
- stochastic table: три algorithm seeds отдельно;
- runtime table: stage summaries;
- plots versus SNR, beam FWHM, uv coverage, profile и parameter domain;
- representative TP/FP/FN и failures.

Для полноценной заявленной в PDF VLBI-части остаётся отдельный будущий слой: реальные synthetic visibilities, station/calibration corruptions, CLEAN и RML. Текущий набор достаточен для честной image-domain robustness статьи, но не для visibility-domain superiority claim.
