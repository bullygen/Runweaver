# Черновик решения о качестве алгоритма

## Принцип

Решение принимает `smbh_cv_quality.py`. Оно использует только числа, которые записывает текущий evaluation code:

- article TP, FP, FN по `article_geometric_iou_v1`;
- article precision, recall, F1;
- 95% image-cluster bootstrap intervals;
- false positives per null image (`null_fppi`) и долю null images хотя бы с одним FP;
- худший заранее заданный factor slice при `n_images >= 20`;
- число test/verification/null images;
- при наличии validation -- вычислимый gap `F1_validation - F1_test`.

В решение не входят визуальное впечатление, «физическая правдоподобность», экспертная уверенность или неизвестная вероятность photon ring.

Article match фиксирован до test: `center_error/R <= 0.15`, `radius_error/R <= 0.15`, `annular_IoU >= 0.10`. Исторические `global_*` по seven-parameter cost сохраняются только для сравнения с Excel и в gate не входят.

## Уровни

### `article_ready_strong_under_tested_conditions`

Все условия должны выполниться одновременно:

```text
test precision CI95 low >= 0.90
test recall CI95 low >= 0.82
test F1 CI95 low >= 0.85
test null FPPI CI95 high <= 0.10
test worst prespecified factor F1 CI95 low >= 0.70

verification precision CI95 low >= 0.75
verification recall CI95 low >= 0.65
verification F1 CI95 low >= 0.70
verification null FPPI CI95 high <= 0.25
```

Также нужны ровно предусмотренные объёмы: не менее 500 test, 100 test null, 500 verification и 100 verification null.

Формулировка вывода: «Алгоритм устойчиво обнаруживает кольцеобразную морфологию в пределах объявленных synthetic image-domain условий». Даже этот уровень не разрешает утверждение «обнаружен photon ring».

### `promising_but_limited`

Если strong gate не пройден, но выполнены point-estimate условия:

```text
test precision >= 0.85
test recall >= 0.75
test F1 >= 0.80
test null FPPI <= 0.25
verification F1 >= 0.60
verification null FPPI <= 0.50
```

Формулировка: «Метод перспективен как candidate generator/морфологический detector, но robustness или false-positive control пока недостаточны для сильного утверждения».

### `experimental_not_yet_robust`

Хотя бы одно limited condition нарушено. В статье можно сообщать результаты как отрицательный/диагностический experiment, показывая slices и failure cases, но нельзя заявлять достигнутое общее качество.

### `insufficient_evidence`

Не выполнен минимальный размер выборок или отсутствуют необходимые конечные метрики. Pilot/smoke всегда попадает сюда независимо от красивого F1.

## Почему используются нижние и верхние границы CI

Порог по point estimate 0.90 может быть результатом случайной вариации. Strong gate требует, чтобы нижняя 95% граница precision/recall/F1 прошла порог, а верхняя граница null FPPI оставалась ниже лимита. Bootstrap выполняется по изображениям, а не по отдельным кольцам, поэтому несколько колец одной сцены не притворяются независимыми наблюдениями.

## Команда после финальных запусков

Сначала прочитайте `candidate_id` из `experiments/article_v1/frozen_config.json`. Затем подставьте его в пути:

```bash
.venv/bin/python smbh_cv_quality.py \
  --test experiment_runs/article_v1/test/<candidate_id>/algorithm_seed_239/statistics/summary.json \
  --verification experiment_runs/article_v1/verification/<candidate_id>/algorithm_seed_239/statistics/summary.json \
  --validation experiment_runs/article_v1/validation/<candidate_id>/algorithm_seed_239/statistics/summary.json \
  --thresholds experiments/article_v1/quality_gate.json \
  --out experiment_runs/article_v1/article_quality_decision.json
```

Все пороги находятся в versioned JSON. Если их требуется изменить после просмотра test, нужно создать новый protocol version и новый untouched test set.

## Дополнительные вычислимые результаты для текста статьи

Независимо от tier следует сообщить:

1. confusion counts, а не только ratios;
2. CI и число изображений;
3. factor-wise metrics по SNR, beam, uv proxy, background, variability, profile, domain и overlap;
4. median/IQR/95th percentile ошибок center, R, sigma и остальных параметров из `parameter_errors_long.csv`;
5. D3 geometric candidate recall;
6. null FPPI и null image FPR;
7. stochastic seed standard deviation;
8. stage runtime и peak RSS.

AP/mAP, PR/FROC и calibration metrics пока нельзя честно вывести из бинарного final list без единого confidence score и заранее фиксированного threshold sweep. Их отсутствие следует указать как технический долг, а не заменять непроверяемой оценкой.
