# Выборки `article_v1`: состав, назначение и генерация

## Итоговый объём

Полный набор уже сгенерирован в `datasets/article_v1` и занимает около 2.0 GiB.

| Split | Изображений | Positive | Null | Назначение |
|---|---:|---:|---:|---|
| development | 600 | 480 | 120 | fine-tuning, 5-fold stability |
| validation | 300 | 240 | 60 | выбор среди top-3 и заморозка порогов |
| test | 500 | 375 | 125 | единственная итоговая ID-оценка |
| verification | 500 | 340 | 160 | OOD и совместные stress tests |
| всего | 1900 | 1435 | 465 | |

`development` является «training» только в смысле подбора обычных гиперпараметров. Обучаемой нейросети или весов в текущем D1--D6 нет.

## Почему выбраны такие размеры

- 600 development дают ровно 120 изображений на каждый из пяти folds и позволяют сравнивать кандидатов на одних и тех же сценах.
- 300 validation достаточно для отбора только трёх заранее продвинутых кандидатов, но не используется для широкого поиска.
- 500 test ограничивают image-cluster bootstrap uncertainty и содержат 125 null-сцен. Если на всех 125 null нет ни одного события, rule-of-three даёт верхний порядок около `3/125 = 0.024` для вероятности события на изображение; реальные интервалы вычисляются bootstrap-кодом.
- 500 verification разделены на пять заранее заданных сценариев по 100 изображений; это позволяет не смешивать OOD-причины в один непрозрачный score.

Размеры являются минимальным практическим протоколом для этой вычислительно дорогой версии D5, а не универсальным power calculation. Единицей bootstrap остаётся изображение, потому что кольца внутри изображения зависимы.

## Факторы development/validation/test

Дискретные уровни циклически балансируются, затем перемешиваются независимо внутри каждого split.

| Фактор | Уровни |
|---|---|
| target SNR | 3, 5, 10, 20, 40 |
| Gaussian noise | iid Gaussian; correlated Gaussian с correlation length 2 px |
| background value | 0, 0.001, 0.003, 0.006 |
| background gradient | 0, 0.002, 0.006 со случайным углом |
| background class | исходный jet, two-blob, smooth disk, turbulent |
| beam FWHM | 0, 2, 4, 8 px |
| uv coverage | filled, EHT-like sparse proxy, ngEHT-like dense proxy |
| variability | 0, 0.03, 0.08; усреднение восьми кадров |
| ring profile | gaussian, Lorentzian, variable-width, partial |
| parameter domain | ID core и границы исходного fit domain |
| overlap | random и clustered |
| null type | blobs, knots, disk, crescent, short arcs, turbulent |

SNR вычисляется явно:

```text
SNR = RMS(observed ring-only component) / std(realized noise)
```

`observed` означает применение того же beam/uv transfer, что и к изображению. Для null-сцены уровень шума калибруется по скрытому phantom ring, который не добавляется в изображение. Поэтому null noise distribution сопоставима с positive, но `true_n = 0`.

## Verification scenarios

Каждый блок содержит 100 изображений:

1. `snr_and_background_extreme`: SNR 1.5/2/60 и фон за training range;
2. `held_out_dirty_beam`: unseen `space_vlbi` и `sparse_adversarial`, beam до 12 px;
3. `held_out_dynamics_and_profile`: variability 0.15/0.25, unseen elliptical/top-hat profiles;
4. `parameter_domain_ood`: A, R, sigma, center и background parameters вне fit domain;
5. `held_out_nulls`: только null-сцены с совместными тяжёлыми деградациями.

Verification нельзя использовать для повторного выбора порогов. Если на ней найден провал, он становится ограничением текущей версии и гипотезой следующей, отдельно пререгистрированной серии.

## Parameter domain

- `id_core` отступает от границ исходного fit domain.
- `id_boundary` касается исходных границ и проверяет boundary effects.
- `ood_extended` выходит за исходные fit bounds: например, `R = 42...105` вместо `50...91`, `sigma = 3...14` вместо `5...10`, centers до ±42 px от центра. Детектор намеренно не получает расширенные fit bounds на verification.

## uv-plane и dirty beam: точная граница модели

Генератор строит детерминированную Hermitian-symmetric image-domain Fourier transfer mask из радиальных tracks и применяет её к FFT изображения. В `truth.json` сохраняются `uv_coverage`, доля mask выше 0.5 и peak sidelobe dirty beam.

Это воспроизводимый proxy разных angular resolutions и dirty-beam patterns. Это не симуляция конкретных станций, времени наблюдения, thermal visibility errors, gain/phase corruption, CLEAN или RML. Для сильного VLBI-утверждения статьи нужен отдельный внешний benchmark с физическими `(u,v,t)`-координатами и как минимум двумя reconstruction pipelines.

## Формат одного изображения

```text
datasets/article_v1/<split>/
  design.json
  manifest.json
  images/image_000000/
    image.npy
    clean_image.npy
    truth_clean_pre_degradation.npy
    noise.npy
    truth.json
```

`truth.json` содержит split, fold, уникальные `family_id`/`sample_seed`, истинные параметры колец, все факторы, target/realized SNR и честное описание reconstruction proxy. `manifest.json` содержит SHA-256 пиксельного массива.

## Команды

Посмотреть план без записи данных:

```bash
.venv/bin/python smbh_cv_dataset.py plan --split all --scale full
```

Повторная полная генерация в новый каталог:

```bash
.venv/bin/python smbh_cv_dataset.py generate \
  --plan experiments/article_v1/dataset_plan.json \
  --split all --scale full \
  --out datasets/article_v1_reproduction \
  --workers 4
```

Проверить существующий набор:

```bash
.venv/bin/python smbh_cv_dataset.py validate \
  --plan experiments/article_v1/dataset_plan.json \
  --out datasets/article_v1
```

Генератор отказывается перезаписывать непустой split без явного `--overwrite`. Для экономии места можно добавить `--no-components`; тогда сохраняются только `image.npy` и `truth.json`, достаточные D1--D6 и statistics, но не полной визуализации generation components.

Фактическая проверка текущего набора: `valid = true`, 1900 уникальных family IDs, 1900 уникальных image hashes, пропущенных или non-finite изображений нет. Полный отчёт находится в `datasets/article_v1/validation_report.json`.
