# Graph-JEPA: графы, маски и конфигурация экспериментов

Это руководство описывает графовое маскирование Xela JEPA, добавленные стратегии
target-масок и рекомендуемые для текущей серии экспериментов значения. Источником
истины для валидации параметров остаётся
`tactile_ssl/utils/jepa_masking.py`; этот документ объясняет их смысл и фиксирует
принятые экспериментальные соглашения.

## Рекомендуемая отправная точка

Для нового сопоставимого эксперимента используйте:

```yaml
# @package _global_
defaults:
  - /experiment/xela/jepa_graph
  - _self_

run_name: physical_dijkstra_1c4t_my_method_c50-90_t10-18

algorithm:
  num_context_masks: 1
  num_target_masks: 4
  context_mask_scale: [0.50, 0.90]
  target_mask_scale: [0.10, 0.18]
  masking:
    target:
      groups:
        - count: 4
          strategy: connected_region
          growth: dijkstra
```

Базовый `jepa_graph` уже задаёт:

```yaml
algorithm:
  masking:
    mode: multiblock_graph
    context:
      strategy: connected_region
      growth: dijkstra
    target:
      strategy: connected_region
      growth: dijkstra
    overlap:
      target_target: allow
      context_target: subtract_from_context
      context_context: allow
    min_context_keep_tokens: 32
    min_context_keep_ratio: 0.15
    max_resample_attempts: 32

data:
  graph:
    enabled: true
    type: physical
    topology_mode: per_window
    edge_attr_mode: distance
    params:
      bridge_k: 4
```

Для сравнения только геометрии target-масок рекомендуется не менять остальные
параметры: `1 context`, `4 targets`, context `50–90%`, target `10–18%`,
`physical/per_window/distance`, Dijkstra и одинаковый training recipe.

Краткая таблица зафиксированных значений:

| Ось | Основное значение | Что считать абляцией |
|---|---|---|
| Число масок | `1 context`, `4 targets` | `2c4t`, `1c2t` |
| Размер context | `[0.50, 0.90]` | Старый `[0.40, 0.65]` |
| Размер каждого target | `[0.10, 0.18]` | Старый `[0.05, 0.11]`; multiscale locals |
| Context | `connected_region + dijkstra` | Random context, BFS context |
| Граф | `physical`, `per_window`, distance weights | kNN, threshold, custom |
| Межплощадочные мосты | `bridge_k: 4` | `bridge_k: 2` и другие значения |
| Overlap targets | Разрешён | Ограничения внутри отдельных target groups |
| Context-target | Вычитание union targets | Другие режимы не поддерживаются |
| Минимальный итоговый context | `max(32, ceil(0.15N))`; при `N=368` это 56 | Более строгий порог |
| Пересэмплирования | `32` | Изменять только при измерении времени/failure rate |
| Дальние endpoints | `farthest_quantile: 0.80` | Другие квантили |
| Две endcap-лопасти | `lobe_size_ratio: 0.50`, `min_lobe_tokens: 16` | Неравные лопасти |
| Overlap двух structured globals | Не более `0.50`, если ограничение включено | `1.0` снимает ограничение |

## Режим и обратная совместимость

```yaml
algorithm:
  masking:
    mode: multiblock_graph
```

включает новый sampler. Он требует `XelaGraphSSLDataset`, граф в каждом sample и
`JEPAGraphMaskCollator`, которые уже подключены базовым experiment
`xela/jepa_graph`.

Если секция `algorithm.masking` отсутствует либо имеет `mode: legacy`,
`XelaJEPAModule` использует старый индексный sampler и его прежнее RNG-поведение.
Это необходимо для воспроизводимости старых checkpoint. Добавление graph-config
само по себе не переводит legacy JEPA на графовые маски.

Если все targets однотипны, `target.groups` необязателен:

```yaml
masking:
  target:
    strategy: connected_region
    growth: dijkstra
```

Тогда эти настройки применяются ко всем `num_target_masks`. `groups` нужен для
смеси стратегий, разных scale или отдельных overlap-ограничений.

## Как формируется layout

Для каждого элемента batch:

1. Независимо сэмплируются все targets.
2. Независимо сэмплируется сырой связный context.
3. Из context удаляется объединение всех targets.
4. Если осталось слишком мало context-токенов, context пересэмплируется; затем
   при необходимости пересэмплируется весь layout.
5. После исчерпания `max_resample_attempts` обучение завершается явной ошибкой:
   скрытого fallback с пересечением context и targets нет.
6. Внутри batch все contexts обрезаются до общей минимальной длины. Порядок
   обхода сохраняется до этой обрезки, затем ID сенсоров сортируются для модели.

Поддерживается только следующая I-JEPA-семантика:

| Опция | Поддерживаемое значение | Смысл |
|---|---|---|
| `overlap.target_target` | `allow` | Разные targets могут пересекаться |
| `overlap.context_target` | `subtract_from_context` | Union targets удаляется из context |
| `overlap.context_context` | `allow` | Несколько contexts могут пересекаться |

Связность гарантируется для сырого context и для тех targets, стратегия которых
её обещает. После вычитания targets итоговый context может стать несвязным — это
ожидаемая I-JEPA-семантика.

## Размеры масок

```yaml
algorithm:
  num_context_masks: 1
  num_target_masks: 4
  context_mask_scale: [0.50, 0.90]
  target_mask_scale: [0.10, 0.18]
```

Размер один раз случайно выбирается из указанного диапазона для collated batch.
Перевод доли в токены выполняется как `int(num_sensors * sampled_scale)`.

Для 368 сенсоров принятые диапазоны примерно равны:

| Маска | Принятый диапазон | Число токенов |
|---|---:|---:|
| Context | `0.50–0.90` | `184–331` |
| Каждый target | `0.10–0.18` | `36–66` |
| Малый local в multiscale-вариантах | `0.05–0.11` | `18–40` |

`target.groups[].scale` переопределяет общий `target_mask_scale` только для этой
группы. Поэтому можно совместить local targets размером `5–11%` и global targets
размером `10–18%`:

```yaml
target:
  groups:
    - count: 2
      strategy: connected_region
      growth: dijkstra
      scale: [0.05, 0.11]
    - count: 2
      strategy: geodesic_endcaps
      growth: dijkstra
      scale: [0.10, 0.18]
```

Сумма всех `target.groups[].count` обязана точно равняться
`algorithm.num_target_masks`.

## Context

Для context сейчас рекомендуется:

```yaml
context:
  strategy: connected_region
  growth: dijkstra
```

Поддерживаемые context-стратегии:

| `strategy` | Смысл |
|---|---|
| `connected_region` | Связная область на графе |
| `random` | Независимая случайная выборка сенсоров без требования связности |

Для `connected_region` доступны `growth: dijkstra` и `growth: bfs`.
На основном физическом графе принят Dijkstra.

Итоговый минимальный context:

```text
max(min_context_keep_tokens,
    ceil(min_context_keep_ratio * num_sensors))
```

При принятых `32`, `0.15` и 368 сенсорах порог равен 56 токенам. Эти значения
рекомендуется сохранять. Увеличивать порог стоит только как отдельную абляцию:
слишком строгий порог повышает число пересэмплирований и время collate.

## Dijkstra и BFS

| Обход | Что минимизируется | Использует `edge_attr` | Когда применять |
|---|---|---|---|
| `dijkstra` | Суммарная физическая длина пути | Да | Основной вариант; пространственная геометрия важна |
| `bfs` | Число рёбер, hop-distance | Нет | Абляция топологии или намеренное игнорирование длины |

Оба обхода используют случайный детерминированный tie-break через текущий
`torch.Generator`. Входной `edge_index` трактуется как неориентированный, дубли
рёбер удаляются. Dijkstra требует положительные конечные веса.

Не следует автоматически переводить весь эксперимент на BFS. Полезная чистая
абляция — BFS только для local targets при сохранении weighted endcaps:

```yaml
- count: 2
  strategy: connected_region
  growth: bfs
- count: 2
  strategy: geodesic_endcaps
  growth: dijkstra
```

Если нужно полностью топологическое сравнение, используйте BFS locals вместе с
`topological_endcaps`.

## Стратегии targets

### `connected_region`

Один случайный seed и связное разрастание Dijkstra или BFS до точного бюджета.
Это основной local target.

```yaml
- count: 2
  strategy: connected_region
  growth: dijkstra
```

Рекомендация: Dijkstra; `scale: [0.10, 0.18]` для стандартного сравнения или
`[0.05, 0.11]` для явно более локальной multiscale-постановки.

### `random`

Равномерная выборка `k` сенсоров из всех 368 без связности и без использования
графа. Это «настоящий random», совместимый по идее со старой JEPA.

```yaml
- count: 2
  strategy: random
```

Параметр `growth` для результата несущественен. Random полезен как источник
глобального покрытия, но сам по себе не гарантирует охват разных частей руки.

### `stratified_random`

Случайная несвязная выборка, распределённая по группам сенсоров. Требует
`graph.node_group_id`, который dataset формирует через `node_group_mode`.

```yaml
- count: 2
  strategy: stratified_random
  min_per_group: 1
  remainder_allocation: proportional
```

Здесь поддерживается только `remainder_allocation: proportional`: после
обязательного минимума остаток бюджета распределяется пропорционально доступному
числу сенсоров группы.

Для стратификации по всем физическим площадкам:

```yaml
data:
  graph:
    node_group_mode: link
```

`link` — default и даёт 18 групп. При минимальном target-бюджете 36 токенов
`min_per_group: 1` гарантирует хотя бы один сенсор с каждой площадки.

### `stratified_connected_lobes`

Target является объединением нескольких связных шариков: по одному внутри каждой
группы. Весь target обычно несвязен, но каждый шарик связен и имеет свой случайный
seed.

```yaml
algorithm:
  masking:
    target:
      groups:
        - count: 4
          strategy: stratified_connected_lobes
          growth: dijkstra
          min_per_group: 1
          remainder_allocation: equal

data:
  graph:
    node_group_mode: hypertaxel
```

Для этой стратегии поддерживается только `remainder_allocation: equal`. Бюджет
делится максимально поровну, а остаток случайно раздаётся группам.

Принятый `hypertaxel`-вариант содержит пять групп:

| Гипертаксель | Сенсоров |
|---|---:|
| Большой палец | 62 |
| Указательный палец | 78 |
| Средний палец | 78 |
| Безымянный палец | 78 |
| Ладонь | 72 |

При target `36–66` каждый из пяти шариков получает примерно `7–14` сенсоров.
Для текущего эксперимента принято четыре таких targets без дополнительных local
targets.

`node_group_mode: hand_part` является синонимом `hypertaxel`.

### `graph_farthest_points`

Первый сенсор выбирается случайно, каждый следующий максимизирует минимальную
взвешенную графовую дистанцию до уже выбранных. Получается распределённая по
графу, обычно несвязная маска.

```yaml
- count: 1
  strategy: graph_farthest_points
  growth: dijkstra
  max_previous_overlap_ratio: 0.50
```

Требует Dijkstra и SciPy. Это более дорогая стратегия; разумнее использовать один
такой target, а не все четыре, если целью не является отдельная FPS-абляция.

### `geodesic_corridor`

Выбирается seed, затем вторая точка из дальнего квантиля допустимых вершин.
Восстанавливается кратчайший путь, после чего он расширяется multi-source
Dijkstra до точного target-бюджета.

```yaml
- count: 2
  strategy: geodesic_corridor
  endpoint_sampling: farthest_quantile
  farthest_quantile: 0.80
  growth: dijkstra
  max_pairwise_overlap_ratio: 0.50
```

Принятые значения:

- `endpoint_sampling: farthest_quantile` — пока единственный поддерживаемый режим;
- `farthest_quantile: 0.80` — endpoint случайно выбирается из дальних 20%;
- `max_pairwise_overlap_ratio: 0.50` — два коридора не совпадают более чем
  наполовину.

Коридор имеет точный бюджет и связен. Его ширина отдельно не фиксируется:
сначала бюджет занимает путь, остаток формирует расширение вокруг него.

### `geodesic_endcaps`

Target состоит из двух непересекающихся связных Dijkstra-лопастей вокруг
геодезически далёких концов. Сам target не обязан быть связным: путь между
лопастями в маску не входит.

```yaml
- count: 2
  strategy: geodesic_endcaps
  endpoint_sampling: farthest_quantile
  farthest_quantile: 0.80
  growth: dijkstra
  lobe_size_ratio: 0.50
  min_lobe_tokens: 16
  max_pairwise_overlap_ratio: 0.50
```

Для текущих сравнений приняты `0.80`, равное деление `0.50`, минимум 16 токенов
на лопасть и не более 50% overlap между endcap-targets. При target `36–66`
лопасти обычно получают `18–33` токена.

### `topological_endcaps`

Та же идея двух далёких лопастей, но дальность и разрастание определяются только
hop-distance. Длины рёбер игнорируются.

```yaml
- count: 2
  strategy: topological_endcaps
  endpoint_sampling: farthest_quantile
  farthest_quantile: 0.80
  growth: bfs
  lobe_size_ratio: 0.50
  min_lobe_tokens: 16
  max_pairwise_overlap_ratio: 0.50
```

Стратегия требует `growth: bfs`. Использовать её стоит как чистую BFS-абляцию,
а не как default.

## Target groups и ограничения overlap

Каждая запись `target.groups` задаёт последовательную группу однотипных targets:

```yaml
target:
  groups:
    - count: 2
      strategy: connected_region
      growth: dijkstra
    - count: 2
      strategy: geodesic_endcaps
      growth: dijkstra
      max_pairwise_overlap_ratio: 0.50
      max_previous_overlap_ratio: 1.00
```

| Параметр | Диапазон/default | Смысл |
|---|---|---|
| `count` | положительный `int`, обязателен | Число targets этой группы |
| `strategy` | default из `masking.target.strategy` | Геометрия target |
| `growth` | default из `masking.target.growth` | Dijkstra или BFS |
| `scale` | `null` | Свой диапазон размера вместо общего |
| `max_pairwise_overlap_ratio` | `0–1`, default `1` | Максимальная доля пересечения с предыдущими targets той же группы |
| `max_previous_overlap_ratio` | `0–1`, default `1` | Максимальная доля пересечения с targets всех более ранних групп |

Доля overlap считается относительно размера нового candidate. Значение `1.0`
означает отсутствие ограничения, `0.5` — не более половины candidate.

Глобальная опция `overlap.target_target: allow` разрешает пересечения. Параметры
группы могут сделать это разрешение строже, но не меняют вычитание union targets
из context.

## Граф

```yaml
data:
  dataset_target: tactile_ssl.data.xela_tactile.XelaGraphSSLDataset
  graph:
    enabled: true
    type: physical
    topology_mode: per_window
    edge_attr_mode: distance
    params:
      bridge_k: 4
```

### Принятый основной граф

- `type: physical`;
- `topology_mode: per_window`;
- `edge_attr_mode: distance`;
- `params.bridge_k: 4`.

Внутри каждой физической площадки `physical` использует рёбра четырёхсвязной
сетки. Для каждой известной пары соседних площадок `bridge_k` добавляет до `k`
кратчайших межплощадочных рёбер с разными концами.

То есть `bridge_k: 2` действительно означает до двух связок **на каждую заданную
пару соседних площадок**, а не две связки на всю руку. Это допустимый более
разреженный вариант. Текущий общий default — `4`; менять его следует как
отдельную абляцию и отражать в `run_name`.

`per_window` перестраивает веса и, где применимо, топологию по координатам
текущего окна. Графы кэшируются preprocessing cache. Для графового sampler
batch должен содержать `edge_index`, `edge_count`, а при Dijkstra ещё
`edge_attr`.

### Другие типы графов

| `type` | Основные `params` | Смысл |
|---|---|---|
| `physical` | `bridge_k` | Сетка площадок плюс известные физические мосты |
| `distance_threshold` | `threshold` | Ребро между сенсорами ближе порога в метрах |
| `knn` | `k`, `symmetrize`; либо `k_inner_neighbors`, `k_outer_neighbors` | k ближайших по XYZ |
| `custom` | см. ниже | Композиция локальных, физических и метрических рёбер |

`distance` принимается как устаревший alias для `distance_threshold`.

Параметры `custom`:

```yaml
params:
  link_pads: sparse       # none | sparse | dense
  phys_bridge_k: 4
  distance_threshold: null
  k_inner_neighbors: 0
  k_outer_neighbors: 0
  outer_edge_distance: null
```

`k_neighbors` и `k_extra_neighbors` — устаревшие aliases для
`k_inner_neighbors` и `k_outer_neighbors`.

Sampler не знает `graph.type`: он получает готовые padded `edge_index`,
`edge_attr`, `edge_count` и при необходимости `node_group_id`. Поэтому любую
стратегию можно исследовать на любом builder-графе при соблюдении требований к
весам и связности.

### `node_group_mode`

| Значение | Групп | Назначение |
|---|---:|---|
| `link` | 18 | Физические сенсорные площадки; default |
| `hypertaxel` | 5 | Четыре пальца и ладонь |
| `hand_part` | 5 | Alias для `hypertaxel` |

Этот параметр влияет только на `stratified_random` и
`stratified_connected_lobes`; обычные графовые стратегии его игнорируют.

## Рекомендуемые экспериментальные рецепты

### Чистый graph-local baseline

```yaml
target:
  groups:
    - count: 4
      strategy: connected_region
      growth: dijkstra
```

### Два local + два endcaps

```yaml
target:
  groups:
    - count: 2
      strategy: connected_region
      growth: dijkstra
    - count: 2
      strategy: geodesic_endcaps
      endpoint_sampling: farthest_quantile
      farthest_quantile: 0.80
      growth: dijkstra
      lobe_size_ratio: 0.50
      min_lobe_tokens: 16
      max_pairwise_overlap_ratio: 0.50
```

### Два local + два true-random

```yaml
target:
  groups:
    - count: 2
      strategy: connected_region
      growth: dijkstra
    - count: 2
      strategy: random
```

### Четыре hypertaxel-endcaps

```yaml
target:
  groups:
    - count: 4
      strategy: stratified_connected_lobes
      growth: dijkstra
      min_per_group: 1
      remainder_allocation: equal

data:
  graph:
    node_group_mode: hypertaxel
```

Во всех четырёх рецептах для честного сравнения сохраняются `1c4t`,
`c50–90`, `t10–18` и основной physical graph.

## Taxel-type embedding и координаты

Геометрия масок и входные признаки энкодера — независимые оси эксперимента.

```yaml
algorithm:
  encoder:
    use_taxel_type_embedding: false
```

отключает learned taxel-type embedding, но не отключает графовый sampler.
Такой запуск необходимо явно помечать `no-taxel` в `run_name` и сравнивать с
оригиналом, обученным с тем же downstream protocol. Не следует считать
`node_group_mode: hypertaxel` заменой taxel-type embedding: group IDs нужны
только collator для построения маски и не передаются энкодеру как embedding.

### Режимы подачи XYZ в XelaTransformer

При `data.features.use_spatial_coords: true` dataset возвращает шесть каналов:
три магнитных показания и XYZ каждого сенсора. `algorithm.encoder.input_fusion`
задаёт способ их преобразования:

| Режим | Преобразование |
|---|---|
| `joint` | Один `PatchEmbed1d(6 → embed_dim)`, как в исходном Xela DINO |
| `separate_coordinates` | Отдельные `PatchEmbed1d(3 → embed_dim)` для сигнала и XYZ, concat и `Linear(2D → D)` |
| `fresh_random` | Signal `PatchEmbed1d(3 → D)`, свежий Gaussian `D`-embedding вместо XYZ, concat и `Linear(2D → D)` |

Для `xela_tiny` `embed_dim=192`. Во всех режимах временная свёртка сохраняется:
при `sequence_length=10` и `time_chunk_size=10` один token агрегирует десять
кадров одного сенсора.

Рекомендуемые настройки coordinate-абляции:

```yaml
algorithm:
  encoder:
    in_chans: 6
    input_fusion: separate_coordinates  # joint | fresh_random
    signal_chans: 3
    coordinate_chans: 3
    random_embedding_std: 1.0

data:
  features:
    use_spatial_coords: true
```

В `fresh_random` XYZ намеренно игнорируются. Шум пересэмплируется на каждом
forward, но context и target encoder внутри одного JEPA-forward получают один и
тот же tensor; иначе target содержал бы независимо сэмплированную
непредсказуемую компоненту. На downstream шум также свежий на каждом forward,
поэтому этот вариант является заведомо стохастическим control.

### Coordinate-only JEPA и late fusion

Для независимого coordinate-backbone доступны:

- `coordinates_only_patch`: XYZ каждого сенсора за 10 кадров проходят через
  `PatchEmbed1d(3 channels, length 10, chunk 10)`;
- `coordinates_only_mean_patch`: XYZ усредняются по 10 кадрам и проходят через
  `PatchEmbed1d(3 channels, length 1, chunk 1, padding 0)`.

Оба режима ожидают шесть каналов, полностью игнорируют первые три signal-канала
и не нормализуют абсолютные XYZ статистиками Xela-сигнала.

`XelaLateFusionEncoder` загружает frozen `target_encoder` из signal- и
coordinate-JEPA, конкатенирует их 192D токены каждого сенсора и обучает
`PatchEmbed1d(384 channels, length 1, chunk 1, padding 0)` вместе с downstream
головой. Контроль `fresh_random` заменяет coordinate-токены свежим
`N(0, 1)` шумом при каждом forward.

Полная возобновляемая очередь:

```bash
bash scripts/run_jepa_late_fusion_after_coordinate_queue.sh
```

Она ждёт завершения `run_graph_jepa_coordinate_inputs_queue.sh`, не печатая
сообщение на каждом poll, затем последовательно строит random control, два
coordinate-претрейна, их downstream, накопительную JEPA-таблицу и отдельный
late-fusion отчёт с рангами.

## Запуск полного пайплайна

```bash
bash scripts/run_graph_jepa_experiment.sh \
  xela/jepa_graph_1c4t_4hypertaxelendcaps_c50_90_t10_18
```

Аргумент — Hydra config относительно `config/experiment`, без `.yaml`.
Пайплайн:

1. Проверяет Hydra-compose.
2. Ждёт GPU-lock и устойчиво свободные GPU.
3. Пропускает готовый `epoch-0500.ckpt` либо продолжает незавершённый run из
   `last.ckpt` в той же папке.
4. Обучает force, pose и object downstream.
5. Пропускает downstream, если уже существует `evaluation/test_predictions.npz`.
6. Регистрирует candidate и перестраивает накопительную significance-таблицу.

Основные переменные окружения:

| Переменная | Default | Назначение |
|---|---|---|
| `GPU_IDS` | `0,1,2,3` | Видимые GPU, строго возрастающий список |
| `NUM_DEVICES` | число из `GPU_IDS` | Число trainer devices |
| `PRETRAIN_BATCH_PER_GPU` | `256 / NUM_DEVICES` | Сохраняет global batch 256 |
| `WAIT_SECONDS` | `60` | Проверка уже активного такого же pretrain |
| `GPU_POLL_SECONDS` | `30` | Частота проверки занятых GPU |
| `GPU_FREE_CONFIRM_SECONDS` | `30` | Защита от запуска в паузе между jobs |
| `FINAL_CHECKPOINT_NAME` | `epoch-0500.ckpt` | Критерий завершённого pretrain |
| `DRY_RUN` | `0` | При `1` только compose и печать плана |

Пример безопасной предварительной проверки:

```bash
DRY_RUN=1 bash scripts/run_graph_jepa_experiment.sh \
  xela/jepa_graph_1c4t_4hypertaxelendcaps_c50_90_t10_18
```

Для двух GPU global batches сохраняются автоматически:

```bash
GPU_IDS=1,2 bash scripts/run_graph_jepa_experiment.sh \
  xela/my_experiment
```

## Именование

Имя должно позволять восстановить отличия без чтения checkpoint:

```text
physical_dijkstra_1c4t_<targets>_c50-90_t10-18
```

Добавляйте в него только реально изменённые оси, например:

- `2local2endcaps`;
- `2bfslocal2weightedendcaps`;
- `4hypertaxelendcaps`;
- `no-taxel`;
- `bridge2`;
- `multiscale_..._l05-11_g10-18`.

Не переиспользуйте один `run_name` для разных effective-конфигов: resume-поиск
считает одинаковый `run_name` одним экспериментом.

## Где смотреть реализацию

- Базовая схема: `config/algorithm/xela_jepa_graph.yaml`
- Базовый experiment: `config/experiment/xela/jepa_graph.yaml`
- Sampler и валидация: `tactile_ssl/utils/jepa_masking.py`
- Dataset и `node_group_id`: `tactile_ssl/data/xela_tactile.py`
- Graph builders: `tactile_ssl/graph/builders.py`
- Разметка площадок и гипертакселей: `tactile_ssl/graph/utils.py`
- Тестовые примеры: `tests/test_jepa_graph_masking.py`
- Полный pipeline: `scripts/run_graph_jepa_experiment.sh`
