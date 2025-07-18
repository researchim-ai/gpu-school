# Модуль 1 — Архитектура GPU и коалесцирование памяти

В этом модуле мы изучаем основы архитектуры GPU (SIMT, иерархия памяти, блоки/варпы) и демонстрируем влияние паттернов доступа к памяти на производительность.

## Пример `memory_access.cu`

Программа сравнивает время выполнения двух ядер:

1. **vecAddCoalesced** — потоковая (коалесцированная) загрузка данных;
2. **vecAddStrided** — некоалесцированный доступ с шагом 32 (размер варпа).

### Сборка

Из корня репозитория:

```bash
cd build && cmake --build . -j$(nproc)
```

### Запуск

```bash
./modules/01-architecture/memory_access
```

Ожидаемый вывод (пример):

```text
Vector size: 16777216 elements
Coalesced access time:       1.25 ms
Strided (stride=32) time: 6.80 ms
Strided / Coalesced ratio: 5.44x
```

Коэффициент показывает, насколько дороже некоалесцированный доступ.

## Задания

1. Измените `stride` на 2, 4, 8, 16 и измерьте время.
2. Попробуйте увеличить размер вектора до `1 << 26`.
3. Добавьте использование `cudaOccupancyMaxActiveBlocksPerMultiprocessor` для оценки заполнения.

## Теория

GPU-память делится на уровни (глобальная, shared, регистры). **Коалесцирование** означает, что 32 последовательных 4-байтовых обращения варпа могут быть объединены контроллером памяти в одну транзакцию.

Условия коалесцированного чтения/записи:
1. Потоки варпа обращаются к адресам, лежащим в одном 128-байтовом сегменте (для float32).
2. Адреса упорядочены монотонно: `idx + const`.
3. Размер слова постоянен для всех потоков.

Если хотя бы одно из условий нарушается (пример с `stride = 32`) — каждая нить тянет отдельную транзакцию; пропускная способность падает во столько раз, сколько лишних транзакций.

Формулы:
```
# транзакций при коалесцировании  ≈  N / (warp_size * words_per_transaction)
# транзакций при stride           ≈  N / warp_size * stride
```

Shared memory (on-chip) позволяет полностью избежать проблем коалесцирования, если данные сначала загружаются коалесцированно в блок, а потом переиспользуются.

> Nsight Compute показывает «Memory Throughput» и «Transactions per request» — полезно профилировать. 