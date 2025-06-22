# Модуль 3 — Управление памятью и наивное умножение матриц

В этом модуле мы рассматриваем различия между *host*- и *device*-памятью, способы их выделения (`cudaMalloc`, Unified Memory) и демонстрируем наивное умножение матриц.

## Пример `matmul_naive.cu`

– Наивное ядро (без shared memory).  
– Выделение и копирование данных вручную.  
– Сравнение времени GPU vs CPU.

### Сборка и запуск

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target matmul_naive -j$(nproc)
./build/matmul_naive            # 512×512 (по умолчанию)
./build/matmul_naive 1024       # 1024×1024
```

Пример вывода:

```text
Matrix 512x512
GPU naive matmul:  22.31 ms
CPU reference:     180.45 ms
Speedup:           8.09x
Validation:        PASSED
```

## Задания

1. Включите `-lineinfo` и посмотрите в `nvprof`/`Nsight Compute`, где тратится время.
2. Реализуйте tiled-версию ядра с использованием *shared memory* и сравните ускорение.
3. Попробуйте включить *Unified Memory* (`cudaMallocManaged`) и замерьте разницу. 