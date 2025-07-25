# OpenCL 04-performance — Tiled Matrix Multiplication

`matmul_tiled_cl.cpp` оптимизирует умножение матриц путём тайлинга и
использования локальной памяти.

Отличия от наивной версии:
* Матрицы разбиваются на тайлы `16×16` (константа `TILE`).
* Каждый work-group размера `16×16` загружает соответствующие тайлы `A` и `B`
  в `__local` массивы `As` и `Bs` → меньше обращений в глобальную память.
* После синхронизации (`barrier`) выполняется частичная свёртка по `k`.
* Число проходов по циклу `t` = `ceil(N / TILE)`.

Сценарий проверки: для `N ≤ 512` дополнительно считается CPU-эталон и выводится
ускорение.

Сборка/запуск:
```bash
cmake --build build --target matmul_tiled_cl
./matmul_tiled_cl 256
```

### Локальная сборка в каталоге модуля

```bash
cd modules/opencl/04-performance
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/matmul_tiled_cl 256
``` 