# OpenCL 03-memory — Naive Matrix Multiplication

Пример `matmul_naive_cl.cpp` умножает две квадратные матрицы `A(N×N)` и `B` с
вычислением `C = A × B`.

Структура программы:
1. Подготовка входных данных: `A` — единичная матрица, `B` — линейно
   возрастающие числа. Это упрощает проверку, т.к. `C` должно совпадать с `B`.
2. Создание контекста, очереди, компиляция ядра `matmul`.
3. В ядре каждая work-item вычисляет один элемент `C[row][col]` в
   двойном цикле по `k`.
4. Глобальный NDRange задаётся как `N × N`, локальный — не указывается (оставляем
   автотюнинг драйверу).
5. После выполнения результат копируется назад и сверяется с CPU-эталоном.

Запуск:
```bash
cmake --build build --target matmul_naive_cl
./matmul_naive_cl 256
```

### Локальная сборка в каталоге модуля

```bash
cd modules/opencl/03-memory
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/matmul_naive_cl 256
``` 