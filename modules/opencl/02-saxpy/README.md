# OpenCL 02-basics — SAXPY

Пример `saxpy_cl.cpp` реализует операцию *SAXPY* (Single-Precision A·X + Y) и
служит эквивалентом CUDA-версии из модуля `cuda/02-basics`.

Ключевые этапы:
1. Выделяем и инициализируем векторные данные на CPU.
2. Создаём контекст, очередь команд и компилируем ядро `saxpy`.
3. Копируем входные массивы X и Y на устройство (`clCreateBuffer` + `CL_MEM_COPY_HOST_PTR`).
4. Передаём коэффициент `a` и дескрипторы буферов через `clSetKernelArg`.
5. Запускаем ядро глобальным размером `N`, измеряем время.
6. Читаем результаты `clEnqueueReadBuffer`, проверяем корректность.

Формула проверяется: `Y[i] = a * X[i] + Y[i]`.

## Быстрый старт
```bash
cmake --build build --target saxpy_basic_cl
./saxpy_basic_cl 1048576   # 1 M элементов по умолчанию
```
Вывод наподобие:
```
OpenCL SAXPY PASSED | N=1048576, time=3.21 ms
```

### Локальная сборка в каталоге модуля

```bash
cd modules/opencl/02-basics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/saxpy_basic_cl 1048576
``` 