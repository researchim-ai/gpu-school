# Модуль 2 — Основы CUDA C (Grid/Block, Thread Indexing)

В этом модуле реализуем базовые шаблоны CUDA: вызов ядра `__global__`, конфигурация сетки/блоков и проверка корректности.

## Пример `vector_add.cu`

Складывает два вектора `A` и `B`, получая `C = A + B`.

### Сборка и запуск (из корня репозитория)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vector_add -j$(nproc)
./build/vector_add          # по умолчанию 1М элементов
./build/vector_add 8388608  # произвольный размер
```

При успешной проверке увидите:

```text
VectorAdd PASSED for 1048576 elements
```

## Задания

1. Измените конфигурацию запуска ядра: `<<<blocks, threads>>>`, исследуйте влияние на время.
2. Реализуйте ядро SAXPY: `Y = a * X + Y` с параметром `a`.
3. Добавьте использование `cudaEvent` для измерения времени выполнения. 