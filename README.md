# GPU-School

**GPU-School** — учебный репозиторий, сопровождающий курс по программированию на GPU (CUDA, OpenCL, HIP, SYCL и др.). Проект содержит исходники лабораторных работ, презентации, ноутбуки и дополнительную литературу.

## Структура репозитория

| Путь | Содержимое |
|------|------------|
| `modules/` | Папки с модулями курса (00-intro, 01-architecture …) |
| `modules/cuda/` | CUDA-модули курса (00-intro … 08-profiling) |
| `modules/opencl/` | Аналогичные примеры на OpenCL (00-intro … 06-advanced) |
| `modules/opencl/00-intro` | Минимальный `hello_opencl` |
| `modules/opencl/04-performance` | Оптимизированный tiled MatMul |
| `docs/` | (опционально) сгенерированная документация Sphinx/Markdown |
| `ROADMAP.md` | Двоязычная дорожная карта всех модулей |
| `CMakeLists.txt` | Корневая конфигурация CMake для сборки C/CUDA примеров |

## Быстрый старт

1. Установите:
   • NVIDIA/AMD драйвер с поддержкой CUDA *и/или* OpenCL ≥ 1.2 (≥ 3.0 ещё лучше);  
   • CMake ≥ 3.20, GCC/Clang ≥ 11.

2. Конфигурируйте и соберите **все** примеры одной командой:

```bash
# Корень репозитория
airbuild=build
cmake -S . -B $airbuild -DCMAKE_BUILD_TYPE=Release
cmake --build $airbuild -j$(nproc)
```

3. Запустите единый набор тестов (CUDA + OpenCL):

```bash
ctest --test-dir $airbuild --output-on-failure
```
Вы увидите ~13 тестов `PASSED` — значит всё скомпилировалось и выполнилось.

4. Чтобы собрать и протестировать только один пример:

```bash
cmake --build $airbuild --target hello_opencl -j$(nproc)
./build/hello_opencl
```

Для CUDA-версии замените цель на `hello_gpu` и т.д.

## Дорожная карта
См. файл `ROADMAP.md` — он описывает 17 модулей от введения в CUDA до capstone-проекта и DevOps-развёртывания.

## Как вносить вклад

1. Форкните репозиторий или создайте ветку.
2. Назовите ветку по шаблону `module-XX-feature`.
3. Откройте Pull Request + описание изменений.
4. Запустите `cmake --build . && ctest` (будущие тесты) перед отправкой.

## Лицензия

Все материалы распространяются под лицензией MIT (см. `LICENSE`). Исключение могут составлять сторонние библиотеки, приведённые в примерах — они лицензируются отдельно.  

### Сборка отдельного примера из его каталога

Каждый подмодуль имеет собственный `CMakeLists.txt`, поэтому можно
собрать его автономно, не затрагивая остальной проект. Например для
OpenCL SAXPY:

```bash
cd modules/opencl/02-basics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/saxpy_basic_cl 1048576
```

То же справедливо для CUDA-модулей (`modules/cuda/...`).  
