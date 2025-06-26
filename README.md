# GPU-School

**GPU-School** — учебный репозиторий, сопровождающий курс по программированию на GPU (CUDA, OpenCL, HIP, SYCL и др.). Проект содержит исходники лабораторных работ, презентации, ноутбуки и дополнительную литературу.

## Структура репозитория

| Путь | Содержимое |
|------|------------|
| `modules/` | Директория с модулями курса |
| `modules/cuda/` | CUDA-модули курса (00-intro, 01-memory-access, 02-vector-add, 03-matmul-naive, 04-matmul-tiled, 05-streams, 06-graph-api, 07-cublas, 08-profiling) |
| `modules/opencl/` | Примеры на OpenCL (00-intro, 02-saxpy, 03-matmul-naive, 04-matmul-tiled, 05-streams, 06-reduction-atomic, 07-clblast) |
| `docs/` | (опционально) сгенерированная документация Sphinx/Markdown |
| `ROADMAP.md` | Двоязычная дорожная карта всех модулей |
| `CMakeLists.txt` | Корневая конфигурация CMake для сборки C/CUDA примеров |

## Зависимости

| Компонент | Минимальная версия | Как установить (Ubuntu/Debian) |
|-----------|--------------------|--------------------------------|
| CMake     | 3.20 | `sudo apt install cmake` 
| GCC/Clang | 11   | `sudo apt install build-essential` или `clang` |
| CUDA Toolkit (для модулей CUDA) | 11.4 | `.run`-инсталлер с сайта NVIDIA или пакет `cuda-toolkit-X.Y` |
| OpenCL ICD loader              | 1.2 | `sudo apt install ocl-icd-opencl-dev` |
| GPU-драйвер с OpenCL 1.2+       | —   | NVIDIA ≥ 470, AMD ≥ AMDGPU-Pro 20.x |
| CLBlast (опц.)                 | 1.6 | `sudo apt install libclblast-dev` |

Docker-образ для быстрой проверки:

В репозитории есть `docker/Dockerfile`.

Собрать и запустить:
```bash
# в корне репозитория
docker build -t gpu-school/dev -f docker/Dockerfile .

# запустить контейнер с доступом к GPU (требуется nvidia-docker2)
docker run --gpus all -it --rm -v $(pwd):/workspace gpu-school/dev

# внутри контейнера:
cmake -S /workspace -B /workspace/build -DCMAKE_BUILD_TYPE=Release
cmake --build /workspace/build -j$(nproc)
ctest --test-dir /workspace/build --output-on-failure
```

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

4. Чтобы собрать и протестировать только один пример:

```bash
cmake --build $airbuild --target hello_opencl -j$(nproc)
./build/hello_opencl
```

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
cd modules/opencl/02-saxpy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/saxpy_basic_cl 1048576
```

То же справедливо для CUDA-модулей (`modules/cuda/...`).  
