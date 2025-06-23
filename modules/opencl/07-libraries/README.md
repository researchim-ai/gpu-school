# OpenCL 07-libraries — SGEMM c помощью CLBlast

Этот модуль показывает, как подключить внешнюю библиотеку BLAS на OpenCL —
[CLBlast](https://github.com/CNugteren/CLBlast) — и вызвать функцию SGEMM.

Требования: установлен пакет `CLBlast` (Linux: `libclblast-dev`). Если CMake
не найдёт библиотеку, модуль будет пропущен с предупреждением.

Команда сборки и теста:
```bash
cmake --build build --target gemm_clblast
ctest -R gemm_clblast
```

### Локальная сборка в каталоге модуля

```bash
cd modules/opencl/07-libraries
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCLBlast_DIR=/path/if/needed
cmake --build build -j$(nproc)
./build/gemm_clblast 512
```

Учтите: цель появится только если `find_package(CLBlast)` успешно нашёл библиотеку.

Пример вывода:
```
CLBlast SGEMM PASSED | N=512, time=4.12 ms
```
Сравните с результатами модуля 04-performance (tiled MatMul) для оценки
ускорения готовой библиотеки. 