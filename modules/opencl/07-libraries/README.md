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

Пример вывода:
```
CLBlast SGEMM PASSED | N=512, time=4.12 ms
```
Сравните с результатами модуля 04-performance (tiled MatMul) для оценки
ускорения готовой библиотеки. 