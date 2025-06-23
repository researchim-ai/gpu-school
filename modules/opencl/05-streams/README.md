# OpenCL 05-streams — Overlap Copy & Compute

`stream_overlap_cl.cpp` показывает, как перекрывать обмен данными и вычисления
с помощью нескольких очередей команд (аналог CUDA Streams).

Алгоритм:
1. Большие массивы `X`, `Y` (по умолч. 64 MiB) делятся на `CHUNKS=4` чанка.
2. Для каждого чанка создаётся отдельная очередь `cl_command_queue`.
3. В очереди выполняются по порядку:
   • `clEnqueueWriteBuffer` X, Y  (H→D)
   • `clEnqueueNDRangeKernel`     (SAXPY)
   • `clEnqueueReadBuffer` Y      (D→H)
4. После рассылки команд вызывается `clFinish` для всех очередей.

При достаточной полосе пропускания PCIe и асинхронности драйвера мы получаем
перекрытие, благодаря чему общее время < (H→D + kernel + D→H).

Запуск:
```bash
cmake --build build --target stream_overlap_cl
./stream_overlap_cl 32   # 32 MiB данных
```

### Локальная сборка в каталоге модуля

```bash
cd modules/opencl/05-streams
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/stream_overlap_cl 32
``` 