# OpenCL 06-advanced — Reduction (Atomics + Fallback)

`reduction_atomic_cl.cpp` демонстрирует две стратегии суммирования большого
массива:
1. **Atomic** — однопроходовый, использует `atomic_fetch_add_explicit` для
   записи частичных результатов в глобальной памяти. Требует поддержки
   расширения `cl_khr_global_float_atomics` или OpenCL 2.0 (как core).
2. **Work-group + Host** (fallback) — если атомики float недоступны, ядро
   `reduce_wg` выполняет редукцию внутри work-group, записывает частичные суммы
    в массив `partial`, а финальное сложение выполняется на CPU.

Программа автоматически пытается скомпилировать атомарную версию.
При неуспехе выводится лог компиляции и используется запасной путь.

Вывод указывает, какая стратегия сработала:
```
OpenCL reduction PASSED | N=16777216, sum=16777216.0, time=4.7 ms (wg + host)
```

Сборка/запуск:
```bash
cmake --build build --target reduction_atomic_cl
./reduction_atomic_cl 1048576   # 1M элементов
``` 