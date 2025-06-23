# Модуль 8 — Profiling & Debugging

GPU-приложения редко ограничены лишь арифметикой: пропускная способность памяти, конфликты иерархии кэшей, задержки синхронизаций — всё это важно понимать и оптимизировать. NVIDIA предоставляет два ключевых инструмента:

| Инструмент | Цель | Формат отчёта |
|------------|------|---------------|
| **Nsight Compute (ncu)** | Профилировщик одного ядра: Occupancy, Throughput, латентность памяти | `.ncu-rep` + текст | 
| **Nsight Systems (nsys)** | Таймлайн всего приложения: CPU ↔ GPU ↔ memcpy, Streams, корреляция | `.qdrep` | 

## Скрипт `profile_matmul.sh`

Быстрый способ собрать метрики для оптимизированного `matmul_tiled`:

```bash
chmod +x modules/08-profiling/profile_matmul.sh
modules/08-profiling/profile_matmul.sh
```

Скрипт:
1. Проверяет, что `build/matmul_tiled` существует.  
2. Запускает `ncu --set full` и сохраняет отчёт `matmul_tiled.ncu-rep`.  
3. Запускает `nsys profile` и формирует `matmul_tiled.nsys-rep.qdrep`.

Откройте отчёты в GUI:
```bash
ncu-ui matmul_tiled.ncu-rep
nsys-ui matmul_tiled.nsys-rep.qdrep
```

## Quick-Win Метрики

*Nsight Compute*
- **DRAM Read/Write Throughput** — близко ли к теоретическому пику?  
- **Memory Transactions per Request** — >1 намекает на некоалесцированные обращения.  
- **Shared Store Conflicts** — банки shared memory.  

*Nsight Systems*
- Ищите пустые зоны: CPU idle, GPU idle, PCIe busy.  
- Проверьте, что ядра overlapped с H2D/D2H копиями (см. Module 5).

## Debugging

- `cuda-gdb` поддерживает пошаговый дебаг внутри ядра (breakpoint ↔ thread ↔ warp).  
- `cuda-memcheck` ловит out-of-bounds, инициализация, меж-поточные data race.  

### Пример:
```bash
cuda-gdb --args build/hello_gpu
(cuda-gdb) break hello_kernel
(cuda-gdb) run
(cuda-gdb) thread 5  // переключиться на поток #5 варпа
```

## Задания
1. Откройте `matmul_tiled.ncu-rep`, найдите метрики *achieved occupancy*, *ALU Utilization*. Сравните с `matmul_naive`.  
2. В `nsys-ui` измерьте долю времени, когда GPU неактивен (Idle).  
3. С помощью `cuda-gdb` установите breakpoint внутри `vecAdd` и распечатайте `idx`. 