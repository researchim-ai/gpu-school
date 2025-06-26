# Дорожная карта курса по программированию GPU / GPU Programming Course Roadmap

## Русская версия

### Структура и временная шкала
Каждый модуль рассчитан в среднем на 1-2 недели изучения с практикой.

| # | Модуль | Ключевые темы | Практические задания | Планируемые артефакты |
|---|--------|---------------|----------------------|-----------------------|
| 0 | Введение | История GPGPU, обзор экосистемы, установка инструментов (CUDA Toolkit, драйверы, Nsight) | Проверка окружения, «Hello GPU» | Setup-скрипты, README | 
| 1 | Архитектура GPU | SIMT-модель, блоки/варпы, иерархия памяти | Оптимизация доступа к глобальной памяти | Презентация, шпаргалка архитектур | 
| 2 | Основы CUDA C | Синтаксис `__global__`, конфигурация сеток/блоков, управление потоками | Векторное сложение, SAXPY | Сниппеты кода, unit-тесты | 
| 3 | Управление памятью | Хост vs Device, `cudaMalloc`, `cudaMemcpy`, Unified Memory | Матричное умножение (Naïve) | Бенчмарк, отчёт | 
| 4 | Оптимизация производительности | Коалесcинг, использование shared memory, register pressure | Tile-based матмул | Профиль Nsight, сравнительная таблица | 
| 5 | Потоки и асинхронность | CUDA Streams, Events, Overlap H2D/D2H | Pipeline обработка изображений | Demo-видео | 
| 6 | Продвинутые возможности CUDA | Dynamic Parallelism, Cooperative Groups, Graph API | BFS на графе | Статья блога | 
| 7 | Библиотеки CUDA | cuBLAS, cuFFT, Thrust, cuRAND | Использование cuBLAS для GEMM, FFT-спектрограммы | Jupyter-ноутбуки | 
| 8 | Профилирование и отладка | Nsight Compute, Nsight Systems, cuda-gdb | Оптимизация «узкого места» | Руководство «how-to» | 
| 9 | Межплатформенное программирование | OpenCL основы, отличие от CUDA | Реализация SAXPY на OpenCL | Сравнительный отчёт | 
|10 | Высокоуровневые обёртки | PyCUDA, CuPy, Numba | Нейросетевой inference на CuPy | Jupyter-ноутбуки | 
|11 | Мульти-GPU и распределённые вычисления | NCCL, CUDA-Aware MPI | Red-Black SOR на 2 GPU | Dockerfile, скрипты запуска | 
|12 | Tensor Cores и смешанная точность | WMMA API, матмул FP16 | Перенос матмул на Tensor Cores | Бенчмарк, отчёт | 
|13 | HIP и SYCL | Портирование CUDA-кода, DPC++ | Векторное сложение на HIP | Гайд по портированию | 
|14 | Реальный проект capstone | Выбор темы: трассировка лучей/симуляция Монте-Карло/модель обучения | Полный пайплайн от CPU до GPU | Репозиторий проекта, защита | 
|15 | Развёртывание и DevOps | Docker-образы с GPU, Kubernetes, облачные GPU | CI/CD для CUDA | GitHub Actions, манифесты | 
|16 | Будущее и ресурсы | Roadmap CUDA, Hopper, Grace, литература | – | Список ссылок |

> После каждого модуля — мини-квиз и домашнее задание. Каждые 4 модуля — контрольный проект.

### Методические материалы и формат
- Слайды (PDF + онлайн)
- Лабораторные в форме Jupyter Notebook + CMake-проекты

---

## English Version

### Structure & Timeline
Each module is designed for approximately 1–2 weeks of study with hands-on practice.

| # | Module | Key topics | Practical labs | Deliverables |
|---|--------|-----------|----------------|--------------|
| 0 | Introduction | History of GPGPU, ecosystem overview, toolchain installation (CUDA Toolkit, drivers, Nsight) | Environment check, "Hello GPU" | Setup scripts, README |
| 1 | GPU Architecture | SIMT model, blocks/warps, memory hierarchy | Global-memory access optimization | Slide deck, cheat-sheet |
| 2 | CUDA C Basics | `__global__` syntax, grid/block configuration, thread indexing | Vector addition, SAXPY | Code snippets, unit tests |
| 3 | Memory Management | Host vs Device, `cudaMalloc`, `cudaMemcpy`, Unified Memory | Naïve matrix multiplication | Benchmark report |
| 4 | Performance Optimization | Memory coalescing, shared memory, register pressure | Tiled GEMM | Nsight profile, comparison table |
| 5 | Streams & Asynchrony | CUDA Streams, Events, H2D/D2H overlap | Image processing pipeline | Demo video |
| 6 | Advanced CUDA Features | Dynamic Parallelism, Cooperative Groups, Graph API | Graph BFS | Blog article |
| 7 | CUDA Libraries | cuBLAS, cuFFT, Thrust, cuRAND | GEMM with cuBLAS, FFT spectrograms | Jupyter notebooks |
| 8 | Profiling & Debugging | Nsight Compute, Nsight Systems, cuda-gdb | Bottleneck optimization | "How-to" guide |
| 9 | Cross-platform GPU Programming | OpenCL fundamentals, differences from CUDA | SAXPY in OpenCL | Comparative report |
|10 | High-level Wrappers | PyCUDA, CuPy, Numba | Neural-network inference in CuPy | Jupyter notebooks |
|11 | Multi-GPU & Distributed | NCCL, CUDA-Aware MPI | Red-Black SOR on 2 GPUs | Dockerfile, run scripts |
|12 | Tensor Cores & Mixed Precision | WMMA API, FP16 GEMM | Porting GEMM to Tensor Cores | Benchmark report |
|13 | HIP & SYCL | Porting CUDA code, DPC++ basics | Vector add in HIP | Porting guide |
|14 | Capstone Project | Choose: ray tracing / Monte-Carlo simulation / ML model training | Full CPU-to-GPU pipeline | Project repo, presentation |
|15 | Deployment & DevOps | GPU Docker images, Kubernetes, cloud GPUs | CI/CD for CUDA | GitHub Actions, manifests |
|16 | Future & Resources | CUDA roadmap, Hopper, Grace, literature | – | Link collection |

> After each module — mini-quiz & homework. Every 4 modules — milestone project.

### Teaching Materials & Format
- Slide decks (PDF + online)
- Labs in Jupyter Notebooks plus CMake projects
- Video lectures (YouTube)
- Community channel in Discord/Telegram for Q&A

---

### Next Steps
1. Утвердить содержание дорожной карты и приоритеты модулей.
2. Настроить структуру репозитория: `modules/` с примерами, `docs/` для методических материалов.
3. Реализовать материалы Модуля 0 (скрипты установки, Hello GPU).
4. Поддерживать актуальную структуру репозитория: `modules/cuda/` (00-intro … 08-profiling) и `modules/opencl/` (00-intro, 02-saxpy, 03-matmul-naive, 04-matmul-tiled, 05-streams, 06-reduction-atomic, 07-clblast) + директория `docs/` для методических материалов.
5. Поддерживать актуальную структуру репозитория: `modules/cuda/` (00-intro … 08-profiling) и `modules/opencl/` (00-intro, 02-saxpy, 03-matmul-naive, 04-matmul-tiled, 05-streams, 06-reduction-atomic, 07-clblast) + директория `docs/` для методических материалов.

> Feel free to propose additions or adjustments! 