#!/usr/bin/env bash
# Профилирование примера matmul_tiled с помощью Nsight Compute и Nsight Systems.
# Скрипт ожидает, что бинарник уже собран (см. корневой README).

set -e

BIN=./build/matmul_tiled
N=1024              # размер матрицы (можно изменить)

if [ ! -f "$BIN" ]; then
  echo "Бинарник $BIN не найден. Сначала соберите проект (cmake --build build)." >&2
  exit 1
fi

# ---------- Nsight Compute (детальный GPU-анализ) ----------
# Генерируем отчёт в формате .ncu-rep и краткий txt-summary.

echo "[1/2] Nsight Compute..."
OUTPUT_NCU=matmul_tiled.ncu-rep
ncu --set full -o matmul_tiled --target-processes all -- "$BIN" $N | tee matmul_tiled.ncu.txt

echo "Отчёт Nsight Compute сохранён в $OUTPUT_NCU (и matmul_tiled.ncu.txt)"

# ---------- Nsight Systems (таймлайн CPU↔GPU) ----------

echo "[2/2] Nsight Systems..."
OUTPUT_NSYS=matmul_tiled.nsys-rep
nsys profile --output="$OUTPUT_NSYS" --sample none --trace="cuda,osrt" -- "$BIN" $N

echo "Отчёт Nsight Systems сохранён в $OUTPUT_NSYS.qdrep"

echo "Готово. Откройте отчёты GUI-приложениями: ncu-ui / nsys-ui." 