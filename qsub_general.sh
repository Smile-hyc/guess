#!/bin/bash
#PBS -N guess_general
#PBS -l nodes=2:ppn=8
#PBS -j oe
#PBS -o general.out

cd $PBS_O_WORKDIR

make clean
make general

# 例子：2 个节点 × 8 个进程，每个 MPI 进程内部用 2 个 pthread 线程生成，哈希阶段使用 SIMD 尝试。
# 如果 SIMD4 在当前平台不可用，程序会自动退化为普通 MD5。
mpirun -np 16 ./correctness_guess_general \
  --train /guessdata/Rockyou-singleLined-full.txt \
  --test /guessdata/Rockyou-singleLined-full.txt \
  --test-limit 1000000 \
  --max-guesses 10000000 \
  --hash-threshold 1000000 \
  --batch-size 16 \
  --local-mode pthread \
  --local-threads 2 \
  --hash-mode simd \
  --pt-mpi 1
