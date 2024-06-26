#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <string>

#include "apis_cu.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
 * 本示例程序为：通过4个GPU chiplet
 * 计算随机数矩阵A（400 * 100）与随机数矩阵B（100 * 400）相乘结果。
 * 由矩阵乘法原理可知，我们可将计算任务划分为4个100*100的矩阵相乘，并将结果相加。
 */

#define Row 100
#define Col 100

/**
 * 矩阵乘法的核心函数，由每个线程都会运行一次本函数，
 * 根据线程编号不同计算出位于结果矩阵不同位置的数据。
 */

__global__ void matrix_mul_gpu(int64_t* M, int64_t* N, int64_t* P, int width) {
    int sumNum = threadIdx.x + threadIdx.y * 10;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int64_t sum = 0;
    for (int k = 0; k < width; k++) {
        int64_t a = M[j * width + k];
        int64_t b = N[k * width + i];
        sum += a * b;
    }
    P[sumNum] = sum;
}

int main(int argc, char** argv) {
    // 读取本进程所代表的chiplet编号

    int idX = atoi(argv[1]);
    int idY = atoi(argv[2]);
    int64_t *d_dataA, *d_dataB, *d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(int64_t) * Row * Col);
    cudaMalloc((void**)&d_dataB, sizeof(int64_t) * Row * Col);
    cudaMalloc((void**)&d_dataC, sizeof(int64_t) * Col);

    receiveMessage(idX, idY, 0, 0, d_dataA, sizeof(int64_t) * Row * Col);
    receiveMessage(idX, idY, 0, 0, d_dataB, sizeof(int64_t) * Row * Col);

    // calculate
    dim3 threadPerBlock(10, 10);
    dim3 blockNumber(1);
    matrix_mul_gpu<<<blockNumber, threadPerBlock>>>(d_dataA, d_dataB, d_dataC, Col);

    sendMessage(0, 0, idX, idY, d_dataC, 100 * sizeof(int64_t));
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);
    return 0;
}
