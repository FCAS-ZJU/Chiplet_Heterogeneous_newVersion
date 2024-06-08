#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>

#include "apis_cu.h"

/**
 * 本示例程序为：通过4个GPU chiplet
 * 计算随机数矩阵A（400 * 100）与随机数矩阵B（100 * 400）相乘结果。
 * 由矩阵乘法原理可知，我们可将计算任务划分为4个100*100的矩阵相乘，并将结果相加。
 */

#define Row 100
#define Col 100

int main(int argc, char** argv)
{
	//读取本进程所代表的chiplet编号

	int idX = atoi(argv[1]);
	int idY = atoi(argv[2]);
	int64_t *d_dataA1, *d_dataA2, *d_dataA3, *d_dataB1, *d_dataB2, *d_dataB3;
	cudaMalloc((void**)&d_dataA1, sizeof(int64_t) *Row*Col);
	cudaMalloc((void**)&d_dataA2, sizeof(int64_t) *Row*Col);
	cudaMalloc((void**)&d_dataA3, sizeof(int64_t) *Row*Col);
	cudaMalloc((void**)&d_dataB1, sizeof(int64_t) *Row*Col);
	cudaMalloc((void**)&d_dataB2, sizeof(int64_t) *Row*Col);
	cudaMalloc((void**)&d_dataB3, sizeof(int64_t) *Row*Col);

	int res;
	receiveMessage <<<1,1>>> (idX,idY,0,0,d_dataA1,sizeof(int64_t) *Row*Col,&res);
	receiveMessage <<<1,1>>> (idX,idY,0,1,d_dataA2,sizeof(int64_t) *Row*Col,&res);
	receiveMessage <<<1,1>>> (idX,idY,1,0,d_dataA3,sizeof(int64_t) *Row*Col,&res);
	receiveMessage <<<1,1>>> (idX,idY,0,0,d_dataB1,sizeof(int64_t) *Row*Col,&res);
	receiveMessage <<<1,1>>> (idX,idY,0,1,d_dataB2,sizeof(int64_t) *Row*Col,&res);
	receiveMessage <<<1,1>>> (idX,idY,1,0,d_dataB3,sizeof(int64_t) *Row*Col,&res);

	cudaFree(d_dataA1);
	cudaFree(d_dataA2);
	cudaFree(d_dataA3);
	cudaFree(d_dataB1);
	cudaFree(d_dataB2);
	cudaFree(d_dataB3);
	return 0;
}
