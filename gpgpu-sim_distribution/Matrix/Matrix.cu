#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#define Row 100
#define Col 100

__global__ void matrix_mul_gpu(int *M, int* N, int* P, int width)
{
    int sumNum = threadIdx.x + threadIdx.y*10 ;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int sum = 0;
    for(int k=0;k<width;k++)
    {
        int a = M[j*width+k];
        int b = N[k*width+i];
        sum += a*b;
    }
    P[sumNum] = sum;
}
__global__ void passMessage(int dstX, int dstY, int srcX,int srcY,int* data, int dataSize){
    int para1 = srcX *10000000 + srcY*100000 + dstX*1000+dstY * 10 ;
    for(int i = 0; i<dataSize;i++){
        asm("addc.s32 %0, %1, %2;" : "=r"(data[i]) : "r"(para1) , "r"(data[i]));
    }
}
void readMessage( int srcX,int srcY,int dstX,int dstY,int*data,int dataSize){

    char * fileName = new char[100];
    sprintf(fileName,"./buffer%d_%d_%d_%d",srcX,srcY,dstX,dstY);
    std::ifstream file(fileName);
    int tmpdata = 0;
    for(int i = 0;i<dataSize;i++)
    {
        file>>tmpdata;
        data[i] += tmpdata;
    }
    file.close();
}

int srcX,srcY;
int main(int argc, char** argv)
{
    srcX=atoi(argv[1]);
    srcY=atoi(argv[2]);

    struct timeval start, end;
    gettimeofday( &start, NULL );

    int *A = (int *)malloc(sizeof(int) * Row * Col);
    int *B = (int *)malloc(sizeof(int) * Row * Col);
    int *C = (int *)malloc(sizeof(int) * Row * Col);
    //malloc device memory
    int *d_dataA, *d_dataB, *d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataC, sizeof(int) *Row*Col);
    //set value
    for (int i = 0; i < Row*Col; i++) {
        A[i] = rand() % 51;
        B[i] = rand() % 51;
    }

    cudaMemcpy(d_dataA, A, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, B, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(10,10);
    dim3 blockNumber(1);
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    cudaMemcpy(C, d_dataC, sizeof(int) * Row * Col, cudaMemcpyDeviceToHost);
    /*if(srcX != 0 || srcY != 0)
    {*/
        passMessage << <1,1>> > (0,0,srcX,srcY,d_dataC,100);
    //}
/*    else
    {
        char ready = '0';
        std::cin >>ready;
        readMessage(srcX,srcY,0,1,C,100);
        readMessage(srcX,srcY,1,0,C,100);
        readMessage(srcX,srcY,1,1,C,100);
    }*/

    //拷贝计算数据-一级数据指针

    //释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);

    return 0;
}



