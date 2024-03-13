#include <fstream>
#include <iostream>
#include "interchiplet_app.h"
#define Row 100
#define Col 100

int srcX,srcY;

using namespace std;

using namespace nsInterchiplet;

int main(int argc, char** argv)
{
    srcX=atoi(argv[1]);
    srcY=atoi(argv[2]);

    int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *B = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *C1 = (int64_t *)malloc(sizeof(int64_t) * Col);
    int64_t *C2 = (int64_t *)malloc(sizeof(int64_t) * Col);
    int64_t *C3 = (int64_t *)malloc(sizeof(int64_t) * Col);

    for (int i = 0; i < Row*Col; i++) {
        A[i] = rand() % 51;
        B[i] = rand() % 51;
    }

    std::cout<<"aaa"<<endl;

    sendGpuMessage(0,1,srcX,srcY,A,10000);
    sendGpuMessage(1,0,srcX,srcY,A,10000);
    sendGpuMessage(1,1,srcX,srcY,A,10000);

    sendGpuMessage(0,1,srcX,srcY,B,10000);
    sendGpuMessage(1,0,srcX,srcY,B,10000);
    sendGpuMessage(1,1,srcX,srcY,B,10000);

    readGpuMessage(0,1,srcX,srcY,C1,100);
    readGpuMessage(1,0,srcX,srcY,C2,100);
    readGpuMessage(1,1,srcX,srcY,C3,100);

    for(int i=0;i<100;i++)
    {
        C1[i] += C2[i];
        C1[i] += C3[i];
    }
}
