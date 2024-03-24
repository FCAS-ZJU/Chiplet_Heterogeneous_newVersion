#include <fstream>
#include <iostream>
#include "interchiplet_app.h"
#define Row 100
#define Col 100

int idX,idY;

using namespace std;

using namespace nsInterchiplet;

int main(int argc, char** argv)
{
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *B = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *C1 = (int64_t *)malloc(sizeof(int64_t) * Col);
    int64_t *C2 = (int64_t *)malloc(sizeof(int64_t) * Col);
    int64_t *C3 = (int64_t *)malloc(sizeof(int64_t) * Col);

    for (int i = 0; i < Row*Col; i++) {
        A[i] = rand() % 51;
        B[i] = rand() % 51;
    }

    sendGpuMessage(0,1,idX,idY,A,10000);
    sendGpuMessage(1,0,idX,idY,A,10000);
    sendGpuMessage(1,1,idX,idY,A,10000);

    sendGpuMessage(0,1,idX,idY,B,10000);
    sendGpuMessage(1,0,idX,idY,B,10000);
    sendGpuMessage(1,1,idX,idY,B,10000);

    readGpuMessage(idX,idY,0,1,C1,100);
    readGpuMessage(idX,idY,1,0,C2,100);
    readGpuMessage(idX,idY,1,1,C3,100);

    for(int i=0;i<100;i++)
    {
        C1[i] += C2[i];
        C1[i] += C3[i];
    }
}
