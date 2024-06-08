#include <fstream>
#include <iostream>
#include "apis_c.h"

#define Row 100
#define Col 100

int idX,idY;

int main(int argc, char** argv)
{
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
    int64_t *B = (int64_t *)malloc(sizeof(int64_t) * Row * Col);

    for (int i = 0; i < Row*Col; i++) {
        A[i] = rand() % 51;
        B[i] = rand() % 51;
    }


    InterChiplet::sendMessage(1, 1, idX, idY, A, 10000*sizeof(int64_t));

    InterChiplet::sendMessage(1, 1, idX, idY, B, 10000*sizeof(int64_t));
}
