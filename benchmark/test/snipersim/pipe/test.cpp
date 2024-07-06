#include <fstream>
#include <iostream>
#include <cstring>

#include "apis_c.h"

#define Row 100
#define Col 100

int idX, idY;

int main(int argc, char **argv) {
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    // Test Purpose:
    //  Communication (0,0) -> (0,1) -> (1,0) -> (1,1) -> (0,0)

    // Core (0,0)
    if (idX == 0 && idY == 0) {
        int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
        int64_t *B = (int64_t *)malloc(sizeof(int64_t) * Row * Col);

        // Initialization value.
        for (int i = 0; i < Row * Col; i++) {
            A[i] = rand() % 51;
        }

        // Send message to (0,1)
        InterChiplet::sendMessage(0, 1, idX, idY, A, Row * Col * sizeof(int64_t));
        // Receive message from (1,1)
        InterChiplet::receiveMessage(idX, idY, 1, 1, B, Row * Col * sizeof(int64_t));

        // Check result
        for (int i = 0; i < Row * Col; i++) {
            if (A[i] != B[i]) {
                std::cout << "Data check error!" << std::endl;
                return 1;
            }
        }
        std::cout << "Data check PASS!" << std::endl;

        return 0;
    }
    // Core (0,1)
    else if (idX == 0 && idY == 1) {
        int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
        memset(A, 0, Row * Col * sizeof(int64_t));
        // Receive message from (0,0)
        InterChiplet::receiveMessage(idX, idY, 0, 0, A, Row * Col * sizeof(int64_t));
        // Send message to (1,0)
        InterChiplet::sendMessage(1, 0, idX, idY, A, Row * Col * sizeof(int64_t));

        return 0;
    }
    // Core (1,0)
    else if (idX == 1 && idY == 0) {
        int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
        memset(A, 0, Row * Col * sizeof(int64_t));
        // Receive message from (0,1)
        InterChiplet::receiveMessage(idX, idY, 0, 1, A, Row * Col * sizeof(int64_t));
        // Send message to (1,1)
        InterChiplet::sendMessage(1, 1, idX, idY, A, Row * Col * sizeof(int64_t));

        return 0;
    }
    // Core (1,1)
    else if (idX == 1 && idY == 1) {
        int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);
        memset(A, 0, Row * Col * sizeof(int64_t));
        // Receive message from (0,0)
        InterChiplet::receiveMessage(idX, idY, 1, 0, A, Row * Col * sizeof(int64_t));
        // Send message to (1,0)
        InterChiplet::sendMessage(0, 0, idX, idY, A, Row * Col * sizeof(int64_t));

        return 0;
    }
}
