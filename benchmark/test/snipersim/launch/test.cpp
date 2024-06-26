#include <fstream>
#include <iostream>

#include "apis_c.h"
#include "unistd.h"

#define Row 100
#define Col 100

int idX, idY;

int main(int argc, char **argv) {
    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    // Core (0,0), wait launch
    if (idX == 0 && idY == 0) {
        for (int r = 0; r < 6; r ++) {
            int64_t srcX = -1, srcY = -1;
            InterChiplet::waitLaunch(0, 0, &srcX, &srcY);

            int64_t *A = (int64_t *)malloc(sizeof(int64_t) * Row * Col);

            int64_t sum = 0;
            for (int i = 0; i < Row * Col; i++) {
                sum = sum + A[i];
            }

            InterChiplet::sendMessage(srcX, srcY, 0, 0, &sum, sizeof(int64_t));
        }
    }
    // Core (0,1),(1,0),(1,1), launch
     else {
        int delay_count[2] = {0, 0};

        if (idX == 0 && idY == 0) {
            delay_count[0] = 5000;
            delay_count[1] = 4000;
        } else if (idX == 0 && idY == 1) {
            delay_count[0] = 1000;
            delay_count[1] = 3000;
        } else if (idX == 1 && idY == 0) {
            delay_count[0] = 2000;
            delay_count[1] = 1000;
        } else if (idX == 1 && idY == 1) {
            delay_count[0] = 2000;
            delay_count[1] = 4000;
        }

        int64_t sum = 0;
        for (int r = 0; r < 2; r ++) {
            // Create time gap between threads.
            for (int j = 0; j < delay_count[r]; j++) {
                sum += rand() % 10;
            }

            InterChiplet::launch(0, 0, idX, idY);

            // Read result from Core (0,0)
            int64_t result;
            InterChiplet::receiveMessage(idX, idY, 0, 0, &result, sizeof(int64_t));

            sum = sum + result;
        }
        std::cout << "Sum = " << sum << std::endl;
    }
}
