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

        InterChiplet::barrier(255, idX, idY, 4);
    }
    std::cout << "Sum = " << sum << std::endl;
}
