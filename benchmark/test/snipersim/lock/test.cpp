#include <cstdint>
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

    int delay_count[3] = {0, 0, 0};
    if (idX == 0 && idY == 0) {
        delay_count[0] = 5000;
        delay_count[1] = 4000;
        delay_count[2] = 3000;
    } else if (idX == 0 && idY == 1) {
        delay_count[0] = 1000;
        delay_count[1] = 3000;
        delay_count[2] = 5000;
    } else if (idX == 1 && idY == 0) {
        delay_count[0] = 2000;
        delay_count[1] = 1000;
        delay_count[2] = 0000;
    } else if (idX == 1 && idY == 1) {
        delay_count[0] = 2000;
        delay_count[1] = 4000;
        delay_count[2] = 6000;
    }

    int64_t sum = 0;
    for (int i = 0; i < 3; i++) {
        // Create time gap between threads.
        for (int j = 0; j < delay_count[i]; j++) {
            sum += rand() % 10;
        }

        InterChiplet::lock(255, idX, idY);

        for (int j = 0; j < delay_count[i]; j++) {
            sum += rand() % 10;
        }

        InterChiplet::unlock(255, idX, idY);
    }

    std::cout << "Sum = " << sum << std::endl;

    return 0;
}
