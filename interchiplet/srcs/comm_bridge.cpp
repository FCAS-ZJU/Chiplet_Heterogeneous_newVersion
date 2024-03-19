
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Return fifo name
std::string fifo_name(int id, bool rd)
{
    std::stringstream ss;
    if (rd == 1)
    {
        ss << "buffer_" << id << "_r";
    }
    else
    {
        ss << "buffer_" << id << "_w";
    }
    return ss.str();
}

// Create FIFO with specified name.
int create_fifo(std::string fifo_name)
{
    if (access(fifo_name.c_str(), F_OK) == -1)
    {
        // Report error if FIFO file does not exist and mkfifo error.
        if ((fifo_name.c_str(), 0664) == -1)
        {
            std::cerr << "Cannot create FIFO file " << fifo_name << "." << std::endl;
            return -1;
        }
        // Report success.
        else
        {
            std::cout << "Create FIFO file " << fifo_name << "." << std::endl;
            return 0;
        }
    }
    // Reuse exist FIFO and reports.
    else
    {
        std::cout << "Reuse exist FIFO file " << fifo_name << "." << std::endl;
        return 0;
    }
}

int main(int argc, char** argv)
{
    // Create FIFO for each PE.
    std::vector<int> fifo_rd_fd_list;
    std::vector<int> fifo_wr_fd_list;

    for (int i = 1; i < argc; i ++)
    {
        int node_id = atoi(argv[i]);

        if (create_fifo(fifo_name(node_id, true)) == -1)
        {
            exit(1);
        }
        if (create_fifo(fifo_name(node_id, false)) == -1)
        {
            exit(1);
        }
    }

    // Open FIFO and add to select.
    for (int i = 0; i < argc, i ++)
    {
        
    }
}