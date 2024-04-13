
#ifndef INTERCOMM_H
#define INTERCOMM_H

#include <cstdio>
#include <cstring>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

//#define PIPE_COMMON_DEBUG
#define PIPE_COMMON_UNIT_CAPACITY 4096

namespace nsInterchiplet
{
    class PipeCommUnit
    {
    public:
        PipeCommUnit(const char* file_name, bool read)
        {
            m_file_name = std::string(file_name);
            m_file_id = open(file_name, read ? O_RDONLY : O_WRONLY);
            if (m_file_id == -1)
            {
                std::cerr << "Cannot open pipe file " << m_file_name << "." << std::endl;
                exit(1);
            }
            else
            {
                std::cout << "Open pipe file " << m_file_name << "." << std::endl;
            }

            if (read)
            {
                mp_buf = (uint8_t*)malloc(PIPE_COMMON_UNIT_CAPACITY);
            }
            m_size = 0;
            m_read_ptr = 0;

#ifdef PIPE_COMMON_DEBUG
            std::string debug_file_name = m_file_name + ".hex";
            m_debug_fs.open(debug_file_name.c_str(), std::ios::out);
            m_debug_col = 0;
#endif
        }

        bool valid() const { return m_file_id >= 0; }

        int read_data(void *dst_buf, int nbyte)
        {
            uint8_t* uint8_buf = (uint8_t*)dst_buf;
            int dst_ptr = 0;
            while (dst_ptr < nbyte)
            {
                int iter_nbyte = PIPE_COMMON_UNIT_CAPACITY;
                if ((nbyte - dst_ptr) < iter_nbyte) iter_nbyte = nbyte - dst_ptr;
                iter_nbyte = read_data_iter(uint8_buf + dst_ptr, iter_nbyte);
#ifdef PIPE_COMMON_DEBUG
                std::cout << "[DEBUG read_data] " << dst_ptr << "\t"
                          << (void*)(uint8_buf + dst_ptr)
                          << "\t" << iter_nbyte << std::endl;
#endif
                if (iter_nbyte > 0)
                {
                    dst_ptr += iter_nbyte;
                }
                else if (iter_nbyte == 0)
                {
                    std::cerr << "Read from " << m_file_name << " abort due to EOF." << std::endl;
                    break;
                }
                else
                {
                    std::cerr << "Read from " << m_file_name << " abort due to Error." << std::endl;
                    break;
                }
            }

            std::cout << "Read " << dst_ptr << " B from " << m_file_name << "." << std::endl;
            return dst_ptr;
        }

        int write_data(void *src_buf, int nbyte)
        {
            uint8_t* uint8_buf = (uint8_t*)src_buf;
            int src_ptr = 0;
            while (src_ptr < nbyte)
            {
                int iter_nbyte = PIPE_COMMON_UNIT_CAPACITY;
                if ((nbyte - src_ptr) < iter_nbyte) iter_nbyte = nbyte - src_ptr;
                iter_nbyte = write(m_file_id, uint8_buf + src_ptr, iter_nbyte);
#ifdef PIPE_COMMON_DEBUG
                std::cout << "[DEBUG write_data] " << src_ptr << "\t"
                          << (void*)(uint8_buf + src_ptr)
                          << "\t" << iter_nbyte << std::endl;
#endif
                if (iter_nbyte > 0)
                {
                    src_ptr += iter_nbyte;
                }
                else if (iter_nbyte == 0)
                {
                    std::cerr << "Write to " << m_file_name << " abort due to EOF." << std::endl;
                    break;
                }
                else
                {
                    std::cerr << "Write to " << m_file_name << " abort due to Error." << std::endl;
                    break;
                }
            }

#ifdef PIPE_COMMON_DEBUG
            char byte_hex_l, byte_hex_h;
            for (int i = 0; i < nbyte; i ++)
            {
                uint8_t byte_value = uint8_buf[i];
                uint8_t byte_low = byte_value % 16;
                byte_hex_l = byte_low < 10 ? byte_low + '0' : byte_low + 'A';
                uint8_t byte_high = byte_value / 16;
                byte_hex_h = byte_high < 10 ? byte_high + '0' : byte_high + 'A';

                m_debug_fs << byte_hex_h << byte_hex_l << " ";
                if (m_debug_col == 15)
                {
                    m_debug_fs << "\n";
                    m_debug_col = 0;
                }
                else
                {
                    m_debug_col += 1;
                }
            }
            m_debug_fs.flush();
#endif

            std::cout << "Write " << src_ptr << " B to " << m_file_name << "." << std::endl;
            return src_ptr;
        }

    private:
        int read_data_iter(uint8_t *dst_buf, int nbyte)
        {
            while ((m_size - m_read_ptr) < nbyte)
            {
                int left_size = m_size - m_read_ptr;
                if (left_size > 0 && m_read_ptr > 0)
                {
                    memcpy(mp_buf, mp_buf + m_read_ptr, left_size);
                    m_read_ptr = 0;
                    m_size = left_size;
                }
                else if (left_size == 0 && m_read_ptr > 0)
                {
                    m_read_ptr = 0;
                    m_size = 0;
                }

                int read_size = read(
                    m_file_id, mp_buf + m_size, PIPE_COMMON_UNIT_CAPACITY - m_size);
#ifdef PIPE_COMMON_DEBUG
                std::cerr << "[DEBUG read_data_iter] " << (void*)(mp_buf + m_size) << "\t"
                          << PIPE_COMMON_UNIT_CAPACITY - m_size << "\t"
                          << read_size << std::endl;
#endif
                if (read_size <= 0)
                {
                    return read_size;
                }
                else
                {
                    m_size += read_size;
                }
            }

            memcpy(dst_buf, mp_buf + m_read_ptr, nbyte);
            m_read_ptr += nbyte;
            return nbyte;
        }

    private:
        std::string m_file_name;
        int m_file_id;
        uint8_t *mp_buf;
        int m_read_ptr;
        int m_size;
#ifdef PIPE_COMMON_DEBUG
        std::ofstream m_debug_fs;
        int m_debug_col;
#endif
    };

    class PipeComm
    {
    public:
        PipeComm()
            : m_named_fifo_map()
        {
        }

        int read_data(const char* file_name, void *buf, int nbyte)
        {
            std::string file_name_str = std::string(file_name);
            std::map<std::string, PipeCommUnit *>::iterator it =
                m_named_fifo_map.find(file_name_str);
            if (it == m_named_fifo_map.end())
            {
                PipeCommUnit *recv_unit = new PipeCommUnit(file_name, true);
                m_named_fifo_map[file_name_str] = recv_unit;
                it = m_named_fifo_map.find(file_name_str);
            }

            return it->second->read_data(buf, nbyte);
        }

        int write_data(const char* file_name, void *buf, int nbyte)
        {
            std::string file_name_str = std::string(file_name);
            std::map<std::string, PipeCommUnit *>::iterator it =
                m_named_fifo_map.find(file_name_str);
            if (it == m_named_fifo_map.end())
            {
                PipeCommUnit *recv_unit = new PipeCommUnit(file_name, false);
                m_named_fifo_map[file_name_str] = recv_unit;
                it = m_named_fifo_map.find(file_name_str);
            }

            return it->second->write_data(buf, nbyte);
        }

    private:
        std::map<std::string, PipeCommUnit *> m_named_fifo_map;
    };

    enum SyncCommType
    {
        SC_CYCLE,
        SC_PIPE,
        SC_READ,
        SC_WRITE,
        SC_BARRIER,
        SC_LOCK,
        SC_UNLOCK,
        SC_SYNC,
        SC_RESULT,
    };

    class SyncCommand
    {
    public:
        SyncCommType m_type;
        long long int m_cycle;
        int m_src_x;
        int m_src_y;
        int m_dst_x;
        int m_dst_y;
        int m_nbytes;

        int m_stdin_fd;
    };

    std::string m_cmd_head = "[INTERCMD]";

    void sendCycleCmd(long long int cycle)
    {
        std::cout << m_cmd_head << " CYCLE " << cycle << std::endl;
    }

    void sendPipeCmd(int src_x, int src_y, int dst_x, int dst_y)
    {
        std::cout << m_cmd_head << " PIPE " << 0 << " "
            << src_x << " " << src_y << " " << dst_x << " " << dst_y
            << std::endl;
    }

    void sendReadCmd(long long int cycle, int src_x, int src_y, int dst_x, int dst_y, int nbyte)
    {
        std::cout << m_cmd_head << " READ " << cycle << " "
            << src_x << " " << src_y << " " << dst_x << " " << dst_y << " " << nbyte
            << std::endl;
    }

    void sendWriteCmd(long long int cycle, int src_x, int src_y, int dst_x, int dst_y, int nbyte)
    {
        std::cout << m_cmd_head << " WRITE " << cycle << " "
            << src_x << " " << src_y << " " << dst_x << " " << dst_y << " " << nbyte
            << std::endl;
    }

    void sendSyncCmd(long long int cycle)
    {
        std::cout << m_cmd_head << " SYNC " << cycle << std::endl;
    }

    SyncCommand parseCmd(const std::string& __message)
    {
        std::string message;
        if (__message.substr(0, 10) == m_cmd_head)
        {
            message = __message.substr(11);
        }
        else
        {
            message = __message;
        }
        std::stringstream ss(message);
        std::string command;
        long long int cycle;
        ss >> command >> cycle;

        SyncCommand cmd;
        cmd.m_cycle = cycle;
        cmd.m_type = command == "CYCLE" ? SC_CYCLE :
                     command == "PIPE" ? SC_PIPE :
                     command == "READ" ? SC_READ :
                     command == "WRITE" ? SC_WRITE :
                     command == "BARRIER" ? SC_BARRIER :
                     command == "LOCK" ? SC_LOCK :
                     command == "UNLOCK" ? SC_UNLOCK :
                     command == "SYNC" ? SC_SYNC : SC_CYCLE;

        if (cmd.m_type == SC_PIPE || cmd.m_type == SC_READ || cmd.m_type == SC_WRITE)
        {
            ss >> cmd.m_src_x >> cmd.m_src_y >> cmd.m_dst_x >> cmd.m_dst_y >> cmd.m_nbytes;
        }

        return cmd;
    }

    long long int cycleSync(long long int cycle)
    {
        sendCycleCmd(cycle);

        char* message = new char[1024];
        while(read(STDIN_FILENO, message, 1024) == 0);
        for (std::size_t i = 0; i < strlen(message); i ++) if (message[i] == '\n') message[i + 1] = 0;
        SyncCommand resp_cmd = parseCmd(std::string(message));
        std::cout << message;
        delete message;

        if (resp_cmd.m_type == SC_SYNC)
        {
            return resp_cmd.m_cycle;
        }
        else
        {
            return -1;
        }
    }

    long long int pipeSync(int src_x, int src_y, int dst_x, int dst_y)
    {
        sendPipeCmd(src_x, src_y, dst_x, dst_y);

        char* message = new char[1024];
        while(read(STDIN_FILENO, message, 1024) == 0);
        for (std::size_t i = 0; i < strlen(message); i ++) if (message[i] == '\n') message[i + 1] = 0;
        SyncCommand resp_cmd = parseCmd(std::string(message));
        std::cout << message;
        delete message;

        if (resp_cmd.m_type == SC_SYNC)
        {
            return resp_cmd.m_cycle;
        }
        else
        {
            return -1;
        }
    }

    long long int readSync(long long int cycle, int src_x, int src_y, int dst_x, int dst_y, int nbyte)
    {
        sendReadCmd(cycle, src_x, src_y, dst_x, dst_y, nbyte);

        char* message = new char[1024];
        while(read(STDIN_FILENO, message, 1024) == 0);
        for (std::size_t i = 0; i < strlen(message); i ++) if (message[i] == '\n') message[i + 1] = 0;
        SyncCommand resp_cmd = parseCmd(std::string(message));
        std::cout << message;
        delete message;

        if (resp_cmd.m_type == SC_SYNC)
        {
            return resp_cmd.m_cycle;
        }
        else
        {
            return -1;
        }
    }

    long long int writeSync(long long int cycle, int src_x, int src_y, int dst_x, int dst_y, int nbyte)
    {
        sendWriteCmd(cycle, src_x, src_y, dst_x, dst_y, nbyte);

        char* message = new char[1024];
        while(read(STDIN_FILENO, message, 1024) == 0);
        for (std::size_t i = 0; i < strlen(message); i ++) if (message[i] == '\n') message[i + 1] = 0;
        SyncCommand resp_cmd = parseCmd(std::string(message));
        std::cout << message;
        delete message;

        if (resp_cmd.m_type == SC_SYNC)
        {
            return resp_cmd.m_cycle;
        }
        else
        {
            return -1;
        }
    }

    char* pipeName(int __src_x, int __src_y, int __dst_x, int __dst_y)
    {
        char * fileName = new char[100];
        sprintf(fileName, "./buffer%d_%d_%d_%d", __src_x, __src_y, __dst_x, __dst_y);
        return fileName;
    }

    // Return fifo name
    std::string pipeNameString(int __src_x, int __src_y, int __dst_x, int __dst_y)
    {
        std::stringstream ss;
        ss << "./buffer" << __src_x << "_" << __src_y << "_" << __dst_x << "_" << __dst_y;
        return ss.str();
    }
}

#endif
