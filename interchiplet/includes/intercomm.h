#include <cstdio>
#include <cstring>

#include <iostream>
#include <vector>
#include <map>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define PIPE_COMMON_DEBUG true
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
                if (PIPE_COMMON_DEBUG)
                {
                    std::cout << dst_ptr << "\t"
                              << (void*)(uint8_buf + dst_ptr)
                              << "\t" << iter_nbyte << std::endl;
                }
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

            if (PIPE_COMMON_DEBUG)
            {
                std::cout << "Read " << dst_ptr << " B from " << m_file_name << "." << std::endl;
            }
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
                if (PIPE_COMMON_DEBUG)
                {
                    std::cout << src_ptr << "\t"
                              << (void*)(uint8_buf + src_ptr)
                              << "\t" << iter_nbyte << std::endl;
                }
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

            if (PIPE_COMMON_DEBUG)
            {
                std::cout << "Write " << src_ptr << " B to " << m_file_name << "." << std::endl;
            }
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
                if (PIPE_COMMON_DEBUG)
                {
                    std::cerr << "[DEBUG] " << (void*)(mp_buf + m_size) << "\t"
                              << PIPE_COMMON_UNIT_CAPACITY - m_size << "\t"
                              << read_size << std::endl;
                }
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
}