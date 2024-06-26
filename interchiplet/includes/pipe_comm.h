
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "global_define.h"
#include "sync_protocol.h"

// #define PIPE_COMMON_DEBUG
#define PIPE_COMMON_UNIT_CAPACITY 4096
#define NSINTERCHIPLET_CMD_HEAD "[INTERCMD]"

namespace InterChiplet {
/**
 * @defgroup pipe_comm
 * @brief Pipe communication interface.
 * @{
 */
/**
 * @brief Structure for Single Pipe communication.
 */
class PipeCommUnit {
   public:
    PipeCommUnit(const char *file_name, bool read) {
        m_file_name = std::string(file_name);
        m_file_id = open(file_name, read ? O_RDONLY : O_WRONLY);
        if (m_file_id == -1) {
            std::cerr << "Cannot open pipe file " << m_file_name << "." << std::endl;
            exit(1);
        } else {
            std::cout << "Open pipe file " << m_file_name << "." << std::endl;
        }

        if (read) {
            mp_buf = (uint8_t *)malloc(PIPE_COMMON_UNIT_CAPACITY);
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

    int read_data(void *dst_buf, int nbyte) {
        uint8_t *uint8_buf = (uint8_t *)dst_buf;
        int dst_ptr = 0;
        while (dst_ptr < nbyte) {
            int iter_nbyte = PIPE_COMMON_UNIT_CAPACITY;
            if ((nbyte - dst_ptr) < iter_nbyte) iter_nbyte = nbyte - dst_ptr;
            iter_nbyte = read_data_iter(uint8_buf + dst_ptr, iter_nbyte);
#ifdef PIPE_COMMON_DEBUG
            std::cout << "[DEBUG read_data] " << dst_ptr << "\t" << (void *)(uint8_buf + dst_ptr)
                      << "\t" << iter_nbyte << std::endl;
#endif
            if (iter_nbyte > 0) {
                dst_ptr += iter_nbyte;
            } else if (iter_nbyte == 0) {
                std::cerr << "Read from " << m_file_name << " abort due to EOF." << std::endl;
                break;
            } else {
                std::cerr << "Read from " << m_file_name << " abort due to Error." << std::endl;
                break;
            }
        }

        std::cout << "Read " << dst_ptr << " B from " << m_file_name << "." << std::endl;
        return dst_ptr;
    }

    int write_data(void *src_buf, int nbyte) {
        uint8_t *uint8_buf = (uint8_t *)src_buf;
        int src_ptr = 0;
        while (src_ptr < nbyte) {
            int iter_nbyte = PIPE_COMMON_UNIT_CAPACITY;
            if ((nbyte - src_ptr) < iter_nbyte) iter_nbyte = nbyte - src_ptr;
            iter_nbyte = write(m_file_id, uint8_buf + src_ptr, iter_nbyte);
#ifdef PIPE_COMMON_DEBUG
            std::cout << "[DEBUG write_data] " << src_ptr << "\t" << (void *)(uint8_buf + src_ptr)
                      << "\t" << iter_nbyte << std::endl;
#endif
            if (iter_nbyte > 0) {
                src_ptr += iter_nbyte;
            } else if (iter_nbyte == 0) {
                std::cerr << "Write to " << m_file_name << " abort due to EOF." << std::endl;
                break;
            } else {
                std::cerr << "Write to " << m_file_name << " abort due to Error." << std::endl;
                break;
            }
        }

#ifdef PIPE_COMMON_DEBUG
        char byte_hex_l, byte_hex_h;
        for (int i = 0; i < nbyte; i++) {
            uint8_t byte_value = uint8_buf[i];
            uint8_t byte_low = byte_value % 16;
            byte_hex_l = byte_low < 10 ? byte_low + '0' : byte_low + 'A';
            uint8_t byte_high = byte_value / 16;
            byte_hex_h = byte_high < 10 ? byte_high + '0' : byte_high + 'A';

            m_debug_fs << byte_hex_h << byte_hex_l << " ";
            if (m_debug_col == 15) {
                m_debug_fs << "\n";
                m_debug_col = 0;
            } else {
                m_debug_col += 1;
            }
        }
        m_debug_fs.flush();
#endif

        std::cout << "Write " << src_ptr << " B to " << m_file_name << "." << std::endl;
        return src_ptr;
    }

   private:
    int read_data_iter(uint8_t *dst_buf, int nbyte) {
        while ((m_size - m_read_ptr) < nbyte) {
            int left_size = m_size - m_read_ptr;
            if (left_size > 0 && m_read_ptr > 0) {
                memcpy(mp_buf, mp_buf + m_read_ptr, left_size);
                m_read_ptr = 0;
                m_size = left_size;
            } else if (left_size == 0 && m_read_ptr > 0) {
                m_read_ptr = 0;
                m_size = 0;
            }

            int read_size = read(m_file_id, mp_buf + m_size, PIPE_COMMON_UNIT_CAPACITY - m_size);
#ifdef PIPE_COMMON_DEBUG
            std::cerr << "[DEBUG read_data_iter] " << (void *)(mp_buf + m_size) << "\t"
                      << PIPE_COMMON_UNIT_CAPACITY - m_size << "\t" << read_size << std::endl;
#endif
            if (read_size <= 0) {
                return read_size;
            } else {
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

/**
 * @brief Pipe communication structure.
 */
class PipeComm {
   public:
    PipeComm() : m_named_fifo_map() {}

    int read_data(const char *file_name, void *buf, int nbyte) {
        std::string file_name_str = std::string(file_name);
        std::map<std::string, PipeCommUnit *>::iterator it = m_named_fifo_map.find(file_name_str);
        if (it == m_named_fifo_map.end()) {
            PipeCommUnit *recv_unit = new PipeCommUnit(file_name, true);
            m_named_fifo_map[file_name_str] = recv_unit;
            it = m_named_fifo_map.find(file_name_str);
        }

        return it->second->read_data(buf, nbyte);
    }

    int write_data(const char *file_name, void *buf, int nbyte) {
        std::string file_name_str = std::string(file_name);
        std::map<std::string, PipeCommUnit *>::iterator it = m_named_fifo_map.find(file_name_str);
        if (it == m_named_fifo_map.end()) {
            PipeCommUnit *recv_unit = new PipeCommUnit(file_name, false);
            m_named_fifo_map[file_name_str] = recv_unit;
            it = m_named_fifo_map.find(file_name_str);
        }

        return it->second->write_data(buf, nbyte);
    }

   private:
    std::map<std::string, PipeCommUnit *> m_named_fifo_map;
};
/**
 * @}
 */
}  // namespace InterChiplet
