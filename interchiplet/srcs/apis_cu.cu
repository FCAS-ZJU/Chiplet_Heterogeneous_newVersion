
#include "global_define.h"
#include "apis_cu.h"

__global__ void passMessage(
    int __dst_x, int __dst_y, int __src_x, int __srx_y, void* __addr, int __nbyte, int* __res)
{
    // Split address into 32 bits.
	uint32_t lo_data_ptr = ((uint64_t)__addr) & 0xFFFFFFFF;
	uint32_t hi_data_ptr = ((uint64_t)__addr >> 32) & 0xFFFFFFFF;

    // Initialize return value.
	int t_res;
    *__res = 0;

    // Set syscall parameters.
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__dst_x) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__dst_y) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__src_x) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__srx_y) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(lo_data_ptr) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(hi_data_ptr) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__nbyte) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
    // Send syscall command.
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(InterChiplet::SYSCALL_READ_FROM_GPU),
                                                "r"(InterChiplet::CUDA_SYSCALL_CMD));
	*__res += t_res;
}

__global__ void readMessage(
    int __dst_x, int __dst_y, int __src_x, int __srx_y, void* __addr, int __nbyte, int* __res)
{
    // Split address into 32 bits.
	uint32_t lo_data_ptr = ((uint64_t)__addr) & 0xFFFFFFFF;
	uint32_t hi_data_ptr = ((uint64_t)__addr >> 32) & 0xFFFFFFFF;

    // Initialize return value.
	int t_res;
    *__res = 0;

    // Set syscall parameters.
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__dst_x) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__dst_y) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__src_x) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__srx_y) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(lo_data_ptr) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(hi_data_ptr) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(__nbyte) , "r"(InterChiplet::CUDA_SYSCALL_ARG));
	*__res += t_res;
    // Send syscall command.
	asm("addc.u32 %0, %1, %2;" : "=r"(t_res) : "r"(InterChiplet::SYSCALL_SEND_TO_GPU),
                                                "r"(InterChiplet::CUDA_SYSCALL_CMD));
	*__res += t_res;
}
