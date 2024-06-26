
# APIs

You can learn the following topics from this page:

- Create or modify the source code of a benchmark to execute on the LegoSim.

## Create Source Codes of Benchmarks

Considering the complexity of the software stack of heterogeneous systems, it should not be expected that there will be a standard software stack available for every experimental platform on LegoSim. Hence, task partitioning and task management should be done manually. Each simulator should have one individual executable file. For example, different executable files should be provided to SniperSim and GPGPUSim. Moreover, imported simulators can share the same executable file if they perform the same task on various datasets, like Same-Task-Multiple-Data (STMD).

Take the *matmul* benchmark as an example. In *matmul* benchmark, the CPU generates source matrixes and sends data to GPGPUs, while three GPGPUs perform the matrix multiplex operation. Each GPGPU handles a part of the data. The *matmul* benchmark provides two executable files for SniperSim and GPGPUSim. Simulation processes of GPGPUSim share the same executable file.

Several APIs should be added to benchmarks to communicate and synchronize processes. In reality, similar APIs are also injected by one complete software stack (like CUDA).

## API Lists

> TODO: non-blocking APIs.

### Communication

The source sends data to the destination by `sendMessage` while the destination receives data from the source by `receiveMessage`. The source and destination addresses are the same among `sendMessage` and `receiveMessage`.

**APIs for CPU**

```cpp
syscall_return_t sendMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y, void* __addr, int64_t __nbyte);
syscall_return_t receiveMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y, void* __addr, int64_t __nbyte);
```

**APIs for CUDA**

```cpp
cudaError_t sendMessage(int __dst_x, int __dst_y, int __src_x, int __srx_y, void* __addr, int __nbyte);
cudaError_t receiveMessage(int __dst_x, int __dst_y, int __src_x, int __srx_y, void* __addr, int __nbyte);
```

**Arguments**

- `__dst_x` and `__dst_y` specify the destination address.
- `__src_x` and `__src_y` specify the source address.
- `__addr` specifies the pointer to the data array.
- `__nbyte` determines the number of bytes in the data array.

**Return value**

- APIs for CPU return the result of the operation.
    - 0 means the transmission operation succeeds.
    - 1 means the transmission operation fails.
- APIs for GPU return one value of `cudaError_t`.

### Lock and unlock

`lock` and `unlock` are used to manage the critical region. `lock` blocks the process until the mutex is locked by this request. `unlock` releases the mutex.

**APIs for CPU**

```cpp
syscall_return_t lock(int64_t __uid, int64_t __src_x, int64_t __src_y);
syscall_return_t unlock(int64_t __uid, int64_t __src_x, int64_t __src_y);
```

**APIs for CUDA**

```cpp
cudaError_t lock(int __uid, int __src_x, int __src_y);
cudaError_t unlock(int __uid, int __src_x, int __src_y);
```

**Arguments**

- `__uid` specifies one unique ID of the mutex. `__uid` should not be the same as any one address in the system.
- `__src_x` and `__src_y` specify the source address.

**Return value**

- APIs for CPU return the result of the operation.
    - 0 means the transmission operation succeeds.
    - 1 means the transmission operation fails.
- APIs for GPU return one value of `cudaError_t`.

## Barrier

When calling `barrier`, processes are blocked until a certain number of processes enter the barrier.

**APIs for CPU**

```cpp
syscall_return_t barrier(int64_t __uid, int64_t __src_x, int64_t __src_y, int64_t __count = 0);
```

**APIs for CUDA**

```cpp
cudaError_t barrier(int __uid, int __src_x, int __src_y, int __count = 0);
```

**Arguments**

- `__uid` specifies one unique ID of the barrier. `__uid` should not be the same as any one address in the system.
- `__src_x` and `__src_y` specify the source address.
- `__count` specifies the number of threads for the barrier. If `__count` is greater than 0, the number of processes is overridden when the barrier overflows.

**Return value**

- APIs for CPU return the result of the operation.
    - 0 means the transmission operation succeeds.
    - 1 means the transmission operation fails.
- APIs for GPU return one value of `cudaError_t`.

### Launch

When several masters share a computation resource, masters should send launch requests to the shared computation resource to start tasks.

The program on the master calls `launch` to launch one task on the shared computation resource. `launch` blocks the program on the master until the shared computation resource has been triggered by this request.

The program on the shared computation resources calls `waitlaunch` to get the launcher. `waitlaunch` blocks the task until one master triggers the task.

The source and destination addresses are the same among `launch` and `waitlaunch`.

**APIs for CPU**

```cpp
syscall_return_t launch(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y);
syscall_return_t waitLaunch(int64_t __dst_x, int64_t __dst_y, int64_t* __src_x, int64_t* __src_y);
```

**APIs for CUDA**

```cpp
cudaError_t launch(int __dst_x, int __dst_y, int __src_x, int __src_y);
cudaError_t waitLaunch(int __dst_x, int __dst_y, int* __src_x, int* __src_y);
```

**Arguments**

- `__dst_x` and `__dst_y` specify the destination address.
- `__src_x` and `__src_y` specify the source address.

**Return value**

- `waitLaunch` returns the source of the launch command through `__src_x` and `__src_y`.

- APIs for CPU return the result of the operation.
    - 0 means the transmission operation succeeds.
    - 1 means the transmission operation fails.
- APIs for GPU return one value of `cudaError_t`.

> TODO: A more flexible way to specify the source and the destination address.

**Example**

```C++
//
// master (0,1)
//
...
// Launch task on the slave.
launch(0, 0, 0, 1);
// Send data to the slave.
sendMessage(0, 0, 0, 1, src_data, 1024);
// Wait and receive data from the slave.
receiveMessage(0, 0, 0, 1, dst_data, 8);
...

//
// slave (0, 0)
//
...
// Wait launcher
int64_t src_x = -1, src_y = -1;
waitlaunch(0, 0, &src_x, &src_y);
// Receive data from the master.
receiveMessage(0, 0, 0, 1, src_data, 1024);
// Run task.
...
// Send data to the master.
sendMessage(0, 0, 0, 1, dst_data, 8);
...
```

## API Declaration and Implementation

### APIs for CPU

The declaration of APIs for CPUs is provided in [`$SIMULATOR_ROOT/interchiplet/includes/apis_c.h`](../../apiProject1/apis__c_8h.md). The implementation of these APIs is compiled into a static library `$SIMULATOR_ROOT/interchiplet/lib/libinterchiplet_c.a`, which should be linked to the benchmark.

APIs on CPUs are implemented by system calls. System Calls can have a number of arguments. The mapping between APIs and system calls is listed below:

| API              | System Call ID         |
| ---------------- | ---------------------- |
| `launch`         | `SYSCALL_LAUNCH`       |
| `waitLaunch`     | `SYSCALL_WAITLAUNCH`   |
| `lock`           | `SYSCALL_LOCK`         |
| `unlock`         | `SYSCALL_UNLOCK`       |
| `barrier`        | `SYSCALL_BARRIER`      |
| `sendMessage`    | `SYSCALL_REMOTE_WRITE` |
| `receiveMessage` | `SYSCALL_REMOTE_READ`  |

### APIs for CUDA

APIs on CUDA platforms are implemented by built-in CUDA APIs. The declaration of these APIs is provided in [`$SIMULATOR_ROOT/interchiplet/includes/apis_cu.h`](../../apiProject1/apis__cu_8h.md). The implementation of these APIs is provided by the CUDA simulator, like GPGPU-Sim. Hence, when compiling executable files for CUDA platforms, the specified CUDA library should be provided as below:

```makefile
# CUDA language target
CUDA_target: $(CUDA_OBJS)
	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) --cudart shared $(CUDA_OBJS) -o $(CUDA_TARGET)
```

> TODO: A more flexible way to specify the source and the destination address.
